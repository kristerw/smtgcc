// This file contains a simple optimization pass that tracks how many
// leading and trailing zeros each SSA variable has. The main usage of
// this pass is to remove redundant UB checks generated from GIMPLE
// range information.
#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

class Vrp
{
  std::map<Inst *, uint32_t> leading_zeros_map;
  std::map<Inst *, uint32_t> trailing_zeros_map;
  Function *func;

  uint32_t leading_zeros(Inst *inst);
  uint32_t trailing_zeros(Inst *inst);
  void handle_add(Inst *inst);
  void handle_and(Inst *inst);
  void handle_ashr(Inst *inst);
  void handle_extract(Inst *inst);
  void handle_concat(Inst *inst);
  void handle_lshr(Inst *inst);
  void handle_memory(Inst *inst);
  void handle_mov(Inst *inst);
  void handle_mul(Inst *inst);
  void handle_neg(Inst *inst);
  void handle_or(Inst *inst);
  void handle_phi(Inst *inst);
  void handle_sadd_wraps(Inst *inst);
  void handle_sext(Inst *inst);
  void handle_sgt(Inst *inst);
  void handle_shl(Inst *inst);
  void handle_ssub_wraps(Inst *inst);
  void handle_ugt(Inst *inst);
  void handle_value(Inst *inst);
  void handle_xor(Inst *inst);
  void handle_zext(Inst *inst);
  void handle_inst(Inst *inst);

  public:
  Vrp(Function *func);
};

uint32_t Vrp::leading_zeros(Inst *inst)
{
  auto I = leading_zeros_map.find(inst);
  if (I != leading_zeros_map.end())
    return I->second;
  return 0;
}

uint32_t Vrp::trailing_zeros(Inst *inst)
{
  auto I = trailing_zeros_map.find(inst);
  if (I != trailing_zeros_map.end())
    return I->second;
  return 0;
}

void Vrp::handle_add(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  uint32_t lz = std::min(leading_zeros(arg1), leading_zeros(arg2));
  uint32_t tz = std::min(trailing_zeros(arg1), trailing_zeros(arg2));
  if (tz > 0)
    trailing_zeros_map.insert({inst, tz});
  if (lz > 1)
    leading_zeros_map.insert({inst, lz - 1});
}

void Vrp::handle_and(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  uint32_t lz = std::max(leading_zeros(arg1), leading_zeros(arg2));
  uint32_t tz = std::max(trailing_zeros(arg1), trailing_zeros(arg2));
  if (arg2->op == Op::VALUE)
    {
      unsigned __int128 value = arg2->value();

      unsigned __int128 lz_mask = -1;
      uint32_t lz_shift = (128 - inst->bitsize) + lz;
      assert(lz_shift <= 128);
      lz_mask = (lz_shift < 128) ? lz_mask >> lz_shift : 0;
      value = value & lz_mask;

      unsigned __int128 tz_mask = -1;
      uint32_t tz_shift = tz;
      assert(tz_shift <= 128);
      tz_mask = (tz_shift < 128) ? tz_mask << tz_shift : 0;
      value = value & tz_mask;

      if (value != arg2->value())
	{
	  Inst *v = inst->bb->value_inst(value, inst->bitsize);
	  handle_inst(v);
	  Inst *new_inst = create_inst(Op::AND, arg1, v);
	  new_inst->insert_before(inst);
	  inst->replace_all_uses_with(new_inst);
	  handle_inst(new_inst);
	  return;
	}
    }

  if (tz > 0)
    trailing_zeros_map.insert({inst, tz});
  if (lz > 0)
    leading_zeros_map.insert({inst, lz});
}

void Vrp::handle_ashr(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  uint32_t lz = leading_zeros(arg1);
  uint32_t tz = trailing_zeros(arg1);

  if (lz > 0)
    {
      Inst *lshr = create_inst(Op::LSHR, arg1, arg2);
      lshr->insert_before(inst);
      inst->replace_all_uses_with(lshr);
      handle_inst(lshr);
      return;
    }

  if (arg2->op == Op::VALUE)
    {
      uint32_t shift = arg2->value();
      if (tz > shift)
	trailing_zeros_map.insert({inst, tz - shift});
    }
}

void Vrp::handle_extract(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  uint32_t hi = inst->args[1]->value();
  uint32_t lo = inst->args[2]->value();

  uint32_t lz = leading_zeros(arg1);
  uint32_t tz = trailing_zeros(arg1);

  if (hi < tz || lo > (arg1->bitsize - lz))
    {
      Inst *zero = inst->bb->value_inst(0, inst->bitsize);
      handle_inst(zero);
      inst->replace_all_uses_with(zero);
      return;
    }

  if (lo < tz)
    trailing_zeros_map.insert({inst, tz - lo});
  if (hi >= arg1->bitsize - lz)
    leading_zeros_map.insert({inst, hi - (arg1->bitsize - 1 - lz)});
}

void Vrp::handle_concat(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  if (uint32_t lz = leading_zeros(arg1); lz > 0)
    {
      if (lz == arg1->bitsize)
	lz += leading_zeros(arg2);
      leading_zeros_map.insert({inst, lz});
    }

  if (uint32_t tz = trailing_zeros(arg2); tz > 0)
    {
      if (tz == arg2->bitsize)
	tz += trailing_zeros(arg1);
      trailing_zeros_map.insert({inst, tz});
    }
}

void Vrp::handle_lshr(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  uint32_t lz = leading_zeros(arg1);
  uint32_t tz = trailing_zeros(arg1);

  if (arg2->op == Op::VALUE)
    {
      uint32_t shift = arg2->value();
      if (lz + shift > 0)
	leading_zeros_map.insert({inst, std::min(lz + shift, inst->bitsize)});
      if (tz > shift)
	trailing_zeros_map.insert({inst, tz - shift});
    }
}

void Vrp::handle_memory(Inst *inst)
{
  if (inst->bb->func->module->ptr_offset_low == 0)
    trailing_zeros_map.insert({inst, inst->bb->func->module->ptr_offset_bits});
}

void Vrp::handle_mov(Inst *inst)
{
  Inst *const arg1 = inst->args[0];

  if (uint32_t tz = trailing_zeros(arg1); tz > 0)
    trailing_zeros_map.insert({inst, tz});
  if (uint32_t lz = leading_zeros(arg1); lz > 0)
    leading_zeros_map.insert({inst, lz});
}

void Vrp::handle_mul(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  uint32_t tz = trailing_zeros(arg1) + trailing_zeros(arg2);
  if (tz > 0)
    trailing_zeros_map.insert({inst, tz});
}

void Vrp::handle_neg(Inst *inst)
{
  Inst *const arg1 = inst->args[0];

  if (uint32_t tz = trailing_zeros(arg1); tz > 0)
    trailing_zeros_map.insert({inst, tz});
}

void Vrp::handle_or(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  uint32_t lz = std::min(leading_zeros(arg1), leading_zeros(arg2));
  uint32_t tz = std::min(trailing_zeros(arg1), trailing_zeros(arg2));
  if (tz > 0)
    trailing_zeros_map.insert({inst, tz});
  if (lz > 0)
    leading_zeros_map.insert({inst, lz});
}

void Vrp::handle_phi(Inst *inst)
{
  assert(!inst->phi_args.empty());
  bool all_has_lz = true;
  bool all_has_tz = true;
  for (auto& phi_arg : inst->phi_args)
    {
      all_has_lz = all_has_lz && leading_zeros(phi_arg.inst) > 0;
      all_has_tz = all_has_tz && trailing_zeros(phi_arg.inst) > 0;
    }

  if (all_has_lz)
    {
      uint32_t lz = inst->bitsize;
      for (auto& phi_arg : inst->phi_args)
	{
	  lz = std::min(lz, leading_zeros(phi_arg.inst));
	}
      if (lz > 0)
	leading_zeros_map.insert({inst, lz});
    }

  if (all_has_tz)
    {
      uint32_t tz = inst->bitsize;
      for (auto& phi_arg : inst->phi_args)
	{
	  tz = std::min(tz, trailing_zeros(phi_arg.inst));
	}
      if (tz > 0)
	trailing_zeros_map.insert({inst, tz});
    }
}

void Vrp::handle_sadd_wraps(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  if (leading_zeros(arg1) > 1 && leading_zeros(arg2) > 1)
    {
      Inst *zero = inst->bb->value_inst(0, 1);
      handle_inst(zero);
      inst->replace_all_uses_with(zero);
      return;
    }
}

void Vrp::handle_sext(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  if (leading_zeros(arg1) > 0)
    {
      Inst *zext = create_inst(Op::ZEXT, arg1, arg2);
      zext->insert_before(inst);
      inst->replace_all_uses_with(zext);
      handle_inst(zext);
      return;
    }

  if (uint32_t tz = trailing_zeros(arg1); tz > 0)
    trailing_zeros_map.insert({inst, tz});
}

void Vrp::handle_sgt(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // sgt 0, x -> 0 if x has leading zeros.
  if (is_value_zero(arg1) && leading_zeros(arg2) > 0)
    {
      Inst *zero = inst->bb->value_inst(0, 1);
      handle_inst(zero);
      inst->replace_all_uses_with(zero);
      return;
    }

  // sgt x, c -> false if c >= the maximum value of x (that is, the value
  // where all bits not being leading or trailing zeros are set to 1).
  if (arg2->op == Op::VALUE && leading_zeros(arg1) > 0)
    {
      unsigned __int128 lz_mask = -1;
      uint32_t lz_shift = ((128 - arg1->bitsize) + leading_zeros(arg1));
      assert(lz_shift <= 128);
      lz_mask = (lz_shift < 128) ? lz_mask >> lz_shift : 0;

      unsigned __int128 tz_mask = -1;
      uint32_t tz_shift = trailing_zeros(arg1);
      assert(tz_shift <= 128);
      tz_mask = (tz_shift < 128) ? tz_mask << tz_shift : 0;

      __int128 max_value = lz_mask & tz_mask;
      if (arg2->signed_value() >= max_value)
	{
	  Inst *zero = inst->bb->value_inst(0, 1);
	  handle_inst(zero);
	  inst->replace_all_uses_with(zero);
	  return;
	}
    }
}

void Vrp::handle_shl(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  uint32_t lz = leading_zeros(arg1);
  uint32_t tz = trailing_zeros(arg1);

  if (arg2->op == Op::VALUE)
    {
      uint32_t shift = arg2->value();
      if (lz > shift)
	leading_zeros_map.insert({inst, lz - shift});
      if (tz + shift > 0)
	trailing_zeros_map.insert({inst, std::min(tz + shift, inst->bitsize)});
    }
}

void Vrp::handle_ssub_wraps(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  if (leading_zeros(arg1) > 1 && leading_zeros(arg2) > 1)
    {
      Inst *zero = inst->bb->value_inst(0, 1);
      handle_inst(zero);
      inst->replace_all_uses_with(zero);
      return;
    }
}

void Vrp::handle_ugt(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // ugt x, c -> false if c >= the maximum value of x (that is, the value
  // where all bits not being leading or trailing zeros are set to 1).
  if (arg2->op == Op::VALUE)
    {
      unsigned __int128 lz_mask = -1;
      uint32_t lz_shift = (128 - arg1->bitsize) + leading_zeros(arg1);
      assert(lz_shift <= 128);
      lz_mask = (lz_shift < 128) ? lz_mask >> lz_shift : 0;

      unsigned __int128 tz_mask = -1;
      uint32_t tz_shift = trailing_zeros(arg1);
      assert(tz_shift <= 128);
      tz_mask = (tz_shift < 128) ? tz_mask << tz_shift : 0;

      unsigned __int128 max_value = lz_mask & tz_mask;
      if (arg2->value() >= max_value)
	{
	  Inst *zero = inst->bb->value_inst(0, 1);
	  handle_inst(zero);
	  inst->replace_all_uses_with(zero);
	  return;
	}
    }
}

void Vrp::handle_value(Inst *inst)
{
  uint64_t lz = clz(inst->value()) - (128 - inst->bitsize);
  if (lz > 0)
    leading_zeros_map.insert({inst, lz});
  uint64_t tz = std::min(ctz(inst->value()), inst->bitsize);
  if (tz > 0)
    trailing_zeros_map.insert({inst, tz});
}

void Vrp::handle_xor(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  uint32_t lz = std::min(leading_zeros(arg1), leading_zeros(arg2));
  uint32_t tz = std::min(trailing_zeros(arg1), trailing_zeros(arg2));
  if (tz > 0)
    trailing_zeros_map.insert({inst, tz});
  if (lz > 0)
    leading_zeros_map.insert({inst, lz});
}

void Vrp::handle_zext(Inst *inst)
{
  Inst *const arg1 = inst->args[0];

  if (uint32_t tz = trailing_zeros(arg1); tz > 0)
    trailing_zeros_map.insert({inst, tz});

  uint32_t lz = leading_zeros(arg1) + (inst->bitsize - arg1->bitsize);
  leading_zeros_map.insert({inst, lz});
}

void Vrp::handle_inst(Inst *inst)
{
  if (!inst->has_lhs())
    return;

  switch (inst->op)
    {
    case Op::ADD:
      handle_add(inst);
      break;
    case Op::AND:
      handle_and(inst);
      break;
    case Op::ASHR:
      handle_ashr(inst);
      break;
    case Op::EXTRACT:
      handle_extract(inst);
      break;
    case Op::CONCAT:
      handle_concat(inst);
      break;
    case Op::LSHR:
      handle_lshr(inst);
      break;
    case Op::MEMORY:
      handle_memory(inst);
      break;
    case Op::MOV:
      handle_mov(inst);
      break;
    case Op::MUL:
      handle_mul(inst);
      break;
    case Op::NEG:
      handle_neg(inst);
      break;
    case Op::OR:
      handle_or(inst);
      break;
    case Op::PHI:
      handle_phi(inst);
      break;
    case Op::SADD_WRAPS:
      handle_sadd_wraps(inst);
      break;
    case Op::SEXT:
      handle_sext(inst);
      break;
    case Op::SGT:
      handle_sgt(inst);
      break;
    case Op::SHL:
      handle_shl(inst);
      break;
    case Op::SSUB_WRAPS:
      handle_ssub_wraps(inst);
      break;
    case Op::UGT:
      handle_ugt(inst);
      break;
    case Op::VALUE:
      handle_value(inst);
      break;
    case Op::XOR:
      handle_xor(inst);
      break;
    case Op::ZEXT:
      handle_zext(inst);
      break;
    default:
      break;
    }

  // Replace the value with 0 if the leading/trailing zeros cover the
  // full bitwidth.
  if (inst->op != Op::VALUE
      && leading_zeros(inst) + trailing_zeros(inst) >= inst->bitsize)
    {
      Inst *zero = inst->bb->value_inst(0, inst->bitsize);
      handle_inst(zero);
      inst->replace_all_uses_with(zero);
    }
}

Vrp::Vrp(Function *func) : func{func}
{
  assert(!has_loops(func));
  for (Basic_block *bb : func->bbs)
    {
      for (auto phi : bb->phis)
	{
	  handle_inst(phi);
	}
      for (Inst *inst = bb->first_inst; inst; inst = inst->next)
	{
	  handle_inst(inst);
	}
    }
}

} // end anonymous namespace

void vrp(Function *func)
{
  Vrp vrp_pass(func);
}

void vrp(Module *module)
{
  for (auto func : module->functions)
    vrp(func);
}

} // end namespace smtgcc
