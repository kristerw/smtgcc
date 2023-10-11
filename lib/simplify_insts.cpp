// This file contains peephole optimizations and constant folding.
// We do not want to optimize "everything" in this optimization pass as
// that risks introducing new bugs/hiding GCC bugs. Instead, we aim to
// just eliminate common cases where our translations from GIMPLE introduce
// lots of extra instructions. For example, the UB checks for constant
// shift amount, or constant pointer arithmetic.

#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

Instruction *simplify_inst(Instruction *inst);

bool is_value_zero(Instruction *inst)
{
  return inst->opcode == Op::VALUE && inst->value() == 0;
}

bool is_value_one(Instruction *inst)
{
  return inst->opcode == Op::VALUE && inst->value() == 1;
}

bool is_value_signed_min(Instruction *inst)
{
  unsigned __int128 smin = ((unsigned __int128)1) << (inst->bitsize - 1);
  return inst->opcode == Op::VALUE && inst->value() == smin;
}

bool is_value_signed_max(Instruction *inst)
{
  unsigned __int128 smax = (((unsigned __int128)1) << (inst->bitsize - 1)) - 1;
  return inst->opcode == Op::VALUE && inst->value() == smax;
}

bool is_value_m1(Instruction *inst)
{
  unsigned __int128 m1 = ~((unsigned __int128)0);
  m1 = (m1 << (128 - inst->bitsize)) >> (128 - inst->bitsize);
  return inst->opcode == Op::VALUE && inst->value() == m1;
}

Instruction *cfold_add(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val + arg2_val, inst->bitsize);
}

Instruction *cfold_sub(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val - arg2_val, inst->bitsize);
}

Instruction *cfold_and(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val & arg2_val, inst->bitsize);
}

Instruction *cfold_or(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val | arg2_val, inst->bitsize);
}

Instruction *cfold_xor(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val ^ arg2_val, inst->bitsize);
}

Instruction *cfold_concat(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  unsigned __int128 val = (arg1_val << inst->arguments[1]->bitsize) | arg2_val;
  return inst->bb->value_inst(val, inst->bitsize);
}

Instruction *cfold_mul(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val * arg2_val, inst->bitsize);
}

Instruction *cfold_extract(Instruction *inst)
{
  unsigned __int128 arg_val = inst->arguments[0]->value();
  arg_val >>= inst->arguments[2]->value();
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Instruction *cfold_neg(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  return inst->bb->value_inst(-arg1_val, inst->bitsize);
}

Instruction *cfold_not(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  return inst->bb->value_inst(~arg1_val, inst->bitsize);
}

Instruction *cfold_ne(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val != arg2_val, 1);
}

Instruction *cfold_eq(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val == arg2_val, 1);
}

Instruction *cfold_uge(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val >= arg2_val, 1);
}

Instruction *cfold_ugt(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val > arg2_val, 1);
}

Instruction *cfold_ule(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val <= arg2_val, 1);
}

Instruction *cfold_ult(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val < arg2_val, 1);
}

Instruction *cfold_sge(Instruction *inst)
{
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  __int128 arg1_val = inst->arguments[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->arguments[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val >= arg2_val, 1);
}

Instruction *cfold_sgt(Instruction *inst)
{
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  __int128 arg1_val = inst->arguments[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->arguments[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val > arg2_val, 1);
}

Instruction *cfold_sle(Instruction *inst)
{
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  __int128 arg1_val = inst->arguments[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->arguments[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val <= arg2_val, 1);
}

Instruction *cfold_slt(Instruction *inst)
{
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  __int128 arg1_val = inst->arguments[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->arguments[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val < arg2_val, 1);
}

Instruction *cfold_zext(Instruction *inst)
{
  unsigned __int128 arg_val = inst->arguments[0]->value();
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Instruction *cfold_sext(Instruction *inst)
{
  __int128 arg_val = inst->arguments[0]->value();
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  arg_val <<= shift;
  arg_val >>= shift;
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Instruction *constant_fold_inst(Instruction *inst)
{
  for (uint64_t i = 0; i < inst->nof_args; i++)
    {
      Instruction *arg = inst->arguments[i];
      if (arg->opcode != Op::VALUE)
	return inst;
      if (inst->bitsize > 128)
	return inst;
    }

  switch (inst->opcode)
    {
    case Op::ADD:
      inst = cfold_add(inst);
      break;
    case Op::SUB:
      inst = cfold_sub(inst);
      break;
    case Op::AND:
      inst = cfold_and(inst);
      break;
    case Op::OR:
      inst = cfold_or(inst);
      break;
    case Op::XOR:
      inst = cfold_xor(inst);
      break;
    case Op::CONCAT:
      inst = cfold_concat(inst);
      break;
    case Op::EXTRACT:
      inst = cfold_extract(inst);
      break;
    case Op::MUL:
      inst = cfold_mul(inst);
      break;
    case Op::NE:
      inst = cfold_ne(inst);
      break;
    case Op::EQ:
      inst = cfold_eq(inst);
      break;
    case Op::NEG:
      inst = cfold_neg(inst);
      break;
    case Op::NOT:
      inst = cfold_not(inst);
      break;
    case Op::SEXT:
      inst = cfold_sext(inst);
      break;
    case Op::SGE:
      inst = cfold_sge(inst);
      break;
    case Op::SGT:
      inst = cfold_sgt(inst);
      break;
    case Op::SLE:
      inst = cfold_sle(inst);
      break;
    case Op::SLT:
      inst = cfold_slt(inst);
      break;
    case Op::UGE:
      inst = cfold_uge(inst);
      break;
    case Op::UGT:
      inst = cfold_ugt(inst);
      break;
    case Op::ULE:
      inst = cfold_ule(inst);
      break;
    case Op::ULT:
      inst = cfold_ult(inst);
      break;
    case Op::ZEXT:
      inst = cfold_zext(inst);
      break;
    default:
      break;
    }

  return inst;
}

Instruction *simplify_mem_size(Instruction *inst, const std::map<uint64_t,uint64_t>& id2size)
{
  if (inst->arguments[0]->opcode == Op::VALUE)
    {
      uint64_t id = inst->arguments[0]->value();
      auto it = id2size.find(id);
      if (it != id2size.end())
	return inst->bb->value_inst(it->second, inst->bitsize);
    }
  return inst;
}

Instruction *simplify_add(Instruction *inst)
{
  // add 0, x -> x
  if (is_value_zero(inst->arguments[0]))
    return inst->arguments[1];

  // add x, 0 -> x
  if (is_value_zero(inst->arguments[1]))
    return inst->arguments[0];

  return inst;
}

Instruction *simplify_and(Instruction *inst)
{
  Instruction *arg1 = inst->arguments[0];
  Instruction *arg2 = inst->arguments[1];
  if (arg2->opcode != Op::VALUE)
    std::swap(arg1, arg2);

  // and x, 0 -> 0
  if (is_value_zero(arg2))
    return arg2;

  // and x, -1 -> x
  if (is_value_m1(arg2))
    return arg1;

  // and (zext x, s), c -> 0 if x is Boolean and c is a constant with least
  // significant bit 0.
  // This is rather common in UB checks of range information where a Boolean
  // has been extended to an integer.
  if (arg1->opcode == Op::ZEXT && arg1->arguments[0]->bitsize == 1 &&
      arg2->opcode == Op::VALUE && (arg2->value() & 1) == 0)
    return inst->bb->value_inst(0, inst->bitsize);

  // Optimize UB check for signed Boolean to false when it is obvious it
  // is not UB.
  //   %3 = sext %1, %2
  //   %6 = ne -1, %3
  //   %7 = ne 0, %3
  //   %8 = and %6, %7
  if (arg1->opcode == Op::NE && arg2->opcode == Op::NE)
    {
      Instruction *arg1_arg1 = arg1->arguments[0];
      Instruction *arg1_arg2 = arg1->arguments[1];
      if (arg1_arg2->opcode != Op::VALUE)
	std::swap(arg1_arg1, arg1_arg2);
      Instruction *arg2_arg1 = arg2->arguments[0];
      Instruction *arg2_arg2 = arg2->arguments[1];
      if (arg2_arg2->opcode != Op::VALUE)
	std::swap(arg2_arg1, arg2_arg2);
      if (arg1_arg1 == arg2_arg1
	  && arg1_arg1->opcode == Op::SEXT
	  && arg1_arg1->arguments[0]->bitsize == 1
	  && ((is_value_zero(arg1_arg2) && is_value_m1(arg2_arg2))
	      || (is_value_zero(arg2_arg2) && is_value_m1(arg1_arg2))))
	return inst->bb->value_inst(0, 1);
    }

  return inst;
}

Instruction *simplify_eq(Instruction *inst)
{
  Instruction *arg1 = inst->arguments[0];
  Instruction *arg2 = inst->arguments[1];
  if (arg2->opcode != Op::VALUE)
    std::swap(arg1, arg2);

  // Comparing an MININT with a ZEXT/SEXT value is always false.
  // This is common when code is negating a promoted char/short.
  if (arg2->opcode == Op::VALUE && is_value_signed_min(arg2))
    {
      if (arg1->opcode == Op::ZEXT || arg1->opcode == Op::SEXT)
	{
	  assert(arg1->bitsize > arg1->arguments[0]->bitsize);
	  return inst->bb->value_inst(0, 1);
	}
    }

  // Boolean x == 1 -> x
  if (arg1->bitsize == 1 && is_value_one(arg2))
    return arg1;

  // x == x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Instruction *simplify_ne(Instruction *inst)
{
  Instruction *arg1 = inst->arguments[0];
  Instruction *arg2 = inst->arguments[1];
  if (arg2->opcode != Op::VALUE)
    std::swap(arg1, arg2);

  // Boolean x != 0 -> x
  if (arg1->bitsize == 1 && is_value_zero(arg2))
    return arg1;

  // x != x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Instruction *simplify_ashr(Instruction *inst)
{
  // ashr (shl x y), y -> sext x
  if (inst->arguments[0]->opcode == Op::SHL
      && inst->arguments[1]->opcode == Op::VALUE
      && inst->arguments[1] == inst->arguments[0]->arguments[1])
    {
      uint64_t value = inst->arguments[1]->value();
      // simplify_shl have eliminated the trivial cases.
      assert(value > 0 && value < inst->bitsize);
      Instruction *shl = inst->arguments[0];
      Instruction *high = inst->bb->value_inst(inst->bitsize - value - 1, 32);
      Instruction *low = inst->bb->value_inst(0, 32);
      Instruction *new_inst1 =
	create_inst(Op::EXTRACT, shl->arguments[0], high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *bitsize_inst = inst->bb->value_inst(inst->bitsize, 32);
      Instruction *new_inst2 =
	create_inst(Op::SEXT, new_inst1, bitsize_inst);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Instruction *simplify_lshr(Instruction *inst)
{
  // lshr (shl x y), y -> zext x
  if (inst->arguments[0]->opcode == Op::SHL
      && inst->arguments[1]->opcode == Op::VALUE
      && inst->arguments[1] == inst->arguments[0]->arguments[1])
    {
      uint64_t value = inst->arguments[1]->value();
      // simplify_shl have eliminated the trivial cases.
      assert(value > 0 && value < inst->bitsize);
      Instruction *shl = inst->arguments[0];
      Instruction *high = inst->bb->value_inst(inst->bitsize - value - 1, 32);
      Instruction *low = inst->bb->value_inst(0, 32);
      Instruction *new_inst1 =
	create_inst(Op::EXTRACT, shl->arguments[0], high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *bitsize_inst = inst->bb->value_inst(inst->bitsize, 32);
      Instruction *new_inst2 = create_inst(Op::ZEXT, new_inst1, bitsize_inst);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // lshr x, y -> 0 if y is larger than bitsize
  if (inst->arguments[1]->opcode == Op::VALUE &&
      inst->arguments[1]->value() >= inst->bitsize)
    return inst->bb->value_inst(0, inst->bitsize);

  return inst;
}

Instruction *simplify_or(Instruction *inst)
{
  // or 0, x -> x
  if (is_value_zero(inst->arguments[0]))
    return inst->arguments[1];

  // or x, 0 -> x
  if (is_value_zero(inst->arguments[1]))
    return inst->arguments[0];

  // or -1, x -> -1
  if (is_value_m1(inst->arguments[0]))
    return inst->arguments[0];

  // or x, -1 -> -1
  if (is_value_m1(inst->arguments[1]))
    return inst->arguments[1];

  return inst;
}

Instruction *simplify_mov(Instruction *inst)
{
  return inst->arguments[0];
}

Instruction *simplify_mul(Instruction *inst)
{
  // mul 0, x -> 0
  if (is_value_zero(inst->arguments[0]))
    return inst->arguments[0];

  // mul x, 0 -> 0
  if (is_value_zero(inst->arguments[1]))
    return inst->arguments[1];

  // mul 1, x -> x
  if (is_value_one(inst->arguments[0]))
    return inst->arguments[1];

  // mul x, 1 -> x
  if (is_value_one(inst->arguments[1]))
    return inst->arguments[0];

  return inst;
}

// Helper function for simplify_sadd_wraps and simplify_ssub_wraps.
// Check if inst is a ZEXT/SEXT extending more than one bit, or a
// VALUE that is extended more than one bit.
bool is_ext(Instruction *inst)
{
  if (inst->opcode == Op::ZEXT || inst->opcode == Op::SEXT)
    {
      if (inst->arguments[0]->bitsize < inst->bitsize - 1)
	return true;
    }

  if (inst->opcode == Op::VALUE)
    {
      if (inst->bitsize >= 3)
	{
	  unsigned __int128 top_bits = inst->value() >> (inst->bitsize - 2);
	  if (top_bits == 0 || top_bits == 3)
	    return true;
	}
    }

  return false;
}

Instruction *simplift_sgt(Instruction *inst)
{
  // sgt signed_min_val, x -> false
  if (is_value_signed_min(inst->arguments[0]))
    return inst->bb->value_inst(0, 1);

  // sgt x, signed_max_val -> false
  if (is_value_signed_max(inst->arguments[1]))
    return inst->bb->value_inst(0, 1);

  return inst;
}

Instruction *simplify_sadd_wraps(Instruction *inst)
{
  // sadd_wraps 0, x -> false
  if (is_value_zero(inst->arguments[0]))
    return inst->bb->value_inst(0, 1);

  // sadd_wraps x, 0 -> false
  if (is_value_zero(inst->arguments[1]))
    return inst->bb->value_inst(0, 1);

  // sadd_wraps x, y is always false if x and y are zext/sext that expand
  // more than one bit, or constant that could have been extended in that
  // way. This is a common case for e.g. char/short arithmetic that is
  // promoted to int.
  if (is_ext(inst->arguments[0]) && is_ext(inst->arguments[1]))
    return inst->bb->value_inst(0, 1);

  return inst;
 }

Instruction *simplify_ssub_wraps(Instruction *inst)
{
  // ssub_wraps 0, x -> false
  if (is_value_zero(inst->arguments[0]))
    return inst->bb->value_inst(0, 1);

  // ssub_wraps x, 0 -> false
  if (is_value_zero(inst->arguments[1]))
    return inst->bb->value_inst(0, 1);

  // ssub_wraps x, y is always false if x and y are zext/sext that expand
  // more than one bit, or constant that could have been extended in that
  // way. This is a common case for e.g. char/short arithmetic that is
  // promoted to int.
  if (is_ext(inst->arguments[0]) && is_ext(inst->arguments[1]))
    return inst->bb->value_inst(0, 1);

  return inst;
}

Instruction *simplify_ite(Instruction *inst)
{
  // ite 0, a, b -> b
  if (is_value_zero(inst->arguments[0]))
    return inst->arguments[2];

  // ite 1, a, b -> a
  if (is_value_one(inst->arguments[0]))
    return inst->arguments[1];

  // ite a, b, b -> b
  if (inst->arguments[1] == inst->arguments[2])
    return inst->arguments[1];

  // ite a, 1, 0 -> zext a
  if (is_value_one(inst->arguments[1]) && is_value_zero(inst->arguments[2]))
    {
      Instruction *bs = inst->bb->value_inst(inst->bitsize, 32);
      Instruction *new_inst = create_inst(Op::ZEXT, inst->arguments[0], bs);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Instruction *simplify_ugt(Instruction *inst)
{
  Instruction *arg1 = inst->arguments[0];
  Instruction *arg2 = inst->arguments[1];

  // ugt 0, x -> false
  if (is_value_zero(arg1))
    return inst->bb->value_inst(0, 1);

  // ugt x, -1 -> false
  if (is_value_m1(arg2))
    return inst->bb->value_inst(0, 1);

  // ugt (zext x, s), c -> false if x is Boolean and c is a constant > 0.
  // This is rather common in UB checks of range information where a Boolean
  // has been extended to an integer.
  if (arg1->opcode == Op::ZEXT && arg1->arguments[0]->bitsize == 1 &&
      arg2->opcode == Op::VALUE && arg2->value() > 0)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Instruction *simplify_shl(Instruction *inst)
{
  // shl x, 0 -> x
  if (is_value_zero(inst->arguments[1]))
    return inst->arguments[0];

  // shl x, c -> 0 if c >= bitsize
  if (inst->arguments[1]->opcode == Op::VALUE
      && inst->arguments[1]->value() >= inst->bitsize)
    return inst->bb->value_inst(0, inst->bitsize);

  return inst;
}

Instruction *simplify_memory(Instruction *inst)
{
  uint64_t id = inst->arguments[0]->value();
  uint64_t addr = id << inst->bb->func->module->ptr_id_low;
  return inst->bb->value_inst(addr, inst->bb->func->module->ptr_bits);
}

Instruction *simplify_phi(Instruction *phi)
{
  // If phi only references itself or one other value it can be replaced by
  // that value, e.g. %2 = phi [ %1, .1] [ %2, .2] [%1, .3]

  Instruction *inst = nullptr;
  for (auto phi_arg : phi->phi_args)
    {
      if (phi_arg.inst != phi)
	{
	  if (!inst)
	    inst = phi_arg.inst;
	  else if (phi_arg.inst != inst)
	    return phi;
	}
    }

  return inst;
}

void flatten_concat(Instruction *inst, std::vector<Instruction *>& elems)
{
  assert(inst->opcode == Op::CONCAT);
  if (inst->arguments[1]->opcode == Op::CONCAT)
    flatten_concat(inst->arguments[1], elems);
  else
    elems.push_back(inst->arguments[1]);
  if (inst->arguments[0]->opcode == Op::CONCAT)
    flatten_concat(inst->arguments[0], elems);
  else
    elems.push_back(inst->arguments[0]);
}

Instruction *simplify_extract(Instruction *inst)
{
  Instruction *arg = inst->arguments[0];
  uint32_t high_val = inst->arguments[1]->value();
  uint32_t low_val = inst->arguments[2]->value();

  // extract x -> x if the range completely cover x.
  if (low_val == 0 && high_val == arg->bitsize - 1)
    return arg;

  // "extract (sext x)" and "extract (zext x)" is changed to "extract x" if
  // the range only access bits from x.
  if (arg->opcode == Op::SEXT || arg->opcode == Op::ZEXT)
    {
      Instruction *ext_arg = arg->arguments[0];
      if (low_val == 0 && high_val == ext_arg->bitsize - 1)
	return ext_arg;
      if (high_val < ext_arg->bitsize)
	{
	  Instruction *high = inst->bb->value_inst(high_val, 32);
	  Instruction *low = inst->bb->value_inst(low_val, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, ext_arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // "extract (sext x)" is changed to "sext x" with a smaller extension if
  // the extraction extract from the low part (and similarly for zext).
  if ((arg->opcode == Op::SEXT || arg->opcode == Op::ZEXT) && low_val == 0)
    {
      Instruction *bitsize = arg->bb->value_inst(high_val + 1, 32);
      Instruction *new_inst =
	create_inst(arg->opcode, arg->arguments[0], bitsize);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // "extract (extract x)" is changed to "extract x".
  if (arg->opcode == Op::EXTRACT)
    {
      uint32_t arg_low_val = arg->arguments[2]->value();
      Instruction *high = inst->bb->value_inst(high_val + arg_low_val, 32);
      Instruction *low = inst->bb->value_inst(low_val + arg_low_val, 32);
      Instruction *new_inst =
	create_inst(Op::EXTRACT, arg->arguments[0], high, low);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // "extract (concat x, y)" is changed to "extract x" or "extract y" if the
  // range only access bits from one of the arguments.
  if (arg->opcode == Op::CONCAT)
    {
      // We often have chains of concat for loads and for vectors, so iterate
      // to find the final element instead of needing to recursively simplify
      // the new instruction.
      while (arg->opcode == Op::CONCAT)
	{
	  uint32_t low_bitsize = arg->arguments[1]->bitsize;
	  if (high_val < low_bitsize)
	    arg = arg->arguments[1];
	  else if (low_val >= low_bitsize)
	    {
	      high_val -= low_bitsize;
	      low_val -= low_bitsize;
	      arg = arg->arguments[0];
	    }
	  else
	    break;
	}
      if (arg != inst->arguments[0]
	  || high_val != inst->arguments[1]->value()
	  || low_val != inst->arguments[2]->value())
	{
	  if (low_val == 0 && high_val == arg->bitsize - 1)
	    return arg;
	  Instruction *high = inst->bb->value_inst(high_val, 32);
	  Instruction *low = inst->bb->value_inst(low_val, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // We often have chains of concat (for vectors, structures, etc.) and
  // the extract only need a few elements in the middle which is not
  // handled by the previous "extract (concat x, y)" optimization.
  if (arg->opcode == Op::CONCAT)
    {
      std::vector<Instruction *> elems;
      flatten_concat(arg, elems);

      if (low_val >= elems[0]->bitsize
	  || high_val < (arg->bitsize - elems.back()->bitsize))
	{
	  int i;
	  for (i = 0; low_val >= elems[i]->bitsize; i++)
	    {
	      high_val -= elems[i]->bitsize;
	      low_val -= elems[i]->bitsize;
	    }
	  arg = elems[i++];
	  for (; high_val >= arg->bitsize; i++)
	    {
	      arg = create_inst(Op::CONCAT, elems[i], arg);
	      arg->insert_before(inst);
	      arg = simplify_inst(arg);
	    }

	  if (low_val == 0 && high_val == arg->bitsize - 1)
	    return arg;
	  Instruction *high = inst->bb->value_inst(high_val, 32);
	  Instruction *low = inst->bb->value_inst(low_val, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // extract (lshr x, c) -> extract if the shift is constant.
  if (arg->opcode == Op::LSHR && arg->arguments[1]->opcode == Op::VALUE)
    {
      Instruction *lshr_arg = arg->arguments[0];
      uint64_t shift = arg->arguments[1]->value();
      uint64_t valid_bits = arg->bitsize - shift;
      // Larger shift should have been optimized to 0 before we reach this.
      assert(shift < arg->bitsize);
      if (low_val >= valid_bits)
	{
	  // We only extract from the 0-extended part.
	  return inst->bb->value_inst(0, inst->bitsize);
	}
      else if (high_val < valid_bits)
	{
	  // We only extract from the original value.
	  Instruction *high = inst->bb->value_inst(high_val + shift, 32);
	  Instruction *low = inst->bb->value_inst(low_val + shift, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, lshr_arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
      else
	{
	  // We extract the high part, and zero extend to the correct width.
	  Instruction *high = inst->bb->value_inst(lshr_arg->bitsize - 1, 32);
	  Instruction *low = inst->bb->value_inst(low_val + shift, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, lshr_arg, high, low);
	  new_inst->insert_before(inst);
	  new_inst = simplify_inst(new_inst);
	  assert(inst->bitsize > new_inst->bitsize);
	  Instruction *bitsize = inst->bb->value_inst(inst->bitsize, 32);
	  Instruction *new_inst2 = create_inst(Op::ZEXT, new_inst, bitsize);
	  new_inst2->insert_before(inst);
	  return new_inst2;
	}
    }
  return inst;
}

Instruction *simplify_is_const_mem(Instruction *inst, const std::map<uint64_t,Instruction *>& id2mem_inst, bool has_const_mem)
{
  // We know the memory is not const if the function does not have any const
  // memory.
  if (!has_const_mem)
    return inst->bb->value_inst(0, 1);

  if (inst->arguments[0]->opcode == Op::VALUE)
    {
      uint64_t id = inst->arguments[0]->value();
      if (id2mem_inst.contains(id))
	{
	  Instruction *mem_inst = id2mem_inst.at(id);
	  uint32_t flags = mem_inst->arguments[2]->value();
	  bool is_const_mem = (flags & MEM_CONST) != 0;
	  return inst->bb->value_inst(is_const_mem, 1);
	}
    }
  return inst;
}

Instruction *simplify_inst(Instruction *inst)
{
  Instruction *original_inst = inst;

  switch (inst->opcode)
    {
    case Op::ADD:
      inst = simplify_add(inst);
      break;
    case Op::AND:
      inst = simplify_and(inst);
      break;
    case Op::ASHR:
      inst = simplify_ashr(inst);
      break;
    case Op::EQ:
      inst = simplify_eq(inst);
      break;
    case Op::NE:
      inst = simplify_ne(inst);
      break;
    case Op::EXTRACT:
      inst = simplify_extract(inst);
      break;
    case Op::LSHR:
      inst = simplify_lshr(inst);
      break;
    case Op::MOV:
      inst = simplify_mov(inst);
      break;
    case Op::MUL:
      inst = simplify_mul(inst);
      break;
    case Op::OR:
      inst = simplify_or(inst);
      break;
    case Op::SGT:
      inst = simplift_sgt(inst);
      break;
    case Op::SADD_WRAPS:
      inst = simplify_sadd_wraps(inst);
      break;
    case Op::ITE:
      inst = simplify_ite(inst);
      break;
    case Op::SHL:
      inst = simplify_shl(inst);
      break;
    case Op::SSUB_WRAPS:
      inst = simplify_ssub_wraps(inst);
      break;
    case Op::UGT:
      inst = simplify_ugt(inst);
      break;
    default:
      break;
    }

  if (inst != original_inst)
    inst = simplify_inst(inst);
  else
    inst = constant_fold_inst(inst);

  return inst;
}

void destroy(Instruction *inst)
{
  // Memory removal is done in memory-specific passes.
  if (inst->opcode == Op::MEMORY)
    return;

  destroy_instruction(inst);
}

} // end anonymous namespace

void simplify_insts(Function *func)
{
  for (Basic_block *bb : func->bbs)
    {
      for (auto phi : bb->phis)
	{
	  Instruction *res = simplify_phi(phi);
	  if (res != phi)
	    phi->replace_all_uses_with(res);
	}
      for (Instruction *inst = bb->first_inst; inst;)
	{
	  Instruction *next_inst = inst->next;
	  if (inst->has_lhs())
	    {
	      if (!inst->used_by.empty())
		{
		  Instruction *res = simplify_inst(inst);
		  if (res != inst)
		    {
		      inst->replace_all_uses_with(res);
		      destroy(inst);
		    }
		}
	      else
		destroy(inst);
	    }
	  inst = next_inst;
	}
    }
}

void simplify_insts(Module *module)
{
  for (auto func : module->functions)
    simplify_insts(func);
}

void simplify_mem(Function *func)
{
  std::map<uint64_t,Instruction *> id2mem_inst;
  std::map<uint64_t,uint64_t> id2size;
  bool has_const_mem = false;
  for (Instruction *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->opcode == Op::MEMORY)
	{
	  uint64_t id = inst->arguments[0]->value();
	  uint64_t size = inst->arguments[1]->value();
	  uint32_t flags = inst->arguments[2]->value();
	  assert(!id2mem_inst.contains(id));
	  id2mem_inst[id] = inst;
	  id2size[id] = size;

	  has_const_mem |= (flags & MEM_CONST) != 0;
	}
    }

  for (Basic_block *bb : func->bbs)
    {
      for (auto phi : bb->phis)
	{
	  Instruction *res = simplify_phi(phi);
	  if (res != phi)
	    phi->replace_all_uses_with(res);
	}
      for (Instruction *inst = bb->first_inst; inst;)
	{
	  Instruction *next_inst = inst->next;

	  if (inst->has_lhs() && inst->used_by.empty())
	    {
	      destroy(inst);
	      inst = next_inst;
	      continue;
	    }

	  Instruction *res;
	  switch (inst->opcode)
	    {
	    case Op::IS_CONST_MEM:
	      res = simplify_is_const_mem(inst, id2mem_inst, has_const_mem);
	      break;
	    case Op::MEM_SIZE:
	      res = simplify_mem_size(inst, id2size);
	      break;
	    case Op::FREE:
	      // Remove the size value for an ID when the size changes -- we
	      // are iterating in reverse post order, so this prevents use
	      // of incorrect values.
	      // Note: We must remove the value (i.e. not update it) as
	      // some future blocks may use the old value if they are later
	      // in the iteration order, but not dominated by this BB.
	      if (inst->arguments[0]->opcode == Op::VALUE)
		{
		  uint64_t id = inst->arguments[0]->value();
		  id2size.erase(id);
		}
	      else
		id2size.clear();
	      res = inst;
	      break;
	    case Op::MEMORY:
	      res = simplify_memory(inst);
	      break;
	    default:
	      res = simplify_inst(inst);
	      break;
	    }
	  if (res != inst)
	    {
	      inst->replace_all_uses_with(res);
	      destroy(inst);
	    }

	  inst = next_inst;
	}
    }
}

void simplify_mem(Module *module)
{
  for (auto func : module->functions)
    simplify_mem(func);
}

} // end namespace smtgcc
