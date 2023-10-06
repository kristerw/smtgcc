#include "config.h"

#if HAVE_LIBZ3
#include <cassert>
#include <cinttypes>
#include <z3++.h>

#include "smtgcc.h"

using namespace std::string_literals;

namespace smtgcc {

namespace {

enum class mem_kind {memory, memory_flag, memory_undef, memory_sizes};

class Common {
public:
  std::map<int, z3::expr> index2param;
  z3::context& ctx;
  std::map<mem_kind, z3::expr> mem;
  Module *module;
  Common(z3::context& ctx, Module *module)
    : ctx{ctx}
    , module{module}
  {
    z3::sort address_sort = ctx.bv_sort(module->ptr_bits);
    z3::sort byte_sort = ctx.bv_sort(8);
    z3::sort array_sort = ctx.array_sort(address_sort, byte_sort);
    z3::expr memory = ctx.constant(".memory", array_sort);
    mem.insert({mem_kind::memory, memory});
    z3::expr memory_flag =
      z3::expr(ctx, Z3_mk_const_array(ctx, address_sort, ctx.bv_val(0, 1)));
    mem.insert({mem_kind::memory_flag, memory_flag});
    z3::expr memory_undef =
      z3::expr(ctx, Z3_mk_const_array(ctx, address_sort, ctx.bv_val(0, 8)));
    mem.insert({mem_kind::memory_undef, memory_undef});

    z3::sort id_sort = ctx.bv_sort(module->ptr_id_bits);
    z3::expr zero_offset = ctx.bv_val(0, module->ptr_offset_bits);
    z3::expr memory_sizes =
      z3::expr(ctx, Z3_mk_const_array(ctx, id_sort, zero_offset));
    mem.insert({mem_kind::memory_sizes, memory_sizes});
  }
};

class Converter {
  std::map<const Instruction *, z3::expr> inst2bv;
  std::map<const Instruction *, z3::expr> inst2fp;
  std::map<const Instruction *, z3::expr> inst2bool;

  // Maps basic blocks to an expression telling if it is executed.
  std::map<const Basic_block *, z3::expr> bb2cond;

  // Maps basic blocks to an expression determining if it contain UB.
  std::map<const Basic_block *, z3::expr> bb2ub;

  // Maps basic blocks to an expression determining if it contain an
  // assertion failure.
  std::map<const Basic_block *, z3::expr> bb2not_assert;

  // List of the mem_id for the constant memory blocks.
  std::vector<Instruction *> const_ids;

  z3::expr ite(z3::expr c, z3::expr a, z3::expr b);
  z3::expr bool_or(z3::expr a, z3::expr b);
  z3::expr bool_and(z3::expr a, z3::expr b);
  void add_ub(const Basic_block *bb, z3::expr cond);
  void add_assert(const Basic_block *bb, z3::expr cond);
  Z3_sort fp_sort(uint32_t bitsize);
  void build_bv_comparison_smt(const Instruction *inst);
  void build_fp_comparison_smt(const Instruction *inst);
  void build_bv_unary_smt(const Instruction *inst);
  void build_fp_unary_smt(const Instruction *inst);
  void build_bv_binary_smt(const Instruction *inst);
  void build_fp_binary_smt(const Instruction *inst);
  void build_ternary_smt(const Instruction *inst);
  void build_conversion_smt(const Instruction *inst);
  void build_special_smt(const Instruction *inst);
  void build_smt(const Instruction *inst);
  z3::expr get_full_edge_cond(const Basic_block *src, const Basic_block *dest);
  void build_mem_state(const Basic_block *bb, std::map<const Basic_block *, z3::expr>& map);
  void generate_bb2cond(const Basic_block *bb);
  void convert_ir();

  z3::context& ctx;
  Common& common;
  const Function *func;

public:
  Converter(Common& common, const Function *func)
    : ctx{common.ctx}
    , common{common}
    , func{func}
  {
    convert_ir();
  }
  z3::expr inst_as_bv(const Instruction *inst);
  z3::expr inst_as_fp(const Instruction *inst);
  z3::expr inst_as_bool(const Instruction *inst);
  z3::expr generate_ub();
  z3::expr generate_assert();
  Instruction *retval = nullptr;
  Instruction *retval_undef = nullptr;

  std::map<const Basic_block *, z3::expr> bb2memory;
  std::map<const Basic_block *, z3::expr> bb2memory_sizes;
  std::map<const Basic_block *, z3::expr> bb2memory_flag;
  std::map<const Basic_block *, z3::expr> bb2memory_undef;
};

z3::expr Converter::ite(z3::expr c, z3::expr a, z3::expr b)
{
  if (z3::eq(a, b))
    return a;
  return z3::ite(c, a, b);
}

z3::expr Converter::bool_or(z3::expr a, z3::expr b)
{
  if (a.is_true())
    return a;
  if (a.is_false())
    return b;
  if (b.is_true())
    return b;
  if (b.is_false())
    return a;
  return a || b;
}

z3::expr Converter::bool_and(z3::expr a, z3::expr b)
{
  if (a.is_true())
    return b;
  if (a.is_false())
    return a;
  if (b.is_true())
    return a;
  if (b.is_false())
    return b;
  return a && b;
}

z3::expr Converter::inst_as_bv(const Instruction *inst)
{
  auto I = inst2bv.find(inst);
  if (I != inst2bv.end())
    return I->second;

  if (inst->bitsize == 1)
    {
      // We did not have a bitvector value for inst. This means there must
      // exist a Boolean value for this instruction. Convert it to bitvector.
      z3::expr expr =
	z3::ite(inst2bool.at(inst), ctx.bv_val(1, 1), ctx.bv_val(0, 1));
      inst2bv.insert({inst, expr});
      return expr;
    }
  else
    {
      // We did not have a bitvector value for inst. This means there must
      // exist a floating point value for this instruction. Convert it to
      // bitvector.
      z3::expr expr =
	z3::expr(ctx, Z3_mk_fpa_to_ieee_bv(ctx, inst2fp.at(inst)));
      inst2bv.insert({inst, expr});
      return expr;
    }
}

z3::expr Converter::inst_as_bool(const Instruction *inst)
{
  assert(inst->bitsize == 1);
  auto I = inst2bool.find(inst);
  if (I != inst2bool.end())
    return I->second;

  // We did not have a Boolean value for inst. This means there must exist
  // a bitvector value for this instruction. Convert it to Boolean.
  z3::expr bv = inst2bv.at(inst);
  z3::expr expr = bv != ctx.bv_val(0, 1);
  if (Z3_is_numeral_ast(ctx, bv))
    expr = expr.simplify();
  inst2bool.insert({inst, expr});
  return expr;
}

z3::expr Converter::inst_as_fp(const Instruction *inst)
{
  auto I = inst2fp.find(inst);
  if (I != inst2fp.end())
    return I->second;

  // We did not have a floating point value for inst. This means there must
  // exist a bitvector value for this instruction. Convert it to floating
  // point.
  z3::expr bv = inst2bv.at(inst);
  Z3_sort sort = fp_sort(inst->bitsize);
  Z3_ast r = Z3_mk_fpa_to_fp_bv(ctx, bv, sort);
  z3::expr expr = z3::expr(ctx, r);
  if (Z3_is_numeral_ast(ctx, bv))
    expr = expr.simplify();
  inst2fp.insert({inst, expr});
  return expr;
}

void Converter::build_bv_comparison_smt(const Instruction *inst)
{
  assert(inst->nof_args == 2);

  if (inst->arguments[0]->bitsize == 1
      && (inst->opcode == Op::EQ || inst->opcode == Op::NE)
      && (inst2bool.contains(inst->arguments[0])
	  && inst2bool.contains(inst->arguments[1])))
    {
      z3::expr arg1 = inst_as_bool(inst->arguments[0]);
      z3::expr arg2 = inst_as_bool(inst->arguments[1]);

      if (inst->opcode == Op::EQ)
	inst2bool.insert({inst, arg1 == arg2});
      else
	inst2bool.insert({inst, arg1 != arg2});
      return;
    }

  z3::expr arg1 = inst_as_bv(inst->arguments[0]);
  z3::expr arg2 = inst_as_bv(inst->arguments[1]);
  switch (inst->opcode)
    {
    case Op::EQ:
      inst2bool.insert({inst, arg1 == arg2});
      break;
    case Op::NE:
      inst2bool.insert({inst, arg1 != arg2});
      break;
    case Op::SGE:
      inst2bool.insert({inst, z3::sge(arg1, arg2)});
      break;
    case Op::SGT:
      inst2bool.insert({inst, z3::sgt(arg1, arg2)});
      break;
    case Op::SLE:
      inst2bool.insert({inst, z3::sle(arg1, arg2)});
      break;
    case Op::SLT:
      inst2bool.insert({inst, z3::slt(arg1, arg2)});
      break;
    case Op::UGE:
      inst2bool.insert({inst, z3::uge(arg1, arg2)});
      break;
    case Op::UGT:
      inst2bool.insert({inst, z3::ugt(arg1, arg2)});
      break;
    case Op::ULE:
      inst2bool.insert({inst, z3::ule(arg1, arg2)});
      break;
    case Op::ULT:
      inst2bool.insert({inst, z3::ult(arg1, arg2)});
      break;
    default:
      throw Not_implemented("build_comparison_smt: "s + inst->name());
    }
}

Z3_sort Converter::fp_sort(uint32_t bitsize)
{
  switch (bitsize)
    {
    case 16:
      return Z3_mk_fpa_sort_16(ctx);
    case 32:
      return Z3_mk_fpa_sort_32(ctx);
    case 64:
      return Z3_mk_fpa_sort_64(ctx);
    case 128:
      return Z3_mk_fpa_sort_128(ctx);
    default:
      throw Not_implemented("fp_sort: f" + std::to_string(bitsize));
    }
}

void Converter::add_ub(const Basic_block *bb, z3::expr cond)
{
  if (bb2ub.contains(bb))
    {
      cond = bool_or(bb2ub.at(bb), cond);
      bb2ub.erase(bb);
    }
  bb2ub.insert({bb, cond});
}

void Converter::add_assert(const Basic_block *bb, z3::expr cond)
{
  cond = !cond;
  if (bb2not_assert.contains(bb))
    {
      cond = bool_or(bb2not_assert.at(bb), cond);
      bb2not_assert.erase(bb);
    }
  bb2not_assert.insert({bb, cond});
}

void Converter::build_fp_comparison_smt(const Instruction *inst)
{
  assert(inst->nof_args == 2);
  z3::expr arg1 = inst_as_fp(inst->arguments[0]);
  z3::expr arg2 = inst_as_fp(inst->arguments[1]);

  switch (inst->opcode)
    {
    case Op::FEQ:
      inst2bool.insert({inst, z3::fp_eq(arg1, arg2)});
      break;
    case Op::FNE:
      inst2bool.insert({inst, !z3::fp_eq(arg1, arg2)});
      break;
    case Op::FGE:
      inst2bool.insert({inst, arg1 >= arg2});
      break;
    case Op::FGT:
      inst2bool.insert({inst, arg1 > arg2});
      break;
    case Op::FLE:
      inst2bool.insert({inst, arg1 <= arg2});
      break;
    case Op::FLT:
      inst2bool.insert({inst, arg1 < arg2});
      break;
    default:
      throw Not_implemented("build_comparison_smt: "s + inst->name());
    }
}

void Converter::build_bv_unary_smt(const Instruction *inst)
{
  assert(inst->nof_args == 1);

  // Do the "not" operation as Boolean if the argument is Boolean. This avoids
  // multiple conversions between bitvector and Boolean and for the typical
  // case where the 1-bit values is used in comparisons etc.
  if (inst->bitsize == 1
      && inst->opcode == Op::NOT
      && inst2bool.contains(inst->arguments[0]))
    {
      z3::expr arg1 = inst_as_bool(inst->arguments[0]);
      inst2bool.insert({inst, !arg1});
      return;
    }

  // The UB processing assumes the argument is Boolean.
  if (inst->opcode == Op::UB)
    {
      z3::expr ub = inst_as_bool(inst->arguments[0]);
      add_ub(inst->bb, ub);
      return;
    }

  // The ASSERT processing assumes the argument is Boolean.
  if (inst->opcode == Op::ASSERT)
    {
      z3::expr assrt = inst_as_bool(inst->arguments[0]);
      add_assert(inst->bb, assrt);
      return;
    }

  z3::expr arg1 = inst_as_bv(inst->arguments[0]);
  switch (inst->opcode)
    {
    case Op::SYMBOLIC:
      {
	static int symbolic_idx = 0;
	char name[100];
	sprintf(name, ".symbolic%d", symbolic_idx++);
	z3::expr symbolic = ctx.bv_const(name, inst->bitsize);
	inst2bv.insert({inst, symbolic});
      }
      break;
    case Op::GET_MEM_FLAG:
      {
	z3::expr memory_flag = bb2memory_flag.at(inst->bb);
	inst2bv.insert({inst, z3::select(memory_flag, arg1)});
      }
      break;
    case Op::GET_MEM_UNDEF:
      {
	z3::expr memory_undef = bb2memory_undef.at(inst->bb);
	inst2bv.insert({inst, z3::select(memory_undef, arg1)});
      }
      break;
    case Op::FREE:
      {
	uint32_t ptr_offset_bits = func->module->ptr_offset_bits;
	z3::expr sizes = bb2memory_sizes.at(inst->bb);
	sizes = z3::store(sizes, arg1, ctx.bv_val(0, ptr_offset_bits));
	bb2memory_sizes.erase(inst->bb);
	bb2memory_sizes.insert({inst->bb, sizes});
      }
      break;
    case Op::IS_CONST_MEM:
      {
	z3::expr is_const = ctx.bool_val(false);
	for (Instruction *id : const_ids)
	  {
	    z3::expr cond = arg1 == inst_as_bv(id);
	    is_const = bool_or(is_const, cond);
	  }
	inst2bool.insert({inst, is_const});
      }
      break;
    case Op::IS_NAN:
      {
	z3::expr farg1 = inst_as_fp(inst->arguments[0]);
	z3::expr is_nan = z3::expr(ctx, Z3_mk_fpa_is_nan(ctx, farg1));
	inst2bool.insert({inst, is_nan});
      }
      break;
    case Op::IS_NONCANONICAL_NAN:
      {
	z3::expr farg1 = inst_as_fp(inst->arguments[0]);
	z3::expr is_nan = z3::expr(ctx, Z3_mk_fpa_is_nan(ctx, farg1));
	Z3_sort sort = fp_sort(inst->arguments[0]->bitsize);
	z3::expr nan = z3::expr(ctx, Z3_mk_fpa_nan(ctx, sort));
	z3::expr nan_bv = z3::expr(ctx, Z3_mk_fpa_to_ieee_bv(ctx, nan));
	inst2bool.insert({inst, is_nan && (nan_bv != arg1)});
      }
      break;
    case Op::LOAD:
      {
	z3::expr memory = bb2memory.at(inst->bb);
	inst2bv.insert({inst, z3::select(memory, arg1)});
      }
      break;
    case Op::MEM_SIZE:
      {
	z3::expr sizes = bb2memory_sizes.at(inst->bb);
	inst2bv.insert({inst, z3::select(sizes, arg1)});
      }
      break;
    case Op::MOV:
      inst2bv.insert({inst, arg1});
      break;
    case Op::NEG:
      inst2bv.insert({inst, -arg1});
      break;
    case Op::NOT:
      inst2bv.insert({inst, ~arg1});
      break;
    default:
      throw Not_implemented("build_bv_unary_smt: "s + inst->name());
    }
}

void Converter::build_fp_unary_smt(const Instruction *inst)
{
  z3::expr arg1 = inst_as_fp(inst->arguments[0]);
  switch (inst->opcode)
    {
    case Op::FABS:
      inst2fp.insert({inst, z3::abs(arg1)});
      break;
    case Op::FNEG:
      inst2fp.insert({inst, -arg1});
      break;
    case Op::NAN:
      {
	Z3_sort sort = fp_sort(inst->arguments[0]->value());
	inst2fp.insert({inst, z3::expr(ctx, Z3_mk_fpa_nan(ctx, sort))});
      }
      break;
    default:
      throw Not_implemented("build_fp_unary_smt: "s + inst->name());
    }
}

void Converter::build_bv_binary_smt(const Instruction *inst)
{
  assert(inst->nof_args == 2);

  // Do and/or/xor operations as Boolean if at least one of the arguments
  // is Boolean. This avoids multiple conversions between bitvector and
  // Boolean and for the typical case where the 1-bit values is used in
  // comparisons etc.
  if (inst->bitsize == 1 &&
      (inst->opcode == Op::AND
       || inst->opcode == Op::OR
       || inst->opcode == Op::XOR)
      && (inst2bool.contains(inst->arguments[0])
	  || inst2bool.contains(inst->arguments[1])))
    {
      z3::expr arg1 = inst_as_bool(inst->arguments[0]);
      z3::expr arg2 = inst_as_bool(inst->arguments[1]);
      if (inst->opcode == Op::AND)
	inst2bool.insert({inst, arg1 && arg2});
      else if (inst->opcode == Op::OR)
	inst2bool.insert({inst, arg1 || arg2});
      else
	inst2bool.insert({inst, arg1 ^ arg2});
      return;
    }

  z3::expr arg1 = inst_as_bv(inst->arguments[0]);
  z3::expr arg2 = inst_as_bv(inst->arguments[1]);
  switch (inst->opcode)
    {
    case Op::ADD:
      inst2bv.insert({inst, arg1 + arg2});
      break;
    case Op::SUB:
      inst2bv.insert({inst, arg1 - arg2});
      break;
    case Op::MUL:
      inst2bv.insert({inst, arg1 * arg2});
      break;
    case Op::PARAM:
      {
	uint32_t index = inst->arguments[0]->value();
	if (common.index2param.contains(index))
	  {
	    inst2bv.insert({inst, common.index2param.at(index)});
	  }
	else
	  {
	    char name[100];
	    sprintf(name, ".param%d", index);
	    z3::expr param = ctx.bv_const(name, inst->bitsize);
	    inst2bv.insert({inst, param});
	    common.index2param.insert({index, param});
	  }
      }
      break;
    case Op::SDIV:
      inst2bv.insert({inst, arg1 / arg2});
      break;
    case Op::UDIV:
      inst2bv.insert({inst, z3::udiv(arg1, arg2)});
      break;
    case Op::UMAX:
      inst2bv.insert({inst, ite(z3::uge(arg1, arg2), arg1, arg2)});
      break;
    case Op::UMIN:
      inst2bv.insert({inst, ite(z3::ult(arg1, arg2), arg1, arg2)});
      break;
    case Op::SADD_WRAPS:
      {
	z3::expr earg1 = z3::sext(arg1, 1);
	z3::expr earg2 = z3::sext(arg2, 1);
	z3::expr eadd = earg1 + earg2;
	z3::expr eres = z3::sext(arg1 + arg2, 1);
	inst2bool.insert({inst, eadd != eres});
      }
      break;
    case Op::SMAX:
      inst2bv.insert({inst, ite(z3::sge(arg1, arg2), arg1, arg2)});
      break;
    case Op::SMIN:
      inst2bv.insert({inst, ite(z3::slt(arg1, arg2), arg1, arg2)});
      break;
    case Op::SMUL_WRAPS:
      {
	z3::expr earg1 = z3::sext(arg1, inst->arguments[0]->bitsize);
	z3::expr earg2 = z3::sext(arg2, inst->arguments[0]->bitsize);
	z3::expr emul = earg1 * earg2;
	z3::expr eres = z3::sext(arg1 * arg2, inst->arguments[0]->bitsize);
	inst2bool.insert({inst, emul != eres});
      }
      break;
    case Op::SREM:
      inst2bv.insert({inst, z3::srem(arg1, arg2)});
      break;
    case Op::SSUB_WRAPS:
      {
	z3::expr earg1 = z3::sext(arg1, 1);
	z3::expr earg2 = z3::sext(arg2, 1);
	z3::expr esub = earg1 - earg2;
	z3::expr eres = z3::sext(arg1 - arg2, 1);
	inst2bool.insert({inst, esub != eres});
      }
      break;
    case Op::UREM:
      inst2bv.insert({inst, z3::urem(arg1, arg2)});
      break;
    case Op::ASHR:
      inst2bv.insert({inst, z3::ashr(arg1, arg2)});
      break;
    case Op::LSHR:
      inst2bv.insert({inst, z3::lshr(arg1, arg2)});
      break;
    case Op::SET_MEM_FLAG:
      {
	z3::expr memory_flag = bb2memory_flag.at(inst->bb);
	memory_flag = z3::store(memory_flag, arg1, arg2);
	bb2memory_flag.erase(inst->bb);
	bb2memory_flag.insert({inst->bb, memory_flag});
      }
      break;
    case Op::SET_MEM_UNDEF:
      {
	z3::expr memory_undef = bb2memory_undef.at(inst->bb);
	memory_undef = z3::store(memory_undef, arg1, arg2);
	bb2memory_undef.erase(inst->bb);
	bb2memory_undef.insert({inst->bb, memory_undef});
      }
      break;
    case Op::SHL:
      inst2bv.insert({inst, z3::shl(arg1, arg2)});
      break;
    case Op::AND:
      inst2bv.insert({inst, arg1 & arg2});
      break;
    case Op::OR:
      inst2bv.insert({inst, arg1 | arg2});
      break;
    case Op::XOR:
      inst2bv.insert({inst, arg1 ^ arg2});
      break;
    case Op::CONCAT:
      {
	z3::expr res = z3::concat(arg1, arg2);
	// We may concat two large constants that cannot be constant folded
	// at IR level (as the IR limits constant width to 128 bits), so
	// we fold it here to get nicer SMT2 when debugging.
	if (Z3_is_numeral_ast(ctx, arg1) && Z3_is_numeral_ast(ctx, arg2))
	  res = res.simplify();
	inst2bv.insert({inst, res});
      }
      break;
    case Op::STORE:
      {
	z3::expr memory = bb2memory.at(inst->bb);
	memory = z3::store(memory, arg1, arg2);
	bb2memory.erase(inst->bb);
	bb2memory.insert({inst->bb, memory});
      }
      break;
    default:
      throw Not_implemented("build_binary_smt: "s + inst->name());
    }
}

void Converter::build_fp_binary_smt(const Instruction *inst)
{
  assert(inst->nof_args == 2);
  z3::expr arg1 = inst_as_fp(inst->arguments[0]);
  z3::expr arg2 = inst_as_fp(inst->arguments[1]);
  switch (inst->opcode)
    {
    case Op::FADD:
      inst2fp.insert({inst, arg1 + arg2});
      break;
    case Op::FSUB:
      inst2fp.insert({inst, arg1 - arg2});
      break;
    case Op::FMUL:
      inst2fp.insert({inst, arg1 * arg2});
      break;
    case Op::FDIV:
      inst2fp.insert({inst, arg1 / arg2});
      break;
    default:
      throw Not_implemented("build_binary_smt: "s + inst->name());
    }
}

void Converter::build_ternary_smt(const Instruction *inst)
{
  assert(inst->nof_args == 3);
  switch (inst->opcode)
    {
    case Op::EXTRACT:
      {
	z3::expr arg = inst_as_bv(inst->arguments[0]);
	uint32_t high = inst->arguments[1]->value();
	uint32_t low = inst->arguments[2]->value();
	inst2bv.insert({inst, arg.extract(high, low)});
      }
      break;
    case Op::MEMORY:
      // Memory has already been processed.
      assert(inst2bv.contains(inst));
      break;
    case Op::ITE:
      if (inst->bitsize == 1 &&
	  (inst2bool.contains(inst->arguments[1])
	   && inst2bool.contains(inst->arguments[2])))
	{
	  z3::expr arg1 = inst_as_bool(inst->arguments[0]);
	  z3::expr arg2 = inst_as_bool(inst->arguments[1]);
	  z3::expr arg3 = inst_as_bool(inst->arguments[2]);
	  inst2bool.insert({inst, ite(arg1, arg2, arg3)});
	}
      else
	{
	  z3::expr arg1 = inst_as_bool(inst->arguments[0]);
	  z3::expr arg2 = inst_as_bv(inst->arguments[1]);
	  z3::expr arg3 = inst_as_bv(inst->arguments[2]);
	  inst2bv.insert({inst, ite(arg1, arg2, arg3)});
	}
      break;
    default:
      throw Not_implemented("build_ternary_smt: "s + inst->name());
    }
}

void Converter::build_conversion_smt(const Instruction *inst)
{
  switch (inst->opcode)
    {
    case Op::SEXT:
      {
	z3::expr arg = inst_as_bv(inst->arguments[0]);
	uint32_t arg_bitsize = inst->arguments[0]->bitsize;
	assert(arg_bitsize < inst->bitsize);
	inst2bv.insert({inst, z3::sext(arg, inst->bitsize - arg_bitsize)});
      }
      break;
    case Op::ZEXT:
      {
	z3::expr arg = inst_as_bv(inst->arguments[0]);
	uint32_t arg_bitsize = inst->arguments[0]->bitsize;
	assert(arg_bitsize < inst->bitsize);
	inst2bv.insert({inst, z3::zext(arg, inst->bitsize - arg_bitsize)});
      }
      break;
    case Op::F2U:
      {
	z3::expr arg = inst_as_fp(inst->arguments[0]);
	z3::expr rtz = to_expr(ctx, Z3_mk_fpa_rtz(ctx));
	Z3_ast r = Z3_mk_fpa_to_ubv(ctx, rtz, arg, inst->bitsize);
	inst2bv.insert({inst, z3::expr(ctx, r)});
      }
      break;
    case Op::F2S:
      {
	z3::expr arg = inst_as_fp(inst->arguments[0]);
	z3::expr rtz = to_expr(ctx, Z3_mk_fpa_rtz(ctx));
	Z3_ast r = Z3_mk_fpa_to_sbv(ctx, rtz, arg, inst->bitsize);
	inst2bv.insert({inst, z3::expr(ctx, r)});
      }
      break;
    case Op::S2F:
      {
	z3::expr arg = inst_as_bv(inst->arguments[0]);
	z3::expr rne = to_expr(ctx, Z3_mk_fpa_rne(ctx));
	Z3_sort sort = fp_sort(inst->bitsize);
	Z3_ast r = Z3_mk_fpa_to_fp_signed(ctx, rne, arg, sort);
	inst2fp.insert({inst, z3::expr(ctx, r)});
      }
      break;
    case Op::U2F:
      {
	z3::expr arg = inst_as_bv(inst->arguments[0]);
	z3::expr rne = to_expr(ctx, Z3_mk_fpa_rne(ctx));
	Z3_sort sort = fp_sort(inst->bitsize);
	Z3_ast r = Z3_mk_fpa_to_fp_unsigned(ctx, rne, arg, sort);
	inst2fp.insert({inst, z3::expr(ctx, r)});
      }
      break;
    case Op::FCHPREC:
      {
	z3::expr arg = inst_as_fp(inst->arguments[0]);
	z3::expr rne = to_expr(ctx, Z3_mk_fpa_rne(ctx));
	Z3_sort sort = fp_sort(inst->bitsize);
	Z3_ast r = Z3_mk_fpa_to_fp_float(ctx, rne, arg, sort);
	inst2fp.insert({inst, z3::expr(ctx, r)});
      }
      break;
    default:
      throw Not_implemented("build_conversion_smt: "s + inst->name());
    }
}

void Converter::build_special_smt(const Instruction *inst)
{
  switch (inst->opcode)
    {
    case Op::BR:
      // Nothing to do for branches here (they are handled by the end
      // of basic block code).
      break;
    case Op::RET:
      assert(inst->nof_args <= 2);
      if (inst->nof_args == 1)
	retval = inst->arguments[0];
      else if (inst->nof_args == 2)
	{
	  retval = inst->arguments[0];
	  retval_undef = inst->arguments[1];
	}
      break;
    case Op::VALUE:
      {
	uint64_t low = inst->value();
	uint64_t high = inst->value() >> 64;
	if (inst->bitsize > 64)
	  {
	    z3::expr lo = ctx.bv_val(low, 64);
	    z3::expr hi = ctx.bv_val(high, inst->bitsize - 64);
	    inst2bv.insert({inst, z3::concat(hi, lo).simplify()});
	  }
	else
	  {
	    inst2bv.insert({inst, ctx.bv_val(low, inst->bitsize)});
	    if (inst->bitsize == 1)
	      inst2bool.insert({inst, ctx.bool_val(low != 0)});
	  }
      }
      break;
    default:
      throw Not_implemented("build_special_smt: "s + inst->name());
    }
}

void Converter::build_smt(const Instruction *inst)
{
  switch (inst->iclass())
    {
    case Inst_class::icomparison:
      build_bv_comparison_smt(inst);
      break;
    case Inst_class::fcomparison:
      build_fp_comparison_smt(inst);
      break;
    case Inst_class::iunary:
      build_bv_unary_smt(inst);
      break;
    case Inst_class::funary:
      build_fp_unary_smt(inst);
      break;
    case Inst_class::ibinary:
      build_bv_binary_smt(inst);
      break;
    case Inst_class::fbinary:
      build_fp_binary_smt(inst);
      break;
    case Inst_class::ternary:
      build_ternary_smt(inst);
      break;
    case Inst_class::conv:
      build_conversion_smt(inst);
      break;
    case Inst_class::special:
      build_special_smt(inst);
      break;
    default:
      throw Not_implemented("build_smt: "s + inst->name());
    }
}

z3::expr Converter::get_full_edge_cond(const Basic_block *src, const Basic_block *dest)
{
  if (src->succs.size() == 1)
    return bb2cond.at(src);
  assert(src->succs.size() == 2);
  assert(src->last_inst->opcode == Op::BR);
  assert(src->last_inst->nof_args == 1);
  z3::expr cond = inst_as_bool(src->last_inst->arguments[0]);
  if (dest != src->succs[0])
    cond = !cond;
  z3::expr src_cond = bb2cond.at(src);
  return bool_and(src_cond, cond);
}

void Converter::build_mem_state(const Basic_block *bb, std::map<const Basic_block *, z3::expr>& map)
{
  assert(bb->preds.size() > 0);
  z3::expr expr = map.at(bb->preds[0]);
  for (size_t i = 1; i < bb->preds.size(); i++)
    {
      const Basic_block *pred_bb = bb->preds[i];
      expr = ite(bb2cond.at(pred_bb), map.at(pred_bb), expr);
    }
  map.insert({bb, expr});
}

void Converter::generate_bb2cond(const Basic_block *bb)
{
  Basic_block *dominator = nearest_dominator(bb);
  if (dominator && post_dominates(bb, dominator))
    {
      // If the dominator is post dominated by bb, then they have identical
      // conditions.
      bb2cond.insert({bb, bb2cond.at(dominator)});
    }
  else
    {
      // We must build a new condition that reflect the path(s) to this bb.
      z3::expr cond = ctx.bool_val(false);
      for (auto pred_bb : bb->preds)
	{
	  z3::expr edge_cond = get_full_edge_cond(pred_bb, bb);
	  cond = bool_or(cond, edge_cond);
	}
      bb2cond.insert({bb, cond});
    }
}

void Converter::convert_ir()
{
  for (auto bb : func->bbs)
    {
      if (bb == func->bbs[0])
	{
	  bb2cond.insert({bb, ctx.bool_val(true)});
	  bb2memory.insert({bb, common.mem.at(mem_kind::memory)});
	  bb2memory_flag.insert({bb, common.mem.at(mem_kind::memory_flag)});
	  z3::expr undef = common.mem.at(mem_kind::memory_undef);
	  z3::expr sizes = common.mem.at(mem_kind::memory_sizes);
	  for (Instruction *inst = bb->first_inst; inst; inst = inst->next)
	    {
	      if (inst->opcode != Op::MEMORY)
		continue;
	      assert(inst->bitsize == func->module->ptr_bits);
	      uint64_t id = inst->arguments[0]->value();
	      uint64_t ptr_val = id << func->module->ptr_id_low;
	      inst2bv.insert({inst, ctx.bv_val(ptr_val, inst->bitsize)});

	      uint32_t ptr_bits = func->module->ptr_bits;
	      uint32_t ptr_offset_bits = func->module->ptr_offset_bits;
	      uint32_t ptr_id_bits = func->module->ptr_id_bits;
	      z3::expr mem_id = ctx.bv_val(id, ptr_id_bits);
	      uint64_t size_val = inst->arguments[1]->value();
	      z3::expr size = ctx.bv_val(size_val, ptr_offset_bits);
	      sizes = z3::store(sizes, mem_id, size);

	      uint32_t flags = inst->arguments[2]->value();
	      if (flags & MEM_CONST)
		const_ids.push_back(inst->arguments[0]);
	      if (flags & MEM_UNINIT)
		{
		  z3::expr byte = ctx.bv_val(255, 8);
		  for (uint64_t i = 0; i < size_val; i++)
		    {
		      z3::expr ptr = ctx.bv_val(ptr_val + i, ptr_bits);
		      undef = z3::store(undef, ptr, byte);
		    }
		}
	    }

	  bb2memory_sizes.insert({bb, sizes});
	  bb2memory_undef.insert({bb, undef});
	}
      else
	{
	  generate_bb2cond(bb);

	  build_mem_state(bb, bb2memory);
	  build_mem_state(bb, bb2memory_sizes);
	  build_mem_state(bb, bb2memory_flag);
	  build_mem_state(bb, bb2memory_undef);
	}

      for (auto phi : bb->phis)
	{
	  // We want to create the phi as a Boolean if any element is Boolean
	  // (as that in general reduces conversion between bool and bv.
	  // Doing this for floating point would also reduce the number of
	  // conversion, but that gives lots of problems with inconsistent
	  // NaN handling.
	  bool is_bool = false;
	  for (auto phi_arg : phi->phi_args)
	    {
	      is_bool = is_bool || inst2bool.contains(phi_arg.inst);
	    }
	  if (is_bool)
	    {
	      z3::expr phi_expr = inst_as_bool(phi->phi_args[0].inst);
	      assert(phi->phi_args.size() == bb->preds.size());
	      for (unsigned i = 1; i < phi->phi_args.size(); i++)
		{
		  const Basic_block *pred_bb = phi->phi_args[i].bb;
		  z3::expr cond = get_full_edge_cond(pred_bb, bb);
		  z3::expr expr = inst_as_bool(phi->phi_args[i].inst);
		  phi_expr = ite(cond, expr, phi_expr);
		}
	      inst2bool.insert({phi, phi_expr});
	    }
	  else
	    {
	      z3::expr phi_expr = inst_as_bv(phi->phi_args[0].inst);
	      assert(phi->phi_args.size() == bb->preds.size());
	      for (unsigned i = 1; i < phi->phi_args.size(); i++)
		{
		  const Basic_block *pred_bb = phi->phi_args[i].bb;
		  z3::expr cond = get_full_edge_cond(pred_bb, bb);
		  z3::expr expr = inst_as_bv(phi->phi_args[i].inst);
		  phi_expr = ite(cond, expr, phi_expr);
		}
	      inst2bv.insert({phi, phi_expr});
	    }
	}

      for (Instruction *inst = bb->first_inst; inst; inst = inst->next)
	{
	  build_smt(inst);
	}
    }
}

z3::expr Converter::generate_ub()
{
  z3::expr ub = ctx.bool_val(false);
  for (auto bb : func->bbs)
    {
      if (!bb2ub.contains(bb))
	continue;

      z3::expr is_ub = bool_and(bb2ub.at(bb), bb2cond.at(bb));
      ub = bool_or(ub, is_ub);
    }
  return ub;
}

z3::expr Converter::generate_assert()
{
  z3::expr assrt = ctx.bool_val(false);
  for (auto bb : func->bbs)
    {
      if (!bb2not_assert.contains(bb))
	continue;

      z3::expr is_assrt = bool_and(bb2not_assert.at(bb), bb2cond.at(bb));
      assrt = bool_or(assrt, is_assrt);
    }
  return assrt;
}

Solver_result run_solver(z3::solver& s, const char *str)
{
  switch (s.check()) {
  case z3::unsat:
    return {Result_status::correct, {}};
  case z3::sat:
    {
      std::string msg = "Transformation is not correct ("s + str + ")\n";
      z3::model m = s.get_model();
      for (unsigned i = 0; i < m.size(); i++)
	{
	  z3::func_decl v = m[i];
	  std::string name = v.name().str();
	  std::string value = m.get_const_interp(v).to_string();
	  msg = msg + name + " = " + value + "\n";
	}
      return {Result_status::incorrect, msg};
    }
  case z3::unknown:
    std::string msg = "Analysis timed out ("s + str + ")\n";
    return {Result_status::unknown, msg};
  }

  throw Not_implemented("run_solver: unknown solver.check return");
}

} // end anonymous namespace

std::pair<SStats, Solver_result> check_refine_z3(Function *src, Function *tgt)
{
  char buf[32];
  sprintf(buf, "%d", config.timeout);
  Z3_global_param_set("timeout", buf);
  sprintf(buf, "%" PRIu64, (uint64_t)config.memory_limit * 1024 * 1024);
  Z3_global_param_set("memory_high_watermark", buf);

  SStats stats;
  stats.skipped = false;
  z3::context ctx;

  assert(src->module == tgt->module);
  Module *module = src->module;
  Common common(ctx, module);
  Converter conv_src(common, src);
  z3::expr src_ub_expr = conv_src.generate_ub();

  Converter conv_tgt(common, tgt);
  z3::expr tgt_ub_expr = conv_tgt.generate_ub();

  std::string warning;
  if (conv_src.retval || conv_tgt.retval)
    {
      assert(conv_src.retval && conv_tgt.retval);
      z3::expr src_expr = conv_src.inst_as_bv(conv_src.retval);
      z3::expr tgt_expr = conv_tgt.inst_as_bv(conv_tgt.retval);
      z3::expr is_more_undef = ctx.bool_val(false);
      z3::expr src_undef = ctx.bv_val(0, conv_src.retval->bitsize);
      z3::expr tgt_undef = ctx.bv_val(0, conv_tgt.retval->bitsize);
      if (conv_src.retval_undef)
	{
	  src_undef = conv_src.inst_as_bv(conv_src.retval_undef);
	  z3::expr src_mask = ~src_undef;
	  src_expr = src_expr & src_mask;
	  tgt_expr = tgt_expr & src_mask;

	  // Check that tgt is not more undef than src.
	  if (conv_tgt.retval_undef)
	    {
	      tgt_undef = conv_tgt.inst_as_bv(conv_tgt.retval_undef);
	      is_more_undef = (src_mask & tgt_undef) != 0;
	    }
	}

      z3::solver solver(ctx);
      solver.add(!src_ub_expr);
      solver.add((src_expr != tgt_expr) || is_more_undef);
      uint64_t start_time = get_time();
      Solver_result solver_result = run_solver(solver, "retval");
      stats.time[0] = std::max(get_time() - start_time, (uint64_t)1);
      if (solver_result.status == Result_status::incorrect)
	{
	  assert(solver_result.message);
	  z3::model model = solver.get_model();
	  std::string msg = *solver_result.message;
	  msg = msg + "src retval: " + model.eval(src_expr).to_string() + "\n";
	  msg = msg + "tgt retval: " + model.eval(tgt_expr).to_string() + "\n";
	  if (conv_src.retval_undef || conv_tgt.retval_undef)
	    {
	      msg = msg + "src undef: " + model.eval(src_undef).to_string() + "\n";
	      msg = msg +  "tgt undef: " + model.eval(tgt_undef).to_string() + "\n";
	    }
	  Solver_result result = {Result_status::incorrect, msg};
	  return std::pair<SStats, Solver_result>(stats, result);
	}
      if (solver_result.status == Result_status::unknown)
	{
	  assert(solver_result.message);
	  warning = warning + *solver_result.message;
	}
    }

  {
    Basic_block *src_exit_bb = src->bbs[src->bbs.size() - 1];
    z3::expr src_mem = conv_src.bb2memory.at(src_exit_bb);
    z3::expr src_mem_sizes = conv_src.bb2memory_sizes.at(src_exit_bb);
    z3::expr src_mem_undef = conv_src.bb2memory_undef.at(src_exit_bb);

    Basic_block *tgt_exit_bb = tgt->bbs[tgt->bbs.size() - 1];
    z3::expr tgt_mem = conv_tgt.bb2memory.at(tgt_exit_bb);
    z3::expr tgt_mem_undef = conv_tgt.bb2memory_undef.at(tgt_exit_bb);

    z3::expr ptr = ctx.bv_const(".ptr", module->ptr_bits);
    uint32_t ptr_id_high = module->ptr_id_high;
    uint32_t ptr_id_low = module->ptr_id_low;
    z3::expr id = ptr.extract(ptr_id_high, ptr_id_low);
    uint32_t ptr_offset_high = module->ptr_offset_high;
    uint32_t ptr_offset_low = module->ptr_offset_low;
    z3::expr offset = ptr.extract(ptr_offset_high, ptr_offset_low);

    z3::solver solver(ctx);
    solver.add(!src_ub_expr);

    // Only check global memory.
    solver.add(id > 0);

    // Only check memory within a memory block.
    solver.add(z3::ult(offset, z3::select(src_mem_sizes, id)));

    // Check that src and tgt are the same for the bits where src is defined
    // and that tgt is not more undefined than src.
    z3::expr src_mask = ~z3::select(src_mem_undef, ptr);
    z3::expr src_value = z3::select(src_mem, ptr) & src_mask;
    z3::expr tgt_value = z3::select(tgt_mem, ptr) & src_mask;
    z3::expr tgt_more_undef = (z3::select(tgt_mem_undef, ptr) & src_mask) != 0;
    solver.add(src_value != tgt_value || tgt_more_undef);

    uint64_t start_time = get_time();
    Solver_result solver_result = run_solver(solver, "Memory");
    stats.time[1] = std::max(get_time() - start_time, (uint64_t)1);
    if (solver_result.status == Result_status::incorrect)
      {
	assert(solver_result.message);
	z3::model model = solver.get_model();
	z3::expr src_byte = model.eval(z3::select(src_mem, ptr));
	z3::expr tgt_byte = model.eval(z3::select(tgt_mem, ptr));
	z3::expr src_undef = model.eval(z3::select(src_mem_undef, ptr));
	z3::expr tgt_undef = model.eval(z3::select(tgt_mem_undef, ptr));
	std::string msg = *solver_result.message;
	msg = msg + "\n.ptr = " + model.eval(ptr).to_string() + "\n";
	msg = msg + "src *.ptr: " + src_byte.to_string() + "\n";
	msg = msg + "tgt *.ptr: " + tgt_byte.to_string() + "\n";
	msg = msg + "src undef: " + src_undef.to_string() + "\n";
	msg = msg + "tgt undef: " + tgt_undef.to_string() + "\n";
	Solver_result result = {Result_status::incorrect, msg};
	return std::pair<SStats, Solver_result>(stats, result);
      }
    if (solver_result.status == Result_status::unknown)
      {
	assert(solver_result.message);
	warning = warning + *solver_result.message;
      }
  }

  // Check that tgt does not have UB that is not in src.
  //
  // This should be the last check as UB that does not change the result
  // has lowe priority.
  {
    z3::solver solver(ctx);
    solver.add(!src_ub_expr);
    solver.add(tgt_ub_expr);
    solver.add(src_ub_expr != tgt_ub_expr);
    uint64_t start_time = get_time();
    Solver_result solver_result = run_solver(solver, "UB");
    stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
    if (solver_result.status == Result_status::incorrect)
      return std::pair<SStats, Solver_result>(stats, solver_result);
    if (solver_result.status == Result_status::unknown)
      {
	assert(solver_result.message);
	warning = warning + *solver_result.message;
      }
  }

  if (!warning.empty())
    {
      Solver_result result = {Result_status::unknown, warning};
      return std::pair<SStats, Solver_result>(stats, result);
    }
  return std::pair<SStats, Solver_result>(stats, {Result_status::correct, {}});
}

std::pair<SStats, Solver_result> check_ub_z3(Function *func)
{
  char buf[32];
  sprintf(buf, "%d", config.timeout);
  Z3_global_param_set("timeout", buf);
  sprintf(buf, "%" PRIu64, (uint64_t)config.memory_limit * 1024 * 1024);
  Z3_global_param_set("memory_high_watermark", buf);

  SStats stats;
  stats.skipped = false;
  z3::context ctx;

  Module *module = func->module;
  Common common(ctx, module);
  Converter conv_func(common, func);
  z3::expr ub_expr = conv_func.generate_ub();

  z3::solver solver(ctx);
  solver.add(ub_expr);
  uint64_t start_time = get_time();
  Solver_result solver_result = run_solver(solver, "UB");
  stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
  return std::pair<SStats, Solver_result>(stats, solver_result);
}

std::pair<SStats, Solver_result> check_assert_z3(Function *func)
{
  char buf[32];
  sprintf(buf, "%d", config.timeout);
  Z3_global_param_set("timeout", buf);
  sprintf(buf, "%" PRIu64, (uint64_t)config.memory_limit * 1024 * 1024);
  Z3_global_param_set("memory_high_watermark", buf);

  SStats stats;
  stats.skipped = false;
  z3::context ctx;

  Module *module = func->module;
  Common common(ctx, module);
  Converter conv_func(common, func);
  z3::expr ub_expr = conv_func.generate_ub();
  z3::expr assert_expr = conv_func.generate_assert();

  z3::solver solver(ctx);
  solver.add(!ub_expr);
  solver.add(assert_expr);
  uint64_t start_time = get_time();
  Solver_result solver_result = run_solver(solver, "ASSERT");
  stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
  return std::pair<SStats, Solver_result>(stats, solver_result);
}

} // end namespace smtgcc

#else

#include "smtgcc.h"

namespace smtgcc {

SStats verify_z3(Function *, Function *)
{
  throw Not_implemented("z3 is not available");
}

} // end namespace smtgcc

#endif
