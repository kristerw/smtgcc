#include "config.h"

#if HAVE_LIBZ3
#include <cassert>
#include <cinttypes>
#include <z3++.h>

#include "smtgcc.h"

using namespace std::string_literals;

namespace smtgcc {

namespace {

class Converter {
  std::map<const Instruction *, z3::expr> inst2array;
  std::map<const Instruction *, z3::expr> inst2bv;
  std::map<const Instruction *, z3::expr> inst2fp;
  std::map<const Instruction *, z3::expr> inst2bool;

  z3::expr ite(z3::expr c, z3::expr a, z3::expr b);
  Z3_sort fp_sort(uint32_t bitsize);
  void build_bv_comparison_smt(const Instruction *inst);
  void build_fp_comparison_smt(const Instruction *inst);
  void build_nullary_smt(const Instruction *inst);
  void build_bv_unary_smt(const Instruction *inst);
  void build_fp_unary_smt(const Instruction *inst);
  void build_bv_binary_smt(const Instruction *inst);
  void build_fp_binary_smt(const Instruction *inst);
  void build_ternary_smt(const Instruction *inst);
  void build_conversion_smt(const Instruction *inst);
  void build_special_smt(const Instruction *inst);
  void build_smt(const Instruction *inst);
  void convert_function();

  z3::context& ctx;
  const Function *func;

public:
  Converter(z3::context& ctx, const Function *func)
    : ctx{ctx}
    , func{func}
  {
    convert_function();
  }
  z3::expr inst_as_array(const Instruction *inst);
  z3::expr inst_as_bv(const Instruction *inst);
  z3::expr inst_as_fp(const Instruction *inst);
  z3::expr inst_as_bool(const Instruction *inst);

  std::vector<const Instruction *> print;

  Instruction *src_assert = nullptr;
  Instruction *src_memory = nullptr;
  Instruction *src_memory_size = nullptr;
  Instruction *src_memory_undef = nullptr;
  Instruction *src_retval = nullptr;
  Instruction *src_retval_undef = nullptr;
  Instruction *src_unique_ub = nullptr;
  Instruction *src_common_ub = nullptr;

  Instruction *tgt_assert = nullptr;
  Instruction *tgt_memory = nullptr;
  Instruction *tgt_memory_size = nullptr;
  Instruction *tgt_memory_undef = nullptr;
  Instruction *tgt_retval = nullptr;
  Instruction *tgt_retval_undef = nullptr;
  Instruction *tgt_unique_ub = nullptr;
  Instruction *tgt_common_ub = nullptr;
};

z3::expr Converter::ite(z3::expr c, z3::expr a, z3::expr b)
{
  if (z3::eq(a, b))
    return a;
  return z3::ite(c, a, b);
}

z3::expr Converter::inst_as_array(const Instruction *inst)
{
  return inst2array.at(inst);
}

z3::expr Converter::inst_as_bv(const Instruction *inst)
{
  auto I = inst2bv.find(inst);
  if (I != inst2bv.end())
    return I->second;

  if (inst->bitsize == 1)
    {
      // We do not have a bitvector value for inst. This means there must
      // be a Boolean value for this instruction. Convert it to a bitvector.
      z3::expr expr =
	z3::ite(inst2bool.at(inst), ctx.bv_val(1, 1), ctx.bv_val(0, 1));
      inst2bv.insert({inst, expr});
      return expr;
    }
  else
    {
      // We do not have a bitvector value for inst. This means there must
      // be a floating-point value for this instruction. Convert it to a
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

  // We do not have a Boolean value for inst. This means there must be
  // a bitvector value for this instruction. Convert it to a Boolean.
  z3::expr bv = inst2bv.at(inst);
  z3::expr expr = bv != ctx.bv_val(0, 1);
  if (Z3_is_numeral_ast(ctx, bv))
    inst2bool.insert({inst, expr.simplify()});
  else
    inst2bool.insert({inst, expr});
  return expr;
}

z3::expr Converter::inst_as_fp(const Instruction *inst)
{
  auto I = inst2fp.find(inst);
  if (I != inst2fp.end())
    return I->second;

  // We do not have a floating-point value for inst. This means there must
  // be a bitvector value for this instruction. Convert it to floating
  // point.
  z3::expr bv = inst2bv.at(inst);
  Z3_sort sort = fp_sort(inst->bitsize);
  Z3_ast r = Z3_mk_fpa_to_fp_bv(ctx, bv, sort);
  z3::expr expr = z3::expr(ctx, r);
  if (Z3_is_numeral_ast(ctx, bv))
    inst2fp.insert({inst, expr.simplify()});
  else
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

void Converter::build_nullary_smt(const Instruction *inst)
{
  switch (inst->opcode)
    {
    case Op::MEM_ARRAY:
      {
	z3::sort address_sort = ctx.bv_sort(func->module->ptr_bits);
	z3::sort byte_sort = ctx.bv_sort(8);
	z3::sort array_sort = ctx.array_sort(address_sort, byte_sort);
	z3::expr memory = ctx.constant(".memory", array_sort);
	inst2array.insert({inst, memory});
      }
      break;
    case Op::MEM_FLAG_ARRAY:
      {
	z3::sort address_sort = ctx.bv_sort(func->module->ptr_bits);
	z3::expr memory_flag =
	  z3::expr(ctx, Z3_mk_const_array(ctx, address_sort, ctx.bv_val(0, 1)));
	inst2array.insert({inst, memory_flag});
      }
      break;
    case Op::MEM_SIZE_ARRAY:
      {
	z3::sort id_sort = ctx.bv_sort(func->module->ptr_id_bits);
	z3::expr zero_offset = ctx.bv_val(0, func->module->ptr_offset_bits);
	z3::expr memory_size =
	  z3::expr(ctx, Z3_mk_const_array(ctx, id_sort, zero_offset));
	inst2array.insert({inst, memory_size});
      }
      break;
    case Op::MEM_UNDEF_ARRAY:
      {
	z3::sort address_sort = ctx.bv_sort(func->module->ptr_bits);
	z3::expr memory_undef =
	  z3::expr(ctx, Z3_mk_const_array(ctx, address_sort, ctx.bv_val(0, 8)));
	inst2array.insert({inst, memory_undef});
      }
      break;
    default:
      throw Not_implemented("build_nullary_smt: "s + inst->name());
    }
}

void Converter::build_bv_unary_smt(const Instruction *inst)
{
  assert(inst->nof_args == 1);

  // Perform the NOT operation as Boolean if the argument is Boolean. This
  // avoids multiple conversions between bitvector and Boolean for the
  // typical case where 1-bit values are used in logical expressions.
  if (inst->bitsize == 1
      && inst->opcode == Op::NOT
      && inst2bool.contains(inst->arguments[0]))
    {
      z3::expr arg1 = inst_as_bool(inst->arguments[0]);
      inst2bool.insert({inst, !arg1});
      return;
    }

  z3::expr arg1 = inst_as_bv(inst->arguments[0]);
  switch (inst->opcode)
    {
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
    case Op::MOV:
      inst2bv.insert({inst, arg1});
      break;
    case Op::NEG:
      inst2bv.insert({inst, -arg1});
      break;
    case Op::NOT:
      inst2bv.insert({inst, ~arg1});
      break;
    case Op::SRC_ASSERT:
      assert(!src_assert);
      src_assert = inst->arguments[0];
      break;
    case Op::TGT_ASSERT:
      assert(!tgt_assert);
      tgt_assert = inst->arguments[0];
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

  switch (inst->opcode)
    {
    case Op::ARRAY_GET_FLAG:
    case Op::ARRAY_GET_SIZE:
    case Op::ARRAY_GET_UNDEF:
    case Op::ARRAY_LOAD:
      {
	z3::expr arg1 = inst_as_array(inst->arguments[0]);
	z3::expr arg2 = inst_as_bv(inst->arguments[1]);
	inst2bv.insert({inst, z3::select(arg1, arg2)});
      }
      return;
    case Op::SRC_RETVAL:
      assert(!src_retval);
      assert(!src_retval_undef);
      src_retval = inst->arguments[0];
      src_retval_undef = inst->arguments[1];
      return;
    case Op::TGT_RETVAL:
      assert(!tgt_retval);
      assert(!tgt_retval_undef);
      tgt_retval = inst->arguments[0];
      tgt_retval_undef = inst->arguments[1];
      return;
    case Op::SRC_UB:
      assert(!src_unique_ub && !src_common_ub);
      src_common_ub = inst->arguments[0];
      src_unique_ub = inst->arguments[1];
      return;
    case Op::TGT_UB:
      assert(!tgt_unique_ub && !tgt_common_ub);
      tgt_common_ub = inst->arguments[0];
      tgt_unique_ub = inst->arguments[1];
      return;
    default:
      break;
    }

  // Perform AND/OR/XOR operations as Boolean if at least one of the
  // arguments are Boolean. This avoids multiple conversions between
  // bitvector and Boolean for the typical case where 1-bit values
  // are used in logical expressions.
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
	char name[100];
	sprintf(name, ".param%" PRIu32, index);
	z3::expr param = ctx.bv_const(name, inst->bitsize);
	inst2bv.insert({inst, param});
      }
      break;
    case Op::PRINT:
      print.push_back(inst);
      break;
    case Op::SYMBOLIC:
      {
	uint32_t index = inst->arguments[0]->value();
	char name[100];
	sprintf(name, ".symbolic%" PRIu32, index);
	z3::expr symbolic = ctx.bv_const(name, inst->bitsize);
	inst2bv.insert({inst, symbolic});
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
	// We may concatenate two large constants that cannot be constant
	// folded at IR level (as the IR limits constant width to 128 bits),
	// so we we fold it here to get nicer SMT2 code when debugging.
	if (Z3_is_numeral_ast(ctx, arg1) && Z3_is_numeral_ast(ctx, arg2))
	  inst2bv.insert({inst, res.simplify()});
	else
	  inst2bv.insert({inst, res});
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
    case Op::ARRAY_SET_FLAG:
    case Op::ARRAY_SET_SIZE:
    case Op::ARRAY_SET_UNDEF:
    case Op::ARRAY_STORE:
      {
	z3::expr arg1 = inst_as_array(inst->arguments[0]);
	z3::expr arg2 = inst_as_bv(inst->arguments[1]);
	z3::expr arg3 = inst_as_bv(inst->arguments[2]);
	inst2array.insert({inst, z3::store(arg1, arg2, arg3)});
      }
      break;
    case Op::EXTRACT:
      {
	z3::expr arg = inst_as_bv(inst->arguments[0]);
	uint32_t high = inst->arguments[1]->value();
	uint32_t low = inst->arguments[2]->value();
	inst2bv.insert({inst, arg.extract(high, low)});
      }
      break;
    case Op::ITE:
      if (inst2array.contains(inst->arguments[1]))
	{
	  z3::expr arg1 = inst_as_bool(inst->arguments[0]);
	  z3::expr arg2 = inst_as_array(inst->arguments[1]);
	  z3::expr arg3 = inst_as_array(inst->arguments[2]);
	  inst2array.insert({inst, ite(arg1, arg2, arg3)});
	}
      else if (inst->bitsize == 1 &&
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
    case Op::SRC_MEM:
      assert(!src_memory);
      assert(!src_memory_size);
      src_memory = inst->arguments[0];
      src_memory_size = inst->arguments[1];
      src_memory_undef = inst->arguments[2];
      return;
    case Op::TGT_MEM:
      assert(!tgt_memory);
      assert(!tgt_memory_size);
      tgt_memory = inst->arguments[0];
      tgt_memory_size = inst->arguments[1];
      tgt_memory_undef = inst->arguments[2];
      return;
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
    case Op::RET:
      assert(inst->nof_args == 0);
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
    case Inst_class::nullary:
      build_nullary_smt(inst);
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

void Converter::convert_function()
{
  for (auto bb : func->bbs)
    {
      assert(bb->phis.empty());
      for (Instruction *inst = bb->first_inst; inst; inst = inst->next)
	{
	  build_smt(inst);
	}
    }

  // If both src and tgt retval_undef is 0, then it is the same as no
  // retval_undef.
  if (src_retval_undef
      && src_retval_undef == tgt_retval_undef
      && src_retval_undef->opcode == Op::VALUE
      && !src_retval_undef->value())
    {
      src_retval_undef = nullptr;
      tgt_retval_undef = nullptr;
    }
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

void set_solver_limits()
{
  char buf[32];
  sprintf(buf, "%d", config.timeout);
  Z3_global_param_set("timeout", buf);
  sprintf(buf, "%" PRIu64, (uint64_t)config.memory_limit * 1024 * 1024);
  Z3_global_param_set("memory_high_watermark", buf);
}

void add_print(std::string& msg, Converter& conv, z3::solver& solver)
{
  if (conv.print.empty())
    return;

  z3::model model = solver.get_model();
  for (auto inst : conv.print)
    {
      z3::expr id = conv.inst_as_bv(inst->arguments[0]);
      z3::expr value = conv.inst_as_bv(inst->arguments[1]);
      msg += "print " + model.eval(id).to_string() + ": ";
      msg += model.eval(value).to_string() + "\n";
    }
}

} // end anonymous namespace

std::pair<SStats, Solver_result> check_refine_z3(Function *func)
{
  assert(func->bbs.size() == 1);

  z3::context ctx;
  set_solver_limits();

  SStats stats;
  stats.skipped = false;

  Converter conv(ctx, func);
  z3::expr src_common_ub_expr = conv.inst_as_bool(conv.src_common_ub);
  z3::expr src_unique_ub_expr = conv.inst_as_bool(conv.src_unique_ub);
  z3::expr tgt_unique_ub_expr = conv.inst_as_bool(conv.tgt_unique_ub);

  std::string warning;
  if (conv.src_retval != conv.tgt_retval
      || conv.src_retval_undef != conv.tgt_retval_undef)
    {
      assert(conv.src_retval && conv.tgt_retval);
      z3::expr src_expr = conv.inst_as_bv(conv.src_retval);
      z3::expr tgt_expr = conv.inst_as_bv(conv.tgt_retval);

      z3::expr is_more_undef = ctx.bool_val(false);
      if (conv.src_retval_undef)
	{
	  z3::expr src_undef = conv.inst_as_bv(conv.src_retval_undef);
	  z3::expr tgt_undef = conv.inst_as_bv(conv.tgt_retval_undef);
	  z3::expr src_mask = ~src_undef;
	  z3::expr new_src_expr = src_expr & src_mask;
	  src_expr = new_src_expr;
	  z3::expr new_tgt_expr = tgt_expr & src_mask;
	  tgt_expr = new_tgt_expr;

	  // Check that tgt is not more undef than src.
	  if (conv.tgt_retval_undef != conv.src_retval_undef)
	    {
	      z3::expr new_tgt_undef = conv.inst_as_bv(conv.tgt_retval_undef);
	      tgt_undef = new_tgt_undef;
	      z3::expr new_is_more_undef = (src_mask & tgt_undef) != 0;
	      is_more_undef = new_is_more_undef;
	    }
	}

      z3::solver solver(ctx);
      solver.add(!src_common_ub_expr);
      solver.add(!src_unique_ub_expr);
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
	  if (conv.src_retval_undef)
	    {
	      z3::expr src_undef = conv.inst_as_bv(conv.src_retval_undef);
	      z3::expr tgt_undef = conv.inst_as_bv(conv.tgt_retval_undef);
	      msg = msg + "src undef: " + model.eval(src_undef).to_string() + "\n";
	      msg = msg +  "tgt undef: " + model.eval(tgt_undef).to_string() + "\n";
	    }
	  add_print(msg, conv, solver);
	  Solver_result result = {Result_status::incorrect, msg};
	  return std::pair<SStats, Solver_result>(stats, result);
	}
      if (solver_result.status == Result_status::unknown)
	{
	  assert(solver_result.message);
	  warning = warning + *solver_result.message;
	}
    }

  // Check that the global memory is consistent for src and tgt.
  if (conv.src_memory != conv.tgt_memory
      || conv.src_memory_size != conv.tgt_memory_size
      || conv.src_memory_undef != conv.tgt_memory_undef)
  {
    z3::expr src_mem = conv.inst_as_array(conv.src_memory);
    z3::expr src_mem_size = conv.inst_as_array(conv.src_memory_size);
    z3::expr src_mem_undef = conv.inst_as_array(conv.src_memory_undef);

    z3::expr tgt_mem = conv.inst_as_array(conv.tgt_memory);
    z3::expr tgt_mem_undef = conv.inst_as_array(conv.tgt_memory_undef);

    z3::expr ptr = ctx.bv_const(".ptr", func->module->ptr_bits);
    uint32_t ptr_id_high = func->module->ptr_id_high;
    uint32_t ptr_id_low = func->module->ptr_id_low;
    z3::expr id = ptr.extract(ptr_id_high, ptr_id_low);
    uint32_t ptr_offset_high = func->module->ptr_offset_high;
    uint32_t ptr_offset_low = func->module->ptr_offset_low;
    z3::expr offset = ptr.extract(ptr_offset_high, ptr_offset_low);

    z3::solver solver(ctx);
    solver.add(!src_common_ub_expr);
    solver.add(!src_unique_ub_expr);

    // Only check global memory.
    solver.add(id > 0);

    // Only check memory within a memory block.
    solver.add(z3::ult(offset, z3::select(src_mem_size, id)));

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
	add_print(msg, conv, solver);
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
  // has low priority.
  assert(conv.src_common_ub == conv.tgt_common_ub);
  if (conv.src_unique_ub != conv.tgt_unique_ub
      && !(conv.tgt_unique_ub->opcode == Op::VALUE
	   && conv.tgt_unique_ub->value() == 0))
  {
    z3::solver solver(ctx);
    solver.add(!src_common_ub_expr);
    solver.add(!src_unique_ub_expr);
    solver.add(tgt_unique_ub_expr);
    uint64_t start_time = get_time();
    Solver_result solver_result = run_solver(solver, "UB");
    stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
    if (solver_result.status == Result_status::incorrect)
      {
	assert(solver_result.message);
	std::string msg = *solver_result.message;
	add_print(msg, conv, solver);
	Solver_result result = {Result_status::incorrect, msg};
	return std::pair<SStats, Solver_result>(stats, result);
      }
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
  assert(func->bbs.size() == 1);

  z3::context ctx;
  set_solver_limits();

  SStats stats;
  stats.skipped = false;

  Converter conv(ctx, func);
  z3::expr src_unique_ub_expr = conv.inst_as_bool(conv.src_unique_ub);
  z3::expr src_common_ub_expr = conv.inst_as_bool(conv.src_common_ub);

  z3::solver solver(ctx);
  solver.add(src_common_ub_expr || src_unique_ub_expr);
  uint64_t start_time = get_time();
  Solver_result solver_result = run_solver(solver, "UB");
  stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
  return std::pair<SStats, Solver_result>(stats, solver_result);
}

std::pair<SStats, Solver_result> check_assert_z3(Function *func)
{
  assert(func->bbs.size() == 1);

  z3::context ctx;
  set_solver_limits();

  SStats stats;
  stats.skipped = false;

  Converter conv(ctx, func);
  z3::expr src_unique_ub_expr = conv.inst_as_bool(conv.src_unique_ub);
  z3::expr src_common_ub_expr = conv.inst_as_bool(conv.src_common_ub);
  z3::expr assert_expr = conv.inst_as_bool(conv.src_assert);

  z3::solver solver(ctx);
  solver.add(!src_common_ub_expr);
  solver.add(!src_unique_ub_expr);
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
