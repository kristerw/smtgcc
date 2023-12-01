#include "config.h"

#if HAVE_LIBCVC5
#include <cassert>
#include <cinttypes>
#include <cvc5/cvc5.h>

#include "smtgcc.h"

using namespace std::string_literals;

namespace smtgcc {

namespace {

class Converter {
  std::map<const Instruction *, cvc5::Term> inst2array;
  std::map<const Instruction *, cvc5::Term> inst2bv;
  std::map<const Instruction *, cvc5::Term> inst2fp;
  std::map<const Instruction *, cvc5::Term> inst2bool;

  cvc5::Term ite(cvc5::Term c, cvc5::Term a, cvc5::Term b);
  cvc5::Op fp_sort(cvc5::Kind kind, uint32_t bitsize);
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

  cvc5::Solver& solver;
  const Function *func;

public:
  Converter(cvc5::Solver& solver, const Function *func)
    : solver{solver}
    , func{func}
  {
    convert_function();
  }
  cvc5::Term inst_as_array(const Instruction *inst);
  cvc5::Term inst_as_bv(const Instruction *inst);
  cvc5::Term inst_as_fp(const Instruction *inst);
  cvc5::Term inst_as_bool(const Instruction *inst);

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

cvc5::Term Converter::ite(cvc5::Term c, cvc5::Term a, cvc5::Term b)
{
  if (a == b)
    return a;
  return solver.mkTerm(cvc5::ITE, {c, a, b});
}

cvc5::Term Converter::inst_as_array(const Instruction *inst)
{
  return inst2array.at(inst);
}

cvc5::Term Converter::inst_as_bv(const Instruction *inst)
{
  auto I = inst2bv.find(inst);
  if (I != inst2bv.end())
    return I->second;

  if (inst->bitsize == 1)
    {
      // We do not have a bitvector value for inst. This means there must
      // be a Boolean value for this instruction. Convert it to a bitvector.
      cvc5::Term term =
	ite(inst2bool.at(inst), solver.mkBitVector(1, 1),
	    solver.mkBitVector(1, 0));
      inst2bv.insert({inst, term});
      return term;
    }
  else
    {
      // We do not have a bitvector value for inst. This means there must
      // be a floating point value for this instruction. Convert it to a
      // bitvector.
      throw Not_implemented("inst_as_bv: convert fp to bitvector");
    }
}

cvc5::Term Converter::inst_as_bool(const Instruction *inst)
{
  assert(inst->bitsize == 1);
  auto I = inst2bool.find(inst);
  if (I != inst2bool.end())
    return I->second;

  // We do not have a Boolean value for inst. This means there must be
  // a bitvector value for this instruction. Convert it to a Boolean.
  cvc5::Term bv = inst2bv.at(inst);
  cvc5::Term term = solver.mkTerm(cvc5::EQUAL, {bv, solver.mkBitVector(1, 1)});
  inst2bool.insert({inst, term});
  return term;
}

cvc5::Term Converter::inst_as_fp(const Instruction *inst)
{
  auto I = inst2fp.find(inst);
  if (I != inst2fp.end())
    return I->second;

  // We do not have a floating-point value for inst. This means there
  // be a bitvector value for this instruction. Convert it to floating
  // point.
  throw Not_implemented("inst_as_fp: bitvector to fp");
}

void Converter::build_bv_comparison_smt(const Instruction *inst)
{
  assert(inst->nof_args == 2);

  if (inst->arguments[0]->bitsize == 1
      && (inst->opcode == Op::EQ || inst->opcode == Op::NE)
      && (inst2bool.contains(inst->arguments[0])
	  && inst2bool.contains(inst->arguments[1])))
    {
      cvc5::Term arg1 = inst_as_bool(inst->arguments[0]);
      cvc5::Term arg2 = inst_as_bool(inst->arguments[1]);

      if (inst->opcode == Op::EQ)
	inst2bool.insert({inst, solver.mkTerm(cvc5::EQUAL, {arg1, arg2})});
      else
	inst2bool.insert({inst, solver.mkTerm(cvc5::DISTINCT, {arg1, arg2})});
      return;
    }

  cvc5::Term arg1 = inst_as_bv(inst->arguments[0]);
  cvc5::Term arg2 = inst_as_bv(inst->arguments[1]);
  switch (inst->opcode)
    {
    case Op::EQ:
      inst2bool.insert({inst, solver.mkTerm(cvc5::EQUAL, {arg1, arg2})});
      break;
    case Op::NE:
      inst2bool.insert({inst, solver.mkTerm(cvc5::DISTINCT, {arg1, arg2})});
      break;
    case Op::SGE:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SGE, {arg1, arg2})});
      break;
    case Op::SGT:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SGT, {arg1, arg2})});
      break;
    case Op::SLE:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SLE, {arg1, arg2})});
      break;
    case Op::SLT:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SLT, {arg1, arg2})});
      break;
    case Op::UGE:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_UGE, {arg1, arg2})});
      break;
    case Op::UGT:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_UGT, {arg1, arg2})});
      break;
    case Op::ULE:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_ULE, {arg1, arg2})});
      break;
    case Op::ULT:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_ULT, {arg1, arg2})});
      break;
    default:
      throw Not_implemented("build_comparison_smt: "s + inst->name());
    }
}

cvc5::Op Converter::fp_sort(cvc5::Kind kind, uint32_t bitsize)
{
  switch (bitsize)
    {
      // TODO: 16 and 128 bits
    case 32:
      return solver.mkOp(kind, {8, 24});
    case 64:
      return solver.mkOp(kind, {11, 53});
    default:
      throw Not_implemented("fp_sort: f" + std::to_string(bitsize));
    }
}

void Converter::build_fp_comparison_smt(const Instruction *inst)
{
  assert(inst->nof_args == 2);
  cvc5::Term arg1 = inst_as_fp(inst->arguments[0]);
  cvc5::Term arg2 = inst_as_fp(inst->arguments[1]);

  switch (inst->opcode)
    {
    case Op::FEQ:
      {
	cvc5::Term eq = solver.mkTerm(cvc5::FLOATINGPOINT_EQ, {arg1, arg2});
	inst2bool.insert({inst, eq});
      }
      break;
    case Op::FNE:
      {
	cvc5::Term eq = solver.mkTerm(cvc5::FLOATINGPOINT_EQ, {arg1, arg2});
	inst2bool.insert({inst, solver.mkTerm(cvc5::NOT, {eq})});
      }
      break;
    case Op::FGE:
      {
	cvc5::Term geq = solver.mkTerm(cvc5::FLOATINGPOINT_GEQ, {arg1, arg2});
	inst2bool.insert({inst, geq});
      }
      break;
    case Op::FGT:
      {
	cvc5::Term gt = solver.mkTerm(cvc5::FLOATINGPOINT_GT, {arg1, arg2});
	inst2bool.insert({inst, gt});
      }
      break;
    case Op::FLE:
      {
	cvc5::Term leq = solver.mkTerm(cvc5::FLOATINGPOINT_LEQ, {arg1, arg2});
	inst2bool.insert({inst, leq});
      }
      break;
    case Op::FLT:
      {
	cvc5::Term lt = solver.mkTerm(cvc5::FLOATINGPOINT_LT, {arg1, arg2});
	inst2bool.insert({inst, lt});
      }
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
	cvc5::Sort address_sort =
	  solver.mkBitVectorSort(func->module->ptr_bits);
	cvc5::Sort byte_sort = solver.mkBitVectorSort(8);
	cvc5::Sort array_sort = solver.mkArraySort(address_sort, byte_sort);
	cvc5::Term memory = solver.mkConst(array_sort, ".memory");
	inst2array.insert({inst, memory});
      }
      break;
    case Op::MEM_FLAG_ARRAY:
      {
	cvc5::Sort address_sort =
	  solver.mkBitVectorSort(func->module->ptr_bits);
	cvc5::Sort bit_sort = solver.mkBitVectorSort(1);
	cvc5::Sort array_sort = solver.mkArraySort(address_sort, bit_sort);
	cvc5::Term memory_flag =
	  solver.mkConstArray(array_sort, solver.mkBitVector(1, 0));
	inst2array.insert({inst, memory_flag});
      }
      break;
    case Op::MEM_SIZE_ARRAY:
      {
	uint32_t ptr_id_bits = func->module->ptr_id_bits;
	uint32_t ptr_offset_bits = func->module->ptr_offset_bits;
	cvc5::Sort id_sort = solver.mkBitVectorSort(ptr_id_bits);
	cvc5::Sort offset_sort = solver.mkBitVectorSort(ptr_offset_bits);
	cvc5::Sort array_sort = solver.mkArraySort(id_sort, offset_sort);
	cvc5::Term memory_size =
	  solver.mkConstArray(array_sort,
			      solver.mkBitVector(ptr_offset_bits, 0));
	inst2array.insert({inst, memory_size});
      }
      break;
    case Op::MEM_UNDEF_ARRAY:
      {
	cvc5::Sort address_sort =
	  solver.mkBitVectorSort(func->module->ptr_bits);
	cvc5::Sort byte_sort = solver.mkBitVectorSort(8);
	cvc5::Sort array_sort = solver.mkArraySort(address_sort, byte_sort);
	cvc5::Term memory_undef =
	  solver.mkConstArray(array_sort, solver.mkBitVector(8, 0));
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

  cvc5::Term arg1 = inst_as_bv(inst->arguments[0]);
  switch (inst->opcode)
    {
    case Op::IS_NAN:
      // TODO: Implement Op::IS_NAN
      throw Not_implemented("build_bv_unary_smt: "s + inst->name());
    case Op::IS_NONCANONICAL_NAN:
      // TODO: Implement Op::IS_NONCANONICAL_NAN
      throw Not_implemented("build_bv_unary_smt: "s + inst->name());
    case Op::MOV:
      inst2bv.insert({inst, arg1});
      break;
    case Op::NEG:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_NEG, {arg1})});
      break;
    case Op::NOT:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_NOT, {arg1})});
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
  cvc5::Term arg1 = inst_as_fp(inst->arguments[0]);
  switch (inst->opcode)
    {
    case Op::FABS:
      inst2fp.insert({inst, solver.mkTerm(cvc5::FLOATINGPOINT_ABS, {arg1})});
      break;
    case Op::FNEG:
      inst2fp.insert({inst, solver.mkTerm(cvc5::FLOATINGPOINT_NEG, {arg1})});
      break;
    case Op::NAN:
      // TODO: Implement Op::NAN
      throw Not_implemented("build_fp_unary_smt: "s + inst->name());
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
	cvc5::Term arg1 = inst_as_array(inst->arguments[0]);
	cvc5::Term arg2 = inst_as_bv(inst->arguments[1]);
	inst2bv.insert({inst, solver.mkTerm(cvc5::SELECT, {arg1, arg2})});
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

  cvc5::Term arg1 = inst_as_bv(inst->arguments[0]);
  cvc5::Term arg2 = inst_as_bv(inst->arguments[1]);
  switch (inst->opcode)
    {
    case Op::ADD:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_ADD, {arg1, arg2})});
      break;
    case Op::SUB:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SUB, {arg1, arg2})});
      break;
    case Op::MUL:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_MULT, {arg1, arg2})});
      break;
    case Op::PARAM:
      {
	uint32_t index = inst->arguments[0]->value();
	char name[100];
	sprintf(name, ".param%" PRIu32, index);
	cvc5::Sort sort = solver.mkBitVectorSort(inst->bitsize);
	cvc5::Term param = solver.mkConst(sort, name);
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
	cvc5::Sort sort = solver.mkBitVectorSort(inst->bitsize);
	cvc5::Term symbolic = solver.mkConst(sort, name);
	inst2bv.insert({inst, symbolic});
      }
      break;
    case Op::SDIV:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SDIV, {arg1, arg2})});
      break;
    case Op::UDIV:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_UDIV, {arg1, arg2})});
      break;
    case Op::UMAX:
      {
	cvc5::Term cond = solver.mkTerm(cvc5::BITVECTOR_UGE, {arg1, arg2});
	inst2bv.insert({inst, ite(cond, arg1, arg2)});
      }
      break;
    case Op::UMIN:
      {
	cvc5::Term cond = solver.mkTerm(cvc5::BITVECTOR_ULT, {arg1, arg2});
	inst2bv.insert({inst, ite(cond, arg1, arg2)});
      }
      break;
    case Op::SADD_WRAPS:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SADDO, {arg1, arg2})});
      break;
    case Op::SMAX:
      {
	cvc5::Term cond = solver.mkTerm(cvc5::BITVECTOR_SGE, {arg1, arg2});
	inst2bv.insert({inst, ite(cond, arg1, arg2)});
      }
      break;
    case Op::SMIN:
      {
	cvc5::Term cond = solver.mkTerm(cvc5::BITVECTOR_SLT, {arg1, arg2});
	inst2bv.insert({inst, ite(cond, arg1, arg2)});
      }
      break;
    case Op::SMUL_WRAPS:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SMULO, {arg1, arg2})});
      break;
    case Op::SREM:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SREM, {arg1, arg2})});
      break;
    case Op::SSUB_WRAPS:
      inst2bool.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SSUBO, {arg1, arg2})});
      break;
    case Op::UREM:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_UREM, {arg1, arg2})});
      break;
    case Op::ASHR:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_ASHR, {arg1, arg2})});
      break;
    case Op::LSHR:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_LSHR, {arg1, arg2})});
      break;
    case Op::SHL:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_SHL, {arg1, arg2})});
      break;
    case Op::AND:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_AND, {arg1, arg2})});
      break;
    case Op::OR:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_OR, {arg1, arg2})});
      break;
    case Op::XOR:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_XOR, {arg1, arg2})});
      break;
    case Op::CONCAT:
      {
	cvc5::Term res = solver.mkTerm(cvc5::BITVECTOR_CONCAT, {arg1, arg2});
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
  cvc5::Term arg1 = inst_as_fp(inst->arguments[0]);
  cvc5::Term arg2 = inst_as_fp(inst->arguments[1]);
  cvc5::Term rm = solver.mkRoundingMode(cvc5::ROUND_NEAREST_TIES_TO_EVEN);
  switch (inst->opcode)
    {
    case Op::FADD:
      {
	cvc5::Term term =
	  solver.mkTerm(cvc5::FLOATINGPOINT_ADD, {rm, arg1, arg2});
	inst2fp.insert({inst, term});
      }
      break;
    case Op::FSUB:
      {
	cvc5::Term term =
	  solver.mkTerm(cvc5::FLOATINGPOINT_SUB, {rm, arg1, arg2});
	inst2fp.insert({inst, term});
      }
      break;
    case Op::FMUL:
      {
	cvc5::Term term =
	  solver.mkTerm(cvc5::FLOATINGPOINT_MULT, {rm, arg1, arg2});
	inst2fp.insert({inst, term});
      }
      break;
    case Op::FDIV:
      {
	cvc5::Term term =
	  solver.mkTerm(cvc5::FLOATINGPOINT_DIV, {rm, arg1, arg2});
	inst2fp.insert({inst, term});
      }
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
	cvc5::Term arg1 = inst_as_array(inst->arguments[0]);
	cvc5::Term arg2 = inst_as_bv(inst->arguments[1]);
	cvc5::Term arg3 = inst_as_bv(inst->arguments[2]);
	cvc5::Term array = solver.mkTerm(cvc5::STORE, {arg1, arg2, arg3});
	inst2array.insert({inst, array});
      }
      break;
    case Op::EXTRACT:
      {
	cvc5::Term arg = inst_as_bv(inst->arguments[0]);
	uint32_t high = inst->arguments[1]->value();
	uint32_t low = inst->arguments[2]->value();
	cvc5::Op extract = solver.mkOp(cvc5::BITVECTOR_EXTRACT, {high, low});
	inst2bv.insert({inst, solver.mkTerm(extract, {arg})});
      }
      break;
    case Op::ITE:
      if (inst2array.contains(inst->arguments[1]))
	{
	  cvc5::Term arg1 = inst_as_bool(inst->arguments[0]);
	  cvc5::Term arg2 = inst_as_array(inst->arguments[1]);
	  cvc5::Term arg3 = inst_as_array(inst->arguments[2]);
	  inst2array.insert({inst, ite(arg1, arg2, arg3)});
	}
      else
	{
	  cvc5::Term arg1 = inst_as_bool(inst->arguments[0]);
	  cvc5::Term arg2 = inst_as_bv(inst->arguments[1]);
	  cvc5::Term arg3 = inst_as_bv(inst->arguments[2]);
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
	cvc5::Term arg = inst_as_bv(inst->arguments[0]);
	assert(inst->arguments[0]->bitsize < inst->bitsize);
	unsigned bits =  inst->bitsize - inst->arguments[0]->bitsize;
	cvc5::Op op = solver.mkOp(cvc5::BITVECTOR_SIGN_EXTEND, {bits});
	inst2bv.insert({inst, solver.mkTerm(op, {arg})});
      }
      break;
    case Op::ZEXT:
      {
	cvc5::Term arg = inst_as_bv(inst->arguments[0]);
	assert(inst->arguments[0]->bitsize < inst->bitsize);
	unsigned bits =  inst->bitsize - inst->arguments[0]->bitsize;
	cvc5::Op op = solver.mkOp(cvc5::BITVECTOR_ZERO_EXTEND, {bits});
	inst2bv.insert({inst, solver.mkTerm(op, {arg})});
      }
      break;
    case Op::F2U:
      {
	cvc5::Term arg = inst_as_fp(inst->arguments[0]);
	cvc5::Term rtz = solver.mkRoundingMode(cvc5::ROUND_TOWARD_ZERO);
	cvc5::Op op = solver.mkOp(cvc5::FLOATINGPOINT_TO_UBV, {inst->bitsize});
	inst2bv.insert({inst, solver.mkTerm(op, {rtz, arg})});
      }
      break;
    case Op::F2S:
      {
	cvc5::Term arg = inst_as_fp(inst->arguments[0]);
	cvc5::Term rtz = solver.mkRoundingMode(cvc5::ROUND_TOWARD_ZERO);
	cvc5::Op op = solver.mkOp(cvc5::FLOATINGPOINT_TO_SBV, {inst->bitsize});
	inst2bv.insert({inst, solver.mkTerm(op, {rtz, arg})});
      }
      break;
    case Op::S2F:
      {
	cvc5::Term arg = inst_as_bv(inst->arguments[0]);
	cvc5::Term rm = solver.mkRoundingMode(cvc5::ROUND_NEAREST_TIES_TO_EVEN);
	cvc5::Op op =
	  fp_sort(cvc5::FLOATINGPOINT_TO_FP_FROM_SBV, inst->bitsize);
	inst2fp.insert({inst, solver.mkTerm(op, {rm, arg})});
      }
      break;
    case Op::U2F:
      {
	cvc5::Term arg = inst_as_bv(inst->arguments[0]);
	cvc5::Term rm = solver.mkRoundingMode(cvc5::ROUND_NEAREST_TIES_TO_EVEN);
	cvc5::Op op =
	  fp_sort(cvc5::FLOATINGPOINT_TO_FP_FROM_UBV, inst->bitsize);
	inst2fp.insert({inst, solver.mkTerm(op, {rm, arg})});
      }
      break;
    case Op::FCHPREC:
      // TODO: Implement Op::FCHPREC
      throw Not_implemented("build_conversion_smt: "s + inst->name());
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
	    cvc5::Term lo = solver.mkBitVector(64, low);
	    cvc5::Term hi = solver.mkBitVector(inst->bitsize - 64, high);
	    inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_CONCAT, {hi, lo})});
	  }
	else
	  inst2bv.insert({inst, solver.mkBitVector(inst->bitsize, low)});
	if (inst->bitsize == 1)
	  inst2bool.insert({inst, solver.mkBoolean(low != 0)});
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

Solver_result run_solver(cvc5::Solver& solver, const char *str)
{
  cvc5::Result result = solver.checkSat();
  if (result.isUnsat())
    {
      return {Result_status::correct, {}};
    }
  else if (result.isSat())
    {
      std::string msg = "Transformation is not correct ("s + str + ")\n";
      // TODO: Display the model.
      return {Result_status::incorrect, msg};
    }
  else if (result.isUnknown())
    {
      std::string msg = "Analysis timed out ("s + str + ")\n";
      return {Result_status::unknown, msg};
    }

  throw Not_implemented("run_solver: unknown solver.check return");
}

void set_solver_limits(cvc5::Solver& solver)
{
  solver.setOption("produce-models", "true");
  char buf[32];
  sprintf(buf, "%d", config.timeout);
  solver.setOption("tlimit-per", buf);
  solver.setLogic("QF_ABVFP");
}

} // end anonymous namespace

std::pair<SStats, Solver_result> check_refine_cvc5(Function *func)
{
  assert(func->bbs.size() == 1);

  cvc5::Solver solver;
  set_solver_limits(solver);

  SStats stats;
  stats.skipped = false;

  Converter conv(solver, func);
  cvc5::Term src_common_ub_term = conv.inst_as_bool(conv.src_common_ub);
  cvc5::Term src_unique_ub_term = conv.inst_as_bool(conv.src_unique_ub);
  cvc5::Term not_src_common_ub_term =
    solver.mkTerm(cvc5::NOT, {src_common_ub_term});
  cvc5::Term not_src_unique_ub_term =
    solver.mkTerm(cvc5::NOT, {src_unique_ub_term});
  cvc5::Term tgt_unique_ub_term = conv.inst_as_bool(conv.tgt_unique_ub);

  std::string warning;
  if (conv.src_retval != conv.tgt_retval
      || conv.src_retval_undef != conv.tgt_retval_undef)
    {
      solver.push();
      assert(conv.src_retval && conv.tgt_retval);
      cvc5::Term src_term = conv.inst_as_bv(conv.src_retval);
      cvc5::Term tgt_term = conv.inst_as_bv(conv.tgt_retval);

      cvc5::Term is_more_undef = solver.mkBoolean(false);
      if (conv.src_retval_undef)
	{
	  cvc5::Term src_undef = conv.inst_as_bv(conv.src_retval_undef);
	  cvc5::Term src_mask = solver.mkTerm(cvc5::BITVECTOR_NOT, {src_undef});
	  src_term = solver.mkTerm(cvc5::BITVECTOR_AND, {src_term, src_mask});
	  tgt_term = solver.mkTerm(cvc5::BITVECTOR_AND, {tgt_term, src_mask});

	  // Check that tgt is not more undef than src.
	  if (conv.tgt_retval_undef != conv.src_retval_undef)
	    {
	      cvc5::Term tgt_undef = conv.inst_as_bv(conv.tgt_retval_undef);
	      cvc5::Term undef_result =
		solver.mkTerm(cvc5::BITVECTOR_AND, {tgt_undef, src_mask});
	      cvc5::Term zero = solver.mkBitVector(conv.tgt_retval->bitsize, 0);
	      is_more_undef =
		solver.mkTerm(cvc5::DISTINCT, {undef_result, zero});
	    }
	}

      solver.assertFormula(not_src_common_ub_term);
      solver.assertFormula(not_src_unique_ub_term);
      cvc5::Term res1 = solver.mkTerm(cvc5::DISTINCT, {src_term, tgt_term});
      cvc5::Term res2 = solver.mkTerm(cvc5::OR, {res1, is_more_undef});
      solver.assertFormula(res2);
      uint64_t start_time = get_time();
      Solver_result solver_result = run_solver(solver, "retval");
      stats.time[0] = std::max(get_time() - start_time, (uint64_t)1);
      if (solver_result.status == Result_status::incorrect)
	{
	  assert(solver_result.message);
	  cvc5::Term src_val = solver.getValue(src_term);
	  cvc5::Term tgt_val = solver.getValue(tgt_term);
	  std::string msg = *solver_result.message;
	  msg = msg + "src retval: " + src_val.getBitVectorValue(16) + "\n";
	  msg = msg + "tgt retval: " + tgt_val.getBitVectorValue(16) + "\n";
	  if (conv.src_retval_undef)
	    {
	      cvc5::Term src_undef = conv.inst_as_bv(conv.src_retval_undef);
	      cvc5::Term tgt_undef = conv.inst_as_bv(conv.tgt_retval_undef);
	      cvc5::Term src_undef_val = solver.getValue(src_undef);
	      cvc5::Term tgt_undef_val = solver.getValue(tgt_undef);
	      msg = msg + "src undef: " + src_undef_val.getBitVectorValue(16) + "\n";
	      msg = msg + "tgt undef: " + tgt_undef_val.getBitVectorValue(16) + "\n";
	    }
	  Solver_result result = {Result_status::incorrect, msg};
	  return std::pair<SStats, Solver_result>(stats, result);
	}
      if (solver_result.status == Result_status::unknown)
	{
	  assert(solver_result.message);
	  warning = warning + *solver_result.message;
	}
      solver.pop();
    }

  // Check that the global memory is consistent for src and tgt.
  if (conv.src_memory != conv.tgt_memory
      || conv.src_memory_size != conv.tgt_memory_size
      || conv.src_memory_undef != conv.tgt_memory_undef)
  {
    solver.push();
    solver.assertFormula(not_src_common_ub_term);
    solver.assertFormula(not_src_unique_ub_term);
    cvc5::Term src_mem = conv.inst_as_array(conv.src_memory);
    cvc5::Term src_mem_size = conv.inst_as_array(conv.src_memory_size);
    cvc5::Term src_mem_undef = conv.inst_as_array(conv.src_memory_undef);

    cvc5::Term tgt_mem = conv.inst_as_array(conv.tgt_memory);
    cvc5::Term tgt_mem_undef = conv.inst_as_array(conv.tgt_memory_undef);

    cvc5::Sort ptr_sort = solver.mkBitVectorSort(func->module->ptr_bits);
    cvc5::Term ptr = solver.mkConst(ptr_sort, ".ptr");
    uint32_t ptr_id_high = func->module->ptr_id_high;
    uint32_t ptr_id_low = func->module->ptr_id_low;
    cvc5::Op id_op =
      solver.mkOp(cvc5::BITVECTOR_EXTRACT, {ptr_id_high, ptr_id_low});
    cvc5::Term id = solver.mkTerm(id_op, {ptr});
    uint32_t ptr_offset_high = func->module->ptr_offset_high;
    uint32_t ptr_offset_low = func->module->ptr_offset_low;
    cvc5::Op offset_op =
      solver.mkOp(cvc5::BITVECTOR_EXTRACT, {ptr_offset_high, ptr_offset_low});
    cvc5::Term offset = solver.mkTerm(offset_op, {ptr});

    // Only check global memory.
    cvc5::Term zero_id = solver.mkBitVector(func->module->ptr_id_bits, 0);
    cvc5::Term cond1 = solver.mkTerm(cvc5::BITVECTOR_SGT, {id, zero_id});
    solver.assertFormula(cond1);

    // Only check memory within a memory block.
    cvc5::Term mem_size = solver.mkTerm(cvc5::SELECT, {src_mem_size, id});
    cvc5::Term cond2 = solver.mkTerm(cvc5::BITVECTOR_ULT, {offset, mem_size});
    solver.assertFormula(cond2);

    // Check that src and tgt are the same for the bits where src is defined
    // and that tgt is not more undefined than src.
    cvc5::Term src_undef = solver.mkTerm(cvc5::SELECT, {src_mem_undef, ptr});
    cvc5::Term src_mask = solver.mkTerm(cvc5::BITVECTOR_NOT, {src_undef});
    cvc5::Term src_byte = solver.mkTerm(cvc5::SELECT, {src_mem, ptr});
    src_byte = solver.mkTerm(cvc5::BITVECTOR_AND, {src_byte, src_mask});
    cvc5::Term tgt_byte = solver.mkTerm(cvc5::SELECT, {tgt_mem, ptr});
    tgt_byte = solver.mkTerm(cvc5::BITVECTOR_AND, {tgt_byte, src_mask});
    cvc5::Term cond3 = solver.mkTerm(cvc5::DISTINCT, {src_byte, tgt_byte});
    cvc5::Term tgt_undef = solver.mkTerm(cvc5::SELECT, {tgt_mem_undef, ptr});
    cvc5::Term tgt_more_undef =
      solver.mkTerm(cvc5::BITVECTOR_AND, {tgt_undef, src_mask});
    cvc5::Term zero_byte = solver.mkBitVector(8, 0);
    cvc5::Term cond4 = solver.mkTerm(cvc5::DISTINCT, {tgt_more_undef, zero_byte});
    solver.assertFormula(solver.mkTerm(cvc5::OR, {cond3, cond4}));

    // TODO: Should make a better getBitVectorValue that prints values as
    // hex, etc.
    uint64_t start_time = get_time();
    Solver_result solver_result = run_solver(solver, "Memory");
    stats.time[1] = std::max(get_time() - start_time, (uint64_t)1);
    if (solver_result.status == Result_status::incorrect)
      {
	assert(solver_result.message);
	cvc5::Term ptr_val = solver.getValue(ptr);
	cvc5::Term src_byte_val = solver.getValue(src_byte);
	cvc5::Term tgt_byte_val = solver.getValue(tgt_byte);
	cvc5::Term src_undef_val = solver.getValue(src_undef);
	cvc5::Term tgt_undef_val = solver.getValue(tgt_undef);
	std::string msg = *solver_result.message;
	msg = msg + "\n.ptr = " + ptr_val.getBitVectorValue(16) + "\n";
	msg = msg + "src *.ptr: " + src_byte_val.getBitVectorValue(16) + "\n";
	msg = msg + "tgt *.ptr: " + tgt_byte_val.getBitVectorValue(16) + "\n";
	msg = msg + "src undef: " + src_undef_val.getBitVectorValue(16) + "\n";
	msg = msg + "tgt undef: " + tgt_undef_val.getBitVectorValue(16) + "\n";
	Solver_result result = {Result_status::incorrect, msg};
	return std::pair<SStats, Solver_result>(stats, result);
      }
    if (solver_result.status == Result_status::unknown)
      {
	assert(solver_result.message);
	warning = warning + *solver_result.message;
      }
    solver.pop();
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
    solver.push();
    solver.assertFormula(not_src_common_ub_term);
    solver.assertFormula(not_src_unique_ub_term);
    solver.assertFormula(tgt_unique_ub_term);
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
    solver.pop();
  }

  if (!warning.empty())
    {
      Solver_result result = {Result_status::unknown, warning};
      return std::pair<SStats, Solver_result>(stats, result);
    }
  return std::pair<SStats, Solver_result>(stats, {Result_status::correct, {}});
}

std::pair<SStats, Solver_result> check_ub_cvc5(Function *func)
{
  assert(func->bbs.size() == 1);

  cvc5::Solver solver;
  set_solver_limits(solver);

  SStats stats;
  stats.skipped = false;

  Converter conv(solver, func);
  cvc5::Term common_ub_term = conv.inst_as_bool(conv.src_common_ub);
  cvc5::Term unique_ub_term = conv.inst_as_bool(conv.src_unique_ub);
  cvc5::Term ub_term = solver.mkTerm(cvc5::BITVECTOR_OR,
				     {common_ub_term, unique_ub_term});
  solver.push();
  solver.assertFormula(ub_term);
  uint64_t start_time = get_time();
  Solver_result solver_result = run_solver(solver, "UB");
  stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
  return std::pair<SStats, Solver_result>(stats, solver_result);
}

std::pair<SStats, Solver_result> check_assert_cvc5(Function *func)
{
  assert(func->bbs.size() == 1);

  cvc5::Solver solver;
  set_solver_limits(solver);

  SStats stats;
  stats.skipped = false;

  Converter conv(solver, func);
  cvc5::Term common_ub_term = conv.inst_as_bool(conv.src_common_ub);
  cvc5::Term unique_ub_term = conv.inst_as_bool(conv.src_unique_ub);
  cvc5::Term not_common_ub_term = solver.mkTerm(cvc5::NOT, {common_ub_term});
  cvc5::Term not_unique_ub_term = solver.mkTerm(cvc5::NOT, {unique_ub_term});
  cvc5::Term assert_term = conv.inst_as_bool(conv.src_assert);

  solver.push();
  solver.assertFormula(not_common_ub_term);
  solver.assertFormula(not_unique_ub_term);
  solver.assertFormula(assert_term);
  uint64_t start_time = get_time();
  Solver_result solver_result = run_solver(solver, "UB");
  stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
  return std::pair<SStats, Solver_result>(stats, solver_result);
}

} // end namespace smtgcc

#else

#include "smtgcc.h"

namespace smtgcc {

SStats verify_cvc5(Function *, Function *)
{
  throw Not_implemented("cvc5 is not available");
}

} // end namespace smtgcc

#endif
