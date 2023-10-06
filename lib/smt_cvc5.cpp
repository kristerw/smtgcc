#include "config.h"

#if HAVE_LIBCVC5
#include <cassert>
#include <iostream>
#include <cvc5/cvc5.h>

#include "smtgcc.h"

using namespace std::string_literals;

namespace smtgcc {

namespace {

class Common {
public:
  std::map<int, cvc5::Term> index2param;
  cvc5::Solver& solver;
  cvc5::Term memory;
  cvc5::Term memory_flag;
  cvc5::Term memory_undef;
  cvc5::Term memory_sizes;
  Module *module;
  Common(cvc5::Solver& solver, Module *module)
    : solver{solver}
    , module{module}
  {
    cvc5::Sort address_sort = solver.mkBitVectorSort(module->ptr_bits);
    cvc5::Sort bit_sort = solver.mkBitVectorSort(1);
    cvc5::Sort byte_sort = solver.mkBitVectorSort(8);
    cvc5::Sort array_sort1 = solver.mkArraySort(address_sort, bit_sort);
    cvc5::Sort array_sort2 = solver.mkArraySort(address_sort, byte_sort);
    memory = solver.mkConst(array_sort2, ".memory");
    memory_flag = solver.mkConstArray(array_sort1, solver.mkBitVector(1, 0));
    memory_undef = solver.mkConstArray(array_sort2, solver.mkBitVector(8, 0));

    cvc5::Sort id_sort = solver.mkBitVectorSort(module->ptr_id_bits);
    cvc5::Sort offset_sort = solver.mkBitVectorSort(module->ptr_offset_bits);
    cvc5::Sort array_sort3 = solver.mkArraySort(id_sort, offset_sort);
    memory_sizes = solver.mkConstArray(array_sort3, solver.mkBitVector(module->ptr_offset_bits, 0));
  }
};

class Converter {
  std::map<const Instruction *, cvc5::Term> inst2bv;
  std::map<const Instruction *, cvc5::Term> inst2fp;
  std::map<const Instruction *, cvc5::Term> inst2bool;

  // Maps basic blocks to an expression telling if it is executed.
  std::map<const Basic_block *, cvc5::Term> bb2cond;

  // Maps basic blocks to an expression determining if it contain UB.
  std::map<const Basic_block *, cvc5::Term> bb2ub;

  // Maps basic blocks to an expression determining if it contain an
  // assertion failure.
  std::map<const Basic_block *, cvc5::Term> bb2not_assert;

  // List of the mem_id for the constant memory blocks.
  std::vector<Instruction *> const_ids;

  cvc5::Term ite(cvc5::Term c, cvc5::Term a, cvc5::Term b);
  cvc5::Term bool_or(cvc5::Term a, cvc5::Term b);
  cvc5::Term bool_and(cvc5::Term a, cvc5::Term b);
  void add_ub(const Basic_block *bb, cvc5::Term cond);
  void add_assert(const Basic_block *bb, cvc5::Term cond);
  cvc5::Op fp_sort(cvc5::Kind kind, uint32_t bitsize);
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
  cvc5::Term get_full_edge_cond(const Basic_block *src, const Basic_block *dest);
  void build_mem_state(const Basic_block *bb, std::map<const Basic_block *, cvc5::Term>& map);
  void generate_bb2cond(const Basic_block *bb);
  void convert_ir();

  cvc5::Solver& solver;
  Common& common;
  const Function *func;

public:
  Converter(Common& common, const Function *func)
    : solver{common.solver}
    , common{common}
    , func{func}
  {
    convert_ir();
  }
  cvc5::Term inst_as_bv(const Instruction *inst);
  cvc5::Term inst_as_fp(const Instruction *inst);
  cvc5::Term inst_as_bool(const Instruction *inst);
  cvc5::Term generate_ub();
  cvc5::Term generate_assert();
  Instruction *retval = nullptr;
  Instruction *retval_undef = nullptr;

  std::map<const Basic_block *, cvc5::Term> bb2memory;
  std::map<const Basic_block *, cvc5::Term> bb2memory_sizes;
  std::map<const Basic_block *, cvc5::Term> bb2memory_flag;
  std::map<const Basic_block *, cvc5::Term> bb2memory_undef;
};

cvc5::Term Converter::ite(cvc5::Term c, cvc5::Term a, cvc5::Term b)
{
  if (a == b)
    return a;
  return solver.mkTerm(cvc5::ITE, {c, a, b});
}

cvc5::Term Converter::bool_or(cvc5::Term a, cvc5::Term b)
{
  if (a.isBooleanValue())
    {
      if (a.getBooleanValue())
	return a;
      else
	return b;
    }
  if (b.isBooleanValue())
    {
      if (b.getBooleanValue())
	return b;
      else
	return a;
    }
  return solver.mkTerm(cvc5::OR, {a, b});
}

cvc5::Term Converter::bool_and(cvc5::Term a, cvc5::Term b)
{
  if (a.isBooleanValue())
    {
      if (a.getBooleanValue())
	return b;
      else
	return a;
    }
  if (b.isBooleanValue())
    {
      if (b.getBooleanValue())
	return a;
      else
	return b;
    }
  return solver.mkTerm(cvc5::AND, {a, b});
}

cvc5::Term Converter::inst_as_bv(const Instruction *inst)
{
  auto I = inst2bv.find(inst);
  if (I != inst2bv.end())
    return I->second;

  if (inst->bitsize == 1)
    {
      // We did not have a bitvector value for inst. This means there must
      // exist a Boolean value for this instruction. Convert it to bitvector.
      cvc5::Term expr =
	ite(inst2bool.at(inst), solver.mkBitVector(1, 1),
	    solver.mkBitVector(1, 0));
      inst2bv.insert({inst, expr});
      return expr;
    }
  else
    {
      // We did not have a bitvector value for inst. This means there must
      // exist a floating point value for this instruction. Convert it to
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

  // We did not have a Boolean value for inst. This means there must exist
  // a bitvector value for this instruction. Convert it to Boolean.
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

  // We did not have a floating point value for inst. This means there must
  // exist a bitvector value for this instruction. Convert it to floating
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

void Converter::add_ub(const Basic_block *bb, cvc5::Term cond)
{
  if (bb2ub.contains(bb))
    {
      cond = bool_or(bb2ub.at(bb), cond);
      bb2ub.erase(bb);
    }
  bb2ub.insert({bb, cond});
}

void Converter::add_assert(const Basic_block *bb, cvc5::Term cond)
{
  cond = solver.mkTerm(cvc5::NOT, {cond});
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

void Converter::build_bv_unary_smt(const Instruction *inst)
{
  assert(inst->nof_args == 1);

  // The UB processing assumes the argument is Boolean.
  if (inst->opcode == Op::UB)
    {
      cvc5::Term ub = inst_as_bool(inst->arguments[0]);
      add_ub(inst->bb, ub);
      return;
    }

  // The ASSERT processing assumes the argument is Boolean.
  if (inst->opcode == Op::ASSERT)
    {
      cvc5::Term assrt = inst_as_bool(inst->arguments[0]);
      add_assert(inst->bb, assrt);
      return;
    }

  cvc5::Term arg1 = inst_as_bv(inst->arguments[0]);
  switch (inst->opcode)
    {
    case Op::SYMBOLIC:
      {
	static int symbolic_idx = 0;
	char name[100];
	sprintf(name, ".symbolic%d", symbolic_idx++);
	cvc5::Sort sort = solver.mkBitVectorSort(inst->bitsize);
	cvc5::Term symbolic = solver.mkConst(sort, name);
	inst2bv.insert({inst, symbolic});
      }
      break;
    case Op::GET_MEM_FLAG:
      {
	cvc5::Term mem_flag = bb2memory_flag.at(inst->bb);
	inst2bv.insert({inst, solver.mkTerm(cvc5::SELECT, {mem_flag, arg1})});
      }
      break;
    case Op::GET_MEM_UNDEF:
      {
	cvc5::Term mem_undef = bb2memory_undef.at(inst->bb);
	inst2bv.insert({inst, solver.mkTerm(cvc5::SELECT, {mem_undef, arg1})});
      }
      break;
    case Op::FREE:
      {
	uint32_t ptr_offset_bits = func->module->ptr_offset_bits;
	cvc5::Term sizes = bb2memory_sizes.at(inst->bb);
	cvc5::Term zero = solver.mkBitVector(ptr_offset_bits, 0);
	sizes = solver.mkTerm(cvc5::STORE, {sizes, arg1, zero});
	bb2memory_sizes.erase(inst->bb);
	bb2memory_sizes.insert({inst->bb, sizes});
      }
      break;
    case Op::IS_CONST_MEM:
      {
	cvc5::Term is_const = solver.mkBoolean(false);
	for (Instruction *id : const_ids)
	  {
	    cvc5::Term id_bv = inst_as_bv(id);
	    cvc5::Term cond = solver.mkTerm(cvc5::EQUAL, {arg1, id_bv});
	    is_const = bool_or(is_const, cond);
	  }
	inst2bool.insert({inst, is_const});
      }
      break;
    case Op::IS_NAN:
      // TODO: Implement Op::IS_NAN
      throw Not_implemented("build_bv_unary_smt: "s + inst->name());
    case Op::IS_NONCANONICAL_NAN:
      // TODO: Implement Op::IS_NONCANONICAL_NAN
      throw Not_implemented("build_bv_unary_smt: "s + inst->name());
    case Op::LOAD:
      {
	cvc5::Term memory = bb2memory.at(inst->bb);
	inst2bv.insert({inst, solver.mkTerm(cvc5::SELECT, {memory, arg1})});
      }
      break;
    case Op::MEM_SIZE:
      {
	cvc5::Term sizes = bb2memory_sizes.at(inst->bb);
	inst2bv.insert({inst, solver.mkTerm(cvc5::SELECT, {sizes, arg1})});
      }
      break;
    case Op::MOV:
      inst2bv.insert({inst, arg1});
      break;
    case Op::NEG:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_NEG, {arg1})});
      break;
    case Op::NOT:
      inst2bv.insert({inst, solver.mkTerm(cvc5::BITVECTOR_NOT, {arg1})});
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
	if (common.index2param.contains(index))
	  {
	    inst2bv.insert({inst, common.index2param.at(index)});
	  }
	else
	  {
	    char name[100];
	    sprintf(name, ".param%d", index);
	    cvc5::Sort sort = solver.mkBitVectorSort(inst->bitsize);
	    cvc5::Term param = solver.mkConst(sort, name);
	    inst2bv.insert({inst, param});
	    common.index2param.insert({index, param});
	  }
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
    case Op::SET_MEM_FLAG:
      {
	cvc5::Term memory_flag = bb2memory_flag.at(inst->bb);
	memory_flag = solver.mkTerm(cvc5::STORE, {memory_flag, arg1, arg2});
	bb2memory_flag.erase(inst->bb);
	bb2memory_flag.insert({inst->bb, memory_flag});
      }
      break;
    case Op::SET_MEM_UNDEF:
      {
	cvc5::Term memory_undef = bb2memory_undef.at(inst->bb);
	memory_undef = solver.mkTerm(cvc5::STORE, {memory_undef, arg1, arg2});
	bb2memory_undef.erase(inst->bb);
	bb2memory_undef.insert({inst->bb, memory_undef});
      }
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
    case Op::STORE:
      {
	cvc5::Term memory = bb2memory.at(inst->bb);
	memory = solver.mkTerm(cvc5::STORE, {memory, arg1, arg2});
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
    case Op::EXTRACT:
      {
	cvc5::Term arg = inst_as_bv(inst->arguments[0]);
	uint32_t high = inst->arguments[1]->value();
	uint32_t low = inst->arguments[2]->value();
	cvc5::Op extract = solver.mkOp(cvc5::BITVECTOR_EXTRACT, {high, low});
	inst2bv.insert({inst, solver.mkTerm(extract, {arg})});
      }
      break;
    case Op::MEMORY:
      // Memory has already been processed.
      assert(inst2bv.contains(inst));
      break;
    case Op::ITE:
      {
	cvc5::Term arg1 = inst_as_bool(inst->arguments[0]);
	cvc5::Term arg2 = inst_as_bv(inst->arguments[1]);
	cvc5::Term arg3 = inst_as_bv(inst->arguments[2]);
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

cvc5::Term Converter::get_full_edge_cond(const Basic_block *src, const Basic_block *dest)
{
  if (src->succs.size() == 1)
    return bb2cond.at(src);
  assert(src->succs.size() == 2);
  assert(src->last_inst->opcode == Op::BR);
  assert(src->last_inst->nof_args == 1);
  cvc5::Term cond = inst_as_bool(src->last_inst->arguments[0]);
  if (dest != src->succs[0])
    cond = solver.mkTerm(cvc5::NOT, {cond});
  cvc5::Term src_cond = bb2cond.at(src);
  return bool_and(src_cond, cond);
}

void Converter::build_mem_state(const Basic_block *bb, std::map<const Basic_block *, cvc5::Term>& map)
{
  assert(bb->preds.size() > 0);
  cvc5::Term expr = map.at(bb->preds[0]);
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
      cvc5::Term cond = solver.mkBoolean(false);
      for (auto pred_bb : bb->preds)
	{
	  cvc5::Term edge_cond = get_full_edge_cond(pred_bb, bb);
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
	  bb2cond.insert({bb, solver.mkBoolean(true)});
	  bb2memory.insert({bb, common.memory});
	  bb2memory_flag.insert({bb, common.memory_flag});
	  cvc5::Term undef = common.memory_undef;
	  cvc5::Term sizes = common.memory_sizes;
	  for (Instruction *inst = bb->first_inst; inst; inst = inst->next)
	    {
	      if (inst->opcode != Op::MEMORY)
		continue;
	      assert(inst->bitsize == func->module->ptr_bits);
	      uint64_t id = inst->arguments[0]->value();
	      uint64_t ptr_val = id << func->module->ptr_id_low;
	      inst2bv.insert({inst, solver.mkBitVector(inst->bitsize, ptr_val)});

	      uint32_t ptr_bits = func->module->ptr_bits;
	      uint32_t ptr_offset_bits = func->module->ptr_offset_bits;
	      uint32_t ptr_id_bits = func->module->ptr_id_bits;
	      cvc5::Term mem_id = solver.mkBitVector(ptr_id_bits, id);
	      uint64_t size_val = inst->arguments[1]->value();
	      cvc5::Term size = solver.mkBitVector(ptr_offset_bits, size_val);
	      sizes = solver.mkTerm(cvc5::STORE, {sizes, mem_id, size});

	      uint32_t flags = inst->arguments[2]->value();
	      if (flags & MEM_CONST)
		const_ids.push_back(inst->arguments[0]);
	      if (flags & MEM_UNINIT)
		{
		  cvc5::Term byte = solver.mkBitVector(8, 255);
		  for (uint64_t i = 0; i < size_val; i++)
		    {
		      cvc5::Term ptr =
			solver.mkBitVector(ptr_bits, ptr_val + i);
		      undef = solver.mkTerm(cvc5::STORE, {undef, ptr, byte});
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
	  cvc5::Term phi_expr = inst_as_bv(phi->phi_args[0].inst);
	  assert(phi->phi_args.size() == bb->preds.size());
	  for (unsigned i = 1; i < phi->phi_args.size(); i++)
	    {
	      const Basic_block *pred_bb = phi->phi_args[i].bb;
	      cvc5::Term cond = get_full_edge_cond(pred_bb, bb);
	      cvc5::Term expr = inst_as_bv(phi->phi_args[i].inst);
	      phi_expr = ite(cond, expr, phi_expr);
	    }
	  inst2bv.insert({phi, phi_expr});
	}

      for (Instruction *inst = bb->first_inst; inst; inst = inst->next)
	{
	  build_smt(inst);
	}
    }
}

cvc5::Term Converter::generate_ub()
{
  cvc5::Term ub = solver.mkBoolean(false);
  for (auto bb : func->bbs)
    {
      if (!bb2ub.contains(bb))
	continue;

      cvc5::Term is_ub = bool_and(bb2ub.at(bb), bb2cond.at(bb));
      ub = bool_or(ub, is_ub);
    }

  return ub;
}

cvc5::Term Converter::generate_assert()
{
  cvc5::Term assrt = solver.mkBoolean(false);
  for (auto bb : func->bbs)
    {
      if (!bb2not_assert.contains(bb))
	continue;

      cvc5::Term is_assrt = bool_and(bb2not_assert.at(bb), bb2cond.at(bb));
      assrt = bool_or(assrt, is_assrt);
    }

  return assrt;
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

} // end anonymous namespace

std::pair<SStats, Solver_result> check_refine_cvc5(Function *src, Function *tgt)
{
  cvc5::Solver solver;
  solver.setOption("produce-models", "true");
  char buf[32];
  sprintf(buf, "%d", config.timeout);
  solver.setOption("tlimit-per", buf);
  solver.setLogic("QF_ABVFP");

  SStats stats;
  stats.skipped = false;

  assert(src->module == tgt->module);
  Module *module = src->module;
  Common common(solver, module);
  Converter conv_src(common, src);
  cvc5::Term src_ub_expr = conv_src.generate_ub();
  cvc5::Term not_src_ub_expr = solver.mkTerm(cvc5::NOT, {src_ub_expr});

  Converter conv_tgt(common, tgt);
  cvc5::Term tgt_ub_expr = conv_tgt.generate_ub();

  std::string warning;
  if (conv_src.retval || conv_tgt.retval)
    {
      solver.push();
      assert(conv_src.retval && conv_tgt.retval);
      cvc5::Term src_expr = conv_src.inst_as_bv(conv_src.retval);
      cvc5::Term tgt_expr = conv_tgt.inst_as_bv(conv_tgt.retval);
      cvc5::Term is_more_undef = solver.mkBoolean(false);
      cvc5::Term src_undef = solver.mkBitVector(conv_src.retval->bitsize, 0);
      cvc5::Term tgt_undef = solver.mkBitVector(conv_tgt.retval->bitsize, 0);
      if (conv_src.retval_undef)
	{
	  src_undef = conv_src.inst_as_bv(conv_src.retval_undef);
	  cvc5::Term src_mask = solver.mkTerm(cvc5::BITVECTOR_NOT, {src_undef});
	  src_expr = solver.mkTerm(cvc5::BITVECTOR_AND, {src_expr, src_mask});
	  tgt_expr = solver.mkTerm(cvc5::BITVECTOR_AND, {tgt_expr, src_mask});

	  // Check that tgt is not more undef than src.
	  if (conv_tgt.retval_undef)
	    {
	      tgt_undef = conv_tgt.inst_as_bv(conv_tgt.retval_undef);
	      cvc5::Term undef_result =
		solver.mkTerm(cvc5::BITVECTOR_AND, {tgt_undef, src_mask});
	      cvc5::Term zero = solver.mkBitVector(conv_tgt.retval->bitsize, 0);
	      is_more_undef =
		solver.mkTerm(cvc5::DISTINCT, {undef_result, zero});
	    }
	}

      solver.assertFormula(not_src_ub_expr);
      cvc5::Term res1 = solver.mkTerm(cvc5::DISTINCT, {src_expr, tgt_expr});
      cvc5::Term res2 = solver.mkTerm(cvc5::OR, {res1, is_more_undef});
      solver.assertFormula(res2);
      uint64_t start_time = get_time();
      Solver_result solver_result = run_solver(solver, "retval");
      stats.time[0] = std::max(get_time() - start_time, (uint64_t)1);
      if (solver_result.status == Result_status::incorrect)
	{
	  assert(solver_result.message);
	  cvc5::Term src_val = solver.getValue(src_expr);
	  cvc5::Term tgt_val = solver.getValue(tgt_expr);
	  std::string msg = *solver_result.message;
	  msg = msg + "src retval: " + src_val.getBitVectorValue(16) + "\n";
	  msg = msg + "tgt retval: " + tgt_val.getBitVectorValue(16) + "\n";
	  if (conv_src.retval_undef || conv_tgt.retval_undef)
	    {
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

  {
    solver.push();
    solver.assertFormula(not_src_ub_expr);
    Basic_block *src_exit_bb = src->bbs[src->bbs.size() - 1];
    cvc5::Term src_mem = conv_src.bb2memory.at(src_exit_bb);
    cvc5::Term src_mem_sizes = conv_src.bb2memory_sizes.at(src_exit_bb);
    cvc5::Term src_mem_undef = conv_src.bb2memory_undef.at(src_exit_bb);

    Basic_block *tgt_exit_bb = tgt->bbs[tgt->bbs.size() - 1];
    cvc5::Term tgt_mem = conv_tgt.bb2memory.at(tgt_exit_bb);
    cvc5::Term tgt_mem_undef = conv_tgt.bb2memory_undef.at(tgt_exit_bb);

    cvc5::Sort ptr_sort = solver.mkBitVectorSort(module->ptr_bits);
    cvc5::Term ptr = solver.mkConst(ptr_sort, ".ptr");
    uint32_t ptr_id_high = module->ptr_id_high;
    uint32_t ptr_id_low = module->ptr_id_low;
    cvc5::Op id_op =
      solver.mkOp(cvc5::BITVECTOR_EXTRACT, {ptr_id_high, ptr_id_low});
    cvc5::Term id = solver.mkTerm(id_op, {ptr});
    uint32_t ptr_offset_high = module->ptr_offset_high;
    uint32_t ptr_offset_low = module->ptr_offset_low;
    cvc5::Op offset_op =
      solver.mkOp(cvc5::BITVECTOR_EXTRACT, {ptr_offset_high, ptr_offset_low});
    cvc5::Term offset = solver.mkTerm(offset_op, {ptr});

    // Only check global memory.
    cvc5::Term zero_id = solver.mkBitVector(module->ptr_id_bits, 0);
    cvc5::Term cond1 = solver.mkTerm(cvc5::BITVECTOR_SGT, {id, zero_id});
    solver.assertFormula(cond1);

    // Only check memory within a memory block.
    cvc5::Term mem_size = solver.mkTerm(cvc5::SELECT, {src_mem_sizes, id});
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
  // has lowe priority.
  {
    solver.push();
    solver.assertFormula(not_src_ub_expr);
    solver.assertFormula(tgt_ub_expr);
    cvc5::Term res1 = solver.mkTerm(cvc5::DISTINCT, {src_ub_expr, tgt_ub_expr});
    solver.assertFormula(res1);
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
  cvc5::Solver solver;
  solver.setOption("produce-models", "true");
  char buf[32];
  sprintf(buf, "%d", config.timeout);
  solver.setOption("tlimit-per", buf);
  solver.setLogic("QF_ABVFP");

  SStats stats;
  stats.skipped = false;

  Module *module = func->module;
  Common common(solver, module);
  Converter conv_func(common, func);
  cvc5::Term ub_expr = conv_func.generate_ub();

  solver.push();
  solver.assertFormula(ub_expr);
  uint64_t start_time = get_time();
  Solver_result solver_result = run_solver(solver, "UB");
  stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
  return std::pair<SStats, Solver_result>(stats, solver_result);
}

std::pair<SStats, Solver_result> check_assert_cvc5(Function *func)
{
  cvc5::Solver solver;
  solver.setOption("produce-models", "true");
  char buf[32];
  sprintf(buf, "%d", config.timeout);
  solver.setOption("tlimit-per", buf);
  solver.setLogic("QF_ABVFP");

  SStats stats;
  stats.skipped = false;

  Module *module = func->module;
  Common common(solver, module);
  Converter conv_func(common, func);
  cvc5::Term ub_expr = conv_func.generate_ub();
  cvc5::Term not_ub_expr = solver.mkTerm(cvc5::NOT, {ub_expr});
  cvc5::Term assert_expr = conv_func.generate_assert();

  solver.push();
  solver.assertFormula(not_ub_expr);
  solver.assertFormula(assert_expr);
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
