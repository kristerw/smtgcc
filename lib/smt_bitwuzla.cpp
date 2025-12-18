#include "config.h"

#if HAVE_LIBBITWUZLA
#include <cassert>
#include <cinttypes>
#include <bitwuzla/cpp/bitwuzla.h>

#include "smtgcc.h"

using namespace bitwuzla;
using namespace std::string_literals;

namespace smtgcc {

namespace {

class Converter {
  std::map<const Inst *, Term> inst2array;
  std::map<const Inst *, Term> inst2bv;
  std::map<const Inst *, Term> inst2bool;

  Term ite(Term c, Term a, Term b);
  void build_bv_comparison_smt(const Inst *inst);
  void build_memory_state_smt(const Inst *inst);
  void build_bv_unary_smt(const Inst *inst);
  void build_bv_binary_smt(const Inst *inst);
  void build_ternary_smt(const Inst *inst);
  void build_conversion_smt(const Inst *inst);
  void build_solver_smt(const Inst *inst);
  void build_special_smt(const Inst *inst);
  void build_smt(const Inst *inst);
  void convert_function();

  TermManager& tm;
  const Function *func;

public:
  Converter(TermManager& tm, const Function *func)
    : tm{tm}
    , func{func}
  {
    convert_function();
  }
  Term inst_as_array(const Inst *inst);
  Term inst_as_bv(const Inst *inst);
  Term inst_as_bool(const Inst *inst);

  std::vector<const Inst *> print;

  Inst *src_assert = nullptr;
  Inst *src_memory = nullptr;
  Inst *src_memory_size = nullptr;
  Inst *src_memory_indef = nullptr;
  Inst *src_retval = nullptr;
  Inst *src_retval_indef = nullptr;
  Inst *src_unique_ub = nullptr;
  Inst *src_common_ub = nullptr;
  Inst *src_abort  = nullptr;
  Inst *src_exit = nullptr;
  Inst *src_exit_val  = nullptr;

  Inst *tgt_assert = nullptr;
  Inst *tgt_memory = nullptr;
  Inst *tgt_memory_size = nullptr;
  Inst *tgt_memory_indef = nullptr;
  Inst *tgt_retval = nullptr;
  Inst *tgt_retval_indef = nullptr;
  Inst *tgt_unique_ub = nullptr;
  Inst *tgt_common_ub = nullptr;
  Inst *tgt_abort  = nullptr;
  Inst *tgt_exit = nullptr;
  Inst *tgt_exit_val  = nullptr;
};

Term Converter::ite(Term c, Term a, Term b)
{
  if (a == b)
    return a;
  return tm.mk_term(Kind::ITE, {c, a, b});
}

Term Converter::inst_as_array(const Inst *inst)
{
  return inst2array.at(inst);
}

Term Converter::inst_as_bv(const Inst *inst)
{
  auto I = inst2bv.find(inst);
  if (I != inst2bv.end())
    return I->second;

  // We do not have a bitvector value for inst. This means there must
  // be a Boolean value for this instruction. Convert it to a bitvector.
  assert(inst->bitsize == 1);
  Sort bv1 = tm.mk_bv_sort(1);
  Term term = ite(inst2bool.at(inst), tm.mk_bv_one(bv1), tm.mk_bv_zero(bv1));
  inst2bv.insert({inst, term});
  return term;
}

Term Converter::inst_as_bool(const Inst *inst)
{
  assert(inst->bitsize == 1);
  auto I = inst2bool.find(inst);
  if (I != inst2bool.end())
    return I->second;

  // We do not have a Boolean value for inst. This means there must be
  // a bitvector value for this instruction. Convert it to a Boolean.
  Term bv = inst2bv.at(inst);
  Sort bv1 = tm.mk_bv_sort(1);
  Term term = tm.mk_term(Kind::EQUAL, {bv, tm.mk_bv_one(bv1)});
  inst2bool.insert({inst, term});
  return term;
}

void Converter::build_bv_comparison_smt(const Inst *inst)
{
  assert(inst->nof_args == 2);

  if (inst->args[0]->bitsize == 1
      && (inst->op == Op::EQ || inst->op == Op::NE)
      && (inst2bool.contains(inst->args[0])
	  && inst2bool.contains(inst->args[1])))
    {
      Term arg1 = inst_as_bool(inst->args[0]);
      Term arg2 = inst_as_bool(inst->args[1]);

      if (inst->op == Op::EQ)
	inst2bool.insert({inst, tm.mk_term(Kind::EQUAL, {arg1, arg2})});
      else
	inst2bool.insert({inst, tm.mk_term(Kind::DISTINCT, {arg1, arg2})});
      return;
    }

  Term arg1 = inst_as_bv(inst->args[0]);
  Term arg2 = inst_as_bv(inst->args[1]);
  switch (inst->op)
    {
    case Op::EQ:
      inst2bool.insert({inst, tm.mk_term(Kind::EQUAL, {arg1, arg2})});
      break;
    case Op::NE:
      inst2bool.insert({inst, tm.mk_term(Kind::DISTINCT, {arg1, arg2})});
      break;
    case Op::SLE:
      inst2bool.insert({inst, tm.mk_term(Kind::BV_SLE, {arg1, arg2})});
      break;
    case Op::SLT:
      inst2bool.insert({inst, tm.mk_term(Kind::BV_SLT, {arg1, arg2})});
      break;
    case Op::ULE:
      inst2bool.insert({inst, tm.mk_term(Kind::BV_ULE, {arg1, arg2})});
      break;
    case Op::ULT:
      inst2bool.insert({inst, tm.mk_term(Kind::BV_ULT, {arg1, arg2})});
      break;
    default:
      throw Not_implemented("build_comparison_smt: "s + inst->name());
    }
}

void Converter::build_memory_state_smt(const Inst *inst)
{
  switch (inst->op)
    {
    case Op::MEM_ARRAY:
      {
	Sort address_sort = tm.mk_bv_sort(func->module->ptr_bits);
	Sort byte_sort = tm.mk_bv_sort(8);
	Sort array_sort = tm.mk_array_sort(address_sort, byte_sort);
	Term memory = tm.mk_const(array_sort, ".memory");
	inst2array.insert({inst, memory});
      }
      break;
    case Op::MEM_FLAG_ARRAY:
      {
	Sort address_sort = tm.mk_bv_sort(func->module->ptr_bits);
	Sort bit_sort = tm.mk_bv_sort(1);
	Sort array_sort = tm.mk_array_sort(address_sort, bit_sort);
	Term memory_flag =
	  tm.mk_const_array(array_sort, tm.mk_bv_zero(bit_sort));
	inst2array.insert({inst, memory_flag});
      }
      break;
    case Op::MEM_SIZE_ARRAY:
      {
	uint32_t ptr_id_bits = func->module->ptr_id_bits;
	uint32_t ptr_offset_bits = func->module->ptr_offset_bits;
	Sort id_sort = tm.mk_bv_sort(ptr_id_bits);
	Sort offset_sort = tm.mk_bv_sort(ptr_offset_bits);
	Sort array_sort = tm.mk_array_sort(id_sort, offset_sort);
	Term memory_size =
	  tm.mk_const_array(array_sort, tm.mk_bv_zero(offset_sort));
	inst2array.insert({inst, memory_size});
      }
      break;
    case Op::MEM_INDEF_ARRAY:
      {
	Sort address_sort = tm.mk_bv_sort(func->module->ptr_bits);
	Sort byte_sort = tm.mk_bv_sort(8);
	Sort array_sort = tm.mk_array_sort(address_sort, byte_sort);
	Term memory_indef =
	  tm.mk_const_array(array_sort, tm.mk_bv_zero(byte_sort));
	inst2array.insert({inst, memory_indef});
      }
      break;
    default:
      throw Not_implemented("build_memory_state_smt: "s + inst->name());
    }
}

void Converter::build_bv_unary_smt(const Inst *inst)
{
  assert(inst->nof_args == 1);

  // Perform the NOT operation as Boolean if the argument is Boolean. This
  // avoids multiple conversions between bitvector and Boolean for the
  // typical case where 1-bit values are used in logical expressions.
  if (inst->bitsize == 1
      && inst->op == Op::NOT
      && inst2bool.contains(inst->args[0]))
    {
      Term arg1 = inst_as_bool(inst->args[0]);
      inst2bool.insert({inst, tm.mk_term(Kind::NOT, {arg1})});
      return;
    }

  Term arg1 = inst_as_bv(inst->args[0]);
  switch (inst->op)
    {
    case Op::IS_INF:
    case Op::IS_NAN:
    case Op::IS_NONCANONICAL_NAN:
      throw Not_implemented("floating-point support in smt_bitwuzla");
    case Op::MOV:
      inst2bv.insert({inst, arg1});
      break;
    case Op::NEG:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_NEG, {arg1})});
      break;
    case Op::NOT:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_NOT, {arg1})});
      break;
    default:
      throw Not_implemented("build_bv_unary_smt: "s + inst->name());
    }
}

void Converter::build_bv_binary_smt(const Inst *inst)
{
  assert(inst->nof_args == 2);

  switch (inst->op)
    {
    case Op::ARRAY_GET_FLAG:
    case Op::ARRAY_GET_SIZE:
    case Op::ARRAY_GET_INDEF:
    case Op::ARRAY_LOAD:
      {
	Term arg1 = inst_as_array(inst->args[0]);
	Term arg2 = inst_as_bv(inst->args[1]);
	inst2bv.insert({inst, tm.mk_term(Kind::ARRAY_SELECT, {arg1, arg2})});
      }
      return;
    default:
      break;
    }

  // Perform AND/OR/XOR operations as Boolean if at least one of the
  // arguments are Boolean. This avoids multiple conversions between
  // bitvector and Boolean for the typical case where 1-bit values
  // are used in logical expressions.
  if (inst->bitsize == 1 &&
      (inst->op == Op::AND
       || inst->op == Op::OR
       || inst->op == Op::XOR)
      && (inst2bool.contains(inst->args[0])
	  || inst2bool.contains(inst->args[1])))
    {
      Term arg1 = inst_as_bool(inst->args[0]);
      Term arg2 = inst_as_bool(inst->args[1]);
      if (inst->op == Op::AND)
	inst2bool.insert({inst, tm.mk_term(Kind::AND, {arg1, arg2})});
      else if (inst->op == Op::OR)
	inst2bool.insert({inst, tm.mk_term(Kind::OR, {arg1, arg2})});
      else
	inst2bool.insert({inst, tm.mk_term(Kind::XOR, {arg1, arg2})});
      return;
    }

  Term arg1 = inst_as_bv(inst->args[0]);
  Term arg2 = inst_as_bv(inst->args[1]);
  switch (inst->op)
    {
    case Op::ADD:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_ADD, {arg1, arg2})});
      break;
    case Op::SUB:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_SUB, {arg1, arg2})});
      break;
    case Op::MUL:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_MUL, {arg1, arg2})});
      break;
    case Op::SDIV:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_SDIV, {arg1, arg2})});
      break;
    case Op::UDIV:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_UDIV, {arg1, arg2})});
      break;
    case Op::SADD_WRAPS:
      inst2bool.insert({inst, tm.mk_term(Kind::BV_SADD_OVERFLOW, {arg1, arg2})});
      break;
    case Op::SMUL_WRAPS:
      inst2bool.insert({inst, tm.mk_term(Kind::BV_SMUL_OVERFLOW, {arg1, arg2})});
      break;
    case Op::SREM:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_SREM, {arg1, arg2})});
      break;
    case Op::SSUB_WRAPS:
      inst2bool.insert({inst, tm.mk_term(Kind::BV_SSUB_OVERFLOW, {arg1, arg2})});
      break;
    case Op::UREM:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_UREM, {arg1, arg2})});
      break;
    case Op::ASHR:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_ASHR, {arg1, arg2})});
      break;
    case Op::LSHR:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_SHR, {arg1, arg2})});
      break;
    case Op::SHL:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_SHL, {arg1, arg2})});
      break;
    case Op::AND:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_AND, {arg1, arg2})});
      break;
    case Op::OR:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_OR, {arg1, arg2})});
      break;
    case Op::XOR:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_XOR, {arg1, arg2})});
      break;
    case Op::CONCAT:
      inst2bv.insert({inst, tm.mk_term(Kind::BV_CONCAT, {arg1, arg2})});
      break;
    default:
      throw Not_implemented("build_binary_smt: "s + inst->name());
    }
}

void Converter::build_ternary_smt(const Inst *inst)
{
  assert(inst->nof_args == 3);
  switch (inst->op)
    {
    case Op::ARRAY_SET_FLAG:
    case Op::ARRAY_SET_SIZE:
    case Op::ARRAY_SET_INDEF:
    case Op::ARRAY_STORE:
      {
	Term arg1 = inst_as_array(inst->args[0]);
	Term arg2 = inst_as_bv(inst->args[1]);
	Term arg3 = inst_as_bv(inst->args[2]);
	Term array = tm.mk_term(Kind::ARRAY_STORE, {arg1, arg2, arg3});
	inst2array.insert({inst, array});
      }
      break;
    case Op::EXTRACT:
      {
	Term arg = inst_as_bv(inst->args[0]);
	uint32_t high = inst->args[1]->value();
	uint32_t low = inst->args[2]->value();
	Term res = tm.mk_term(Kind::BV_EXTRACT, {arg}, {high, low});
	inst2bv.insert({inst, res});
      }
      break;
    case Op::ITE:
      if (inst2array.contains(inst->args[1]))
	{
	  Term arg1 = inst_as_bool(inst->args[0]);
	  Term arg2 = inst_as_array(inst->args[1]);
	  Term arg3 = inst_as_array(inst->args[2]);
	  inst2array.insert({inst, ite(arg1, arg2, arg3)});
	}
      else if (inst->bitsize == 1
	       && inst2bool.contains(inst->args[1])
	       && inst2bool.contains(inst->args[2]))
	{
	  Term arg1 = inst_as_bool(inst->args[0]);
	  Term arg2 = inst_as_bool(inst->args[1]);
	  Term arg3 = inst_as_bool(inst->args[2]);
	  inst2bool.insert({inst, ite(arg1, arg2, arg3)});
	}
      else
	{
	  Term arg1 = inst_as_bool(inst->args[0]);
	  Term arg2 = inst_as_bv(inst->args[1]);
	  Term arg3 = inst_as_bv(inst->args[2]);
	  inst2bv.insert({inst, ite(arg1, arg2, arg3)});
	}
      break;
    default:
      throw Not_implemented("build_ternary_smt: "s + inst->name());
    }
}

void Converter::build_conversion_smt(const Inst *inst)
{
  switch (inst->op)
    {
    case Op::SEXT:
      {
	Term arg = inst_as_bv(inst->args[0]);
	assert(inst->args[0]->bitsize < inst->bitsize);
	unsigned bits =  inst->bitsize - inst->args[0]->bitsize;
	Term res = tm.mk_term(Kind::BV_SIGN_EXTEND, {arg}, {bits});
	inst2bv.insert({inst, res});
      }
      break;
    case Op::ZEXT:
      {
	Term arg = inst_as_bv(inst->args[0]);
	assert(inst->args[0]->bitsize < inst->bitsize);
	unsigned bits =  inst->bitsize - inst->args[0]->bitsize;
	Term res = tm.mk_term(Kind::BV_ZERO_EXTEND, {arg}, {bits});
	inst2bv.insert({inst, res});
      }
      break;
    case Op::F2U:
    case Op::F2S:
    case Op::S2F:
    case Op::U2F:
    case Op::FCHPREC:
      throw Not_implemented("floating-point support in smt_bitwuzla");
    default:
      throw Not_implemented("build_conversion_smt: "s + inst->name());
    }
}

void Converter::build_solver_smt(const Inst *inst)
{
  if (inst->nof_args == 1)
    {
      switch (inst->op)
	{
	case Op::SRC_ASSERT:
	  assert(!src_assert);
	  src_assert = inst->args[0];
	  break;
	case Op::TGT_ASSERT:
	  assert(!tgt_assert);
	  tgt_assert = inst->args[0];
	  break;
	default:
	  throw Not_implemented("build_solver_smt: "s + inst->name());
	}
    }
  else if (inst->nof_args == 2)
    {
      switch (inst->op)
	{
	case Op::PARAM:
	  {
	    uint32_t index = inst->args[0]->value();
	    char name[100];
	    sprintf(name, ".param%" PRIu32, index);
	    Sort sort = tm.mk_bv_sort(inst->bitsize);
	    Term param = tm.mk_const(sort, name);
	    inst2bv.insert({inst, param});
	  }
	  break;
	case Op::PRINT:
	  print.push_back(inst);
	  break;
	case Op::SRC_RETVAL:
	  assert(!src_retval);
	  assert(!src_retval_indef);
	  src_retval = inst->args[0];
	  src_retval_indef = inst->args[1];
	  return;
	case Op::SRC_UB:
	  assert(!src_unique_ub && !src_common_ub);
	  src_common_ub = inst->args[0];
	  src_unique_ub = inst->args[1];
	  return;
	case Op::SYMBOLIC:
	  {
	    uint32_t index = inst->args[0]->value();
	    char name[100];
	    sprintf(name, ".symbolic%" PRIu32, index);
	    Sort sort = tm.mk_bv_sort(inst->bitsize);
	    Term symbolic = tm.mk_const(sort, name);
	    inst2bv.insert({inst, symbolic});
	  }
	  break;
	case Op::TGT_RETVAL:
	  assert(!tgt_retval);
	  assert(!tgt_retval_indef);
	  tgt_retval = inst->args[0];
	  tgt_retval_indef = inst->args[1];
	  return;
	case Op::TGT_UB:
	  assert(!tgt_unique_ub && !tgt_common_ub);
	  tgt_common_ub = inst->args[0];
	  tgt_unique_ub = inst->args[1];
	  return;
	default:
	  throw Not_implemented("build_solver_smt: "s + inst->name());
	}
    }
  else if (inst->nof_args == 3)
    {
      switch (inst->op)
	{
	case Op::SRC_MEM:
	  assert(!src_memory);
	  assert(!src_memory_size);
	  src_memory = inst->args[0];
	  src_memory_size = inst->args[1];
	  src_memory_indef = inst->args[2];
	  return;
	case Op::TGT_MEM:
	  assert(!tgt_memory);
	  assert(!tgt_memory_size);
	  tgt_memory = inst->args[0];
	  tgt_memory_size = inst->args[1];
	  tgt_memory_indef = inst->args[2];
	  return;
	case Op::SRC_EXIT:
	  assert(!src_abort);
	  assert(!src_exit);
	  assert(!src_exit_val);
	  src_abort = inst->args[0];
	  src_exit = inst->args[1];
	  src_exit_val = inst->args[2];
	  return;
	case Op::TGT_EXIT:
	  assert(!tgt_abort);
	  assert(!tgt_exit);
	  assert(!tgt_exit_val);
	  tgt_abort = inst->args[0];
	  tgt_exit = inst->args[1];
	  tgt_exit_val = inst->args[2];
	  return;
	default:
	  throw Not_implemented("build_solver_smt: "s + inst->name());
	}
    }
  else
    throw Not_implemented("build_solver_smt: "s + inst->name());
}

void Converter::build_special_smt(const Inst *inst)
{
  switch (inst->op)
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
	    Sort lo_sort = tm.mk_bv_sort(64);
	    Term lo = tm.mk_bv_value_uint64(lo_sort, low);
	    Sort hi_sort = tm.mk_bv_sort(inst->bitsize - 64);
	    Term hi = tm.mk_bv_value_uint64(hi_sort, high);
	    inst2bv.insert({inst, tm.mk_term(Kind::BV_CONCAT, {hi, lo})});
	  }
	else
	  {
	    Sort lo_sort = tm.mk_bv_sort(inst->bitsize);
	    inst2bv.insert({inst, tm.mk_bv_value_uint64(lo_sort, low)});
	  }
	if (inst->bitsize == 1)
	  {
	    if (low)
	      inst2bool.insert({inst, tm.mk_true()});
	    else
	      inst2bool.insert({inst, tm.mk_false()});
	  }
      }
      break;
    default:
      throw Not_implemented("build_special_smt: "s + inst->name());
    }
}

void Converter::build_smt(const Inst *inst)
{
  switch (inst->iclass())
    {
    case Inst_class::icomparison:
      build_bv_comparison_smt(inst);
      break;
    case Inst_class::mem_nullary:
      build_memory_state_smt(inst);
      break;
    case Inst_class::iunary:
      build_bv_unary_smt(inst);
      break;
    case Inst_class::ibinary:
      build_bv_binary_smt(inst);
      break;
    case Inst_class::ternary:
      build_ternary_smt(inst);
      break;
    case Inst_class::conv:
      build_conversion_smt(inst);
      break;
    case Inst_class::solver_unary:
    case Inst_class::solver_binary:
    case Inst_class::solver_ternary:
      build_solver_smt(inst);
      break;
    case Inst_class::special:
      build_special_smt(inst);
      break;
    case Inst_class::fcomparison:
    case Inst_class::funary:
    case Inst_class::fbinary:
      throw Not_implemented("floating-point support in smt_bitwuzla");
    default:
      throw Not_implemented("build_smt: "s + inst->name());
    }
}

void Converter::convert_function()
{
  for (auto bb : func->bbs)
    {
      assert(bb->phis.empty());
      for (Inst *inst = bb->first_inst; inst; inst = inst->next)
	{
	  build_smt(inst);
	}
    }

  // If both src and tgt retval_indef is 0, then it is the same as no
  // retval_indef.
  if (src_retval_indef
      && src_retval_indef == tgt_retval_indef
      && src_retval_indef->op == Op::VALUE
      && !src_retval_indef->value())
    {
      src_retval_indef = nullptr;
      tgt_retval_indef = nullptr;
    }

  if (!src_abort && tgt_abort)
    {
      Basic_block *bb = func->bbs[0];
      src_abort = bb->value_inst(0, 1);
      src_exit = bb->value_inst(0, 1);
      src_exit_val = bb->value_inst(0, tgt_exit_val->bitsize);
    }
  if (!tgt_abort && src_abort)
    {
      Basic_block *bb = func->bbs[0];
      tgt_abort = bb->value_inst(0, 1);
      tgt_exit = bb->value_inst(0, 1);
      tgt_exit_val = bb->value_inst(0, src_exit_val->bitsize);
    }
}

Solver_result run_solver(Bitwuzla& solver, const char *str)
{
  if (config.verbose > 2)
    {
      fprintf(stderr, "SMTGCC: SMTLIB2 for %s:\n", str);
      solver.print_formula(std::cerr);
    }
  Result result = solver.check_sat();
  if (result == Result::UNSAT)
    {
      return {Result_status::correct, {}};
    }
  else if (result == Result::SAT)
    {
      std::string msg = "Transformation is not correct ("s + str + ")\n";
      // TODO: Display the model.
      return {Result_status::incorrect, msg};
    }
  else if (result == Result::UNKNOWN)
    {
      std::string msg = "Analysis timed out ("s + str + ")\n";
      return {Result_status::unknown, msg};
    }

  throw Not_implemented("run_solver: unknown solver.check_sat return");
}

void set_solver_limits(Options& options)
{
  options.set(Option::PRODUCE_MODELS, true);
  options.set(Option::TIME_LIMIT_PER, config.timeout);
  options.set(Option::MEMORY_LIMIT, config.memory_limit);
}

} // end anonymous namespace

std::pair<SStats, Solver_result> check_refine_bitwuzla(Function *func)
{
  assert(func->bbs.size() == 1);

  TermManager tm;
  Options options;
  set_solver_limits(options);
  Bitwuzla solver(tm, options);

  SStats stats;
  stats.skipped = false;

  Converter conv(tm, func);
  Term src_common_ub_term = conv.inst_as_bool(conv.src_common_ub);
  Term src_unique_ub_term = conv.inst_as_bool(conv.src_unique_ub);
  Term not_src_common_ub_term = tm.mk_term(Kind::NOT, {src_common_ub_term});
  Term not_src_unique_ub_term = tm.mk_term(Kind::NOT, {src_unique_ub_term});
  Term tgt_unique_ub_term = conv.inst_as_bool(conv.tgt_unique_ub);

  std::string warning;

  // Check that tgt does not have UB that is not in src.
  assert(conv.src_common_ub == conv.tgt_common_ub);
  if (config.optimize_ub
      && conv.src_unique_ub != conv.tgt_unique_ub
      && !(conv.tgt_unique_ub->op == Op::VALUE
	   && conv.tgt_unique_ub->value() == 0))
    {
      solver.push(1);
      solver.assert_formula(not_src_common_ub_term);
      solver.assert_formula(not_src_unique_ub_term);
      solver.assert_formula(tgt_unique_ub_term);
      uint64_t start_time = get_time();
      Solver_result solver_result = run_solver(solver, "UB");
      stats.time[3] = std::max(get_time() - start_time, (uint64_t)1);
      if (solver_result.status == Result_status::incorrect)
	return std::pair<SStats, Solver_result>(stats, solver_result);
      if (solver_result.status == Result_status::unknown)
	{
	  assert(solver_result.message);
	  warning = warning + *solver_result.message;
	}
      solver.pop(1);
    }

  // Check that the function calls abort/exit identically for src and tgt.
  if (conv.src_abort != conv.tgt_abort
      || conv.src_exit != conv.tgt_exit
      || conv.src_exit_val != conv.tgt_exit_val)
    {
      solver.push(1);
      Term src_abort_term = conv.inst_as_bool(conv.src_abort);
      Term tgt_abort_term = conv.inst_as_bool(conv.tgt_abort);
      Term abort_differ =
	tm.mk_term(Kind::XOR, {src_abort_term, tgt_abort_term});
      Term src_exit_term = conv.inst_as_bool(conv.src_exit);
      Term tgt_exit_term = conv.inst_as_bool(conv.tgt_exit);
      Term exit_differ =
	tm.mk_term(Kind::XOR, {src_exit_term, tgt_exit_term});
      Term src_exit_val_term = conv.inst_as_bv(conv.src_exit_val);
      Term tgt_exit_val_term = conv.inst_as_bv(conv.tgt_exit_val);
      Term exit_val_differ =
	tm.mk_term(Kind::DISTINCT, {src_exit_val_term, tgt_exit_val_term});
      exit_val_differ = tm.mk_term(Kind::AND, {src_exit_term, exit_val_differ});
      Term differ = tm.mk_term(Kind::OR, {abort_differ, exit_differ});
      differ = tm.mk_term(Kind::OR, {differ, exit_val_differ});
      solver.assert_formula(not_src_common_ub_term);
      solver.assert_formula(not_src_unique_ub_term);
      solver.assert_formula(differ);
      uint64_t start_time = get_time();
      Solver_result solver_result = run_solver(solver, "abort/exit");
      stats.time[0] = std::max(get_time() - start_time, (uint64_t)1);
      if (solver_result.status == Result_status::incorrect)
	return std::pair<SStats, Solver_result>(stats, solver_result);
      if (solver_result.status == Result_status::unknown)
	{
	  assert(solver_result.message);
	  warning = warning + *solver_result.message;
	}
      solver.pop(1);
    }

  // Check that the returned value (if any) is the same for src and tgt.
  if ((conv.src_retval != conv.tgt_retval
       || conv.src_retval_indef != conv.tgt_retval_indef)
      && !(conv.src_retval_indef && is_value_m1(conv.src_retval_indef)))
    {
      solver.push(1);
      assert(conv.src_retval && conv.tgt_retval);
      Term src_term = conv.inst_as_bv(conv.src_retval);
      Term tgt_term = conv.inst_as_bv(conv.tgt_retval);

      Term is_more_indef = tm.mk_false();
      if (conv.src_retval_indef)
	{
	  Term src_indef = conv.inst_as_bv(conv.src_retval_indef);
	  Term src_mask = tm.mk_term(Kind::BV_NOT, {src_indef});
	  src_term = tm.mk_term(Kind::BV_AND, {src_term, src_mask});
	  tgt_term = tm.mk_term(Kind::BV_AND, {tgt_term, src_mask});

	  // Check that tgt is not more indef than src.
	  if (conv.tgt_retval_indef != conv.src_retval_indef)
	    {
	      Term tgt_indef = conv.inst_as_bv(conv.tgt_retval_indef);
	      Term indef_result =
		tm.mk_term(Kind::BV_AND, {tgt_indef, src_mask});
	      Term zero =
		tm.mk_bv_zero(tm.mk_bv_sort(conv.tgt_retval->bitsize));
	      is_more_indef = tm.mk_term(Kind::DISTINCT, {indef_result, zero});
	    }
	}

      solver.assert_formula(not_src_common_ub_term);
      solver.assert_formula(not_src_unique_ub_term);
      if (conv.src_abort)
	{
	  Term aborted_term = conv.inst_as_bool(conv.src_abort);
	  Term not_aborted =
	    tm.mk_term(Kind::NOT, {conv.inst_as_bool(conv.src_abort)});
	  solver.assert_formula(not_aborted);
	}
      if (conv.src_exit)
	{
	  Term exited_term = conv.inst_as_bool(conv.src_exit);
	  Term not_exited =
	    tm.mk_term(Kind::NOT, {conv.inst_as_bool(conv.src_exit)});
	  solver.assert_formula(not_exited);
	}
      Term res1 = tm.mk_term(Kind::DISTINCT, {src_term, tgt_term});
      Term res2 = tm.mk_term(Kind::OR, {res1, is_more_indef});
      solver.assert_formula(res2);
      uint64_t start_time = get_time();
      Solver_result solver_result = run_solver(solver, "retval");
      stats.time[1] = std::max(get_time() - start_time, (uint64_t)1);
      if (solver_result.status == Result_status::incorrect)
	{
#if 0
	  assert(solver_result.message);
	  Term src_val = solver.getValue(src_term);
	  Term tgt_val = solver.getValue(tgt_term);
	  std::string msg = *solver_result.message;
	  msg = msg + "src retval: " + src_val.getBitVectorValue(16) + "\n";
	  msg = msg + "tgt retval: " + tgt_val.getBitVectorValue(16) + "\n";
	  if (conv.src_retval_indef)
	    {
	      Term src_indef = conv.inst_as_bv(conv.src_retval_indef);
	      Term tgt_indef = conv.inst_as_bv(conv.tgt_retval_indef);
	      Term src_indef_val = solver.getValue(src_indef);
	      Term tgt_indef_val = solver.getValue(tgt_indef);
	      msg = msg + "src indef: " + src_indef_val.getBitVectorValue(16) + "\n";
	      msg = msg + "tgt indef: " + tgt_indef_val.getBitVectorValue(16) + "\n";
	    }
#else
	  std::string msg = *solver_result.message;
#endif
	  Solver_result result = {Result_status::incorrect, msg};
	  return std::pair<SStats, Solver_result>(stats, result);
	}
      if (solver_result.status == Result_status::unknown)
	{
	  assert(solver_result.message);
	  warning = warning + *solver_result.message;
	}
      solver.pop(1);
    }

  // Check that the global memory is consistent for src and tgt.
  if (conv.src_memory != conv.tgt_memory
      || conv.src_memory_size != conv.tgt_memory_size
      || conv.src_memory_indef != conv.tgt_memory_indef)
    {
      solver.push(1);
      solver.assert_formula(not_src_common_ub_term);
      solver.assert_formula(not_src_unique_ub_term);
      Term src_mem = conv.inst_as_array(conv.src_memory);
      Term src_mem_size = conv.inst_as_array(conv.src_memory_size);
      Term src_mem_indef = conv.inst_as_array(conv.src_memory_indef);

      Term tgt_mem = conv.inst_as_array(conv.tgt_memory);
      Term tgt_mem_indef = conv.inst_as_array(conv.tgt_memory_indef);

      Sort ptr_sort = tm.mk_bv_sort(func->module->ptr_bits);
      Term ptr = tm.mk_const(ptr_sort, ".ptr");
      uint32_t ptr_id_high = func->module->ptr_id_high;
      uint32_t ptr_id_low = func->module->ptr_id_low;
      Term id = tm.mk_term(Kind::BV_EXTRACT, {ptr}, {ptr_id_high, ptr_id_low});
      uint32_t ptr_offset_high = func->module->ptr_offset_high;
      uint32_t ptr_offset_low = func->module->ptr_offset_low;
      Term offset =
	tm.mk_term(Kind::BV_EXTRACT, {ptr}, {ptr_offset_high, ptr_offset_low});

      // Only check global memory.
      Term zero_id = tm.mk_bv_zero(tm.mk_bv_sort(func->module->ptr_id_bits));
      Term cond1 = tm.mk_term(Kind::BV_SGT, {id, zero_id});
      solver.assert_formula(cond1);

      // Only check memory within a memory block.
      Term mem_size = tm.mk_term(Kind::ARRAY_SELECT, {src_mem_size, id});
      Term cond2 = tm.mk_term(Kind::BV_ULT, {offset, mem_size});
      solver.assert_formula(cond2);

      // Check that src and tgt are the same for the bits where src is defined
      // and that tgt is not more indefinite than src.
      Term src_indef = tm.mk_term(Kind::ARRAY_SELECT, {src_mem_indef, ptr});
      Term src_mask = tm.mk_term(Kind::BV_NOT, {src_indef});
      Term src_byte = tm.mk_term(Kind::ARRAY_SELECT, {src_mem, ptr});
      src_byte = tm.mk_term(Kind::BV_AND, {src_byte, src_mask});
      Term tgt_byte = tm.mk_term(Kind::ARRAY_SELECT, {tgt_mem, ptr});
      tgt_byte = tm.mk_term(Kind::BV_AND, {tgt_byte, src_mask});
      Term cond3 = tm.mk_term(Kind::DISTINCT, {src_byte, tgt_byte});
      Term tgt_indef = tm.mk_term(Kind::ARRAY_SELECT, {tgt_mem_indef, ptr});
      Term tgt_more_indef = tm.mk_term(Kind::BV_AND, {tgt_indef, src_mask});
      Term zero_byte = tm.mk_bv_zero(tm.mk_bv_sort(8));
      Term cond4 = tm.mk_term(Kind::DISTINCT, {tgt_more_indef, zero_byte});
      solver.assert_formula(tm.mk_term(Kind::OR, {cond3, cond4}));

      // TODO: Should make a better getBitVectorValue that prints values as
      // hex, etc.
      uint64_t start_time = get_time();
      Solver_result solver_result = run_solver(solver, "Memory");
      stats.time[2] = std::max(get_time() - start_time, (uint64_t)1);
      if (solver_result.status == Result_status::incorrect)
	{
#if 0
	  assert(solver_result.message);
	  Term ptr_val = solver.getValue(ptr);
	  Term src_byte_val = solver.getValue(src_byte);
	  Term tgt_byte_val = solver.getValue(tgt_byte);
	  Term src_indef_val = solver.getValue(src_indef);
	  Term tgt_indef_val = solver.getValue(tgt_indef);
	  std::string msg = *solver_result.message;
	  msg = msg + "\n.ptr = " + ptr_val.getBitVectorValue(16) + "\n";
	  msg = msg + "src *.ptr: " + src_byte_val.getBitVectorValue(16) + "\n";
	  msg = msg + "tgt *.ptr: " + tgt_byte_val.getBitVectorValue(16) + "\n";
	  msg = msg + "src indef: "
	    + src_indef_val.getBitVectorValue(16) + "\n";
	  msg = msg + "tgt indef: "
	    + tgt_indef_val.getBitVectorValue(16) + "\n";
#else
	  std::string msg = *solver_result.message;
#endif
	  Solver_result result = {Result_status::incorrect, msg};
	  return std::pair<SStats, Solver_result>(stats, result);
	}
      if (solver_result.status == Result_status::unknown)
	{
	  assert(solver_result.message);
	  warning = warning + *solver_result.message;
	}
      solver.pop(1);
    }

  // Check that tgt does not have UB that is not in src.
  assert(conv.src_common_ub == conv.tgt_common_ub);
  if (!config.optimize_ub
      && conv.src_unique_ub != conv.tgt_unique_ub
      && !(conv.tgt_unique_ub->op == Op::VALUE
	   && conv.tgt_unique_ub->value() == 0))
    {
      solver.push(1);
      solver.assert_formula(not_src_common_ub_term);
      solver.assert_formula(not_src_unique_ub_term);
      solver.assert_formula(tgt_unique_ub_term);
      uint64_t start_time = get_time();
      Solver_result solver_result = run_solver(solver, "UB");
      stats.time[3] = std::max(get_time() - start_time, (uint64_t)1);
      if (solver_result.status == Result_status::incorrect)
	return std::pair<SStats, Solver_result>(stats, solver_result);
      if (solver_result.status == Result_status::unknown)
	{
	  assert(solver_result.message);
	  warning = warning + *solver_result.message;
	}
      solver.pop(1);
    }

  if (!warning.empty())
    {
      Solver_result result = {Result_status::unknown, warning};
      return std::pair<SStats, Solver_result>(stats, result);
    }
  return std::pair<SStats, Solver_result>(stats, {Result_status::correct, {}});
}

} // end namespace smtgcc

#else

#include "smtgcc.h"

namespace smtgcc {

std::pair<SStats, Solver_result> check_refine_bitwuzla(Function *)
{
  throw Not_implemented("bitwuzla is not available");
}

} // end namespace smtgcc

#endif
