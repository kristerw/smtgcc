#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <time.h>

#include "config.h"
#include "smtgcc.h"

using namespace std::string_literals;

namespace smtgcc {

const std::array<Inst_info, 97> inst_info{{
  // Integer Comparison
  {"eq", Op::EQ, Inst_class::icomparison, true, true},
  {"ne", Op::NE, Inst_class::icomparison, true, true},
  {"sle", Op::SLE, Inst_class::icomparison, true, false},
  {"slt", Op::SLT, Inst_class::icomparison, true, false},
  {"ule", Op::ULE, Inst_class::icomparison, true, false},
  {"ult", Op::ULT, Inst_class::icomparison, true, false},

  // Floating-point comparison
  {"feq", Op::FEQ, Inst_class::fcomparison, true, true},
  {"fle", Op::FLE, Inst_class::fcomparison, true, false},
  {"flt", Op::FLT, Inst_class::fcomparison, true, false},
  {"fne", Op::FNE, Inst_class::fcomparison, true, true},

  // Integer unary
  {"is_const_mem", Op::IS_CONST_MEM, Inst_class::iunary, true, false},
  {"is_inf", Op::IS_INF, Inst_class::iunary, true, false},
  {"is_nan", Op::IS_NAN, Inst_class::iunary, true, false},
  {"is_noncanonical_nan", Op::IS_NONCANONICAL_NAN, Inst_class::iunary, true, false},
  {"mov", Op::MOV, Inst_class::iunary, true, false},
  {"neg", Op::NEG, Inst_class::iunary, true, false},
  {"not", Op::NOT, Inst_class::iunary, true, false},
  {"simp_barrier", Op::SIMP_BARRIER, Inst_class::iunary, true, false},

  // Floating-point unary
  {"fabs", Op::FABS, Inst_class::funary, true, false},
  {"fneg", Op::FNEG, Inst_class::funary, true, false},
  {"nan", Op::NAN, Inst_class::funary, true, false},

  // Integer binary
  {"add", Op::ADD, Inst_class::ibinary, true, true},
  {"and", Op::AND, Inst_class::ibinary, true, true},
  {"array_get_flag", Op::ARRAY_GET_FLAG, Inst_class::ibinary, true, false},
  {"array_get_indef", Op::ARRAY_GET_INDEF, Inst_class::ibinary, true, false},
  {"array_get_size", Op::ARRAY_GET_SIZE, Inst_class::ibinary, true, false},
  {"array_load", Op::ARRAY_LOAD, Inst_class::ibinary, true, false},
  {"ashr", Op::ASHR, Inst_class::ibinary, true, false},
  {"concat", Op::CONCAT, Inst_class::ibinary, true, false},
  {"lshr", Op::LSHR, Inst_class::ibinary, true, false},
  {"mul", Op::MUL, Inst_class::ibinary, true, true},
  {"or", Op::OR, Inst_class::ibinary, true, true},
  {"sadd_wraps", Op::SADD_WRAPS, Inst_class::ibinary, true, true},
  {"sdiv", Op::SDIV, Inst_class::ibinary, true, false},
  {"shl", Op::SHL, Inst_class::ibinary, true, false},
  {"smul_wraps", Op::SMUL_WRAPS, Inst_class::ibinary, true, true},
  {"srem", Op::SREM, Inst_class::ibinary, true, false},
  {"ssub_wraps", Op::SSUB_WRAPS, Inst_class::ibinary, true, false},
  {"sub", Op::SUB, Inst_class::ibinary, true, false},
  {"udiv", Op::UDIV, Inst_class::ibinary, true, false},
  {"urem", Op::UREM, Inst_class::ibinary, true, false},
  {"xor", Op::XOR, Inst_class::ibinary, true, true},

  // Floating-point binary
  {"fadd", Op::FADD, Inst_class::fbinary, true, true},
  {"fdiv", Op::FDIV, Inst_class::fbinary, true, false},
  {"fmul", Op::FMUL, Inst_class::fbinary, true, true},
  {"fsub", Op::FSUB, Inst_class::fbinary, true, false},

  // Ternary
  {"array_set_flag", Op::ARRAY_SET_FLAG, Inst_class::ternary, true, false},
  {"array_set_indef", Op::ARRAY_SET_INDEF, Inst_class::ternary, true, false},
  {"array_set_size", Op::ARRAY_SET_SIZE, Inst_class::ternary, true, false},
  {"array_store", Op::ARRAY_STORE, Inst_class::ternary, true, false},
  {"extract", Op::EXTRACT, Inst_class::ternary, true, false},
  {"ite", Op::ITE, Inst_class::ternary, true, false},

  // Conversions
  {"f2s", Op::F2S, Inst_class::conv, true, false},
  {"f2u", Op::F2U, Inst_class::conv, true, false},
  {"fchprec", Op::FCHPREC, Inst_class::conv, true, false},
  {"s2f", Op::S2F, Inst_class::conv, true, false},
  {"sext", Op::SEXT, Inst_class::conv, true, false},
  {"u2f", Op::U2F, Inst_class::conv, true, false},
  {"zext", Op::ZEXT, Inst_class::conv, true, false},

  // Memory state
  {"memory", Op::MEMORY, Inst_class::mem_ternary, true, false},
  {"mem_array", Op::MEM_ARRAY, Inst_class::mem_nullary, true, false},
  {"mem_flag_array", Op::MEM_FLAG_ARRAY, Inst_class::mem_nullary, true, false},
  {"mem_indef_array", Op::MEM_INDEF_ARRAY, Inst_class::mem_nullary, true, false},
  {"mem_size_array", Op::MEM_SIZE_ARRAY, Inst_class::mem_nullary, true, false},

  // Load/store
  {"free", Op::FREE, Inst_class::ls_unary, false, false},
  {"get_mem_flag", Op::GET_MEM_FLAG, Inst_class::ls_unary, true, false},
  {"get_mem_indef", Op::GET_MEM_INDEF, Inst_class::ls_unary, true, false},
  {"get_mem_size", Op::GET_MEM_SIZE, Inst_class::ls_unary, true, false},
  {"load", Op::LOAD, Inst_class::ls_unary, true, false},
  {"memmove", Op::MEMMOVE, Inst_class::ls_ternary, false, false},
  {"memset", Op::MEMSET, Inst_class::ls_ternary, false, false},
  {"set_mem_flag", Op::SET_MEM_FLAG, Inst_class::ls_binary, false, false},
  {"set_mem_indef", Op::SET_MEM_INDEF, Inst_class::ls_binary, false, false},
  {"store", Op::STORE, Inst_class::ls_binary, false, false},

  // Register
  {"read", Op::READ, Inst_class::reg_unary, true, false},
  {"register", Op::REGISTER, Inst_class::reg_unary, true, false},
  {"write", Op::WRITE, Inst_class::reg_binary, false, false},

  // Solver
  {"assert", Op::ASSERT, Inst_class::solver_unary, false, false},
  {"exit", Op::EXIT, Inst_class::solver_ternary, false, false},
  {"param", Op::PARAM, Inst_class::solver_binary, true, false},
  {"print", Op::PRINT, Inst_class::solver_binary, false, false},
  {"src_assert", Op::SRC_ASSERT, Inst_class::solver_unary, false, false},
  {"src_exit", Op::SRC_EXIT, Inst_class::solver_ternary, false, false},
  {"src_mem", Op::SRC_MEM, Inst_class::solver_ternary, false, false},
  {"src_retval", Op::SRC_RETVAL, Inst_class::solver_binary, false, false},
  {"src_ub", Op::SRC_UB, Inst_class::solver_binary, false, false},
  {"symbolic", Op::SYMBOLIC, Inst_class::solver_binary, true, false},
  {"tgt_assert", Op::TGT_ASSERT, Inst_class::solver_unary, false, false},
  {"tgt_exit", Op::TGT_EXIT, Inst_class::solver_ternary, false, false},
  {"tgt_mem", Op::TGT_MEM, Inst_class::solver_ternary, false, false},
  {"tgt_retval", Op::TGT_RETVAL, Inst_class::solver_binary, false, false},
  {"tgt_ub", Op::TGT_UB, Inst_class::solver_binary, false, false},
  {"ub", Op::UB, Inst_class::solver_unary, false, false},

  // Special
  {"br", Op::BR, Inst_class::special, false, false},
  {"phi", Op::PHI, Inst_class::special, true, false},
  {"ret", Op::RET, Inst_class::special, false, false},
  {"value", Op::VALUE, Inst_class::special, true, false},
}};

#ifndef NDEBUG
// We are indexing into inst_info using the opcode. Validate that the
// array is sorted correctly.
struct Inst_info_validator
{
  Inst_info_validator()
  {
    for (unsigned i = 0; i < inst_info.size(); i++)
      {
	assert(i == (int)inst_info[i].op);
      }
  }
};
Inst_info_validator inst_info_validator;
#endif

Config::Config()
{
  if (char *p = getenv("SMTGCC_VERBOSE"))
    verbose = atoi(p);

  if (char *p = getenv("SMTGCC_TIMEOUT"))
    timeout = atoi(p);

  if (char *p = getenv("SMTGCC_MEMORY_LIMIT"))
    memory_limit = atoi(p);

  if (char *p = getenv("SMTGCC_CACHE"))
    {
      if (!strcmp(p, "redis"))
	redis_cache = true;
      else
	throw Not_implemented("Unknown SMTGCC_CACHE");
    }

#if HAVE_LIBZ3
  smt_solver = SmtSolver::z3;
#elif HAVE_LIBCVC5
  smt_solver = SmtSolver::cvc5;
#else
#error "No SMT solver was configured"
#endif
  if (char *p = getenv("SMTGCC_SMT_SOLVER"))
    {
      if (!strcmp(p, "Z3") || !strcmp(p, "z3"))
	smt_solver = SmtSolver::z3;
      else if (!strcmp(p, "cvc5") || !strcmp(p, "CVC5"))
	smt_solver = SmtSolver::cvc5;
      else
	throw Not_implemented("Unknown SMTGCC_SMT_SOLVER");
    }
}

Config config;

Inst *create_inst(Op op)
{
  Inst *inst = new Inst;
  inst->op = op;
  inst->nof_args = 0;
  inst->bitsize = 0;
  return inst;
}

Inst *create_inst(Op op, Inst *arg)
{
  Inst *inst = new Inst;
  inst->op = op;
  inst->nof_args = 1;
  inst->args[0] = arg;
  if (op == Op::IS_CONST_MEM
      || op == Op::IS_INF
      || op == Op::IS_NAN
      || op == Op::IS_NONCANONICAL_NAN
      || op == Op::GET_MEM_FLAG)
    inst->bitsize = 1;
  else if (op == Op::GET_MEM_INDEF || op == Op::LOAD)
    inst->bitsize = 8;
  else if (op == Op::GET_MEM_SIZE)
    inst->bitsize = arg->bb->func->module->ptr_offset_bits;
  else if (op == Op::NAN || op == Op::REGISTER)
    inst->bitsize = arg->value();
  else if (op == Op::READ)
    {
      assert(arg->op == Op::REGISTER);
      inst->bitsize = arg->bitsize;
    }
  else
    inst->bitsize = arg->bitsize;
  return inst;
}

Inst::Inst()
{
  static uint32_t next_id = 0;
  id = next_id++;
}

Inst *create_inst(Op op, Inst *arg1, Inst *arg2)
{
  if (inst_info[(int)op].is_commutative
      && arg2->op != Op::VALUE && arg1->op == Op::VALUE)
    std::swap(arg1, arg2);
  Inst *inst = new Inst;
  inst->op = op;
  inst->nof_args = 2;
  inst->args[0] = arg1;
  inst->args[1] = arg2;
  Inst_class iclass = inst_info[(int)op].iclass;
  if (iclass == Inst_class::icomparison
      || iclass == Inst_class::fcomparison
      || op == Op::SADD_WRAPS
      || op == Op::SSUB_WRAPS
      || op == Op::SMUL_WRAPS)
    {
      assert(arg1->bitsize == arg2->bitsize);
      inst->bitsize = 1;
    }
  else if (iclass == Inst_class::conv)
    {
      inst->bitsize = arg2->value();
      if (op == Op::SEXT || op == Op::ZEXT)
	{
	  assert(inst->bitsize > arg1->bitsize);
	}
    }
  else if (op == Op::CONCAT)
    {
      inst->bitsize = arg1->bitsize + arg2->bitsize;
    }
  else if (op == Op::PARAM || op == Op::SYMBOLIC)
    {
      assert(arg1->op == Op::VALUE);
      assert(arg2->op == Op::VALUE);
      inst->bitsize = arg2->value();
    }
  else if (op == Op::ARRAY_LOAD || op == Op::ARRAY_GET_INDEF)
    inst->bitsize = 8;
  else if (op == Op::ARRAY_GET_FLAG)
    inst->bitsize = 1;
  else if (op == Op::ARRAY_GET_SIZE)
    inst->bitsize = arg2->bb->func->module->ptr_offset_bits;
  else if (op == Op::STORE || op == Op::SET_MEM_INDEF)
    {
      assert(arg1->bitsize == arg1->bb->func->module->ptr_bits);
      assert(arg2->bitsize == 8);
      inst->bitsize = 0;
    }
  else if (op == Op::SET_MEM_FLAG)
    {
      assert(arg1->bitsize == arg1->bb->func->module->ptr_bits);
      assert(arg2->bitsize == 1);
      inst->bitsize = 0;
    }
  else if (op == Op::WRITE)
    {
      assert(arg1->op == Op::REGISTER);
      assert(arg1->args[0]->value() == arg2->bitsize);
      inst->bitsize = 0;
    }
  else if (op == Op::PRINT)
    {
      inst->bitsize = 0;
    }
  else
    {
      assert(arg1->bitsize == arg2->bitsize);
      inst->bitsize = arg1->bitsize;
    }
  return inst;
}

Inst *create_inst(Op op, Inst *arg1, uint32_t arg2_val)
{
  assert(inst_info[(int)op].iclass == Inst_class::conv);
  Inst *arg2 = arg1->bb->value_inst(arg2_val, 32);
  return create_inst(op, arg1, arg2);
}

Inst *create_inst(Op op, Inst *arg1, Inst *arg2, Inst *arg3)
{
  Inst *inst = new Inst;
  inst->op = op;
  inst->nof_args = 3;
  inst->args[0] = arg1;
  inst->args[1] = arg2;
  inst->args[2] = arg3;
  if (op == Op::EXTRACT)
    {
      uint32_t high = arg2->value();
      uint32_t low = arg3->value();
      assert(high >= low);
      assert(high < arg1->bitsize);
      inst->bitsize = 1 + high - low;
    }
  else if (op == Op::ARRAY_SET_FLAG
	   || op == Op::ARRAY_SET_SIZE
	   || op == Op::ARRAY_SET_INDEF
	   || op == Op::ARRAY_STORE
	   || op == Op::MEMMOVE
	   || op == Op::MEMSET
	   || op == Op::SRC_MEM
	   || op == Op::TGT_MEM)
    {
      inst->bitsize = 0;
    }
  else if (op == Op::MEMORY)
    {
      assert(arg1->bitsize == arg1->bb->func->module->ptr_id_bits);
      assert(arg1->op == Op::VALUE);
      assert(arg2->bitsize == arg2->bb->func->module->ptr_offset_bits);
      assert(arg2->op == Op::VALUE);
      assert(arg3->op == Op::VALUE);
      inst->bitsize = arg1->bb->func->module->ptr_bits;
    }
  else if (op == Op::ITE)
    {
      assert(arg1->bitsize == 1);
      assert(arg2->bitsize == arg3->bitsize);
      inst->bitsize = arg2->bitsize;
    }
  else
    {
      assert(op == Op::EXIT || op == Op::SRC_EXIT || op == Op::TGT_EXIT);
      assert(arg1->bitsize == 1);
      assert(arg2->bitsize == 1);
    }
  return inst;
}

Inst *create_inst(Op op, Inst *arg1, uint32_t arg2_val, uint32_t arg3_val)
{
  assert(op == Op::EXTRACT);
  Inst *arg2 = arg1->bb->value_inst(arg2_val, 32);
  Inst *arg3 = arg1->bb->value_inst(arg3_val, 32);
  return create_inst(op, arg1, arg2, arg3);
}

Inst *create_br_inst(Basic_block *dest_bb)
{
  Inst *inst = new Inst;
  inst->op = Op::BR;
  inst->u.br1.dest_bb = dest_bb;
  return inst;
}

Inst *create_phi_inst(int bitsize)
{
  Inst *inst = new Inst;
  inst->op = Op::PHI;
  inst->bitsize = bitsize;
  return inst;
}

Inst *Inst::get_phi_arg(Basic_block *bb)
{
  auto it = std::find_if(phi_args.begin(), phi_args.end(), [bb](const Phi_arg& arg) {
    return arg.bb == bb;
  });
  assert(it != phi_args.end());
  return (*it).inst;
}

void Inst::update_phi_arg(Inst *inst, Basic_block *bb)
{
  auto it = std::find_if(phi_args.begin(), phi_args.end(), [bb](const Phi_arg& arg) {
    return arg.bb == bb;
  });
  assert(it != phi_args.end());
  Inst *orig_arg_inst = (*it).inst;

  (*it).inst = inst;
  inst->used_by.insert(this);

  // Remove this phi nodes from the original arg_inst if it is not used
  // any longer.
  it = std::find_if(phi_args.begin(), phi_args.end(), [orig_arg_inst](const Phi_arg& arg) {
    return arg.inst == orig_arg_inst;
  });
  if (it == phi_args.end())
    orig_arg_inst->used_by.erase(this);
}

void Inst::add_phi_arg(Inst *inst, Basic_block *bb)
{
  Phi_arg phi_arg;
  phi_arg.inst = inst;
  phi_arg.bb = bb;
  this->phi_args.push_back(phi_arg);
  assert(inst->bitsize == this->bitsize);
  inst->used_by.insert(this);
}

void Inst::remove_phi_arg(Basic_block *bb)
{
  auto it = std::find_if(phi_args.begin(), phi_args.end(), [bb](const Phi_arg& arg) {
    return arg.bb == bb;
  });
  assert(it != phi_args.end());
  Inst *arg_inst = (*it).inst;
  phi_args.erase(it);

  // Remove this phi nodes from the arg_inst used_by if it is not used by
  // some other of the phi's arguments.
  it = std::find_if(phi_args.begin(), phi_args.end(), [arg_inst](const Phi_arg& arg) {
    return arg.inst == arg_inst;
  });
  if (it == phi_args.end())
    arg_inst->used_by.erase(this);
}

void Inst::remove_phi_args()
{
  while (!phi_args.empty())
    {
      remove_phi_arg(phi_args.back().bb);
    }
}

void Inst::print(FILE *stream) const
{
  fprintf(stream, "  ");
  if (has_lhs())
    fprintf(stream, "%%%" PRIu32 " = ", id);
  fprintf(stream, "%s", name());
  for (unsigned i = 0; i < nof_args; i++)
    {
      if (i == 0)
	fprintf(stream, " ");
      else
	fprintf(stream, ", ");
      fprintf(stream, "%%%" PRIu32, args[i]->id);
    }
  if (op == Op::BR)
    {
      if (nof_args == 0)
	fprintf(stream, " .%d", u.br1.dest_bb->id);
      else
	fprintf(stream, ", .%d, .%d", u.br3.true_bb->id,
		u.br3.false_bb->id);
    }
  else if (op == Op::VALUE)
    {
      uint64_t low = value();
      uint64_t high = value() >> 64;
      if (value() < 0x10000)
	fprintf(stream, " %" PRIu64 ", %" PRIu32, low, bitsize);
      else if (value() <= std::numeric_limits<uint32_t>::max())
	fprintf(stream, " 0x%08" PRIx64 ", %" PRIu32, low, bitsize);
      else if (value() <= std::numeric_limits<uint64_t>::max())
	fprintf(stream, " 0x%016" PRIx64 ", %" PRIu32, low, bitsize);
      else
	fprintf(stream, " 0x%016" PRIx64 "%016" PRIx64 ", %" PRIu32,
		high, low, bitsize);
    }
  else if (op == Op::PHI)
    {
      for (auto phi : phi_args)
	{
	  const char *s = (phi.bb != phi_args[0].bb) ? "," : "";
	  fprintf(stream, "%s [ %%%" PRIu32", .%d ]", s, phi.inst->id,
		  phi.bb->id);
	}
    }

  fprintf(stream, "\n");
}

Inst *create_ret_inst()
{
  Inst *inst = new Inst;
  inst->op = Op::RET;
  inst->bitsize = 0;
  return inst;
}

Inst *create_ret_inst(Inst *arg)
{
  Inst *inst = new Inst;
  inst->op = Op::RET;
  inst->nof_args = 1;
  inst->args[0] = arg;
  inst->bitsize = arg->bitsize;
  return inst;
}

Inst *create_ret_inst(Inst *arg1, Inst *arg2)
{
  assert(arg1->bitsize == arg2->bitsize);
  Inst *inst = new Inst;
  inst->op = Op::RET;
  inst->nof_args = 2;
  inst->args[0] = arg1;
  inst->args[1] = arg2;
  inst->bitsize = arg1->bitsize;
  return inst;
}

Inst *create_br_inst(Inst *cond, Basic_block *true_bb, Basic_block *false_bb)
{
  assert(true_bb != false_bb);
  Inst *inst = new Inst;
  inst->op = Op::BR;
  inst->nof_args = 1;
  inst->args[0] = cond;
  inst->u.br3.true_bb = true_bb;
  inst->u.br3.false_bb = false_bb;
  return inst;
}

unsigned __int128 Inst::value() const
{
  assert(op == Op::VALUE);
  return u.value.value;
}

__int128 Inst::signed_value() const
{
  assert(op == Op::VALUE);
  assert(bitsize <= 128);
  __int128 value = u.value.value;
  if (bitsize < 128)
    value = (value << (128 - bitsize)) >> (128 - bitsize);
  return value;
}

void Inst::update_uses()
{
  assert(nof_args <= 3);
  if (nof_args > 0)
    args[0]->used_by.insert(this);
  if (nof_args > 1)
    args[1]->used_by.insert(this);
  if (nof_args > 2)
    args[2]->used_by.insert(this);
}

void Inst::insert_after(Inst *inst)
{
  // self.validate_unused()
  assert(!bb);
  assert(!prev);
  assert(!next);

  if (inst->op == Op::PHI)
    {
      if (inst->bb->first_inst)
	insert_before(inst->bb->first_inst);
      else
	inst->bb->insert_last(this);
      return;
    }

  bb = inst->bb;
  update_uses();
  if (inst->next)
    inst->next->prev = this;
  next = inst->next;
  inst->next = this;
  prev = inst;
  if (inst == bb->last_inst)
    bb->last_inst = this;
}

void Inst::insert_before(Inst *inst)
{
  // self.validate_unused()
  assert(!bb);
  assert(!prev);
  assert(!next);
  bb = inst->bb;
  update_uses();
  if (inst->prev)
    inst->prev->next = this;
  prev = inst->prev;
  inst->prev = this;
  next = inst;
  if (inst == bb->first_inst)
    bb->first_inst = this;
}

void Inst::move_after(Inst *inst)
{
  assert(bb);
  assert(op != Op::PHI);

  // Unlink the instruction from the BB.
  if (this == bb->first_inst)
    bb->first_inst = this->next;
  if (this == bb->last_inst)
    bb->last_inst = this->prev;
  if (this->prev)
    this->prev->next = this->next;
  if (this->next)
    this->next->prev = this->prev;
  next = nullptr;
  prev = nullptr;
  bb = nullptr;

  insert_after(inst);
}

void Inst::move_before(Inst *inst)
{
  assert(bb);
  assert(op != Op::PHI);
  assert(inst->op != Op::PHI);

  // Unlink the instruction from the BB.
  if (this == bb->first_inst)
    bb->first_inst = this->next;
  if (this == bb->last_inst)
    bb->last_inst = this->prev;
  if (this->prev)
    this->prev->next = this->next;
  if (this->next)
    this->next->prev = this->prev;
  next = nullptr;
  prev = nullptr;
  bb = nullptr;

  insert_before(inst);
}

void Inst::replace_use_with(Inst *use, Inst *new_inst)
{
  if (use->op == Op::PHI)
    {
      for (size_t i = 0; i < use->phi_args.size(); i++)
	{
	  if (use->phi_args[i].inst == this)
	    use->phi_args[i].inst = new_inst;
	}
    }
  else
    {
      for (size_t i = 0; i < use->nof_args; i++)
	{
	  if (use->args[i] == this)
	    use->args[i] = new_inst;
	}
    }
  new_inst->used_by.insert(use);

  auto I = std::find(used_by.begin(), used_by.end(), use);
  assert(I != used_by.end());
  used_by.erase(I);
}

void Inst::replace_all_uses_with(Inst *inst)
{
  for (Inst *use : used_by)
    {
      if (use->op == Op::PHI)
	{
	  for (size_t i = 0; i < use->phi_args.size(); i++)
	    {
	      if (use->phi_args[i].inst == this)
		use->phi_args[i].inst = inst;
	    }
	}
      else
	{
	  for (size_t i = 0; i < use->nof_args; i++)
	    {
	      if (use->args[i] == this)
		use->args[i] = inst;
	    }
	}
      inst->used_by.insert(use);
    }
  used_by.clear();
}

// Insert the instruction at the last valid place in the basic block.
// Phi nodes are placed last in the list of phi nodes, even if there are
// already other instructions in the BB.
// Normal instructions are placed last in the BB, but before BR or RET.
void Basic_block::insert_last(Inst *inst)
{
  assert(!inst->bb);
  assert(!inst->prev);
  assert(!inst->next);
  if (inst->op == Op::PHI)
    {
      insert_phi(inst);
      return;
    }
  if (inst->op == Op::BR)
    {
      assert(!last_inst ||
	     (last_inst->op != Op::BR
	      && last_inst->op != Op::RET));
      assert(succs.empty());
      if (inst->nof_args == 0)
	{
	  inst->u.br1.dest_bb->preds.push_back(this);
	  succs.push_back(inst->u.br1.dest_bb);
	}
      else
	{
	  assert(inst->nof_args == 1);
	  inst->u.br3.true_bb->preds.push_back(this);
	  succs.push_back(inst->u.br3.true_bb);
	  inst->u.br3.false_bb->preds.push_back(this);
	  succs.push_back(inst->u.br3.false_bb);
	}
    }
  else if (last_inst)
    {
      if (last_inst->op == Op::BR || last_inst->op == Op::RET)
	{
	  inst->insert_before(last_inst);
	  return;
	}
    }

  // self.validate_unused()
  inst->bb = this;
  inst->update_uses();
  if (last_inst)
    {
      inst->prev = last_inst;
      last_inst->next = inst;
    }
  last_inst = inst;
  if (!first_inst)
    first_inst = inst;
}

void Basic_block::insert_phi(Inst *inst)
{
  assert(!inst->bb);
  assert(!inst->prev);
  assert(!inst->next);
  assert(inst->op == Op::PHI);
  phis.push_back(inst);
  inst->bb = this;
  inst->update_uses();
}

Inst *Basic_block::build_inst(Op op)
{
  Inst *inst = create_inst(op);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_inst(Op op, Inst *arg)
{
  Inst *inst = create_inst(op, arg);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_inst(Op op, uint32_t arg_val)
{
  assert(op == Op::REGISTER || op == Op::NAN);
  Inst *arg = value_inst(arg_val, 32);
  Inst *inst = create_inst(op, arg);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_inst(Op op, Inst *arg1, Inst *arg2)
{
  Inst *inst = create_inst(op, arg1, arg2);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_inst(Op op, Inst *arg1, uint32_t arg2_val)
{
  assert(inst_info[(int)op].iclass == Inst_class::conv);
  Inst *arg2 = value_inst(arg2_val, 32);
  Inst *inst = create_inst(op, arg1, arg2);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_inst(Op op, Inst *arg1, Inst *arg2, Inst *arg3)
{
  Inst *inst = create_inst(op, arg1, arg2, arg3);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_inst(Op op, Inst *arg1, uint32_t arg2_val, uint32_t arg3_val)
{
  assert(op == Op::EXTRACT);
  Inst *arg2 = value_inst(arg2_val, 32);
  Inst *arg3 = value_inst(arg3_val, 32);
  Inst *inst = create_inst(op, arg1, arg2, arg3);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_phi_inst(int bitsize)
{
  Inst *inst = create_phi_inst(bitsize);
  insert_phi(inst);
  return inst;
}

Inst *Basic_block::build_ret_inst()
{
  Inst *inst = create_ret_inst();
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_ret_inst(Inst *arg)
{
  Inst *inst = create_ret_inst(arg);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_ret_inst(Inst *arg1, Inst *arg2)
{
  Inst *inst = create_ret_inst(arg1, arg2);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_br_inst(Basic_block *dest_bb)
{
  Inst *inst = create_br_inst(dest_bb);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::build_br_inst(Inst *cond, Basic_block *true_bb, Basic_block *false_bb)
{
  Inst *inst = create_br_inst(cond, true_bb, false_bb);
  insert_last(inst);
  return inst;
}

Inst *Basic_block::value_inst(unsigned __int128 value, uint32_t bitsize)
{
  return func->value_inst(value, bitsize);
}

Inst *Basic_block::value_m1_inst(uint32_t bitsize)
{
  if (bitsize <= 128)
    return value_inst(-1, bitsize);

  Inst *res = nullptr;
  while (bitsize)
    {
      uint32_t bs = std::min(bitsize, 128u);
      bitsize -= bs;
      Inst *inst = value_inst(-1, bs);
      if (res)
	res = func->bbs[0]->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  return res;
}

Inst *Basic_block::build_extract_id(Inst *arg)
{
  assert(arg->bitsize == func->module->ptr_bits);
  Inst *high = value_inst(func->module->ptr_id_high, 32);
  Inst *low = value_inst(func->module->ptr_id_low, 32);
  return build_inst(Op::EXTRACT, arg, high, low);
}

Inst *Basic_block::build_extract_offset(Inst *arg)
{
  assert(arg->bitsize == func->module->ptr_bits);
  Inst *high = value_inst(func->module->ptr_offset_high, 32);
  Inst *low = value_inst(func->module->ptr_offset_low, 32);
  return build_inst(Op::EXTRACT, arg, high, low);
}

// Convenience function for extracting one bit.
// bit_idx = 0 is the lest significant bit.
Inst *Basic_block::build_extract_bit(Inst *arg, uint32_t bit_idx)
{
  assert(bit_idx < arg->bitsize);
  return build_inst(Op::EXTRACT, arg, bit_idx, bit_idx);
}

// Convenience function for truncating the value to nof_bits bits.
Inst *Basic_block::build_trunc(Inst *arg, uint32_t nof_bits)
{
  assert(nof_bits <= arg->bitsize);
  if (nof_bits == arg->bitsize)
    return arg;
  return build_inst(Op::EXTRACT, arg, nof_bits - 1, 0);
}

void Basic_block::print(FILE *stream) const
{
  fprintf(stream, ".%d:\n", id);
  for (auto phi : phis)
    {
      phi->print(stream);
    }
  for (Inst *inst = first_inst; inst; inst = inst->next)
    {
      inst->print(stream);
    }
}

Basic_block *Function::build_bb()
{
  if (bbs.size() > max_nof_bb)
    throw Not_implemented("too many basic blocks");

  Basic_block *bb = new Basic_block;
  bb->func = this;
  bb->id = next_bb_id++;
  bbs.push_back(bb);
  return bb;
}

Inst *Function::value_inst(unsigned __int128 value, uint32_t bitsize)
{
  assert(bitsize > 0);

  if (bitsize < 128)
    value = (value << (128 - bitsize)) >> (128 - bitsize);
  auto key = std::pair(bitsize, value);
  auto it = values.find(key);
  if (it != values.end())
    return it->second;

  if (bitsize > 128)
    {
      Inst *res = nullptr;
      while (bitsize)
	{
	  uint32_t bs = std::min(bitsize, 128u);
	  bitsize -= bs;
	  Inst *inst = value_inst(value, bs);
	  value = 0;
	  if (res)
	    res = bbs[0]->build_inst(Op::CONCAT, inst, res);
	  else
	    res = inst;
	}
      // We do not insert the result in the values map since this is not
      // a real value instruction, so it will misbehave if the dead code
      // elimination pass removes it.
      return res;
    }

  Inst *new_inst = new Inst;
  new_inst->op = Op::VALUE;
  new_inst->u.value.value = value;
  new_inst->bitsize = bitsize;

  // We must insert the value instructions early in the basic block as they
  // may be used by e.g. memory initialization in the entry block.
  // We also want a consistent order (to make functional identical code
  // identical in the IR, even if there are minor differences in the order
  // that the constants has been created. This happens, for example, in the
  // GCC cpp pass when moving constants into phi-nodes.)
  // So we insert the value instructions at the top of the entry block,
  // sorted after their values.
  auto [it2, _] = values.insert({std::pair(bitsize, value), new_inst});
  if (values.size() == 1)
    {
      if (bbs[0]->first_inst)
	new_inst->insert_before(bbs[0]->first_inst);
      else
	bbs[0]->insert_last(new_inst);
    }
  else if (it2 == values.begin())
    {
      Inst *first_value_inst = (++it2)->second;
      new_inst->insert_before(first_value_inst);
    }
  else
    {
      Inst *prev_value_inst = (--it2)->second;
      new_inst->insert_after(prev_value_inst);
    }
  return new_inst;
}

void Function::rename(const std::string& str)
{
  name = str;
}

void Function::canonicalize()
{
  reset_ir_id();

  for (Basic_block *bb : bbs)
    {
      /*
       * We need to sort the phi node arguments and BB predecessors in a
       * consistent way to prevent random differences between src and tgt
       * when we create ITEs for phi nodes. We basically sort them in
       * reverse post order, but we treat basic blocks having exactly one
       * predecessor and successor as having the same position as its
       * successor.
       *
       * This approach generates the same ITEs for cases such as the GCC
       * pre pass eliminating BB3 as shown below (where BB2 in the original
       * came before BB3 in RPO)
       *
       *        BB1                          BB1
       *        / \                           |\
       *       /   \                          | \
       *     BB3   BB2          ===>          | BB2
       *       \   / \                        | / \
       *        \ /   \                       |/   \
       *        BB4   ...                    BB4   ...
       *         |                            |
       *        ...                          ...
       *
       * which would differ if we used the reverse post order directly.
       */
      for (auto phi : bb->phis)
	{
	  std::sort(phi->phi_args.begin(), phi->phi_args.end(),
		    [](const Phi_arg &a, const Phi_arg &b) {
		      Basic_block *a_bb = a.bb;
		      while (a_bb->preds.size() == 1 &&
			     a_bb->succs.size() == 1)
			{
			  a_bb = a_bb->preds[0];
			}
		      Basic_block *b_bb = b.bb;
		      while (b_bb->preds.size() == 1 &&
			     b_bb->succs.size() == 1)
			{
			  b_bb = b_bb->preds[0];
			}
		      if (a_bb == b_bb)
			return a.bb->id < b.bb->id;
		      return a_bb->id < b_bb->id;
		    });
	}
      std::sort(bb->preds.begin(), bb->preds.end(),
		[](const Basic_block *a, const Basic_block *b) {
		  const Basic_block *a_bb = a;
		  while (a_bb->preds.size() == 1 &&
			 a_bb->succs.size() == 1)
		    {
		      a_bb = a_bb->preds[0];
		    }
		  const Basic_block *b_bb = b;
		  while (b_bb->preds.size() == 1 &&
			 b_bb->succs.size() == 1)
		    {
		      b_bb = b_bb->preds[0];
		    }
		  if (a_bb == b_bb)
		    return a->id < b->id;
		  return a_bb->id < b_bb->id;
		});
    }
}

void Function::reset_ir_id()
{
  int bb_nbr = 0;
  uint32_t inst_nbr = 0;
  for (Basic_block *bb : bbs)
    {
      bb->id = bb_nbr++;
      for (Inst *phi : bb->phis)
	{
	  phi->id = inst_nbr++;
	}
      for (Inst *inst = bb->first_inst; inst; inst = inst->next)
	{
	  inst->id = inst_nbr++;
	}
    }
}

Function *Function::clone(Module *dest_module)
{
  Function *tgt_func = dest_module->build_function(name);

  std::map<Basic_block*, Basic_block*> src2tgt_bb;
  std::map<Inst *, Inst *> src2tgt_inst;
  for (auto src_bb : bbs)
    {
      src2tgt_bb[src_bb] = tgt_func->build_bb();
    }

  for (auto src_bb : bbs)
    {
      Basic_block *tgt_bb = src2tgt_bb.at(src_bb);
      for (auto src_phi : src_bb->phis)
	{
	  src2tgt_inst[src_phi] = tgt_bb->build_phi_inst(src_phi->bitsize);
	}
      for (Inst *src_inst = src_bb->first_inst;
	   src_inst;
	   src_inst = src_inst->next)
	{
	  Inst *tgt_inst = nullptr;
	  Inst_class iclass = src_inst->iclass();
	  switch (iclass)
	    {
	    case Inst_class::mem_nullary:
	      tgt_inst = tgt_bb->build_inst(src_inst->op);
	      break;
	    case Inst_class::iunary:
	    case Inst_class::funary:
	    case Inst_class::ls_unary:
	    case Inst_class::reg_unary:
	    case Inst_class::solver_unary:
	      {
		Inst *arg = src2tgt_inst.at(src_inst->args[0]);
		tgt_inst = tgt_bb->build_inst(src_inst->op, arg);
	      }
	      break;
	    case Inst_class::icomparison:
	    case Inst_class::fcomparison:
	    case Inst_class::ibinary:
	    case Inst_class::fbinary:
	    case Inst_class::conv:
	    case Inst_class::ls_binary:
	    case Inst_class::reg_binary:
	    case Inst_class::solver_binary:
	      {
		Inst *arg1 = src2tgt_inst.at(src_inst->args[0]);
		Inst *arg2 = src2tgt_inst.at(src_inst->args[1]);
		tgt_inst = tgt_bb->build_inst(src_inst->op, arg1, arg2);
	      }
	      break;
	    case Inst_class::ternary:
	    case Inst_class::ls_ternary:
	    case Inst_class::mem_ternary:
	    case Inst_class::solver_ternary:
	      {
		Inst *arg1 = src2tgt_inst.at(src_inst->args[0]);
		Inst *arg2 = src2tgt_inst.at(src_inst->args[1]);
		Inst *arg3 = src2tgt_inst.at(src_inst->args[2]);
		tgt_inst =
		  tgt_bb->build_inst(src_inst->op, arg1, arg2, arg3);
	      }
	      break;
	    case Inst_class::special:
	      if (src_inst->op == Op::BR)
		{
		  if (src_inst->nof_args == 0)
		    {
		      Basic_block *dest_bb =
			src2tgt_bb.at(src_inst->u.br1.dest_bb);
		      tgt_inst = tgt_bb->build_br_inst(dest_bb);
		    }
		  else
		    {
		      assert(src_inst->nof_args == 1);
		      Inst *arg1 =
			src2tgt_inst.at(src_inst->args[0]);
		      Basic_block *true_bb =
			src2tgt_bb.at(src_inst->u.br3.true_bb);
		      Basic_block *false_bb =
			src2tgt_bb.at(src_inst->u.br3.false_bb);
		      tgt_inst = tgt_bb->build_br_inst(arg1, true_bb, false_bb);
		    }
		}
	      else if (src_inst->op == Op::RET)
		{
		  if (src_inst->nof_args == 0)
		    {
		      tgt_inst = tgt_bb->build_ret_inst();
		    }
		  else if (src_inst->nof_args == 1)
		    {
		      Inst *arg1 =
			src2tgt_inst.at(src_inst->args[0]);
		      tgt_inst = tgt_bb->build_ret_inst(arg1);
		    }
		  else
		    {
		      assert(src_inst->nof_args == 2);
		      Inst *arg1 =
			src2tgt_inst.at(src_inst->args[0]);
		      Inst *arg2 =
			src2tgt_inst.at(src_inst->args[1]);
		      tgt_inst = tgt_bb->build_ret_inst(arg1, arg2);
		    }
		}
	      else if (src_inst->op == Op::VALUE)
		{
		  tgt_inst = tgt_bb->value_inst(src_inst->value(),
						src_inst->bitsize);
		}
	      else
		throw Not_implemented("Function::clone: "s + src_inst->name());
	      break;
	    default:
	      throw Not_implemented("Function::clone: "s + src_inst->name());
	    }
	  assert(tgt_inst);
	  src2tgt_inst[src_inst] = tgt_inst;
	}
    }
  for (auto src_bb : bbs)
    {
      for (auto src_phi : src_bb->phis)
	{
	  Inst *tgt_phi = src2tgt_inst.at(src_phi);
	  for (auto [src_arg_inst, src_arg_bb] : src_phi->phi_args)
	    {
	      Inst *arg_inst = src2tgt_inst.at(src_arg_inst);
	      Basic_block *arg_bb = src2tgt_bb.at(src_arg_bb);
	      tgt_phi->add_phi_arg(arg_inst, arg_bb);
	    }
	}
    }
  reverse_post_order(tgt_func);

  return tgt_func;
}

void Function::print(FILE *stream) const
{
  fprintf(stream, "\nfunction %s\n", name.c_str());
  for (auto bb : bbs)
    {
      if (bb != bbs[0])
	fprintf(stream, "\n");
      bb->print(stream);
    }
}

Function *Module::build_function(const std::string& name)
{
  Function *func = new Function;
  func->module = this;
  func->name = name;
  functions.push_back(func);
  return func;
}

void Module::canonicalize()
{
  for (auto func : functions)
    {
      func->canonicalize();
    }
}

Module *Module::clone()
{
  Module *m = create_module(ptr_bits, ptr_id_bits, ptr_offset_bits);
  for (auto func : functions)
    func->clone(m);
  return m;
}

void Module::print(FILE *stream) const
{
  fprintf(stream, "config %" PRIu32 ", %" PRIu32 ", %" PRIu32 "\n",
	  ptr_bits, ptr_id_bits, ptr_offset_bits);

  for (auto func : functions)
    func->print(stream);
}

Module *create_module(uint32_t ptr_bits, uint32_t ptr_id_bits, uint32_t ptr_offset_bits)
{
  assert(ptr_bits == 16 || ptr_bits == 32 || ptr_bits == 64);
  assert(ptr_bits >= ptr_id_bits + ptr_offset_bits);
  Module *module = new Module;
  module->ptr_bits = ptr_bits;
  module->ptr_offset_bits = ptr_offset_bits;
  module->ptr_offset_low = 0;
  module->ptr_offset_high = ptr_offset_bits - 1;
  module->ptr_id_bits = ptr_id_bits;
  module->ptr_id_low = ptr_offset_bits;
  module->ptr_id_high = ptr_offset_bits + ptr_id_bits - 1;
  return module;
}

void destroy_module(struct Module *module)
{
  while (!module->functions.empty())
    destroy_function(module->functions[0]);
  delete module;
}

void destroy_function(Function *func)
{
  // The functions destroying basic blocks and instructions does extra work
  // preserving function invariants (as they are meant to be used by
  // optimization passes etc.). This is not needed when destroying the
  // function, so we'll just delete them.
  for (Basic_block *bb : func->bbs)
    {
      for (Inst *inst : bb->phis)
	delete inst;
      Inst *next_inst = bb->first_inst;
      while (next_inst)
	{
	  Inst *inst = next_inst;
	  next_inst = next_inst->next;
	  delete inst;
	}
      delete bb;
    }

  // Unlink func from module.
  Module *module = func->module;
  auto I = std::find(module->functions.begin(), module->functions.end(), func);
  if (I != module->functions.end())
    module->functions.erase(I);

  delete func;
}

void destroy_basic_block(Basic_block *bb)
{
  // Pointers from predecessor will be dangling after we destroy the BB.
  assert(bb->preds.empty());

  for (Inst *phi : bb->phis)
    {
      phi->remove_phi_args();
    }
  for (Inst *inst = bb->last_inst; inst;)
    {
      Inst *curr_inst = inst;
      inst = inst->prev;
      destroy_instruction(curr_inst);
    }
  while (!bb->phis.empty())
    {
      destroy_instruction(bb->phis.back());
    }
  auto it = std::find(bb->func->bbs.begin(), bb->func->bbs.end(), bb);
  assert(it != bb->func->bbs.end());
  bb->func->bbs.erase(it);
  delete bb;
}

void destroy_instruction(Inst *inst)
{
  assert(inst->used_by.empty());

  if (inst->bb)
    {
      if (inst->op == Op::VALUE)
	{
	  auto key = std::pair(inst->bitsize, inst->value());
	  assert(inst->bb->func->values.contains(key));
	  inst->bb->func->values.erase(key);

	  if (inst->bb->func->last_value_inst == inst)
	    {
	      Inst *prev = inst->prev;
	      if (prev && prev->op == Op::VALUE)
		inst->bb->func->last_value_inst = prev;
	      else
		inst->bb->func->last_value_inst = nullptr;
	    }
	}

      Basic_block *bb = inst->bb;
      if (inst->op == Op::PHI)
	{
	  for (auto [arg_inst, arg_bb] : inst->phi_args)
	    {
	      arg_inst->used_by.erase(inst);
	    }

	  auto it = std::find(bb->phis.begin(), bb->phis.end(), inst);
	  assert(it != bb->phis.end());
	  bb->phis.erase(it);
	}
      else
	{
	  if (inst->op == Op::BR)
	    {
	      for (auto succ : inst->bb->succs)
		{
		  auto it = find(succ->preds.begin(), succ->preds.end(), inst->bb);
		  assert(it != succ->preds.end());
		  succ->preds.erase(it);
		}
	      inst->bb->succs.clear();
	      // Note: phi instructions in the successor basic blocks
	      // will have arguments for the now removed branches.
	      // But we cannot fix that here as the reason the branch is
	      // removed may be because the caller want to add a new, similar,
	      // branch, and removing data from the phi nodes will make
	      // that work harder. So it is up to the caller to update
	      // phi nodes as appropriate.
	    }

	  for (uint64_t i = 0; i < inst->nof_args; i++)
	    {
	      inst->args[i]->used_by.erase(inst);
	    }

	  if (inst == bb->first_inst)
	    bb->first_inst = inst->next;
	  if (inst == bb->last_inst)
	    bb->last_inst = inst->prev;
	  if (inst->prev)
	    inst->prev->next = inst->next;
	  if (inst->next)
	    inst->next->prev = inst->prev;
	}
    }
  delete inst;
}

uint64_t get_time()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

} // end namespace smtgcc
