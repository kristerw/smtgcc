// Functionality for performing the check. The code converts the IR to a
// simplified form and sends it to the SMT solver.
// The simplification consists of:
//  * Removing all control flow
//    - Including changing LOAD/STORE/etc. to ARRAY_LOAD/ARRAY_STORE/etc.
//      that takes the memory state as an input parameter.
//  * For check_refine, it combines the two functions into one (where
//    the values to check are inticated by special instructions, such
//    as SRC_RETVAL and TGT_RETVAL.
//  * Running CSE. This helps the SMT solver, as many GCC optimizations
//    only makes minor changes to the IR, so most of the code is identical
//    for src and tgt.
#include <cassert>
#include <cinttypes>

#include "smtgcc.h"

using namespace std::string_literals;
namespace smtgcc {
namespace {

enum class Function_role {
  src, tgt
};

// Convert the IR of a function to a form having essentially 1-1
// correspondence to what the SMT solver is using. In particular
// eliminate control flow, and change the memory operations to use
// arrays.
class Converter {
  // Maps basic blocks to an expression telling if it is executed.
  std::map<Basic_block *, Instruction *> bb2cond;

  // Maps basic blocks to an expression determining if it contain UB.
  std::map<Basic_block *, Instruction *> bb2ub;

  // Maps basic blocks to an expression determining if it contain an
  // assertion failure.
  std::map<Basic_block *, Instruction *> bb2not_assert;

  // Maps basic blocks to the memory state at the end of the basic block.
  std::map<Basic_block *, Instruction *> bb2memory;
  std::map<Basic_block *, Instruction *> bb2memory_size;
  std::map<Basic_block *, Instruction *> bb2memory_flag;
  std::map<Basic_block *, Instruction *> bb2memory_undef;

  // List of the mem_id for the constant memory blocks.
  std::vector<Instruction *> const_ids;

  // Table for mapping original instructions to the corresponding new
  // instruction in destination function.
  std::map<Instruction*, Instruction*> translate;

  // The memory state before src and tgt. This is used to have identical
  // start state for both src and tgt.
  Instruction *memory;
  Instruction *memory_flag;
  Instruction *memory_size;
  Instruction *memory_undef;

  Basic_block *dest_bb;

  Instruction *ite(Instruction *c, Instruction *a, Instruction *b);
  Instruction *bool_or(Instruction *a, Instruction *b);
  Instruction *bool_and(Instruction *a, Instruction *b);
  void add_ub(Basic_block *bb, Instruction *cond);
  void add_assert(Basic_block *bb, Instruction *cond);
  Instruction *generate_ub(Function *func);
  Instruction *generate_assert(Function *func);
  Instruction *get_full_edge_cond(Basic_block *src, Basic_block *dest);
  void build_mem_state(Basic_block *bb, std::map<Basic_block*, Instruction*>& map);
  void generate_bb2cond(Basic_block *bb);
  void cse();
  void convert(Basic_block *bb, Instruction *inst, Function_role role);

public:
  Converter(Module *m);
  ~Converter()
  {
    if (module)
      destroy_module(module);
  }

  void convert_function(Function *func, Function_role role);
  void finalize();

  Module *module = nullptr;
  Function *dest_func = nullptr;
};

struct Cse_key
{
  Op opcode;
  Instruction *arg1 = nullptr;
  Instruction *arg2 = nullptr;
  Instruction *arg3 = nullptr;

  friend bool operator<(const Cse_key& lhs, const Cse_key& rhs)
  {
    if (lhs.opcode < rhs.opcode) return true;
    if (lhs.opcode > rhs.opcode) return false;

    if (lhs.arg1 < rhs.arg1) return true;
    if (lhs.arg1 > rhs.arg1) return false;

    if (lhs.arg2 < rhs.arg2) return true;
    if (lhs.arg2 > rhs.arg2) return false;

    return lhs.arg3 < rhs.arg3;
 }
};

void Converter::cse()
{
  std::map<Cse_key,Instruction*> x;
  for (Instruction *inst = dest_bb->first_inst; inst; inst = inst->next)
    {
      if (inst->iclass() == Inst_class::special)
	continue;

      if (inst->has_lhs())
	{
	  Cse_key key;
	  key.opcode = inst->opcode;
	  if (inst->nof_args > 0)
	    key.arg1 = inst->arguments[0];
	  if (inst->nof_args > 1)
	    key.arg2 = inst->arguments[1];
	  if (inst->nof_args > 2)
	    key.arg3 = inst->arguments[2];

	  if (x.contains(key))
	    {
	      inst->replace_all_uses_with(x.at(key));
	    }
	  else if (inst->is_commutative())
	    {
	      assert(inst->nof_args == 2);
	      std::swap(key.arg1, key.arg2);
	      if (x.contains(key))
		inst->replace_all_uses_with(x.at(key));
	      else
		x.insert({key, inst});
	    }
	  else
	    x.insert({key, inst});
	}
    }
}

Converter::Converter(Module *m)
{
  module = create_module(m->ptr_bits, m->ptr_id_bits, m->ptr_offset_bits);
  dest_func = module->build_function("check");
  dest_bb = dest_func->build_bb();

  memory = dest_bb->build_inst(Op::MEM_ARRAY);
  memory_undef = dest_bb->build_inst(Op::MEM_UNDEF_ARRAY);
  memory_flag = dest_bb->build_inst(Op::MEM_FLAG_ARRAY);
  memory_size = dest_bb->build_inst(Op::MEM_SIZE_ARRAY);
}

Instruction *Converter::ite(Instruction *c, Instruction *a, Instruction *b)
{
  if (a == b)
    return a;
  return dest_bb->build_inst(Op::ITE, c, a, b);
}

Instruction *Converter::bool_or(Instruction *a, Instruction *b)
{
  if (a->opcode == Op::VALUE)
    {
      if (a->value())
	return a;
      else
	return b;
    }
  if (b->opcode == Op::VALUE)
    {
      if (b->value())
	return b;
      else
	return a;
    }
  return dest_bb->build_inst(Op::OR, a, b);
}

Instruction *Converter::bool_and(Instruction *a, Instruction *b)
{
  if (a->opcode == Op::VALUE)
    {
      if (a->value())
	return b;
      else
	return a;
    }
  if (b->opcode == Op::VALUE)
    {
      if (b->value())
	return a;
      else
	return b;
    }
  return dest_bb->build_inst(Op::AND, a, b);
}

void Converter::add_ub(Basic_block *bb, Instruction *cond)
{
  if (bb2ub.contains(bb))
    {
      cond = bool_or(bb2ub.at(bb), cond);
      bb2ub.erase(bb);
    }
  bb2ub.insert({bb, cond});
}

void Converter::add_assert(Basic_block *bb, Instruction *cond)
{
  cond = dest_bb->build_inst(Op::NOT, cond);
  if (bb2not_assert.contains(bb))
    {
      cond = bool_or(bb2not_assert.at(bb), cond);
      bb2not_assert.erase(bb);
    }
  bb2not_assert.insert({bb, cond});
}

Instruction *Converter::generate_ub(Function *func)
{
  Instruction *ub = dest_bb->value_inst(0, 1);
  for (auto bb : func->bbs)
    {
      if (!bb2ub.contains(bb))
	continue;

      ub = bool_or(ub, bool_and(bb2ub.at(bb), bb2cond.at(bb)));
    }

  return ub;
}

Instruction *Converter::generate_assert(Function *func)
{
  Instruction *assrt = dest_bb->value_inst(0, 1);
  for (auto bb : func->bbs)
    {
      if (!bb2not_assert.contains(bb))
	continue;

      assrt = bool_or(assrt, bool_and(bb2not_assert.at(bb), bb2cond.at(bb)));
    }

  return assrt;
}

Instruction *Converter::get_full_edge_cond(Basic_block *src, Basic_block *dest)
{
  if (src->succs.size() == 1)
    return bb2cond.at(src);
  assert(src->succs.size() == 2);
  assert(src->last_inst->opcode == Op::BR);
  assert(src->last_inst->nof_args == 1);
  Instruction *cond = translate.at(src->last_inst->arguments[0]);
  if (dest != src->succs[0])
    cond = dest_bb->build_inst(Op::NOT, cond);
  return bool_and(bb2cond.at(src), cond);
}

void Converter::build_mem_state(Basic_block *bb, std::map<Basic_block*, Instruction*>& map)
{
  assert(bb->preds.size() > 0);
  Instruction *inst = map.at(bb->preds[0]);
  for (size_t i = 1; i < bb->preds.size(); i++)
    {
      Basic_block *pred_bb = bb->preds[i];
      inst = ite(bb2cond.at(pred_bb), map.at(pred_bb), inst);
    }
  map.insert({bb, inst});
}

void Converter::generate_bb2cond(Basic_block *bb)
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
      Instruction *cond = dest_bb->value_inst(0, 1);
      for (auto pred_bb : bb->preds)
	{
	  cond = bool_or(cond, get_full_edge_cond(pred_bb, bb));
	}
      bb2cond.insert({bb, cond});
    }
}

void Converter::convert(Basic_block *bb, Instruction *inst, Function_role role)
{
  Instruction *new_inst = nullptr;
  if (inst->opcode == Op::VALUE)
    {
      new_inst = dest_bb->value_inst(inst->value(), inst->bitsize);
    }
  else if (inst->opcode == Op::LOAD)
    {
      Instruction *array = bb2memory.at(bb);
      Instruction *ptr = translate.at(inst->arguments[0]);
      new_inst = dest_bb->build_inst(Op::ARRAY_LOAD, array, ptr);
    }
  else if (inst->opcode == Op::STORE)
    {
      Instruction *array = bb2memory.at(bb);
      Instruction *ptr = translate.at(inst->arguments[0]);
      Instruction *value = translate.at(inst->arguments[1]);
      array = dest_bb->build_inst(Op::ARRAY_STORE, array, ptr, value);
      bb2memory[bb] = array;
      return;
    }
  else if (inst->opcode == Op::UB)
    {
      add_ub(bb, translate.at(inst->arguments[0]));
      return;
    }
  else if (inst->opcode == Op::ASSERT)
    {
      add_assert(bb, translate.at(inst->arguments[0]));
      return;
    }
  else if (inst->opcode == Op::SET_MEM_FLAG)
    {
      Instruction *array = bb2memory_flag.at(bb);
      Instruction *ptr = translate.at(inst->arguments[0]);
      Instruction *value = translate.at(inst->arguments[1]);
      array = dest_bb->build_inst(Op::ARRAY_SET_FLAG, array, ptr, value);
      bb2memory_flag[bb] = array;
      return;
    }
  else if (inst->opcode == Op::SET_MEM_UNDEF)
    {
      Instruction *array = bb2memory_undef.at(bb);
      Instruction *ptr = translate.at(inst->arguments[0]);
      Instruction *value = translate.at(inst->arguments[1]);
      array = dest_bb->build_inst(Op::ARRAY_SET_UNDEF, array, ptr, value);
      bb2memory_undef[bb] = array;
      return;
    }
  else if (inst->opcode == Op::FREE)
    {
      Instruction *array = bb2memory_size.at(bb);
      Instruction *arg1 = translate.at(inst->arguments[0]);
      Instruction *zero = dest_bb->value_inst(0, module->ptr_offset_bits);
      array = dest_bb->build_inst(Op::ARRAY_SET_SIZE, array, arg1, zero);
      bb2memory_size[bb] = array;
      return;
    }
   else if (inst->opcode == Op::GET_MEM_UNDEF)
     {
      Instruction *array = bb2memory_undef.at(bb);
      Instruction *arg1 = translate.at(inst->arguments[0]);
      new_inst = dest_bb->build_inst(Op::ARRAY_GET_UNDEF, array, arg1);
     }
   else if (inst->opcode == Op::GET_MEM_FLAG)
     {
      Instruction *array = bb2memory_flag.at(bb);
      Instruction *arg1 = translate.at(inst->arguments[0]);
      new_inst = dest_bb->build_inst(Op::ARRAY_GET_FLAG, array, arg1);
     }
   else if (inst->opcode == Op::GET_MEM_SIZE)
    {
      Instruction *array = bb2memory_size.at(bb);
      Instruction *arg1 = translate.at(inst->arguments[0]);
      new_inst = dest_bb->build_inst(Op::ARRAY_GET_SIZE, array, arg1);
    }
  else if (inst->opcode == Op::IS_CONST_MEM)
    {
      Instruction *arg1 = translate.at(inst->arguments[0]);
      Instruction *is_const = dest_bb->value_inst(0, 1);
      for (Instruction *id : const_ids)
	{
	  is_const = bool_or(is_const, dest_bb->build_inst(Op::EQ, arg1, id));
	}
      new_inst = is_const;
    }
  else if (inst->opcode == Op::MEMORY)
    {
      // All uses of MEMORY should have been changed to constants.
      assert(inst->used_by.empty());

      uint32_t ptr_bits = module->ptr_bits;
      uint32_t ptr_offset_bits = module->ptr_offset_bits;
      uint32_t ptr_id_bits = module->ptr_id_bits;
      uint64_t id = inst->arguments[0]->value();
      uint64_t ptr_val = id << module->ptr_id_low;
      uint64_t size_val = inst->arguments[1]->value();
      Instruction *mem_id = dest_bb->value_inst(id, ptr_id_bits);
      Instruction *size = dest_bb->value_inst(size_val, ptr_offset_bits);
      Instruction *size_array = bb2memory_size.at(bb);
      size_array =
	dest_bb->build_inst(Op::ARRAY_SET_SIZE, size_array, mem_id, size);
      bb2memory_size[bb] = size_array;

      uint32_t flags = inst->arguments[2]->value();
      if (flags & MEM_CONST)
	const_ids.push_back(translate.at(inst->arguments[0]));

      if (flags & MEM_UNINIT)
	{
	  Instruction *undef_array = bb2memory_undef.at(bb);
	  Instruction *byte = dest_bb->value_inst(255, 8);
	  for (uint64_t i = 0; i < size_val; i++)
	    {
	      Instruction *ptr = dest_bb->value_inst(ptr_val + i, ptr_bits);
	      undef_array = dest_bb->build_inst(Op::ARRAY_SET_UNDEF,
						undef_array, ptr, byte);
	    }
	  bb2memory_undef[bb] = undef_array;
	}

      return;
    }
  else if (inst->opcode == Op::BR)
    {
      return;
    }
  else if (inst->opcode == Op::RET)
    {
      if (inst->nof_args > 0)
	{
	  Instruction *arg1 = translate.at(inst->arguments[0]);
	  Instruction *arg2;
	  if (inst->nof_args > 1)
	    arg2 = translate.at(inst->arguments[1]);
	  else
	    arg2 = dest_bb->value_inst(0, inst->arguments[0]->bitsize);
	  Op op = role == Function_role::src ? Op::SRC_RETVAL : Op::TGT_RETVAL;
	  dest_bb->build_inst(op, arg1, arg2);
	}
      return;
    }
  else
    {
      assert(inst->has_lhs());
      Inst_class iclass = inst->iclass();
      switch (iclass)
	{
	case Inst_class::iunary:
	case Inst_class::funary:
	  {
	    Instruction *arg = translate.at(inst->arguments[0]);
	    new_inst = dest_bb->build_inst(inst->opcode, arg);
	  }
	  break;
	case Inst_class::icomparison:
	case Inst_class::fcomparison:
	case Inst_class::ibinary:
	case Inst_class::fbinary:
	case Inst_class::conv:
	  {
	    Instruction *arg1 = translate.at(inst->arguments[0]);
	    Instruction *arg2 = translate.at(inst->arguments[1]);
	    new_inst = dest_bb->build_inst(inst->opcode, arg1, arg2);
	  }
	  break;
	case Inst_class::ternary:
	  {
	    Instruction *arg1 = translate.at(inst->arguments[0]);
	    Instruction *arg2 = translate.at(inst->arguments[1]);
	    Instruction *arg3 = translate.at(inst->arguments[2]);
	    new_inst = dest_bb->build_inst(inst->opcode, arg1, arg2, arg3);
	  }
	  break;
	default:
	  throw Not_implemented("Converter::duplicate: "s + inst->name());
	}
    }
  assert(new_inst);
  translate.insert({inst, new_inst});
}

void Converter::convert_function(Function *func, Function_role role)
{
  for (auto bb : func->bbs)
    {
      if (bb == func->bbs[0])
	{
	  bb2cond.insert({bb, dest_bb->value_inst(1, 1)});
	  bb2memory.insert({bb, memory});
	  bb2memory_size.insert({bb, memory_size});
	  bb2memory_flag.insert({bb, memory_flag});
	  bb2memory_undef.insert({bb, memory_undef});
	}
      else
	{
	  generate_bb2cond(bb);
	  build_mem_state(bb, bb2memory);
	  build_mem_state(bb, bb2memory_size);
	  build_mem_state(bb, bb2memory_flag);
	  build_mem_state(bb, bb2memory_undef);
	}

      for (auto phi : bb->phis)
	{
	  Instruction *phi_inst = translate.at(phi->phi_args[0].inst);
	  assert(phi->phi_args.size() == bb->preds.size());
	  for (unsigned i = 1; i < phi->phi_args.size(); i++)
	    {
	      Basic_block *pred_bb = phi->phi_args[i].bb;
	      Instruction *cond = get_full_edge_cond(pred_bb, bb);
	      Instruction *inst = translate.at(phi->phi_args[i].inst);
	      phi_inst = ite(cond, inst, phi_inst);
	    }
	  translate.insert({phi, phi_inst});
	}

      for (Instruction *inst = bb->first_inst; inst; inst = inst->next)
	{
	  convert(bb, inst, role);
	}
    }

  Op ub_op = role == Function_role::src ? Op::SRC_UB : Op::TGT_UB;
  dest_bb->build_inst(ub_op, generate_ub(func));

  Op assert_op = role == Function_role::src ? Op::SRC_ASSERT : Op::TGT_ASSERT;
  dest_bb->build_inst(assert_op, generate_assert(func));

  Basic_block *exit_block = func->bbs.back();
  Op mem1_op = role == Function_role::src ? Op::SRC_MEM1 : Op::TGT_MEM1;
  Op mem2_op = role == Function_role::src ? Op::SRC_MEM2 : Op::TGT_MEM2;
  dest_bb->build_inst(mem1_op, bb2memory.at(exit_block),
		      bb2memory_size.at(exit_block));
  dest_bb->build_inst(mem2_op, bb2memory_flag.at(exit_block),
		      bb2memory_undef.at(exit_block));

  // Clear the arrays. This is needed for check_refine to get a clean slate
  // when converting the second function. But it also reduces memory usage
  // for the other use cases.
  bb2cond.clear();
  bb2ub.clear();
  bb2not_assert.clear();
  bb2memory.clear();
  bb2memory_size.clear();
  bb2memory_flag.clear();
  bb2memory_undef.clear();
  const_ids.clear();
  translate.clear();
}

void Converter::finalize()
{
  dest_bb->build_inst(Op::RET);

  cse();
  dead_code_elimination(dest_func);
  dest_func->canonicalize();

  if (config.verbose > 1)
    module->print(stderr);
}

} // end anonymous namespace

Solver_result check_refine(Module *module)
{
  struct VStats {
    SStats cvc5;
    SStats z3;
  } stats;

  assert(module->functions.size() == 2);
  Function *src = module->functions[0];
  Function *tgt = module->functions[1];
  if (src->name != "src")
    std::swap(src, tgt);
  assert(src->name == "src" && tgt->name == "tgt");

  if (identical(src, tgt))
    return {};

  if (config.verbose > 1)
    module->print(stderr);

  Converter converter(module);
  converter.convert_function(src, Function_role::src);
  converter.convert_function(tgt, Function_role::tgt);
  converter.finalize();

  Solver_result result = {Result_status::correct, {}};
#if 0
  auto [stats_cvc5, result_cvc5] = check_refine_cvc5(converter.dest_func);
  stats.cvc5 = stats_cvc5;
  if (result_cvc5.status != Result_status::correct)
    result = result_cvc5;
#endif
#if 1
  auto [stats_z3, result_z3] = check_refine_z3(converter.dest_func);
  stats.z3 = stats_z3;
  if (result_z3.status != Result_status::correct)
    result = result_z3;
#endif

  if (config.verbose > 0)
    {
      if (!stats.cvc5.skipped || !stats.z3.skipped)
	{
	  fprintf(stderr, "SMTGCC: time: ");
	  for (int i = 0; i < 3; i++)
	    {
	      fprintf(stderr, "%s%" PRIu64, i ? "," : "", stats.cvc5.time[i]);
	    }
	  for (int i = 0; i < 3; i++)
	    {
	      fprintf(stderr, "%s%" PRIu64, ",", stats.z3.time[i]);
	    }
	  fprintf(stderr, "\n");
	}
    }

  return result;
}

Solver_result check_ub(Function *func)
{
  struct VStats {
    SStats cvc5;
    SStats z3;
  } stats;

  if (config.verbose > 1)
    func->print(stderr);

  Converter converter(func->module);
  converter.convert_function(func, Function_role::src);
  converter.finalize();

  Solver_result result = {Result_status::correct, {}};
#if 0
  auto [stats_cvc5, result_cvc5] = check_ub_cvc5(converter.dest_func);
  stats.cvc5 = stats_cvc5;
  if (result_cvc5.status != Result_status::correct)
    result = result_cvc5;
#endif
#if 1
  auto [stats_z3, result_z3] = check_ub_z3(converter.dest_func);
  stats.z3 = stats_z3;
  if (result_z3.status != Result_status::correct)
    result = result_z3;
#endif

  if (config.verbose > 0)
    {
      if (!stats.cvc5.skipped || !stats.z3.skipped)
	{
	  fprintf(stderr, "SMTGCC: time: ");
	  for (int i = 0; i < 3; i++)
	    {
	      fprintf(stderr, "%s%" PRIu64, i ? "," : "", stats.cvc5.time[i]);
	    }
	  for (int i = 0; i < 3; i++)
	    {
	      fprintf(stderr, "%s%" PRIu64, ",", stats.z3.time[i]);
	    }
	  fprintf(stderr, "\n");
	}
    }

  return result;
}

Solver_result check_assert(Function *func)
{
  struct VStats {
    SStats cvc5;
    SStats z3;
  } stats;

  if (config.verbose > 1)
    func->print(stderr);

  Converter converter(func->module);
  converter.convert_function(func, Function_role::src);
  converter.finalize();

  Solver_result result = {Result_status::correct, {}};
#if 0
  auto [stats_cvc5, result_cvc5] = check_assert_cvc5(converter.dest_func);
  stats.cvc5 = stats_cvc5;
  if (result_cvc5.status != Result_status::correct)
    result = result_cvc5;
#endif
#if 1
  auto [stats_z3, result_z3] = check_assert_z3(converter.dest_func);
  stats.z3 = stats_z3;
  if (result_z3.status != Result_status::correct)
    result = result_z3;
#endif

  if (config.verbose > 0)
    {
      if (!stats.cvc5.skipped || !stats.z3.skipped)
	{
	  fprintf(stderr, "SMTGCC: time: ");
	  for (int i = 0; i < 3; i++)
	    {
	      fprintf(stderr, "%s%" PRIu64, i ? "," : "", stats.cvc5.time[i]);
	    }
	  for (int i = 0; i < 3; i++)
	    {
	      fprintf(stderr, "%s%" PRIu64, ",", stats.z3.time[i]);
	    }
	  fprintf(stderr, "\n");
	}
    }

  return result;
}

} // end namespace smtgcc
