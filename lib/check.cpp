// Functionality for performing the check. The code converts the IR to a
// simplified form and sends it to the SMT solver.
// The simplification consists of:
//  * Removing all control flow
//    - Including changing LOAD/STORE/etc. to ARRAY_LOAD/ARRAY_STORE/etc.
//      that takes the memory state as an input parameter.
//  * For check_refine, it combines the two functions into one (where
//    the values to check are indicated by special instructions, such
//    as SRC_RETVAL and TGT_RETVAL.
//  * Running CSE. This helps the SMT solver, as many GCC optimizations
//    only makes minor changes to the IR, so most of the code is identical
//    for src and tgt.
#include <algorithm>
#include <cassert>
#include <cinttypes>

#include "smtgcc.h"

using namespace std::string_literals;
namespace smtgcc {
namespace {

struct Cse_key
{
  Op opcode;
  Instruction *arg1 = nullptr;
  Instruction *arg2 = nullptr;
  Instruction *arg3 = nullptr;

  Cse_key(Op opcode)
    : opcode{opcode}
  {}
  Cse_key(Op opcode, Instruction *arg1)
    : opcode{opcode}
    , arg1{arg1}
  {}
  Cse_key(Op opcode, Instruction *arg1, Instruction *arg2)
    : opcode{opcode}
    , arg1{arg1}
    , arg2{arg2}
  {}
  Cse_key(Op opcode, Instruction *arg1, Instruction *arg2, Instruction *arg3)
    : opcode{opcode}
    , arg1{arg1}
    , arg2{arg2}
    , arg3{arg3}
  {}

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

  // Maps basic blocks to the expressions determining if it contain UB.
  std::map<Basic_block *, std::vector<Instruction *>> bb2ub;

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

  std::map<Cse_key, Instruction*> key2inst;

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
  void convert(Basic_block *bb, Instruction *inst, Function_role role);

  Instruction *get_inst(const Cse_key& key);
  Instruction *value_inst(unsigned __int128 value, uint32_t bitsize);
  Instruction *build_inst(Op opcode);
  Instruction *build_inst(Op opcode, Instruction *arg);
  Instruction *build_inst(Op opcode, Instruction *arg1, Instruction *arg2);
  Instruction *build_inst(Op opcode, Instruction *arg1, Instruction *arg2,
			  Instruction *arg3);

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

Instruction *Converter::get_inst(const Cse_key& key)
{
  auto I = key2inst.find(key);
  if (I != key2inst.end())
    return I->second;

  if (inst_info[(int)key.opcode].is_commutative)
    {
      Cse_key tmp_key = key;
      std::swap(tmp_key.arg1, tmp_key.arg2);
      I = key2inst.find(tmp_key);
      if (I != key2inst.end())
	return I->second;
    }
  else if (inst_info[(int)key.opcode].iclass == Inst_class::icomparison
	   || inst_info[(int)key.opcode].iclass == Inst_class::fcomparison)
    {
      Op op;
      switch (key.opcode)
	{
	case Op::FGE:
	  op = Op::FLE;
	  break;
	case Op::FGT:
	  op = Op::FLT;
	  break;
	case Op::FLE:
	  op = Op::FGE;
	  break;
	case Op::FLT:
	  op = Op::FGT;
	  break;
	case Op::SGE:
	  op = Op::SLE;
	  break;
	case Op::SGT:
	  op = Op::SLT;
	  break;
	case Op::SLE:
	  op = Op::SGE;
	  break;
	case Op::SLT:
	  op = Op::SGT;
	  break;
	case Op::UGE:
	  op = Op::ULE;
	  break;
	case Op::UGT:
	  op = Op::ULT;
	  break;
	case Op::ULE:
	  op = Op::UGE;
	  break;
	case Op::ULT:
	  op = Op::UGT;
	  break;
	default:
	  throw Not_implemented("Converter::get_inst: unknown comparison");
	}
      Cse_key tmp_key = key;
      tmp_key.opcode = op;
      std::swap(tmp_key.arg1, tmp_key.arg2);
      I = key2inst.find(tmp_key);
      if (I != key2inst.end())
	return I->second;
    }

  return nullptr;
}

Instruction *Converter::value_inst(unsigned __int128 value, uint32_t bitsize)
{
  return dest_bb->value_inst(value, bitsize);
}

Instruction *Converter::build_inst(Op opcode)
{
  const Cse_key key(opcode);
  Instruction *inst = get_inst(key);
  if (!inst)
    {
      inst = dest_bb->build_inst(opcode);
      key2inst.insert({key, inst});
    }
  return inst;
}

Instruction *Converter::build_inst(Op opcode, Instruction *arg)
{
  const Cse_key key(opcode, arg);
  Instruction *inst = get_inst(key);
  if (!inst)
    {
      inst = dest_bb->build_inst(opcode, arg);
      key2inst.insert({key, inst});
    }
  return inst;
}

Instruction *Converter::build_inst(Op opcode, Instruction *arg1, Instruction *arg2)
{
  const Cse_key key(opcode, arg1, arg2);
  Instruction *inst = get_inst(key);
  if (!inst)
    {
      inst = dest_bb->build_inst(opcode, arg1, arg2);
      key2inst.insert({key, inst});
    }
  return inst;
}

Instruction *Converter::build_inst(Op opcode, Instruction *arg1, Instruction *arg2, Instruction *arg3)
{
  const Cse_key key(opcode, arg1, arg2, arg3);
  Instruction *inst = get_inst(key);
  if (!inst)
    {
      inst = dest_bb->build_inst(opcode, arg1, arg2, arg3);
      key2inst.insert({key, inst});
    }
  return inst;
}

Converter::Converter(Module *m)
{
  module = create_module(m->ptr_bits, m->ptr_id_bits, m->ptr_offset_bits);
  dest_func = module->build_function("check");
  dest_bb = dest_func->build_bb();

  memory = build_inst(Op::MEM_ARRAY);
  memory_undef = build_inst(Op::MEM_UNDEF_ARRAY);
  memory_flag = build_inst(Op::MEM_FLAG_ARRAY);
  memory_size = build_inst(Op::MEM_SIZE_ARRAY);
}

Instruction *Converter::ite(Instruction *c, Instruction *a, Instruction *b)
{
  if (a == b)
    return a;
  return build_inst(Op::ITE, c, a, b);
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
  return build_inst(Op::OR, a, b);
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
  return build_inst(Op::AND, a, b);
}

void Converter::add_ub(Basic_block *bb, Instruction *cond)
{
  bb2ub[bb].push_back(cond);
}

void Converter::add_assert(Basic_block *bb, Instruction *cond)
{
  cond = build_inst(Op::NOT, cond);
  if (bb2not_assert.contains(bb))
    {
      cond = bool_or(bb2not_assert.at(bb), cond);
      bb2not_assert.erase(bb);
    }
  bb2not_assert.insert({bb, cond});
}

// Generate the UB expression for the function.
//
// We generate the instructions in the order of conditions and UB check
// instruction ID. This makes it more likely that the generated code
// can be CSE:ed between src and tgt.
Instruction *Converter::generate_ub(Function *func)
{
  auto comp = [](Instruction *a, Instruction *b) { return a->id < b->id; };
  std::map<Instruction *, std::vector<Instruction *>, decltype(comp)> bbcond2ub(comp);

  // Merge the UB from the basic blocks having the same condition.
  for (auto bb : func->bbs)
    {
      if (!bb2ub.contains(bb))
	continue;

      const std::vector<Instruction *>& bb_ub = bb2ub[bb];
      std::vector<Instruction *>& cond_ub = bbcond2ub[bb2cond.at(bb)];
      cond_ub.insert(cond_ub.end(), bb_ub.begin(), bb_ub.end());
    }

  // Generate the UB condition for the function.
  Instruction *ub = value_inst(0, 1);
  for (auto& [cond, ub_vec] : bbcond2ub)
    {
      // Eliminate duplicated UB conditions.
      std::sort(ub_vec.begin(), ub_vec.end(), comp);
      ub_vec.erase(std::unique(ub_vec.begin(), ub_vec.end()), ub_vec.end());

      Instruction *bb_ub = value_inst(0, 1);
      for (auto inst : ub_vec)
	{
	  bb_ub = bool_or(bb_ub, inst);
	}
      ub = bool_or(ub, bool_and(cond, bb_ub));
    }

  return ub;
}

Instruction *Converter::generate_assert(Function *func)
{
  Instruction *assrt = value_inst(0, 1);
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
    cond = build_inst(Op::NOT, cond);
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
      Instruction *cond = value_inst(0, 1);
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
      new_inst = value_inst(inst->value(), inst->bitsize);
    }
  else if (inst->opcode == Op::LOAD)
    {
      Instruction *array = bb2memory.at(bb);
      Instruction *ptr = translate.at(inst->arguments[0]);
      new_inst = build_inst(Op::ARRAY_LOAD, array, ptr);
    }
  else if (inst->opcode == Op::STORE)
    {
      Instruction *array = bb2memory.at(bb);
      Instruction *ptr = translate.at(inst->arguments[0]);
      Instruction *value = translate.at(inst->arguments[1]);
      array = build_inst(Op::ARRAY_STORE, array, ptr, value);
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
      array = build_inst(Op::ARRAY_SET_FLAG, array, ptr, value);
      bb2memory_flag[bb] = array;
      return;
    }
  else if (inst->opcode == Op::SET_MEM_UNDEF)
    {
      Instruction *array = bb2memory_undef.at(bb);
      Instruction *ptr = translate.at(inst->arguments[0]);
      Instruction *value = translate.at(inst->arguments[1]);
      array = build_inst(Op::ARRAY_SET_UNDEF, array, ptr, value);
      bb2memory_undef[bb] = array;
      return;
    }
  else if (inst->opcode == Op::FREE)
    {
      Instruction *array = bb2memory_size.at(bb);
      Instruction *arg1 = translate.at(inst->arguments[0]);
      Instruction *zero = value_inst(0, module->ptr_offset_bits);
      array = build_inst(Op::ARRAY_SET_SIZE, array, arg1, zero);
      bb2memory_size[bb] = array;
      return;
    }
   else if (inst->opcode == Op::GET_MEM_UNDEF)
     {
      Instruction *array = bb2memory_undef.at(bb);
      Instruction *arg1 = translate.at(inst->arguments[0]);
      new_inst = build_inst(Op::ARRAY_GET_UNDEF, array, arg1);
     }
   else if (inst->opcode == Op::GET_MEM_FLAG)
     {
      Instruction *array = bb2memory_flag.at(bb);
      Instruction *arg1 = translate.at(inst->arguments[0]);
      new_inst = build_inst(Op::ARRAY_GET_FLAG, array, arg1);
     }
   else if (inst->opcode == Op::GET_MEM_SIZE)
    {
      Instruction *array = bb2memory_size.at(bb);
      Instruction *arg1 = translate.at(inst->arguments[0]);
      new_inst = build_inst(Op::ARRAY_GET_SIZE, array, arg1);
    }
  else if (inst->opcode == Op::IS_CONST_MEM)
    {
      Instruction *arg1 = translate.at(inst->arguments[0]);
      Instruction *is_const = value_inst(0, 1);
      for (Instruction *id : const_ids)
	{
	  is_const = bool_or(is_const, build_inst(Op::EQ, arg1, id));
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
      Instruction *mem_id = value_inst(id, ptr_id_bits);
      Instruction *size = value_inst(size_val, ptr_offset_bits);
      Instruction *size_array = bb2memory_size.at(bb);
      size_array = build_inst(Op::ARRAY_SET_SIZE, size_array, mem_id, size);
      bb2memory_size[bb] = size_array;

      uint32_t flags = inst->arguments[2]->value();
      if (flags & MEM_CONST)
	const_ids.push_back(translate.at(inst->arguments[0]));

      if (flags & MEM_UNINIT)
	{
	  Instruction *undef_array = bb2memory_undef.at(bb);
	  Instruction *byte = value_inst(255, 8);
	  for (uint64_t i = 0; i < size_val; i++)
	    {
	      Instruction *ptr = value_inst(ptr_val + i, ptr_bits);
	      undef_array = build_inst(Op::ARRAY_SET_UNDEF,
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
	    arg2 = value_inst(0, inst->arguments[0]->bitsize);
	  Op op = role == Function_role::src ? Op::SRC_RETVAL : Op::TGT_RETVAL;
	  build_inst(op, arg1, arg2);
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
	    new_inst = build_inst(inst->opcode, arg);
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
	    new_inst = build_inst(inst->opcode, arg1, arg2);
	  }
	  break;
	case Inst_class::ternary:
	  {
	    Instruction *arg1 = translate.at(inst->arguments[0]);
	    Instruction *arg2 = translate.at(inst->arguments[1]);
	    Instruction *arg3 = translate.at(inst->arguments[2]);
	    new_inst = build_inst(inst->opcode, arg1, arg2, arg3);
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
	  bb2cond.insert({bb, value_inst(1, 1)});
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
  build_inst(ub_op, generate_ub(func));

  Op assert_op = role == Function_role::src ? Op::SRC_ASSERT : Op::TGT_ASSERT;
  build_inst(assert_op, generate_assert(func));

  Basic_block *exit_block = func->bbs.back();
  Op mem1_op = role == Function_role::src ? Op::SRC_MEM1 : Op::TGT_MEM1;
  Op mem2_op = role == Function_role::src ? Op::SRC_MEM2 : Op::TGT_MEM2;
  build_inst(mem1_op, bb2memory.at(exit_block),
		      bb2memory_size.at(exit_block));
  build_inst(mem2_op, bb2memory_flag.at(exit_block),
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
  build_inst(Op::RET);

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
