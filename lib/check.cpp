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

struct Inst_comp {
  bool operator()(const Instruction *a, const Instruction *b) const {
    return a->id < b->id;
  }
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

  std::map<Instruction *, std::vector<Instruction *>, Inst_comp> src_bbcond2ub;
  std::map<Instruction *, std::vector<Instruction *>, Inst_comp> tgt_bbcond2ub;
  bool has_tgt = false;

  // The memory state before src and tgt. This is used to have identical
  // start state for both src and tgt.
  Instruction *memory;
  Instruction *memory_flag;
  Instruction *memory_size;
  Instruction *memory_undef;

  Basic_block *dest_bb;

  Instruction *src_memory = nullptr;
  Instruction *src_memory_flag = nullptr;
  Instruction *src_memory_size = nullptr;
  Instruction *src_memory_undef = nullptr;
  Instruction *src_retval = nullptr;
  Instruction *src_retval_undef = nullptr;
  Instruction *src_unique_ub = nullptr;
  Instruction *src_common_ub = nullptr;
  Instruction *tgt_memory = nullptr;
  Instruction *tgt_memory_flag = nullptr;
  Instruction *tgt_memory_size = nullptr;
  Instruction *tgt_memory_undef = nullptr;
  Instruction *tgt_retval = nullptr;
  Instruction *tgt_retval_undef = nullptr;
  Instruction *tgt_unique_ub = nullptr;
  Instruction *tgt_common_ub = nullptr;

  Instruction *ite(Instruction *c, Instruction *a, Instruction *b);
  Instruction *bool_or(Instruction *a, Instruction *b);
  Instruction *bool_and(Instruction *a, Instruction *b);
  Instruction *bool_not(Instruction *a);
  void add_ub(Basic_block *bb, Instruction *cond);
  void add_assert(Basic_block *bb, Instruction *cond);
  std::map<Instruction *, std::vector<Instruction *>, Inst_comp> prepare_ub(Function *func);
  void generate_ub();
  Instruction *generate_assert(Function *func);
  Instruction *get_full_edge_cond(Basic_block *src, Basic_block *dest);
  void build_mem_state(Basic_block *bb, std::map<Basic_block*, Instruction*>& map);
  void generate_bb2cond(Basic_block *bb);
  void convert(Basic_block *bb, Instruction *inst, Function_role role);

  Instruction *get_inst(const Cse_key& key, bool may_add_insts = true);
  Instruction *value_inst(unsigned __int128 value, uint32_t bitsize);
  Instruction *update_key2inst(Instruction *inst);
  Instruction *simplify(Instruction *inst);
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
  bool need_checking();

  Module *module = nullptr;
  Function *dest_func = nullptr;
};

Instruction *Converter::get_inst(const Cse_key& key, bool may_add_insts)
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

  // Check if this is a negation of an existing comparison.
  if (may_add_insts
      && inst_info[(int)key.opcode].iclass == Inst_class::icomparison)
    {
      Op op;
      switch (key.opcode)
	{
	case Op::EQ:
	  op = Op::NE;
	  break;
	case Op::NE:
	  op = Op::EQ;
	  break;
	case Op::SGE:
	  op = Op::SLT;
	  break;
	case Op::SGT:
	  op = Op::SLE;
	  break;
	case Op::SLE:
	  op = Op::SGT;
	  break;
	case Op::SLT:
	  op = Op::SGE;
	  break;
	case Op::UGE:
	  op = Op::ULT;
	  break;
	case Op::UGT:
	  op = Op::ULE;
	  break;
	case Op::ULE:
	  op = Op::UGT;
	  break;
	case Op::ULT:
	  op = Op::UGE;
	  break;
	default:
	  throw Not_implemented("Converter::get_inst: unknown comparison");
	}
      Cse_key tmp_key = key;
      tmp_key.opcode = op;
      Instruction *inst = get_inst(tmp_key, false);
      if (inst)
	return bool_not(inst);
    }

  return nullptr;
}

Instruction *Converter::value_inst(unsigned __int128 value, uint32_t bitsize)
{
  return dest_bb->value_inst(value, bitsize);
}

// Checks if we already have an equivalent instruction. If we do,
// all uses of 'inst' are changed to use the existing instruction.
// If not, 'inst' is added to the CSE table.
Instruction *Converter::update_key2inst(Instruction *inst)
{
  Cse_key key(inst->opcode);
  switch (inst->iclass())
    {
    case Inst_class::nullary:
      break;
    case Inst_class::iunary:
    case Inst_class::funary:
      key.arg1 = inst->arguments[0];
      break;
    case Inst_class::icomparison:
    case Inst_class::fcomparison:
    case Inst_class::ibinary:
    case Inst_class::fbinary:
    case Inst_class::conv:
      key.arg1 = inst->arguments[0];
      key.arg2 = inst->arguments[1];
      break;
    case Inst_class::ternary:
      key.arg1 = inst->arguments[0];
      key.arg2 = inst->arguments[1];
      key.arg3 = inst->arguments[2];
      break;
    default:
      return inst;
    }

  Instruction *existing_inst = get_inst(key);
  if (existing_inst)
    {
      inst->replace_all_uses_with(existing_inst);
      return existing_inst;
    }

  key2inst.insert({key, inst});
  return inst;
}

Instruction *Converter::simplify(Instruction *inst)
{
  // The canonical form for comparisons in the converter is using
  // Op::NOT for the negation of a comparison instead of changing the
  // opcode, so we do not want simplify_inst optimizing this to a plain
  // comparison.
  if (inst->opcode == Op::NOT
      && inst->arguments[0]->iclass() == Inst_class::icomparison)
    return inst;

  Instruction *orig_inst = inst;
  Instruction *prev_inst = inst->prev;
  inst = simplify_inst(inst);
  if (inst == orig_inst)
    return inst;

  // 'simplify_inst' may have added new instructions, so we must add them to
  // the CSE table (or replace them with an already existing instruction).
  // The instructions are always inserted right before the instruction
  // being simplified, so the instructions to examine are between the
  // 'prev_inst' and 'orig_inst'.
  for (Instruction *i = prev_inst->next; i != orig_inst; i = i->next)
    {
      if (i == inst)
	inst = update_key2inst(i);
      else
	update_key2inst(i);
    }

  return inst;
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
      inst = simplify(inst);
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
      inst = simplify(inst);
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
      inst = simplify(inst);
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
  if (c->opcode == Op::VALUE)
    return c->value() ? a : b;
  return build_inst(Op::ITE, c, a, b);
}

Instruction *Converter::bool_or(Instruction *a, Instruction *b)
{
  if (a->opcode == Op::VALUE)
    return a->value() ? a : b;
  if (b->opcode == Op::VALUE)
    return b->value() ? b : a;
  if (a == b)
    return a;
  if (a->opcode == Op::NOT && a->arguments[0] == b)
    return value_inst(1, 1);
  if (b->opcode == Op::NOT && b->arguments[0] == a)
    return value_inst(1, 1);
  return build_inst(Op::OR, a, b);
}

Instruction *Converter::bool_and(Instruction *a, Instruction *b)
{
  if (a->opcode == Op::VALUE)
    return a->value() ? b : a;
  if (b->opcode == Op::VALUE)
    return b->value() ? a : b;
  if (a == b)
    return a;
  if (a->opcode == Op::NOT && a->arguments[0] == b)
    return value_inst(0, 1);
  if (b->opcode == Op::NOT && b->arguments[0] == a)
    return value_inst(0, 1);
  return build_inst(Op::AND, a, b);
}

Instruction *Converter::bool_not(Instruction *a)
{
  if (a->opcode == Op::NOT)
    return a->arguments[0];
  return build_inst(Op::NOT, a);
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

std::map<Instruction *, std::vector<Instruction *>, Inst_comp> Converter::prepare_ub(Function *func)
{
  Inst_comp comp;
  std::map<Instruction *, std::vector<Instruction *>, Inst_comp> bbcond2ub;

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
  for (auto& [cond, ub_vec] : bbcond2ub)
    {
      // Eliminate duplicated UB conditions.
      std::sort(ub_vec.begin(), ub_vec.end(), comp);
      ub_vec.erase(std::unique(ub_vec.begin(), ub_vec.end()), ub_vec.end());
    }

  return bbcond2ub;
}

// Generate the UB expression for the function.
//
// For basic blocks with a common condition in src and tgt, we split
// the list of UB into three sets:
//  * the ones identical in both
//  * the set unique to src
//  * the set unique to tgt
// The common UB is generated first (so it is CSE'd between src and tgt),
// followed by the other two sets.
//
// This greatly assists the SMT solver when checking that tgt does not
// have more UB than src, especially since it only needs to examine the
// UB that are unique to tgt.
void Converter::generate_ub()
{
  std::vector<Instruction *> src_cond;
  std::vector<Instruction *> tgt_cond;

  for (auto [cond, _] : src_bbcond2ub)
    {
      src_cond.push_back(cond);
    }
  for (auto [cond, _] : tgt_bbcond2ub)
    {
      tgt_cond.push_back(cond);
    }

  Inst_comp comp;
  std::vector<Instruction *> cond_common;
  std::vector<Instruction *> cond_src_unique;
  std::vector<Instruction *> cond_tgt_unique;
  std::set_intersection(src_cond.begin(), src_cond.end(),
			tgt_cond.begin(), tgt_cond.end(),
			std::back_inserter(cond_common), comp);
  std::set_difference(src_cond.begin(), src_cond.end(),
		      cond_common.begin(), cond_common.end(),
		      std::back_inserter(cond_src_unique), comp);
  std::set_difference(tgt_cond.begin(), tgt_cond.end(),
		      cond_common.begin(), cond_common.end(),
		      std::back_inserter(cond_tgt_unique), comp);

  Instruction *src_ub = value_inst(0, 1);
  Instruction *tgt_ub = value_inst(0, 1);
  Instruction *common_ub = value_inst(0, 1);
  for (auto cond : cond_common)
    {
      std::vector<Instruction *>& src_ub_vec = src_bbcond2ub.at(cond);
      std::vector<Instruction *>& tgt_ub_vec = tgt_bbcond2ub.at(cond);

      std::vector<Instruction *> ub_common;
      std::vector<Instruction *> ub_src_unique;
      std::vector<Instruction *> ub_tgt_unique;
      std::set_intersection(src_ub_vec.begin(), src_ub_vec.end(),
			    tgt_ub_vec.begin(), tgt_ub_vec.end(),
			    std::back_inserter(ub_common), comp);
      std::set_difference(src_ub_vec.begin(), src_ub_vec.end(),
			  ub_common.begin(), ub_common.end(),
			  std::back_inserter(ub_src_unique), comp);
      std::set_difference(tgt_ub_vec.begin(), tgt_ub_vec.end(),
			  ub_common.begin(), ub_common.end(),
			  std::back_inserter(ub_tgt_unique), comp);

      Instruction *bb_ub = value_inst(0, 1);
      for (auto inst : ub_common)
	{
	  bb_ub = bool_or(bb_ub, inst);
	}
      common_ub = bool_or(common_ub, bool_and(cond, bb_ub));

      bb_ub = value_inst(0, 1);
      for (auto inst : ub_src_unique)
	{
	  bb_ub = bool_or(bb_ub, inst);
	}
      src_ub = bool_or(src_ub, bool_and(cond, bb_ub));

      bb_ub = value_inst(0, 1);
      for (auto inst : ub_tgt_unique)
	{
	  bb_ub = bool_or(bb_ub, inst);
	}
      tgt_ub = bool_or(tgt_ub, bool_and(cond, bb_ub));
    }

  for (auto cond : cond_src_unique)
    {
      Instruction *bb_ub = value_inst(0, 1);
      for (auto inst : src_bbcond2ub.at(cond))
	{
	  bb_ub = bool_or(bb_ub, inst);
	}
      src_ub = bool_or(src_ub, bool_and(cond, bb_ub));
    }

  for (auto cond : cond_tgt_unique)
    {
      Instruction *bb_ub = value_inst(0, 1);
      for (auto inst : tgt_bbcond2ub.at(cond))
	{
	  bb_ub = bool_or(bb_ub, inst);
	}
      tgt_ub = bool_or(tgt_ub, bool_and(cond, bb_ub));
    }

  build_inst(Op::SRC_UB, common_ub, src_ub);
  src_unique_ub = src_ub;
  src_common_ub = common_ub;
  if (has_tgt)
    {
      build_inst(Op::TGT_UB, common_ub, tgt_ub);
      tgt_unique_ub = tgt_ub;
      tgt_common_ub = common_ub;
    }
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
    cond = bool_not(cond);
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
      Instruction *addr = translate.at(inst->arguments[0]);
      for (;;)
	{
	  if (array->opcode != Op::ARRAY_STORE)
	    break;

	  Instruction *store_array = array->arguments[0];
	  Instruction *store_addr = array->arguments[1];
	  Instruction *store_value = array->arguments[2];
	  if (addr == store_addr)
	    {
	      new_inst = store_value;
	      break;
	    }
	  else if (addr->opcode == Op::VALUE && store_addr->opcode == Op::VALUE
		   && addr != store_addr)
	    array = store_array;
	  else
	    break;
	}
      if (!new_inst)
	new_inst = build_inst(Op::ARRAY_LOAD, array, addr);
    }
  else if (inst->opcode == Op::STORE)
    {
      Instruction *array = bb2memory.at(bb);
      Instruction *addr = translate.at(inst->arguments[0]);
      Instruction *value = translate.at(inst->arguments[1]);

      // Traverse the list of previous updates to the array to determine
      // whether the current value is already the same as 'value'.
      for (Instruction *tmp_array = array;;)
	{
	  if (tmp_array->opcode == Op::ARRAY_STORE)
	    {
	      Instruction *tmp_addr = tmp_array->arguments[1];
	      Instruction *tmp_value = tmp_array->arguments[2];
	      if (addr == tmp_addr && value == tmp_value)
		{
		  // The array element has the correct value already.
		  return;
		}
	      else if (addr != tmp_addr
		       && addr->opcode == Op::VALUE
		       && tmp_addr->opcode == Op::VALUE)
		{
		  // If the addresses are distinct Op::VALUE, then we know
		  // that they do not alias, so we continue traversing the
		  // list.
		  tmp_array = tmp_array->arguments[0];
		  continue;
		}
	    }
	  break;
	}

      array = build_inst(Op::ARRAY_STORE, array, addr, value);
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
      Instruction *addr = translate.at(inst->arguments[0]);
      Instruction *value = translate.at(inst->arguments[1]);

      // Traverse the list of previous updates to the array to determine
      // whether the current value is already the same as 'value'.
      for (Instruction *tmp_array = array;;)
	{
	  if (tmp_array->opcode == Op::MEM_FLAG_ARRAY
	      && value->opcode == Op::VALUE
	      && value->value() == 0)
	    {
	      // The array element has the correct value already.
	      return;
	    }
	  if (tmp_array->opcode == Op::ARRAY_SET_FLAG)
	    {
	      Instruction *tmp_addr = tmp_array->arguments[1];
	      Instruction *tmp_value = tmp_array->arguments[2];
	      if (addr == tmp_addr && value == tmp_value)
		{
		  // The array element has the correct value already.
		  return;
		}
	      else if (addr != tmp_addr
		       && addr->opcode == Op::VALUE
		       && tmp_addr->opcode == Op::VALUE)
		{
		  // If the addresses are distinct Op::VALUE, then we know
		  // that they do not alias, so we continue traversing the
		  // list.
		  tmp_array = tmp_array->arguments[0];
		  continue;
		}
	    }
	  break;
	}

      array = build_inst(Op::ARRAY_SET_FLAG, array, addr, value);
      bb2memory_flag[bb] = array;
      return;
    }
  else if (inst->opcode == Op::SET_MEM_UNDEF)
    {
      Instruction *array = bb2memory_undef.at(bb);
      Instruction *addr = translate.at(inst->arguments[0]);
      Instruction *value = translate.at(inst->arguments[1]);

      // Traverse the list of previous updates to the array to determine
      // whether the current value is already the same as 'value'.
      for (Instruction *tmp_array = array;;)
	{
	  if (tmp_array->opcode == Op::MEM_UNDEF_ARRAY
	      && value->opcode == Op::VALUE
	      && value->value() == 0)
	    {
	      // The array element has the correct value already.
	      return;
	    }
	  if (tmp_array->opcode == Op::ARRAY_SET_UNDEF)
	    {
	      Instruction *tmp_addr = tmp_array->arguments[1];
	      Instruction *tmp_value = tmp_array->arguments[2];
	      if (addr == tmp_addr && value == tmp_value)
		{
		  // The array element has the correct value already.
		  return;
		}
	      else if (addr != tmp_addr
		       && addr->opcode == Op::VALUE
		       && tmp_addr->opcode == Op::VALUE)
		{
		  // If the addresses are distinct Op::VALUE, then we know
		  // that they do not alias, so we continue traversing the
		  // list.
		  tmp_array = tmp_array->arguments[0];
		  continue;
		}
	    }
	  break;
	}

      array = build_inst(Op::ARRAY_SET_UNDEF, array, addr, value);
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
      Instruction *addr = translate.at(inst->arguments[0]);
      for (;;)
	{
	  if (array->opcode == Op::MEM_UNDEF_ARRAY)
	    {
	      new_inst = value_inst(0, 8);
	      break;
	    }

	  if (array->opcode != Op::ARRAY_SET_UNDEF)
	    break;

	  Instruction *tmp_array = array->arguments[0];
	  Instruction *tmp_addr = array->arguments[1];
	  Instruction *tmp_value = array->arguments[2];
	  if (addr == tmp_addr)
	    {
	      new_inst = tmp_value;
	      break;
	    }
	  else if (addr->opcode == Op::VALUE && tmp_addr->opcode == Op::VALUE
		   && addr != tmp_addr)
	    array = tmp_array;
	  else
	    break;
	}
      if (!new_inst)
	new_inst = build_inst(Op::ARRAY_GET_UNDEF, array, addr);
     }
   else if (inst->opcode == Op::GET_MEM_FLAG)
     {
      Instruction *array = bb2memory_flag.at(bb);
      Instruction *addr = translate.at(inst->arguments[0]);
      for (;;)
	{
	  if (array->opcode == Op::MEM_FLAG_ARRAY)
	    {
	      new_inst = value_inst(0, 1);
	      break;
	    }

	  if (array->opcode != Op::ARRAY_SET_FLAG)
	    break;

	  Instruction *tmp_array = array->arguments[0];
	  Instruction *tmp_addr = array->arguments[1];
	  Instruction *tmp_value = array->arguments[2];
	  if (addr == tmp_addr)
	    {
	      new_inst = tmp_value;
	      break;
	    }
	  else if (addr->opcode == Op::VALUE && tmp_addr->opcode == Op::VALUE
		   && addr != tmp_addr)
	    array = tmp_array;
	  else
	    break;
	}
      if (!new_inst)
	new_inst = build_inst(Op::ARRAY_GET_FLAG, array, addr);
     }
   else if (inst->opcode == Op::GET_MEM_SIZE)
    {
      Instruction *array = bb2memory_size.at(bb);
      Instruction *mem_id = translate.at(inst->arguments[0]);
      for (;;)
	{
	  if (array->opcode == Op::MEM_SIZE_ARRAY)
	    {
	      new_inst = value_inst(0, inst->bitsize);
	      break;
	    }

	  if (array->opcode != Op::ARRAY_SET_SIZE)
	    break;

	  Instruction *tmp_array = array->arguments[0];
	  Instruction *tmp_mem_id = array->arguments[1];
	  Instruction *tmp_value = array->arguments[2];
	  if (mem_id == tmp_mem_id)
	    {
	      new_inst = tmp_value;
	      break;
	    }
	  else if (mem_id->opcode == Op::VALUE
		   && tmp_mem_id->opcode == Op::VALUE
		   && mem_id != tmp_mem_id)
	    array = tmp_array;
	  else
	    break;
	}
      if (!new_inst)
	new_inst = build_inst(Op::ARRAY_GET_SIZE, array, mem_id);
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
	  if (role == Function_role::src)
	    {
	      src_retval = arg1;
	      src_retval_undef = arg2;
	    }
	  else
	    {
	      tgt_retval = arg1;
	      tgt_retval_undef = arg2;
	    }
	}
      return;
    }
  else if (inst->opcode == Op::ITE)
    {
      Instruction *arg1 = translate.at(inst->arguments[0]);
      Instruction *arg2 = translate.at(inst->arguments[1]);
      Instruction *arg3 = translate.at(inst->arguments[2]);
      new_inst = ite(arg1, arg2, arg3);
    }
  else if (inst->opcode == Op::AND && inst->bitsize == 1)
    {
      Instruction *arg1 = translate.at(inst->arguments[0]);
      Instruction *arg2 = translate.at(inst->arguments[1]);
      new_inst = bool_and(arg1, arg2);
    }
  else if (inst->opcode == Op::OR && inst->bitsize == 1)
    {
      Instruction *arg1 = translate.at(inst->arguments[0]);
      Instruction *arg2 = translate.at(inst->arguments[1]);
      new_inst = bool_or(arg1, arg2);
    }
  else if (inst->opcode == Op::NOT && inst->bitsize == 1)
    new_inst = bool_not(translate.at(inst->arguments[0]));
  else
    {
      assert(inst->has_lhs());
      Inst_class iclass = inst->iclass();
      switch (iclass)
	{
	case Inst_class::nullary:
	  new_inst = build_inst(inst->opcode);
	  break;
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
  func->canonicalize();

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

  if (role == Function_role::src)
    src_bbcond2ub = prepare_ub(func);
  else
    {
      tgt_bbcond2ub = prepare_ub(func);
      has_tgt = true;
    }

  Op assert_op = role == Function_role::src ? Op::SRC_ASSERT : Op::TGT_ASSERT;
  build_inst(assert_op, generate_assert(func));

  Basic_block *exit_block = func->bbs.back();
  Op mem1_op = role == Function_role::src ? Op::SRC_MEM1 : Op::TGT_MEM1;
  Op mem2_op = role == Function_role::src ? Op::SRC_MEM2 : Op::TGT_MEM2;
  build_inst(mem1_op, bb2memory.at(exit_block),
		      bb2memory_size.at(exit_block));
  build_inst(mem2_op, bb2memory_flag.at(exit_block),
		      bb2memory_undef.at(exit_block));
  if (role == Function_role::src)
    {
      src_memory = bb2memory.at(exit_block);
      src_memory_flag = bb2memory_flag.at(exit_block);
      src_memory_size = bb2memory_size.at(exit_block);
      src_memory_undef = bb2memory_undef.at(exit_block);
    }
  else
    {
      tgt_memory = bb2memory.at(exit_block);
      tgt_memory_flag = bb2memory_flag.at(exit_block);
      tgt_memory_size = bb2memory_size.at(exit_block);
      tgt_memory_undef = bb2memory_undef.at(exit_block);
    }

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
  generate_ub();

  build_inst(Op::RET);

  dead_code_elimination(dest_func);
  dest_func->canonicalize();
}

bool Converter::need_checking()
{
  if (src_retval != tgt_retval
      || src_retval_undef != tgt_retval_undef)
    return true;

  if (src_memory != tgt_memory
      || src_memory_size != tgt_memory_size
      || src_memory_undef != tgt_memory_undef)
    return true;

  assert(src_common_ub == tgt_common_ub);
  if (src_unique_ub != tgt_unique_ub
      && !(tgt_unique_ub->opcode == Op::VALUE
	   && tgt_unique_ub->value() == 0))
    return true;

  return false;
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

  Converter converter(module);
  converter.convert_function(src, Function_role::src);
  converter.convert_function(tgt, Function_role::tgt);
  converter.finalize();

  if (!converter.need_checking())
    return {};

  if (config.verbose > 1)
    {
      module->print(stderr);
      converter.module->print(stderr);
    }

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

  Converter converter(func->module);
  converter.convert_function(func, Function_role::src);
  converter.finalize();

  if (config.verbose > 1)
    {
      func->print(stderr);
      converter.module->print(stderr);
    }

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

  Converter converter(func->module);
  converter.convert_function(func, Function_role::src);
  converter.finalize();

  if (config.verbose > 1)
    {
      func->print(stderr);
      converter.module->print(stderr);
    }

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

void convert(Module *module)
{
  assert(module->functions.size() == 2);
  Function *src = module->functions[0];
  Function *tgt = module->functions[1];
  if (src->name != "src")
    std::swap(src, tgt);
  assert(src->name == "src" && tgt->name == "tgt");

  Converter converter(module);
  converter.convert_function(src, Function_role::src);
  converter.convert_function(tgt, Function_role::tgt);
  converter.finalize();

  destroy_function(module->functions[1]);
  destroy_function(module->functions[0]);
  converter.dest_func->clone(module);
}

} // end namespace smtgcc
