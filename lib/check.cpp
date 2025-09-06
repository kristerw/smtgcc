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
#include <functional>
#include <unordered_map>

#include "smtgcc.h"

using namespace std::string_literals;
namespace smtgcc {
namespace {

struct Cse_key
{
  Op op;
  Inst *arg1 = nullptr;
  Inst *arg2 = nullptr;
  Inst *arg3 = nullptr;

  Cse_key(Op op)
    : op{op}
  {}
  Cse_key(Op op, Inst *arg1)
    : op{op}
    , arg1{arg1}
  {}
  Cse_key(Op op, Inst *arg1, Inst *arg2)
    : op{op}
    , arg1{arg1}
    , arg2{arg2}
  {}
  Cse_key(Op op, Inst *arg1, Inst *arg2, Inst *arg3)
    : op{op}
    , arg1{arg1}
    , arg2{arg2}
    , arg3{arg3}
  {}

  friend bool operator<(const Cse_key& lhs, const Cse_key& rhs)
  {
    if (lhs.op < rhs.op) return true;
    if (lhs.op > rhs.op) return false;

    if (lhs.arg1 < rhs.arg1) return true;
    if (lhs.arg1 > rhs.arg1) return false;

    if (lhs.arg2 < rhs.arg2) return true;
    if (lhs.arg2 > rhs.arg2) return false;

    return lhs.arg3 < rhs.arg3;
  }

  bool operator==(const Cse_key &other) const
  {
    return op == other.op
      && arg1 == other.arg1
      && arg2 == other.arg2
      && arg3 == other.arg3;
  }
};

struct Cse_key_hash {
  std::size_t operator()(const Cse_key& key) const
  {
    size_t h = (size_t)key.op;
    h ^= 0x9e3779b9 + (h << 6) + (h >> 2) + (size_t)key.arg1;
    h ^= 0x9e3779b9 + (h << 6) + (h >> 2) + (size_t)key.arg2;
    h ^= 0x9e3779b9 + (h << 6) + (h >> 2) + (size_t)key.arg3;
    return h;
  }
};

enum class Function_role {
  src, tgt
};

// Comparison function for instructions.
//
// This is mostly used for sets etc. where we need a consistent order.
// But for canonicalization we have higher requirements: We want
// the instructions in the order they are created, but with Op::VALUE
// instructions placed after non-Op::VALUE instructions so that we can
// constant fold the constants without affecting the rest of the sequence.
struct Inst_comp {
  bool operator()(const Inst *a, const Inst *b) const {
    if (a->op == Op::VALUE && b->op != Op::VALUE)
      return false;
    else if (a->op != Op::VALUE && b->op == Op::VALUE)
      return true;
    else
      return a->id < b->id;
  }
};

struct Cse : Simplify_config {
private:
  std::unordered_map<Cse_key, Inst *, Cse_key_hash> key2inst;

  Inst *get_inst(const Cse_key& key);
  bool is_min_max(Inst *arg1, Inst *arg2, Inst *arg3);
  Inst *cse_min_max(Inst *arg1, Inst *arg2, Inst *arg3);
  Inst *cse_icomparison(Op op, Inst *arg1, Inst *arg2);

public:
  Inst *get_inst(Op op)
  {
    const Cse_key key(op);
    return get_inst(key);
  }
  Inst *get_inst(Op op, Inst *arg1) override
  {
    const Cse_key key(op, arg1);
    return get_inst(key);
  }
  Inst *get_inst(Op op, Inst *arg1, Inst *arg2) override
  {
    const Cse_key key(op, arg1, arg2);
    Inst *inst = get_inst(key);
    if (!inst && inst_info[(int)op].iclass == Inst_class::icomparison)
      inst = cse_icomparison(op, arg1, arg2);
    return inst;
  }
  Inst *get_inst(Op op, Inst *arg1, Inst *arg2, Inst *arg3) override
  {
    const Cse_key key(op, arg1, arg2, arg3);
    Inst *inst = get_inst(key);
    if (!inst && op == Op::ITE && is_min_max(arg1, arg2, arg3))
      inst = cse_min_max(arg1, arg2, arg3);
    return inst;
  }
  void set_inst(Inst *inst, Op op)
  {
    const Cse_key key(op);
    key2inst.insert({key, inst});
  }
  void set_inst(Inst *inst, Op op, Inst *arg1) override
  {
    const Cse_key key(op, arg1);
    key2inst.insert({key, inst});
  }
  void set_inst(Inst *inst, Op op, Inst *arg1, Inst *arg2) override
  {
    const Cse_key key(op, arg1, arg2);
    key2inst.insert({key, inst});
  }
  void set_inst(Inst *inst, Op op, Inst *arg1, Inst *arg2, Inst *arg3) override
  {
    const Cse_key key(op, arg1, arg2, arg3);
    key2inst.insert({key, inst});
  }
};

// Convert the IR of a function to a form having essentially 1-1
// correspondence to what the SMT solver is using. In particular
// eliminate control flow, and change the memory operations to use
// arrays.
class Converter {
  // True if simplify_inst should be called while building instructions.
  bool run_simplify_inst;

  // True if we are currently canonicalizing Op::AND, Op::OR, or Op::ADD
  // sequences. Used to prevent recursive calls while building the
  // canonicalized sequence.
  bool processing_canonicalization = false;

  // Maps basic blocks to an expression telling if it is executed.
  std::map<Basic_block *, Inst *> bb2cond;

  // Maps basic blocks to the expressions determining if it contain UB.
  std::map<Basic_block *, std::set<Inst *, Inst_comp>> bb2ub;

  // Maps basic blocks to an expression determining if it contain an
  // assertion failure.
  std::map<Basic_block *, Inst *> bb2not_assert;

  // Maps basic blocks to the memory state at the end of the basic block.
  std::map<Basic_block *, Inst *> bb2memory;
  std::map<Basic_block *, Inst *> bb2memory_size;
  std::map<Basic_block *, Inst *> bb2memory_flag;
  std::map<Basic_block *, Inst *> bb2memory_indef;

  // List of the mem_id for the constant memory blocks.
  std::vector<Inst *> const_ids;

  // Table for mapping original instructions to the corresponding new
  // instruction in destination function.
  std::unordered_map<Inst *, Inst *> translate;

  Cse cse;

  std::map<Inst *, std::set<Inst *, Inst_comp>, Inst_comp> src_bbcond2ub;
  std::map<Inst *, std::set<Inst *, Inst_comp>, Inst_comp> tgt_bbcond2ub;
  bool has_tgt = false;

  // The memory state before src and tgt. This is used to have identical
  // start state for both src and tgt.
  Inst *memory;
  Inst *memory_flag;
  Inst *memory_size;
  Inst *memory_indef;

  Basic_block *dest_bb;

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

  void add_ub(Basic_block *bb, Inst *cond);
  void add_assert(Basic_block *bb, Inst *cond);
  void move_ub_earlier(Function *func);
  void remove_redundant_ub(Function *func);
  std::map<Inst *, std::set<Inst *, Inst_comp>, Inst_comp> prepare_ub(Function *func);
  void generate_ub();
  Inst *generate_assert(Function *func);
  Inst *get_full_edge_cond(Basic_block *src, Basic_block *dest);
  Inst *build_phi_ite(Basic_block *bb, const std::function<Inst *(Basic_block *)>& pred2inst);
  void build_mem_state(Basic_block *bb, std::map<Basic_block*, Inst *>& map);
  void generate_bb2cond(Basic_block *bb);
  std::pair<Inst *, Inst *> simplify_array_access(Inst *array, Inst *addr, std::map<Inst *, std::pair<Inst *, Inst *>>& cache);
  Inst *strip_local_mem(Inst *array, std::map<Inst *, Inst *>& cache);
  bool may_alias(Inst *p1, Inst *p2);
  void convert(Basic_block *bb, Inst *inst, Function_role role);

  Inst *get_cmp_inst(const Cse_key& key);
  Inst *value_inst(unsigned __int128 value, uint32_t bitsize);
  Inst *simplify(Inst *inst);
  void flatten(Op op, Inst *inst, std::set<Inst *, Inst_comp>& elems);
  void flatten(Op op, Inst *inst, std::vector<Inst *>& elems);
  Inst *canonicalize_and_or(Inst *inst);
  Inst *canonicalize_add(Inst *inst);
  Inst *specialize_cond_calc(Inst *cond, Inst *inst, bool is_true_branch);
  Inst *specialize_cond_arg(Inst *cond, Inst *inst, bool is_true_branch,
			    int depth = 0);
  Inst *build_inst(Op op);
  Inst *build_inst(Op op, Inst *arg, bool insert_after = false);
  Inst *build_inst(Op op, Inst *arg1, Inst *arg2);
  Inst *build_inst(Op op, Inst *arg1, Inst *arg2, Inst *arg3);

public:
  Converter(Module *m, bool run_simplify_inst = true);
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

Inst *Cse::get_inst(const Cse_key& key)
{
  auto I = key2inst.find(key);
  if (I != key2inst.end())
    return I->second;

  if (inst_info[(int)key.op].is_commutative)
    {
      Cse_key tmp_key = key;
      std::swap(tmp_key.arg1, tmp_key.arg2);
      I = key2inst.find(tmp_key);
      if (I != key2inst.end())
	return I->second;
    }

  return nullptr;
}

Inst *Converter::value_inst(unsigned __int128 value, uint32_t bitsize)
{
  return dest_bb->value_inst(value, bitsize);
}

Inst *Converter::simplify(Inst *inst)
{
  if (!run_simplify_inst)
    return inst;
  if (!inst->has_lhs())
    return inst;

  if (!processing_canonicalization
      && inst->bitsize == 1
      && (inst->op == Op::AND || inst->op == Op::OR))
    inst = canonicalize_and_or(inst);
  else if (!processing_canonicalization && inst->op == Op::ADD)
    inst = canonicalize_add(inst);
  else
    inst = simplify_inst(inst, &cse);

  // Simplify a sequence of two Op::ITEs with identical conditions and an
  // unrelated operation in the middle:
  //
  //   %1 = ite %a, %b, %c
  //   %2 = op %1, %d
  //   %3 = ite %a, %2, %e
  //
  // to
  //
  //   %2 = op %b, %d
  //   %3 = ite %a, %2, %e
  if (inst->op == Op::ITE)
    {
      Inst *const arg1 = inst->args[0];
      Inst *const arg2 = inst->args[1];
      Inst *const arg3 = inst->args[2];
      Inst *new_arg2 = specialize_cond_arg(arg1, arg2, true);
      Inst *new_arg3 = specialize_cond_arg(arg1, arg3, false);
      if (new_arg2 != arg2 || new_arg3 != arg3)
	return build_inst(Op::ITE, arg1, new_arg2, new_arg3);
    }

  return inst;
}

void Converter::flatten(Op op, Inst *inst, std::set<Inst *, Inst_comp>& elems)
{
  if (inst->op == op)
    {
      flatten(op, inst->args[1], elems);
      flatten(op, inst->args[0], elems);
    }
  else if (inst->op == Op::NOT
	   && ((op == Op::AND && inst->args[0]->op == Op::OR)
	       || (op == Op::OR && inst->args[0]->op == Op::AND)))
    {
      flatten(op, build_inst(Op::NOT, inst->args[0]->args[0]), elems);
      flatten(op, build_inst(Op::NOT, inst->args[0]->args[1]), elems);
    }
  else
    elems.insert(inst);
}

void Converter::flatten(Op op, Inst *inst, std::vector<Inst *>& elems)
{
  if (inst->op == op)
    {
      flatten(op, inst->args[1], elems);
      flatten(op, inst->args[0], elems);
    }
  else if (inst->op == Op::NOT
	   && ((op == Op::AND && inst->args[0]->op == Op::OR)
	       || (op == Op::OR && inst->args[0]->op == Op::AND)))
    {
      flatten(op, build_inst(Op::NOT, inst->args[0]->args[0]), elems);
      flatten(op, build_inst(Op::NOT, inst->args[0]->args[1]), elems);
    }
  else
    elems.push_back(inst);
}

// If this is an unsigned comparison x < constant or x <= constant, return
// the largest value of x where this is true.
std::optional<unsigned __int128> ult_upper_bound(Inst *inst)
{
  if (inst->op == Op::ULT
      && inst->args[1]->op == Op::VALUE
      && !is_value_zero(inst->args[1]))
    {
      return inst->args[1]->value() - 1;
    }
  if (inst->op == Op::NOT
      && inst->args[0]->op == Op::ULT
      && inst->args[0]->args[0]->op == Op::VALUE)
    {
      return inst->args[0]->args[0]->value();
    }
  return {};
}

// If this is an unsigned comparison  constant < x or constant <= x, return
// the smallest value of x where this is true.
std::optional<unsigned __int128> ult_lower_bound(Inst *inst)
{
  if (inst->op == Op::ULT
      && inst->args[0]->op == Op::VALUE
      && !is_value_m1(inst->args[0]))
    {
      return inst->args[0]->value() + 1;
    }
  if (inst->op == Op::NOT
      && inst->args[0]->op == Op::ULT
      && inst->args[0]->args[1]->op == Op::VALUE)
    {
      return inst->args[0]->args[1]->value();
    }
  return {};
}

// Return the smallest upper bound of the comparisons in comps that have an
// lt_upper_bound. This is the largest value for which Op::AND of those
// comparisons is true.
std::optional<unsigned __int128> ult_upper_bound(std::vector<Inst*>& comps)
{
  std::optional<unsigned __int128> ret;
  for (auto comp : comps)
    {
      std::optional<unsigned __int128> bound = ult_upper_bound(comp);
      if (bound && (!ret || (ret && *bound < *ret)))
	ret = bound;
    }
  return ret;
}

// Return the largest lower bound of the comparisons in comps that have an
// lt_lower_bound. This is the smallest value for which Op::AND of those
// comparisons is true.
std::optional<unsigned __int128> ult_lower_bound(std::vector<Inst*>& comps)
{
  std::optional<unsigned __int128> ret;
  for (auto comp : comps)
    {
      std::optional<unsigned __int128> bound = ult_lower_bound(comp);
      if (bound && (!ret || (ret && *ret < *bound)))
	ret = bound;
    }
  return ret;
}

// Canonicalize chains of Op::AND or Op::OR to the form:
//   (and (and (and (and e1, e2), e3), e4) ...
Inst *Converter::canonicalize_and_or(Inst *inst)
{
  assert(inst->op == Op::AND || inst->op == Op::OR);
  assert(inst->bitsize == 1);

  Inst *arg1 = inst->args[0];
  Inst *arg2 = inst->args[1];

  // Collect the elements.
  std::set<Inst *, Inst_comp> elems;
  flatten(inst->op, arg1, elems);
  flatten(inst->op, arg2, elems);

  assert(!elems.empty());
  if (elems.size() == 1)
    return *elems.begin();

  // x & !x -> 0
  // x | !x -> 1
  for (auto elem : elems)
    {
      if (elem->op == Op::NOT && elems.contains(elem->args[0]))
	{
	  if (inst->op == Op::AND)
	    return value_inst(0, arg1->bitsize);
	  else
	    return value_inst(-1, arg1->bitsize);
	}
    }

  // Eliminate redundant comparison with a constant. For example:
  //   x > 3 && x > 4
  // is simplified to x > 4.
  {
    std::map<Inst*, std::vector<Inst*>> ult_comps;
    std::map<Inst*, std::vector<Inst*>> eq_comps;

    // Collect Op::ULT, Op::ULE, and Op::EQ between an instruction and
    // a constant.
    for (auto elem : elems)
      {
	if (elem->op == Op::EQ)
	  {
	    if (elem->args[0]->op == Op::VALUE)
	      eq_comps[elem->args[1]].push_back(elem);
	    if (elem->args[1]->op == Op::VALUE)
	      eq_comps[elem->args[0]].push_back(elem);
	  }
	if (elem->op == Op::ULT)
	  {
	    if (elem->args[0]->op == Op::VALUE)
	      ult_comps[elem->args[1]].push_back(elem);
	    if (elem->args[1]->op == Op::VALUE)
	      ult_comps[elem->args[0]].push_back(elem);
	  }
	if (elem->op == Op::NOT && elem->args[0]->op == Op::ULT)
	  {
	    if (elem->args[0]->args[0]->op == Op::VALUE)
	      ult_comps[elem->args[0]->args[1]].push_back(elem);
	    if (elem->args[0]->args[1]->op == Op::VALUE)
	      ult_comps[elem->args[0]->args[0]].push_back(elem);
	  }
      }

    if (inst->op == Op::AND)
      {
	// The expression is false if we have more than one equality
	// comparison, as the comparisons must then be conflicting,
	// such as
	//   x == 1 && x == 2
	for (auto& [_, comps] : eq_comps)
	  {
	    if (comps.size() > 1)
	      return value_inst(0, 1);
	  }

	// Eliminate redundant comparisons.
	for (auto& [x, comps] : ult_comps)
	  {
	    // Find the value of the equality comparison, if any.
	    std::optional<unsigned __int128> eq_val;
	    if (eq_comps.contains(x))
	      {
		assert(eq_comps[x].size() == 1);
		Inst *eq = eq_comps[x][0];
		if (eq->args[0]->op == Op::VALUE)
		  eq_val = eq->args[0]->value();
		else
		  eq_val = eq->args[1]->value();
	      }

	    // Find the largest and smallest values x may have for the
	    // expression to be true.
	    std::optional<unsigned __int128> upper_bound =
	      ult_upper_bound(comps);
	    std::optional<unsigned __int128> lower_bound =
	      ult_lower_bound(comps);

	    if (eq_val)
	      {
		// The expression is false in cases such as
		//   x == 0 && x > 4
		// where an equality comparison is not consistent with the
		// comparisons for upper_bound and lower_bound.
		if (upper_bound && *upper_bound < *eq_val)
		  return value_inst(0, 1);
		if (lower_bound && *eq_val < *lower_bound)
		  return value_inst(0, 1);

		// The comparisons in ult_comps are consistent with the
		// equality comparison, but this means all of them are
		// redundant.
		for (auto comp : comps)
		  {
		    elems.erase(comp);
		  }
		continue;
	      }

	    // Eliminate the ult_comps comparisons that are redundant with
	    // respect to upper_bound and lower_bound.
	    for (auto comp : comps)
	      {
		std::optional<unsigned __int128> ubnd = ult_upper_bound(comp);
		if (ubnd && upper_bound && *upper_bound < *ubnd)
		  elems.erase(comp);
		std::optional<unsigned __int128> lbnd = ult_lower_bound(comp);
		if (lbnd && lower_bound && *lbnd < *lower_bound)
		  elems.erase(comp);
	      }
	  }
      }
  }

  assert(!elems.empty());
  if (elems.size() == 1)
    return *elems.begin();

  // Generate the sequence.
  bool orig_run_simplify_inst = run_simplify_inst;
  run_simplify_inst = false;
  processing_canonicalization = true;
  auto first = elems.begin();
  auto second = std::next(first);
  Inst *new_inst = build_inst(inst->op, *first, *second);
  for (auto it = std::next(second); it != elems.end(); ++it)
    {
      new_inst = build_inst(inst->op, new_inst, *it);
    }
  processing_canonicalization = false;
  run_simplify_inst = orig_run_simplify_inst;

  return new_inst;
}

// Canonicalize chains of Op::ADD to the form:
//   (add (add (add (add e1, e2), e3), e4) ...
Inst *Converter::canonicalize_add(Inst *inst)
{
  assert(inst->op == Op::ADD);

  Inst *arg1 = inst->args[0];
  Inst *arg2 = inst->args[1];

  // Collect the elements.
  std::vector<Inst *> elems;
  if (arg1->op == inst->op)
    flatten(inst->op, arg1, elems);
  else
    elems.push_back(arg1);
  if (arg2->op == inst->op)
    flatten(inst->op, arg2, elems);
  else
    elems.push_back(arg2);
  Inst_comp comp;
  std::sort(elems.begin(), elems.end(), comp);

  assert(!elems.empty());
  if (elems.size() == 1)
    return *elems.begin();

  // Generate the sequence.
  processing_canonicalization = true;
  auto first = elems.begin();
  auto second = std::next(first);
  Inst *new_inst = build_inst(inst->op, *first, *second);
  for (auto it = std::next(second); it != elems.end(); ++it)
    {
      new_inst = build_inst(inst->op, new_inst, *it);
    }
  processing_canonicalization = false;

  return new_inst;
}

Inst *Converter::specialize_cond_calc(Inst *cond, Inst *inst, bool is_true_branch)
{
  if (inst == cond)
    return value_inst(is_true_branch, 1);
  else if (inst->op == Op::ITE && cond == inst->args[0])
    return is_true_branch ? inst->args[1] : inst->args[2];
  return inst;
}

Inst *Converter::specialize_cond_arg(Inst *cond, Inst *inst, bool is_true_branch, int depth)
{
  switch (inst->iclass())
    {
    case Inst_class::iunary:
    case Inst_class::funary:
    case Inst_class::ibinary:
    case Inst_class::fbinary:
    case Inst_class::icomparison:
    case Inst_class::fcomparison:
    case Inst_class::conv:
    case Inst_class::ternary:
      break;
    default:
      return inst;
    }

  Inst *new_inst = specialize_cond_calc(cond, inst, is_true_branch);
  if (new_inst != inst)
    return new_inst;

  if (depth < 2)
    {
      Inst *args[3];
      assert(inst->nof_args <= 3);
      bool modified = false;
      for (uint i = 0; i < inst->nof_args; i++)
	{
	  Inst *arg = inst->args[i];
	  args[i] = inst->args[i];

	  int next_depth = depth + 1;

	  // The array store instructions are a bit special as they are
	  // chained, so a store of a 32-bit value consists of four store
	  // instructions where the value may be extracted from Op::ITE.
	  // So we need to handle this as a special case to essentially
	  // treat all stores as one instruction.
	  if (inst->op == arg->op
	      && (inst->op == Op::ARRAY_STORE
		  || inst->op == Op::ARRAY_SET_INDEF
		  || inst->op == Op::ARRAY_SET_FLAG))
	    next_depth = depth;

	  // We often have chains of Op::EXTRACT, Op::ZEXT, etc., between
	  // the interesting instructions. Exclude them from the depth.
	  if (arg->iclass() == Inst_class::iunary
	      || arg->iclass() == Inst_class::funary
	      || arg->iclass() == Inst_class::conv
	      || arg->op == Op::EXTRACT)
	    next_depth = depth;

	  arg = specialize_cond_arg(cond, args[i], is_true_branch, next_depth);
	  if (arg != args[i])
	    {
	      args[i] = arg;
	      modified = true;
	    }
	}

      if (modified)
	{
	  switch (inst->nof_args)
	    {
	    case 1:
	      return build_inst(inst->op, args[0]);
	    case 2:
	      return build_inst(inst->op, args[0], args[1]);
	    case 3:
	      return build_inst(inst->op, args[0], args[1], args[2]);
	    default:
	      assert(0);
	      break;
	    }
	}
    }
  return inst;
}

bool Cse::is_min_max(Inst *arg1, Inst *arg2, Inst *arg3)
{
  if (arg1->op == Op::SLT || arg1->op == Op::ULT)
    if ((arg1->args[0] == arg2 && arg1->args[1] == arg3)
	|| (arg1->args[1] == arg2 && arg1->args[0] == arg3))
      return true;
  return false;
}

// min/max can be written in either of two ways:
//   ite (ult x, y), x, y
//   ite (ult y, x), y, x
// Check if we already have an instruction using the other form.
Inst *Cse::cse_min_max(Inst *arg1, Inst *arg2, Inst *arg3)
{
  const Cse_key cmp_key(arg1->op, arg1->args[1], arg1->args[0]);
  Inst *cmp_inst = get_inst(cmp_key);
  if (!cmp_inst)
    return nullptr;

  const Cse_key key(Op::ITE, cmp_inst, arg3, arg2);
  return get_inst(key);
}

Inst *Cse::cse_icomparison(Op op, Inst *arg1, Inst *arg2)
{
  // ult x, c <--> not (ult c-1, x)
  if (op == Op::ULT && arg2->op == Op::VALUE && !is_value_zero(arg2))
    {
      Inst *val = arg1->bb->value_inst(arg2->value() - 1, arg1->bitsize);
      const Cse_key key1(op, val, arg1);
      Inst *cmp = get_inst(key1);
      if (cmp)
	{
	  const Cse_key key2(Op::NOT, cmp);
	  return get_inst(key2);
	}
    }

  // slt x, c <--> not (slt c-1, x)
  if (op == Op::SLT && arg2->op == Op::VALUE && !is_value_signed_min(arg2))
    {
      Inst *val = arg1->bb->value_inst(arg2->signed_value() - 1, arg1->bitsize);
      const Cse_key key1(op, val, arg1);
      Inst *cmp = get_inst(key1);
      if (cmp)
	{
	  const Cse_key key2(Op::NOT, cmp);
	  return get_inst(key2);
	}
    }

  // ult c, x <--> not (ult x, c+1)
  if (op == Op::ULT && arg1->op == Op::VALUE && !is_value_m1(arg1))
    {
      Inst *val = arg1->bb->value_inst(arg1->value() + 1, arg1->bitsize);
      const Cse_key key1(op, arg2, val);
      Inst *cmp = get_inst(key1);
      if (cmp)
	{
	  const Cse_key key2(Op::NOT, cmp);
	  return get_inst(key2);
	}
    }

  // slt c, x <--> not (slt x, c+1)
  if (op == Op::SLT && arg1->op == Op::VALUE && !is_value_signed_max(arg1))
    {
      Inst *val = arg1->bb->value_inst(arg1->signed_value() + 1, arg1->bitsize);
      const Cse_key key1(op, arg2, val);
      Inst *cmp = get_inst(key1);
      if (cmp)
	{
	  const Cse_key key2(Op::NOT, cmp);
	  return get_inst(key2);
	}
    }

  return nullptr;
}

Inst *Converter::build_inst(Op op)
{
  Inst *inst = cse.get_inst(op);
  if (!inst)
    {
      inst = dest_bb->build_inst(op);
      cse.set_inst(inst, op);
    }
  return inst;
}

Inst *Converter::build_inst(Op op, Inst *arg, bool insert_after)
{
  if (op == Op::NOT && arg->op == Op::NOT)
    return arg->args[0];

  Inst *inst = cse.get_inst(op, arg);
  if (!inst)
    {
      inst = dest_bb->build_inst(op, arg);
      if (insert_after)
	inst->move_after(arg);
      inst = simplify(inst);
      cse.set_inst(inst, op, arg);
    }
  return inst;
}

Inst *Converter::build_inst(Op op, Inst *arg1, Inst *arg2)
{
  Inst *inst = cse.get_inst(op, arg1, arg2);
  if (!inst && (op == Op::AND || op == Op::OR))
    {
      if (op == Op::AND)
	{
	  if (is_value_m1(arg1))
	    return arg2;
	  if (is_value_m1(arg2))
	    return arg1;
	  if (is_value_zero(arg1) || is_value_zero(arg2))
	    return value_inst(0, arg1->bitsize);
	}
      else
	{
	  if (is_value_zero(arg1))
	    return arg2;
	  if (is_value_zero(arg2))
	    return arg1;
	  if (is_value_m1(arg1) || is_value_m1(arg2))
	    return value_inst(-1, arg1->bitsize);
	}

      // Check if the expression already exists in a different form.
      Inst *not_arg1;
      if (arg1->op == Op::NOT)
	not_arg1 = arg1->args[0];
      else
	not_arg1 = cse.get_inst(Op::NOT, arg1);
      Inst *not_arg2;
      if (arg2->op == Op::NOT)
	not_arg2 = arg2->args[0];
      else
	not_arg2 = cse.get_inst(Op::NOT, arg2);
      if (not_arg1 && not_arg2)
	{
	  if (op == Op::AND)
	    inst = cse.get_inst(Op::OR, not_arg1, not_arg2);
	  else
	    inst = cse.get_inst(Op::AND, not_arg1, not_arg2);
	  if (inst)
	    {
	      inst = build_inst(Op::NOT, inst);
	      cse.set_inst(inst, op, arg1, arg2);
	      return inst;
	    }
	}
    }
  if (inst_info[(int)op].is_commutative
      && !(arg1->op == op && arg2->op != op))
    {
      // Use the same argument order as canonicalization.
      if (arg1->op != op && arg2->op == op)
	std::swap(arg1, arg2);
      else if (arg1->op == Op::VALUE && arg2->op != Op::VALUE)
	std::swap(arg1, arg2);
      else if (arg1->id > arg2->id)
	std::swap(arg1, arg2);
    }
  if (!inst)
    {
      inst = dest_bb->build_inst(op, arg1, arg2);
      inst = simplify(inst);
      cse.set_inst(inst, op, arg1, arg2);
    }

  // Booleans are often used negated. Create the negated form right after
  // the instruction to reduce differences in the result, independent of
  // later simplifications.
  if (inst->bitsize == 1 && inst->op != Op::NOT)
    build_inst(Op::NOT, inst);

  return inst;
}

Inst *Converter::build_inst(Op op, Inst *arg1, Inst *arg2, Inst *arg3)
{
  Inst *inst = cse.get_inst(op, arg1, arg2, arg3);
  if (!inst)
    {
      inst = dest_bb->build_inst(op, arg1, arg2, arg3);
      inst = simplify(inst);
      cse.set_inst(inst, op, arg1, arg2, arg3);
    }

  // Booleans are often used negated. Create the negated form right after
  // the instruction to reduce differences in the result, independent of
  // later simplifications.
  if (inst->bitsize == 1 && inst->op != Op::NOT)
    build_inst(Op::NOT, inst);

  return inst;
}

Converter::Converter(Module *m, bool run_simplify_inst)
  : run_simplify_inst{run_simplify_inst}
{
  module = create_module(m->ptr_bits, m->ptr_id_bits, m->ptr_offset_bits);
  dest_func = module->build_function("check");
  dest_bb = dest_func->build_bb();

  memory = build_inst(Op::MEM_ARRAY);
  memory_indef = build_inst(Op::MEM_INDEF_ARRAY);
  memory_flag = build_inst(Op::MEM_FLAG_ARRAY);
  memory_size = build_inst(Op::MEM_SIZE_ARRAY);
}

void Converter::add_ub(Basic_block *bb, Inst *cond)
{
  if (cond->op == Op::VALUE && cond->value() == 0)
    return;

  // It is more effective to split Op::OR into two elements to better handle
  // cases where, for example, one of the arguments is common to both src
  // and tgt.
  //
  // Note: This split needs to be done here rather than in earlier passes
  // such as simplify_inst, because it is common to encounter IR of the form
  //   UB(!uninit & (cond1 | cond2))
  // and the uninit check is not eliminated until the memory array
  // optimizations are performed here in the Converter pass.
  if (cond->op == Op::OR)
    {
      add_ub(bb, cond->args[0]);
      add_ub(bb, cond->args[1]);
    }
  if (cond->op == Op::NOT && cond->args[0]->op == Op::AND)
    {
      add_ub(bb, build_inst(Op::NOT, cond->args[0]->args[0]));
      add_ub(bb, build_inst(Op::NOT, cond->args[0]->args[1]));
    }
  else
    bb2ub[bb].insert(cond);
}

void Converter::add_assert(Basic_block *bb, Inst *cond)
{
  cond = build_inst(Op::NOT, cond);
  if (bb2not_assert.contains(bb))
    {
      cond = build_inst(Op::OR, bb2not_assert.at(bb), cond);
      bb2not_assert.erase(bb);
    }
  bb2not_assert.insert({bb, cond});
}

// Duplicate UB checks on dominating BBs when they are checked on all paths.
void Converter::move_ub_earlier(Function *func)
{
  Inst_comp comp;
  Inst *b1 = value_inst(1, 1);
  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];
      std::set<Inst *, Inst_comp>& ub_set = bb2ub[bb];

      // Find the UB checks that are performed on all paths from this BB.
      std::set<Inst *, Inst_comp> checked_ub;
      if (bb->succs.size() == 1)
	checked_ub = bb2ub.at(bb->succs[0]);
      else if (bb->succs.size() > 1)
	{
	  bool always_ub = true;
	  for (auto succ : bb->succs)
	    {
	      std::set<Inst*, Inst_comp>& succ_set = bb2ub.at(succ);
	      if (succ_set.contains(b1))
		continue;
	      if (always_ub)
		{
		  // This is the first successor to handle.
		  always_ub = false;
		  checked_ub = succ_set;
		}
	      else
		{
		  std::set<Inst*, Inst_comp> tmp;
		  std::set_intersection(succ_set.begin(), succ_set.end(),
					checked_ub.begin(), checked_ub.end(),
					std::inserter(tmp, tmp.begin()),
					comp);
		  checked_ub = tmp;
		}
	    }
	  if (always_ub)
	    {
	      assert(checked_ub.empty());
	      checked_ub.insert(b1);
	    }
	}

      // Add the checks to the current BB.
      ub_set.insert(checked_ub.begin(), checked_ub.end());
      if (ub_set.size() > 1 && ub_set.contains(b1))
	{
	  ub_set.clear();
	  ub_set.insert(b1);
	}
    }
}

// For all basic blocks, temove UB that have already been checked on all
// paths to the BB.
void Converter::remove_redundant_ub(Function *func)
{
  Inst_comp comp;
  Inst *b1 = value_inst(1, 1);
  std::map<Basic_block*,std::set<Inst*, Inst_comp>> bb2checked_ub;
  for (auto bb : func->bbs)
    {
      std::set<Inst *, Inst_comp>& ub_set = bb2ub[bb];

      // Find the UB checks that are performed on all paths to this BB.
      std::set<Inst *, Inst_comp> checked_ub;
      if (bb->preds.size() == 1)
	checked_ub = bb2checked_ub.at(bb->preds[0]);
      else if (bb->preds.size() > 1)
	{
	  bool always_ub = true;
	  for (auto pred : bb->preds)
	    {
	      std::set<Inst*, Inst_comp>& pred_set = bb2checked_ub.at(pred);
	      if (pred_set.contains(b1))
		continue;
	      if (always_ub)
		{
		  // This is the first predecessor to handle.
		  always_ub = false;
		  checked_ub = pred_set;
		}
	      else
		{
		  std::set<Inst*, Inst_comp> tmp;
		  std::set_intersection(pred_set.begin(), pred_set.end(),
					checked_ub.begin(), checked_ub.end(),
					std::inserter(tmp, tmp.begin()),
					comp);
		  checked_ub = tmp;
		}
	    }
	  if (always_ub)
	    {
	      assert(checked_ub.empty());
	      checked_ub.insert(b1);
	    }
	}

      // Remove UB that are already checked by the predecessor BBs.
      if (checked_ub.contains(b1))
	{
	  // All predecessors are always UB, so the UB in this
	  // BB does not affect the result.
	  ub_set.clear();
	}
      else
	{
	  std::set<Inst*, Inst_comp> tmp;
	  std::set_difference(ub_set.begin(), ub_set.end(),
			      checked_ub.begin(), checked_ub.end(),
			      std::inserter(tmp, tmp.begin()), comp);
	  ub_set = tmp;
	}

      // Add the UB checks from this BB.
      checked_ub.insert(ub_set.begin(), ub_set.end());
      if (checked_ub.size() > 1 && checked_ub.contains(b1))
	{
	  checked_ub.clear();
	  checked_ub.insert(b1);
	}
      bb2checked_ub.insert({bb, checked_ub});
    }
}

std::map<Inst *, std::set<Inst *, Inst_comp>, Inst_comp> Converter::prepare_ub(Function *func)
{
  // Hoist UB to the dominating BB when all successor paths trigger
  // the same UB.
  move_ub_earlier(func);
  remove_redundant_ub(func);

  // Merge the UB from the basic blocks with the same path condition.
  std::map<Inst *, std::set<Inst *, Inst_comp>, Inst_comp> bbcond2ub;
  for (auto bb : func->bbs)
    {
      if (!bb2ub.contains(bb))
	continue;

      const std::set<Inst *, Inst_comp>& bb_ub = bb2ub[bb];
      std::set<Inst *, Inst_comp>& cond_ub = bbcond2ub[bb2cond.at(bb)];
      cond_ub.insert(bb_ub.begin(), bb_ub.end());
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
  std::vector<Inst *> src_cond;
  std::vector<Inst *> tgt_cond;

  for (auto [cond, _] : src_bbcond2ub)
    {
      src_cond.push_back(cond);
    }
  for (auto [cond, _] : tgt_bbcond2ub)
    {
      tgt_cond.push_back(cond);
    }

  Inst_comp comp;
  std::vector<Inst *> cond_common;
  std::vector<Inst *> cond_src_unique;
  std::vector<Inst *> cond_tgt_unique;
  std::set_intersection(src_cond.begin(), src_cond.end(),
			tgt_cond.begin(), tgt_cond.end(),
			std::back_inserter(cond_common), comp);
  std::set_difference(src_cond.begin(), src_cond.end(),
		      cond_common.begin(), cond_common.end(),
		      std::back_inserter(cond_src_unique), comp);
  std::set_difference(tgt_cond.begin(), tgt_cond.end(),
		      cond_common.begin(), cond_common.end(),
		      std::back_inserter(cond_tgt_unique), comp);

  Inst *src_ub = value_inst(0, 1);
  Inst *tgt_ub = value_inst(0, 1);
  Inst *common_ub = value_inst(0, 1);
  for (auto cond : cond_common)
    {
      std::set<Inst *, Inst_comp>& src_ub_set = src_bbcond2ub.at(cond);
      std::set<Inst *, Inst_comp>& tgt_ub_set = tgt_bbcond2ub.at(cond);

      std::vector<Inst *> ub_common;
      std::vector<Inst *> ub_src_unique;
      std::vector<Inst *> ub_tgt_unique;
      std::set_intersection(src_ub_set.begin(), src_ub_set.end(),
			    tgt_ub_set.begin(), tgt_ub_set.end(),
			    std::back_inserter(ub_common), comp);
      std::set_difference(src_ub_set.begin(), src_ub_set.end(),
			  ub_common.begin(), ub_common.end(),
			  std::back_inserter(ub_src_unique), comp);
      std::set_difference(tgt_ub_set.begin(), tgt_ub_set.end(),
			  ub_common.begin(), ub_common.end(),
			  std::back_inserter(ub_tgt_unique), comp);

      Inst *bb_ub = value_inst(0, 1);
      for (auto inst : ub_common)
	{
	  bb_ub = build_inst(Op::OR, bb_ub, inst);
	}
      common_ub =
	build_inst(Op::OR, common_ub, build_inst(Op::AND, cond, bb_ub));

      bb_ub = value_inst(0, 1);
      for (auto inst : ub_src_unique)
	{
	  bb_ub = build_inst(Op::OR, bb_ub, inst);
	}
      src_ub = build_inst(Op::OR, src_ub, build_inst(Op::AND, cond, bb_ub));

      bb_ub = value_inst(0, 1);
      for (auto inst : ub_tgt_unique)
	{
	  bb_ub = build_inst(Op::OR, bb_ub, inst);
	}
      tgt_ub = build_inst(Op::OR, tgt_ub, build_inst(Op::AND, cond, bb_ub));
    }

  for (auto cond : cond_src_unique)
    {
      Inst *bb_ub = value_inst(0, 1);
      for (auto inst : src_bbcond2ub.at(cond))
	{
	  bb_ub = build_inst(Op::OR, bb_ub, inst);
	}
      src_ub = build_inst(Op::OR, src_ub, build_inst(Op::AND, cond, bb_ub));
    }

  for (auto cond : cond_tgt_unique)
    {
      Inst *bb_ub = value_inst(0, 1);
      for (auto inst : tgt_bbcond2ub.at(cond))
	{
	  bb_ub = build_inst(Op::OR, bb_ub, inst);
	}
      tgt_ub = build_inst(Op::OR, tgt_ub, build_inst(Op::AND, cond, bb_ub));
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

Inst *Converter::generate_assert(Function *func)
{
  Inst *assrt = value_inst(0, 1);
  for (auto bb : func->bbs)
    {
      if (!bb2not_assert.contains(bb))
	continue;

      assrt =
	build_inst(Op::OR, assrt,
		   build_inst(Op::AND, bb2not_assert.at(bb), bb2cond.at(bb)));
    }

  return assrt;
}

Inst *Converter::get_full_edge_cond(Basic_block *src, Basic_block *dest)
{
  if (src->succs.size() == 1)
    return bb2cond.at(src);
  assert(src->succs.size() == 2);
  assert(src->last_inst->op == Op::BR);
  assert(src->last_inst->nof_args == 1);
  Inst *cond = translate.at(src->last_inst->args[0]);
  if (dest != src->succs[0])
    cond = build_inst(Op::NOT, cond);
  return build_inst(Op::AND, bb2cond.at(src), cond);
}

// Build a chain of ite instructions for a phi-node.
//
// If we have a phi-node with three arguments, where the path conditions for
// the elements are:
//   1. a & b & c
//   2. a & b & !c
//   3. !a & d
// we could build the chain as:
//   ite (!a & d), v3, (ite (a & b & !c), v2, v1)
// But we know that if the phi-node value is used, then at least one of the
// paths must be taken. So we can eliminate a & b in the last ite:
//   ite (!a & d), v3, (ite (!c), v2, v1)
// because if 3. is not true, then one of 1. or 2. must be true.
// In general, if all conditions in a sub-chain share the same element,
// that element can be eliminated.
// This optimization makes it possible to CSE more between src and tgt for
// optimizations that modify the CFG.
Inst *Converter::build_phi_ite(Basic_block *bb, const std::function<Inst *(Basic_block *)>& pred2inst)
{
  Inst_comp comp;
  assert(bb->preds.size() > 0);
  Inst *inst = pred2inst(bb->preds[0]);
  if (bb->preds.size() == 1)
    return inst;

  std::vector<Inst *> common;
  flatten(Op::AND, get_full_edge_cond(bb->preds[0], bb), common);
  std::sort(common.begin(), common.end(), comp);
  common.erase(std::unique(common.begin(), common.end()), common.end());

  for (unsigned i = 1; i < bb->preds.size(); i++)
    {
      std::vector<Inst *> conds;
      Inst *full_cond = get_full_edge_cond(bb->preds[i], bb);
      flatten(Op::AND, full_cond, conds);
      std::sort(conds.begin(), conds.end(), comp);
      conds.erase(std::unique(conds.begin(), conds.end()), conds.end());

      std::vector<Inst*> tmp;
      std::set_intersection(common.begin(), common.end(),
			    conds.begin(), conds.end(),
			    std::back_inserter(tmp), comp);
      common = std::move(tmp);

      std::vector<Inst*> needed_conds;
      std::set_difference(conds.begin(), conds.end(),
			  common.begin(), common.end(),
			  std::back_inserter(needed_conds), comp);

      // Create the simplified cond. This may be empty if several
      // successive predecessors have the same condition (this
      // typically happens when simplification after CSE folds them
      // to 0 or 1). Use the original condition if that happens.
      Inst *cond;
      if (needed_conds.empty())
	cond = full_cond;
      else
	{
	  cond = needed_conds[0];
	  for (unsigned j = 0; j < needed_conds.size(); j++)
	    cond = build_inst(Op::AND, cond, needed_conds[j]);
	}

      inst = build_inst(Op::ITE, cond, pred2inst(bb->preds[i]), inst);
    }

  return inst;
}

void Converter::build_mem_state(Basic_block *bb, std::map<Basic_block*, Inst *>& map)
{
  assert(bb->preds.size() > 0);
  auto pred2inst =
    [&map](Basic_block *pred)
    {
      return map.at(pred);
    };
  Inst *inst = build_phi_ite(bb, pred2inst);
  map.insert({bb, inst});
}

void Converter::generate_bb2cond(Basic_block *bb)
{
  Basic_block *dominator = nearest_dominator(bb);
  if (dominator && postdominates(bb, dominator))
    {
      // If the dominator is post dominated by bb, then they have identical
      // conditions.
      bb2cond.insert({bb, bb2cond.at(dominator)});
    }
  else
    {
      // We must build a new condition that reflect the path(s) to this bb.
      Inst *cond = value_inst(0, 1);
      for (auto pred_bb : bb->preds)
	{
	  cond = build_inst(Op::OR, cond, get_full_edge_cond(pred_bb, bb));
	}
      bb2cond.insert({bb, cond});
    }
}

// Simplify instructions that take a memory array as the first parameter by:
//  * Implementing store-to-load forwarding (and set-to-get forwarding).
//  * For load/get instructions, attempt to use an "earlier" array when
//    possible (by bypassing non-aliasing store/set in the instruction
//    sequence that generates the array). This helps in several cases:
//     - Load/get instructions become more likely to be CSE'd, especially
//       when vectorizing instructions like:
//         a[0] = a[0] + 1;
//         a[1] = a[1] + 1;
//       In these cases, all loads now reference the same array, so the load
//       instructions in src and vectorized loads in tgt will be CSE:ed.
//     - Load/get instructions are more likely to use the same array,
//       improving the effectiveness of CSE for the generated ITEs for
//       phi nodes of memory arrays.
//     - Using fewer arrays for load/get operations appears to benefit
//       the SMT solver's performance.
std::pair<Inst *, Inst *> Converter::simplify_array_access(Inst *array, Inst *addr, std::map<Inst *, std::pair<Inst *, Inst *>>& cache)
{
  auto I = cache.find(array);
  if (I != cache.end())
    return I->second;

  Inst *orig_array = array;
  Inst *value = nullptr;
  for (;;)
    {
      if (array->op == Op::ITE)
	{
	  Inst *cond = array->args[0];
	  Inst *arg2 = array->args[1];
	  Inst *arg3 = array->args[2];
	  auto [value2, array2] = simplify_array_access(arg2, addr, cache);
	  auto [value3, array3] = simplify_array_access(arg3, addr, cache);
	  if (value2 && value3)
	    value = build_inst(Op::ITE, cond, value2, value3);
	  if (array2 != arg2 || array3 != arg3)
	    array = build_inst(Op::ITE, cond, array2, array3);
	  break;
	}
      else if (array->op == Op::ARRAY_STORE
	       || array->op == Op::ARRAY_SET_SIZE
	       || array->op == Op::ARRAY_SET_FLAG
	       || array->op == Op::ARRAY_SET_INDEF)
	{
	  Inst *store_array = array->args[0];
	  Inst *store_addr = array->args[1];
	  Inst *store_value = array->args[2];
	  if (addr == store_addr)
	    {
	      value = store_value;
	      // Use the original array when a value is found.
	      // This helps loads after loops for types wider than a byte,
	      // preventing the generation of a distinct chain of ITE
	      // instructions for each byte loaded in situations where
	      // store-to-load forwarding of all values is not possible.
	      array = orig_array;
	      break;
	    }
	  else if (!may_alias(addr, store_addr))
	    array = store_array;
	  else
	    break;
	}
      else if (array->op == Op::MEM_INDEF_ARRAY)
	{
	  value = value_inst(0, 8);
	  break;
	}
      else if (array->op == Op::MEM_FLAG_ARRAY)
	{
	  value = value_inst(0, 1);
	  break;
	}
      else if (array->op == Op::MEM_SIZE_ARRAY)
	{
	  value = value_inst(0, array->bb->func->module->ptr_offset_bits);
	  break;
	}
      else if (array->op == Op::SIMP_BARRIER)
	array = array->args[0];
      else
	break;
    }
  cache.insert({orig_array, {value, array}});
  return {value, array};
}

Inst *Converter::strip_local_mem(Inst *array, std::map<Inst *, Inst *>& cache)
{
  auto I = cache.find(array);
  if (I != cache.end())
    return I->second;

  Inst *orig_array = array;
  for (;;)
    {
      if (array->op == Op::ITE)
	{
	  Inst *cond = array->args[0];
	  Inst *arg2 = array->args[1];
	  Inst *arg3 = array->args[2];
	  Inst *array2 = strip_local_mem(arg2, cache);
	  Inst *array3 = strip_local_mem(arg3, cache);
	  if (array2 != arg2 || array3 != arg3)
	    array = build_inst(Op::ITE, cond, array2, array3);
	  break;
	}
      else if (array->op == Op::ARRAY_STORE
	       || array->op == Op::ARRAY_SET_FLAG
	       || array->op == Op::ARRAY_SET_INDEF)
	{
	  Inst *store_array = array->args[0];
	  Inst *store_addr = array->args[1];
	  if (store_addr->op == Op::VALUE)
	    {
	      uint64_t id = store_addr->value() >> module->ptr_id_low;
	      bool is_local = (id >> (module->ptr_id_bits - 1)) != 0;
	      if (is_local)
		{
		  array = store_array;
		  continue;
		}
	    }
	  break;
	}
      else if (array->op == Op::ARRAY_SET_SIZE)
	{
	  Inst *set_size_array = array->args[0];
	  Inst *set_size_id = array->args[1];
	  if (set_size_id->op == Op::VALUE)
	    {
	      uint64_t id = set_size_id->value();
	      bool is_local = (id >> (module->ptr_id_bits - 1)) != 0;
	      if (is_local)
		{
		  array = set_size_array;
		  continue;
		}
	    }
	  break;
	}
      else if (array->op == Op::SIMP_BARRIER)
	array = array->args[0];
      else
	break;
    }
  cache.insert({orig_array, array});
  return array;
}

bool Converter::may_alias(Inst *p1, Inst *p2)
{
  if (p1 != p2 && p1->op == Op::VALUE && p2->op == Op::VALUE)
    return false;

  // p + const1 cannot alias p + const2 if the constants differ.
  if (p1->op == Op::ADD && p2->op == Op::ADD
      && p1->args[0] == p2->args[0]
      && p1->args[1]->op == Op::VALUE
      && p2->args[1]->op == Op::VALUE
      && p1->args[1] != p2->args[1])
    return false;

  // p cannot alias p + const if const != 0.
  if (p1->op == Op::ADD
      && p1->args[0] == p2
      && p1->args[1]->op == Op::VALUE
      && p1->args[1]->value() != 0)
    return false;
  if (p2->op == Op::ADD
      && p2->args[0] == p1
      && p2->args[1]->op == Op::VALUE
      && p2->args[1]->value() != 0)
    return false;

  return true;
}

void Converter::convert(Basic_block *bb, Inst *inst, Function_role role)
{
  Inst *new_inst = nullptr;
  if (inst->op == Op::VALUE)
    {
      new_inst = value_inst(inst->value(), inst->bitsize);
    }
  else if (inst->op == Op::LOAD)
    {
      std::map<Inst *, std::pair<Inst *, Inst *>> cache;
      Inst *array = bb2memory.at(bb);
      Inst *addr = translate.at(inst->args[0]);
      std::tie(new_inst, array) = simplify_array_access(array, addr, cache);
      if (!new_inst)
	new_inst = build_inst(Op::ARRAY_LOAD, array, addr);
    }
  else if (inst->op == Op::STORE)
    {
      Inst *array = bb2memory.at(bb);
      Inst *addr = translate.at(inst->args[0]);
      Inst *value = translate.at(inst->args[1]);
      array = build_inst(Op::ARRAY_STORE, array, addr, value);
      bb2memory[bb] = array;
      return;
    }
  else if (inst->op == Op::UB)
    {
      add_ub(bb, translate.at(inst->args[0]));
      return;
    }
  else if (inst->op == Op::ASSERT)
    {
      add_assert(bb, translate.at(inst->args[0]));
      return;
    }
  else if (inst->op == Op::SET_MEM_FLAG)
    {
      Inst *array = bb2memory_flag.at(bb);
      Inst *addr = translate.at(inst->args[0]);
      Inst *value = translate.at(inst->args[1]);
      array = build_inst(Op::ARRAY_SET_FLAG, array, addr, value);
      bb2memory_flag[bb] = array;
      return;
    }
  else if (inst->op == Op::SET_MEM_INDEF)
    {
      Inst *array = bb2memory_indef.at(bb);
      Inst *addr = translate.at(inst->args[0]);
      Inst *value = translate.at(inst->args[1]);
      array = build_inst(Op::ARRAY_SET_INDEF, array, addr, value);
      bb2memory_indef[bb] = array;
      return;
    }
  else if (inst->op == Op::FREE)
    {
      Inst *array = bb2memory_size.at(bb);
      Inst *arg1 = translate.at(inst->args[0]);
      Inst *zero = value_inst(0, module->ptr_offset_bits);
      array = build_inst(Op::ARRAY_SET_SIZE, array, arg1, zero);
      bb2memory_size[bb] = array;
      return;
    }
   else if (inst->op == Op::GET_MEM_INDEF)
     {
       std::map<Inst *, std::pair<Inst *, Inst *>> cache;
       Inst *array = bb2memory_indef.at(bb);
       Inst *addr = translate.at(inst->args[0]);
       std::tie(new_inst, array) = simplify_array_access(array, addr, cache);
       if (!new_inst)
	 new_inst = build_inst(Op::ARRAY_GET_INDEF, array, addr);
     }
   else if (inst->op == Op::GET_MEM_FLAG)
     {
       std::map<Inst *, std::pair<Inst *, Inst *>> cache;
       Inst *array = bb2memory_flag.at(bb);
       Inst *addr = translate.at(inst->args[0]);
       std::tie(new_inst, array) = simplify_array_access(array, addr, cache);
       if (!new_inst)
	 new_inst = build_inst(Op::ARRAY_GET_FLAG, array, addr);
     }
   else if (inst->op == Op::GET_MEM_SIZE)
    {
       std::map<Inst *, std::pair<Inst *, Inst *>> cache;
       Inst *array = bb2memory_size.at(bb);
       Inst *addr = translate.at(inst->args[0]);
       std::tie(new_inst, array) = simplify_array_access(array, addr, cache);
       if (!new_inst)
	 new_inst = build_inst(Op::ARRAY_GET_SIZE, array, addr);
    }
  else if (inst->op == Op::IS_CONST_MEM)
    {
      Inst *arg1 = translate.at(inst->args[0]);
      Inst *is_const = value_inst(0, 1);
      for (Inst *id : const_ids)
	{
	  is_const = build_inst(Op::OR, is_const, build_inst(Op::EQ, arg1, id));
	}
      new_inst = is_const;
    }
  else if (inst->op == Op::MEMORY)
    {
      // All uses of MEMORY should have been changed to constants.
      assert(inst->used_by.empty());

      uint32_t ptr_bits = module->ptr_bits;
      uint32_t ptr_offset_bits = module->ptr_offset_bits;
      uint32_t ptr_id_bits = module->ptr_id_bits;
      uint64_t id = inst->args[0]->value();
      uint64_t ptr_val = id << module->ptr_id_low;
      uint64_t size_val = inst->args[1]->value();
      Inst *mem_id = value_inst(id, ptr_id_bits);
      Inst *size = value_inst(size_val, ptr_offset_bits);
      Inst *size_array = bb2memory_size.at(bb);
      size_array = build_inst(Op::ARRAY_SET_SIZE, size_array, mem_id, size);
      bb2memory_size[bb] = size_array;

      uint32_t flags = inst->args[2]->value();
      if (flags & MEM_CONST)
	const_ids.push_back(translate.at(inst->args[0]));

      if (flags & MEM_UNINIT)
	{
	  Inst *indef_array = bb2memory_indef.at(bb);
	  Inst *byte = value_inst(255, 8);
	  for (uint64_t i = 0; i < size_val; i++)
	    {
	      Inst *ptr = value_inst(ptr_val + i, ptr_bits);
	      indef_array = build_inst(Op::ARRAY_SET_INDEF,
				       indef_array, ptr, byte);
	    }
	  bb2memory_indef[bb] = indef_array;
	}

      return;
    }
  else if (inst->op == Op::BR)
    {
      return;
    }
  else if (inst->op == Op::RET)
    {
      if (inst->nof_args > 0)
	{
	  Inst *arg1 = translate.at(inst->args[0]);
	  Inst *arg2;
	  if (inst->nof_args > 1)
	    arg2 = translate.at(inst->args[1]);
	  else
	    arg2 = value_inst(0, inst->args[0]->bitsize);
	  Op op = role == Function_role::src ? Op::SRC_RETVAL : Op::TGT_RETVAL;
	  build_inst(op, arg1, arg2);
	  if (role == Function_role::src)
	    {
	      src_retval = arg1;
	      src_retval_indef = arg2;
	    }
	  else
	    {
	      tgt_retval = arg1;
	      tgt_retval_indef = arg2;
	    }
	}
      return;
    }
  else if (inst->op == Op::EXIT)
    {
      Inst *arg1 = translate.at(inst->args[0]);
      Inst *arg2 = translate.at(inst->args[1]);
      Inst *arg3 = translate.at(inst->args[2]);
      if (!is_value_zero(arg1)
	  || !is_value_zero(arg2)
	  || !is_value_zero(arg3))
	{
	  if (role == Function_role::src)
	    {
	      assert(!src_abort);
	      assert(!src_exit);
	      assert(!src_exit_val);
	      src_abort = arg1;
	      src_exit = arg2;
	      src_exit_val = arg3;
	      build_inst(Op::SRC_EXIT, arg1, arg2, arg3);
	    }
	  else
	    {
	      assert(!tgt_abort);
	      assert(!tgt_exit);
	      assert(!tgt_exit_val);
	      tgt_abort = arg1;
	      tgt_exit = arg2;
	      tgt_exit_val = arg3;
	      build_inst(Op::TGT_EXIT, arg1, arg2, arg3);
	    }
	}
      return;
    }
  else
    {
      assert(inst->op == Op::PRINT || inst->has_lhs());
      Inst_class iclass = inst->iclass();
      switch (iclass)
	{
	case Inst_class::mem_nullary:
	  new_inst = build_inst(inst->op);
	  break;
	case Inst_class::iunary:
	case Inst_class::funary:
	case Inst_class::reg_unary:
	  {
	    Inst *arg = translate.at(inst->args[0]);
	    new_inst = build_inst(inst->op, arg);
	  }
	  break;
	case Inst_class::icomparison:
	case Inst_class::fcomparison:
	case Inst_class::ibinary:
	case Inst_class::fbinary:
	case Inst_class::conv:
	case Inst_class::reg_binary:
	case Inst_class::solver_binary:
	  {
	    Inst *arg1 = translate.at(inst->args[0]);
	    Inst *arg2 = translate.at(inst->args[1]);
	    new_inst = build_inst(inst->op, arg1, arg2);
	  }
	  break;
	case Inst_class::ternary:
	  {
	    Inst *arg1 = translate.at(inst->args[0]);
	    Inst *arg2 = translate.at(inst->args[1]);
	    Inst *arg3 = translate.at(inst->args[2]);
	    new_inst = build_inst(inst->op, arg1, arg2, arg3);
	  }
	  break;
	default:
	  throw Not_implemented("Converter::duplicate: "s + inst->name());
	}
    }
  assert(new_inst);
  translate.insert({inst, new_inst});
}

bool need_phi_barrier(Inst *phi, Inst *phi_inst)
{
  if (phi->phi_args.size() <= 2)
    return false;

  // No need for a barrier if most arguments are constants.
  int nof_nonconst_arg = 0;
  for (auto [inst, _] : phi->phi_args)
    {
      if (inst->op != Op::VALUE)
	nof_nonconst_arg++;
    }
  if (nof_nonconst_arg <= 1)
    return false;

  // Simplification after expanding the phi-node may reduce the result
  // so that we no longer need a barrier, even if the original phi-node
  // looked too complex.
  if (phi_inst->op != Op::ITE)
    return false;
  if (phi_inst->op == Op::ITE
      && phi_inst->args[1]->op != Op::ITE
      && phi_inst->args[2]->op != Op::ITE)
    return false;

  return true;
}

void Converter::convert_function(Function *func, Function_role role)
{
  calculate_dominance(func);

  // Create true and false to give them canonical order when sorted by ID.
  value_inst(0, 1);
  value_inst(1, 1);

  for (auto bb : func->bbs)
    {
      if (bb == func->bbs[0])
	{
	  bb2cond.insert({bb, value_inst(1, 1)});
	  bb2memory.insert({bb, memory});
	  bb2memory_size.insert({bb, memory_size});
	  bb2memory_flag.insert({bb, memory_flag});
	  bb2memory_indef.insert({bb, memory_indef});
	}
      else
	{
	  generate_bb2cond(bb);
	  build_mem_state(bb, bb2memory);
	  build_mem_state(bb, bb2memory_size);
	  build_mem_state(bb, bb2memory_flag);
	  build_mem_state(bb, bb2memory_indef);
	}

      for (auto phi : bb->phis)
	{
	  auto pred2inst =
	    [this, phi](Basic_block *pred)
	    {
	      return translate.at(phi->get_phi_arg(pred));
	    };

	  Inst *phi_inst = build_phi_ite(bb, pred2inst);
	  // simplify_inst may move instructions over Op::ITE. This is
	  // not always a good idea for the chains of ITE we generate
	  // here, as it may generate a large number of extra instructions
	  // for phi nodes with many arguments and/or many users.
	  // We therefore add a barrier to prevent this optimization for
	  // large phi nodes.
	  // TODO: Tune this to better detect when the optimization is
	  // helpful and avoid adding the barrier in those cases.
	  if (need_phi_barrier(phi, phi_inst))
	    phi_inst = build_inst(Op::SIMP_BARRIER, phi_inst);
	  translate.insert({phi, phi_inst});
	}

      for (Inst *inst = bb->first_inst; inst; inst = inst->next)
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
  Op mem_op = role == Function_role::src ? Op::SRC_MEM : Op::TGT_MEM;
  Inst *memory = bb2memory.at(exit_block);
  Inst *memory_size = bb2memory_size.at(exit_block);
  Inst *memory_indef = bb2memory_indef.at(exit_block);
  {
    std::map<Inst *, Inst *> cache;
    memory = strip_local_mem(memory, cache);
  }
  {
    std::map<Inst *, Inst *> cache;
    memory_size = strip_local_mem(memory_size, cache);
  }
  {
    std::map<Inst *, Inst *> cache;
    memory_indef = strip_local_mem(memory_indef, cache);
  }
  build_inst(mem_op, memory, memory_size, memory_indef);
  if (role == Function_role::src)
    {
      src_memory = memory;
      src_memory_size = memory_size;
      src_memory_indef = memory_indef;
    }
  else
    {
      tgt_memory = memory;
      tgt_memory_size = memory_size;
      tgt_memory_indef = memory_indef;
    }

  // Dominance information can be extensive for large functions, and it is
  // no longer needed.
  clear_dominance(func);

  // Clear the arrays. This is needed for check_refine to get a clean slate
  // when converting the second function. But it also reduces memory usage
  // for the other use cases.
  bb2cond.clear();
  bb2ub.clear();
  bb2not_assert.clear();
  bb2memory.clear();
  bb2memory_size.clear();
  bb2memory_flag.clear();
  bb2memory_indef.clear();
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
  if ((src_common_ub->op == Op::VALUE && src_common_ub->value() != 0)
      || (src_unique_ub->op == Op::VALUE && src_unique_ub->value() != 0))
    return false;

  if (src_abort != tgt_abort
      || src_exit != tgt_exit
      || src_exit_val != tgt_exit_val)
    return true;

  if (src_retval != tgt_retval
      || src_retval_indef != tgt_retval_indef)
    return true;

  if (src_memory != tgt_memory
      || src_memory_size != tgt_memory_size
      || src_memory_indef != tgt_memory_indef)
    return true;

  assert(src_common_ub == tgt_common_ub);
  if (src_unique_ub != tgt_unique_ub
      && !(tgt_unique_ub->op == Op::VALUE && tgt_unique_ub->value() == 0))
    return true;

  return false;
}

bool identical(Inst *inst1, Inst *inst2)
{
  if (inst1->op != inst2->op)
    return false;
  if (inst1->bitsize != inst2->bitsize)
    return false;
  if (inst1->nof_args != inst2->nof_args)
    return false;
  if (inst1->is_commutative())
    {
      // Some passes, like ccp, may perform pointless argument swaps.
      assert(inst1->nof_args == 2);
      int nbr1_0 = inst1->args[0]->id;
      int nbr1_1 = inst1->args[1]->id;
      int nbr2_0 = inst2->args[0]->id;
      int nbr2_1 = inst2->args[1]->id;
      if (!((nbr1_0 == nbr2_0 && nbr1_1 == nbr2_1)
	    || (nbr1_0 == nbr2_1 && nbr1_1 == nbr2_0)))
	return false;
    }
  else
    for (size_t i = 0; i < inst1->nof_args; i++)
      {
	if (inst1->args[i]->id != inst2->args[i]->id)
	  return false;
      }

  // The normal instructions are fully checked by the preceding code,
  // but instructions of class "special" require additional checks.
  switch (inst1->op)
    {
    case Op::BR:
      if (inst1->nof_args == 0)
	{
	  if (inst1->u.br1.dest_bb->id != inst2->u.br1.dest_bb->id)
	    return false;
	}
      else
	{
	  if (inst1->u.br3.true_bb->id != inst2->u.br3.true_bb->id)
	    return false;
	  if (inst1->u.br3.false_bb->id != inst2->u.br3.false_bb->id)
	    return false;
	}
      break;
    case Op::PHI:
      if (inst1->phi_args.size() != inst2->phi_args.size())
	return false;
      for (size_t i = 0; i < inst1->phi_args.size(); i++)
	{
	  Phi_arg arg1 = inst1->phi_args[i];
	  Phi_arg arg2 = inst2->phi_args[i];
	  if (arg1.inst->id != arg2.inst->id)
	    return false;
	  if (arg1.bb->id != arg2.bb->id)
	    return false;
	}
      break;
    case Op::RET:
      // This is already checked by the argument check above.
      break;
    case Op::VALUE:
      if (inst1->value() != inst2->value())
	return false;
      break;
    default:
      // If this is an instruction of class "special", then there is a missing
      // case in this switch statement.
      assert(inst1->iclass() != Inst_class::special);
      break;
    }

  return true;
}

} // end anonymous namespace

bool identical(Function *func1, Function *func2)
{
  if (func1->bbs.size() != func2->bbs.size())
    return false;

  for (size_t i = 0; i < func1->bbs.size(); i++)
    {
      Basic_block *bb1 = func1->bbs[i];
      Basic_block *bb2 = func2->bbs[i];
      if (bb1->phis.size() != bb2->phis.size())
	return false;
      for (size_t j = 0; j < bb1->phis.size(); j++)
	{
	  Inst *phi1 = bb1->phis[j];
	  Inst *phi2 = bb2->phis[j];
	  if (!identical(phi1, phi2))
	    return false;
	}
      Inst *inst1 = bb1->first_inst;
      Inst *inst2 = bb2->first_inst;
      while (inst1 && inst2)
	{
	  if (!identical(inst1, inst2))
	    return false;
	  inst1 = inst1->next;
	  inst2 = inst2->next;
	}
      if (inst1 || inst2)
	return false;
    }

  return true;
}

Solver_result check_refine(Module *module, bool run_simplify_inst)
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

  src->canonicalize();
  tgt->canonicalize();
  if (identical(src, tgt))
    return {};

  Converter converter(module, run_simplify_inst);
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

  Cache cache(converter.dest_func);
  if (std::optional<Solver_result> result_cache = cache.get())
    {
      if (config.verbose > 0)
	fprintf(stderr, "SMTGCC: Using cached result\n");
      return *result_cache;
    }

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

  cache.set(result);

  if (config.verbose > 0)
    {
      if (!stats.cvc5.skipped || !stats.z3.skipped)
	{
	  fprintf(stderr, "SMTGCC: time: ");
	  for (int i = 0; i < 4; i++)
	    {
	      fprintf(stderr, "%s%" PRIu64, i ? "," : "", stats.cvc5.time[i]);
	    }
	  for (int i = 0; i < 4; i++)
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

  func->canonicalize();

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

  func->canonicalize();

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
