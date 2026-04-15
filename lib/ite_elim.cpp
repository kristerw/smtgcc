// This pass simplifies a sequence of two Op::ITEs with redundant conditions
// and unrelated operations in between. For example:
//
//   %1 = ite %a, %b, %c
//   %2 = add %1, %d
//   %3 = ite %a, %2, %e
//
// is simplified to:
//
//   %2 = add %b, %d
//   %3 = ite %a, %2, %e
//
#include <bitset>
#include <cassert>
#include <optional>

#include "smtgcc.h"

namespace smtgcc {

namespace {

// The maximum number of different Op::ITE conditions.
const int nof_cond = 256;

// Max recursion depth for specialize_cond.
const int max_depth = 10;

struct Ite_elim
{
  Ite_elim(Function *func)
    : func{func}
  {}
  std::optional<bool> specialize_cond_true(Inst *inst, Inst *cond, int depth = 0);
  std::optional<bool> specialize_cond_false(Inst *inst, Inst *cond, int depth = 0);
  std::optional<bool> specialize_cond_true(Inst *inst, std::bitset<nof_cond>& set);
  std::optional<bool> specialize_cond_false(Inst *inst, std::bitset<nof_cond>& set);
  bool handle_arg_ite(Inst *inst, size_t idx);
  bool handle_arg_bool(Inst *inst, size_t idx);
  void handle_arg(Inst *inst, size_t idx);
  void propagate_from_uses(Inst *inst);
  bool run();

  Function *func;
  std::map<Inst*, std::bitset<nof_cond>> used_true;
  std::map<Inst*, std::bitset<nof_cond>> used_false;
  std::map<Inst*, int> inst2bit_idx;
  std::map<int, Inst*> bit_idx2inst;
  int next_bit_idx = 0;
  bool changed = false;
};

void flatten_and(Inst *inst, std::vector<Inst *>& elems)
{
  if (inst->op == Op::AND)
    {
      flatten_and(inst->args[0], elems);
      flatten_and(inst->args[1], elems);
    }
  else
    elems.push_back(inst);
}

// Return the value of inst provided we know that cond evaluates
// to true.
//
// For example, an inst representing x == 0 is false if we know that
// cond x > 4 evaluates to true.
std::optional<bool> Ite_elim::specialize_cond_true(Inst *inst, Inst *cond, int depth)
{
  if (depth > max_depth)
    return {};

  if (inst->op == Op::VALUE)
    return inst->value();

  if (inst == cond)
    return true;
  if (inst->op == Op::NOT)
    {
      std::optional<bool> value =
	specialize_cond_true(inst->args[0], cond, depth + 1);
      if (value)
	return !*value;
    }
  if (cond->op == Op::NOT)
    {
      std::optional<bool> value =
	specialize_cond_false(inst, cond->args[0], depth + 1);
      if (value)
	return value;
    }

  if (inst->op == Op::AND)
    {
      std::vector<Inst*> conds;
      flatten_and(inst, conds);
      for (auto cnd : conds)
	{
	  std::optional<bool> value =
	    specialize_cond_true(cnd, cond, depth + 1);
	  if (!value)
	    return {};
	  if (!*value)
	    return false;
	}
      return true;
    }

  if (cond->op == Op::AND)
    {
      std::vector<Inst*> conds;
      flatten_and(cond, conds);
      for (auto cnd : conds)
	{
	  std::optional<bool> value =
	    specialize_cond_true(inst, cnd, depth + 1);
	  if (value)
	    return value;
	}
    }

  if (cond->op == Op::ULT && cond->args[0]->op == Op::VALUE)
    {
      if (inst->op == Op::ULT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[1])
	{
	  if (inst->args[0]->value() <= cond->args[0]->value())
	    return true;
	}
      if (inst->op == Op::ULT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[1]
	  && !is_value_zero(inst->args[1]))
	{
	  if (inst->args[1]->value() - 1 <= cond->args[0]->value())
	    return false;
	}
      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[1])
	{
	  if (inst->args[1]->value() <= cond->args[0]->value())
	    return false;
	}
    }
  if (cond->op == Op::ULT && cond->args[1]->op == Op::VALUE)
    {
      if (inst->op == Op::ULT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[0]
	  && !is_value_zero(cond->args[1]))
	{
	  if (inst->args[0]->value() >= cond->args[1]->value() - 1)
	    return false;
	}
      if (inst->op == Op::ULT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	{
	  if (inst->args[1]->value() >= cond->args[1]->value())
	    return true;
	}
      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	{
	  if (inst->args[1]->value() >= cond->args[1]->value())
	    return false;
	}
    }

  if (cond->op == Op::SLT && cond->args[0]->op == Op::VALUE)
    {
      if (inst->op == Op::SLT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[1])
	{
	  if (inst->args[0]->signed_value() <= cond->args[0]->signed_value())
	    return true;
	}
      if (inst->op == Op::SLT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[1]
	  && !is_value_signed_min(inst->args[1]))
	{
	  if (inst->args[1]->signed_value() - 1 <= cond->args[0]->signed_value())
	    return false;
	}
      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[1])
	{
	  if (inst->args[1]->signed_value() <= cond->args[0]->signed_value())
	    return false;
	}
    }
  if (cond->op == Op::SLT && cond->args[1]->op == Op::VALUE)
    {
      if (inst->op == Op::SLT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[0]
	  && !is_value_signed_min(cond->args[1]))
	{
	  if (inst->args[0]->signed_value() >= cond->args[1]->signed_value() - 1)
	    return false;
	}
      if (inst->op == Op::SLT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	{
	  if (inst->args[1]->signed_value() >= cond->args[1]->signed_value())
	    return true;
	}
      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	{
	  if (inst->args[1]->signed_value() >= cond->args[1]->signed_value())
	    return false;
	}
    }

  if (cond->op == Op::EQ && cond->args[1]->op == Op::VALUE)
    {
      if (inst->op == Op::ULT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[0])
	return inst->args[0]->value() < cond->args[1]->value();
      if (inst->op == Op::ULT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	return inst->args[1]->value() > cond->args[1]->value();

      if (inst->op == Op::SLT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[0])
	return inst->args[0]->signed_value() < cond->args[1]->signed_value();
      if (inst->op == Op::SLT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	return inst->args[1]->signed_value() > cond->args[1]->signed_value();

      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	return inst->args[1]->value() == cond->args[1]->value();
    }

  return {};
}

// Return the value of inst provided we know that cond evaluates
// to false.
//
// For example, an inst representing x < 7 is true if we know that
// cond x > 4 evaluates to false.
std::optional<bool> Ite_elim::specialize_cond_false(Inst *inst, Inst *cond, int depth)
{
  if (depth > max_depth)
    return {};

  if (inst->op == Op::VALUE)
    return inst->value();

  if (inst == cond)
    return false;
  if (inst->op == Op::NOT)
    {
      std::optional<bool> value =
	specialize_cond_false(inst->args[0], cond, depth + 1);
      if (value)
	return !*value;
    }
  if (cond->op == Op::NOT)
    {
      std::optional<bool> value =
	specialize_cond_true(inst, cond->args[0], depth + 1);
      if (value)
	return value;
    }

  if (inst->op == Op::AND)
    {
      std::vector<Inst*> conds;
      flatten_and(inst, conds);
      for (auto cnd : conds)
	{
	  std::optional<bool> value =
	    specialize_cond_false(cnd, cond, depth + 1);
	  if (!value)
	    return {};
	  if (!*value)
	    return false;
	}
      return true;
    }

  if (cond->op == Op::ULT && cond->args[0]->op == Op::VALUE)
    {
      if (inst->op == Op::ULT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[1])
	{
	  if (inst->args[0]->value() >= cond->args[0]->value())
	    return false;
	}
      if (inst->op == Op::ULT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[1])
	{
	  if (inst->args[1]->value() > cond->args[0]->value())
	    return true;
	}
      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[1])
	{
	  if (inst->args[1]->value() > cond->args[0]->value())
	    return false;
	}
    }
  if (cond->op == Op::ULT && cond->args[1]->op == Op::VALUE)
    {
      if (inst->op == Op::ULT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[0])
	{
	  if (inst->args[0]->value() < cond->args[1]->value())
	    return true;
	}
      if (inst->op == Op::ULT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	{
	  if (inst->args[1]->value() <= cond->args[1]->value())
	    return false;
	}
      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	{
	  if (inst->args[1]->value() < cond->args[1]->value())
	    return false;
	}
    }

  if (cond->op == Op::SLT && cond->args[0]->op == Op::VALUE)
    {
      if (inst->op == Op::SLT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[1])
	{
	  if (inst->args[0]->signed_value() >= cond->args[0]->signed_value())
	    return false;
	}
      if (inst->op == Op::SLT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[1])
	{
	  if (inst->args[1]->signed_value() > cond->args[0]->signed_value())
	    return true;
	}
      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[1])
	{
	  if (inst->args[1]->signed_value() > cond->args[0]->signed_value())
	    return false;
	}
    }
  if (cond->op == Op::SLT && cond->args[1]->op == Op::VALUE)
    {
      if (inst->op == Op::SLT
	  && inst->args[0]->op == Op::VALUE
	  && inst->args[1] == cond->args[0])
	{
	  if (inst->args[0]->signed_value() < cond->args[1]->signed_value())
	    return true;
	}
      if (inst->op == Op::SLT
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	{
	  if (inst->args[1]->signed_value() <= cond->args[1]->signed_value())
	    return false;
	}
      if (inst->op == Op::EQ
	  && inst->args[1]->op == Op::VALUE
	  && inst->args[0] == cond->args[0])
	{
	  if (inst->args[1]->signed_value() < cond->args[1]->signed_value())
	    return false;
	}
    }

  return {};
}

std::optional<bool> Ite_elim::specialize_cond_true(Inst *inst, std::bitset<nof_cond>& set)
{
  for (int i = 0; i < next_bit_idx; i++)
    {
      if (set[i])
	{
	  Inst *cond = bit_idx2inst.at(i);
	  std::optional<bool> value = specialize_cond_true(inst, cond);
	  if (value)
	    return value;
	}
    }
  return {};
}

std::optional<bool> Ite_elim::specialize_cond_false(Inst *inst, std::bitset<nof_cond>& set)
{
  for (int i = 0; i < next_bit_idx; i++)
    {
      if (set[i])
	{
	  Inst *cond = bit_idx2inst.at(i);
	  std::optional<bool> value = specialize_cond_false(inst, cond);
	  if (value)
	    return value;
	}
    }
  return {};
}

bool Ite_elim::handle_arg_ite(Inst *inst, size_t idx)
{
  Inst *arg = inst->args[idx];
  Inst *cond = arg->args[0];
  std::optional<bool> cond_value;
  if (used_true.contains(inst))
    cond_value = specialize_cond_true(cond, used_true[inst]);
  if (!cond_value && used_false.contains(inst))
    cond_value = specialize_cond_false(cond, used_false[inst]);
  if (cond_value)
    {
      if (*cond_value)
	inst->update_arg(idx, arg->args[1]);
      else
	inst->update_arg(idx, arg->args[2]);
      return true;
    }
  return false;
}

bool Ite_elim::handle_arg_bool(Inst *inst, size_t idx)
{
  Inst *cond = inst->args[idx];
  if (cond->op == Op::VALUE)
    return false;
  std::optional<bool> cond_value;
  if (used_true.contains(inst))
    cond_value = specialize_cond_true(cond, used_true[inst]);
  if (!cond_value && used_false.contains(inst))
    cond_value = specialize_cond_false(cond, used_false[inst]);
  if (cond_value)
    {
      inst->update_arg(idx, inst->bb->value_inst(*cond_value, 1));
      return true;
    }
  return false;
}

void Ite_elim::handle_arg(Inst *inst, size_t idx)
{
  bool modified;
  do {
    modified = false;
    Inst *arg = inst->args[idx];
    if (arg->op == Op::ITE)
      modified = handle_arg_ite(inst, idx);
    else if (arg->bitsize == 1)
      modified = handle_arg_bool(inst, idx);
    changed |= modified;
  } while(modified);
}

void Ite_elim::propagate_from_uses(Inst *inst)
{
  if (inst->used_by.empty())
    return;

  std::bitset<nof_cond> true_conds;
  true_conds.set();
  std::bitset<nof_cond> false_conds;
  false_conds.set();
  for (auto use : inst->used_by)
    {
      // If the use is a value in a true/false branch of an Op::ITE, then
      // we add the Op::ITE condition to the resulting condition bitset.
      if (use->op == Op::ITE
	  && inst != use->args[0]
	  && use->args[1] != use->args[2]
	  && inst2bit_idx.contains(use->args[0]))
	{
	  std::bitset<nof_cond> use_true;
	  if (used_true.contains(use))
	    use_true = used_true[use];
	  std::bitset<nof_cond> use_false;
	  if (used_false.contains(use))
	    use_false = used_false[use];
	  int bit_idx = inst2bit_idx.at(use->args[0]);
	  if (inst == use->args[1])
	    {
	      use_true.set(bit_idx);
	      use_false.reset(bit_idx);
	    }
	  else
	    {
	      use_true.reset(bit_idx);
	      use_false.set(bit_idx);
	    }
	  true_conds &= use_true;
	  false_conds &= use_false;
	}
      else
	{
	  if (!used_true.contains(use))
	    true_conds.reset();
	  else
	    true_conds &= used_true[use];
	  if (!used_false.contains(use))
	    false_conds.reset();
	  else
	    false_conds &= used_false[use];
	}
    }

  if (true_conds.any())
    used_true.emplace(inst, true_conds);
  if (false_conds.any())
    used_false.emplace(inst, false_conds);
}

bool Ite_elim::run()
{
  assert(func->bbs.size() == 1);
  Basic_block *bb = func->bbs[0];

  for (Inst *inst = bb->last_inst; inst;)
    {
      if (inst->op == Op::VALUE)
	break;

      if (inst->has_lhs() && inst->used_by.empty())
	{
	  Inst *orig_inst = inst;
	  inst = inst->prev;
	  destroy_instruction(orig_inst);
	  continue;
	}

      // Create the condition set for this instruction based on its uses.
      propagate_from_uses(inst);

      // Propagate the values from arguments based on the condition set.
      for (size_t i = 0; i < inst->nof_args; i++)
	{
	  handle_arg(inst, i);
	}

      // Ensure Op::ITE instructions are in a canonical form after
      // argument propagation.
      if (inst->op == Op::ITE)
	{
	  if (is_value_zero(inst->args[0]))
	    {
	      inst->replace_all_uses_with(inst->args[2]);
	      continue;
	    }
	  if (is_value_one(inst->args[0]))
	    {
	      inst->replace_all_uses_with(inst->args[1]);
	      continue;
	    }
	  if (inst->args[1] == inst->args[2])
	    {
	      inst->replace_all_uses_with(inst->args[1]);
	      continue;
	    }
	}

      // Ensure Op::ITE conditions have a bit in the condition bit set.
      if (inst->op == Op::ITE
	  && next_bit_idx < nof_cond
	  && !inst2bit_idx.contains(inst->args[0]))
	{
	  inst2bit_idx.emplace(inst->args[0], next_bit_idx);
	  bit_idx2inst.emplace(next_bit_idx, inst->args[0]);
	  next_bit_idx++;
	}

      inst = inst->prev;
    }

  return changed;
}

} // end anonymous namespace

bool ite_elim(Function *func)
{
  Ite_elim pass(func);
  return pass.run();
}

bool ite_elim(Module *module)
{
  bool changed = false;
  for (auto func : module->functions)
    changed |= ite_elim(func);
  return changed;
}

} // end namespace smtgcc
