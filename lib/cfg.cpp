#include <algorithm>
#include <cassert>

#include "smtgcc.h"

namespace smtgcc {
namespace {

void dfs_walk(Basic_block *bb, std::vector<Basic_block *>& bbs, std::set<Basic_block *>& visited)
{
  visited.insert(bb);
  for (auto succ : bb->succs)
    {
      if (!visited.contains(succ))
	dfs_walk(succ, bbs, visited);
    }
  bbs.push_back(bb);
}

void inverse_dfs_walk(Basic_block *bb, std::vector<Basic_block *>& bbs, std::set<Basic_block *>& visited)
{
  visited.insert(bb);
  for (auto pred : bb->preds)
    {
      if (!visited.contains(pred))
	inverse_dfs_walk(pred, bbs, visited);
    }
  bbs.push_back(bb);
}

void remove_dead_bbs(std::vector<Basic_block *>& dead_bbs)
{
  for (auto bb : dead_bbs)
    {
      for (auto succ : bb->succs)
	{
	  for (auto phi : succ->phis)
	    {
	      phi->remove_phi_arg(bb);
	    }
	}
    }

  for (auto bb : dead_bbs)
    {
      for (Inst *phi : bb->phis)
	{
	  phi->remove_phi_args();
	}
    }

  // We must remove instructions in reverse post order, but it is
  // not guaranteed that the dead BBs are in RPO, so we remove the
  // instructions we can remove, and iterate over the BBs until all
  // are removed.
  while (!dead_bbs.empty())
    {
      for (int i = dead_bbs.size() - 1; i >= 0; i--)
	{
	  Basic_block *bb = dead_bbs[i];
	  for (Inst *inst = bb->last_inst; inst;)
	    {
	      Inst *next_inst = inst->prev;
	      if (inst->used_by.empty())
		destroy_instruction(inst);
	      inst = next_inst;
	    }
	}
      while (!dead_bbs.empty())
	{
	  Basic_block *bb = dead_bbs.back();
	  if (bb->last_inst)
	    break;
	  dead_bbs.pop_back();
	  destroy_basic_block(bb);
	}
    }
}

// Remove empty BBs ending in unconditional branch by letting the
// predecessors call the successor.
void remove_empty_bb(Basic_block *bb)
{
  if (bb->first_inst->op != Op::BR
      || bb->first_inst->nof_args != 0
      || bb->phis.size() != 0)
    return;

  Basic_block *dest_bb = bb->succs[0];
  if (bb == dest_bb)
    return;

  if (!dest_bb->phis.empty())
    {
      // We cannot remove this BB if any predecessor already branches to
      // the destination, as that would result in duplicated entries in
      // the phi node.
      for (auto pred : bb->preds)
	{
	  auto it = std::find(pred->succs.begin(), pred->succs.end(), dest_bb);
	  if (it != pred->succs.end())
	    return;
	}
    }

  for (auto phi : dest_bb->phis)
    {
      Inst *inst = phi->get_phi_arg(bb);
      phi->remove_phi_arg(bb);
      for (auto pred : bb->preds)
	{
	  phi->add_phi_arg(inst, pred);
	}
    }

  while (!bb->preds.empty())
    {
      Basic_block *pred = bb->preds.back();
      assert(pred->last_inst->op == Op::BR);
      if (pred->last_inst->nof_args == 0)
	{
	  destroy_instruction(pred->last_inst);
	  pred->build_br_inst(dest_bb);
	}
      else
	{
	  Inst *cond = pred->last_inst->args[0];
	  Basic_block *true_bb = pred->last_inst->u.br3.true_bb;
	  Basic_block *false_bb = pred->last_inst->u.br3.false_bb;
	  if (true_bb == bb)
	    true_bb = dest_bb;
	  if (false_bb == bb)
	    false_bb = dest_bb;

	  if (true_bb == false_bb)
	    {
	      assert(dest_bb->phis.empty());
	      destroy_instruction(pred->last_inst);
	      pred->build_br_inst(dest_bb);
	    }
	  else
	    {
	      destroy_instruction(pred->last_inst);
	      pred->build_br_inst(cond, true_bb, false_bb);
	    }
	}
    }

  destroy_instruction(bb->first_inst);
  bb->build_br_inst(bb);
}

bool is_always_ub(Basic_block *bb)
{
  return bb->first_inst->op == Op::UB && is_value_one(bb->first_inst->args[0]);
}

} // end anonymous namespace

void clear_dominance(Function *func)
{
  func->has_dominance = false;
  func->nearest_dom.clear();
  func->nearest_postdom.clear();
}

// We assume the CFG is loop-free and no dead BBs.
void calculate_dominance(Function *func)
{
  clear_dominance(func);

  // We must set has_dominance early as we call the dominance functions (for
  // cases we know it is safe) while we create the dominance information.
  func->has_dominance = true;

  // Calculate func->nearest_dom
  {
    std::vector<Basic_block *> post;
    post.reserve(func->bbs.size());
    std::set<Basic_block *> visited;
    dfs_walk(func->bbs.front(), post, visited);
    func->nearest_dom.insert({post.back(), post.back()});
    for (size_t i = 1; i < post.size(); i++)
      {
	Basic_block *bb = post[post.size() - i - 1];
	assert(!bb->preds.empty());
	Basic_block *dom = bb->preds[0];
	for (;;)
	  {
	    bool found_dom = true;
	    for (auto pred : bb->preds)
	      {
		found_dom = found_dom && dominates(dom, pred);
	      }
	    if (found_dom)
	      break;
	    dom = func->nearest_dom.at(dom);
	  }
	func->nearest_dom.insert({bb, dom});
      }
  }

  // Calculate func->nearest_postdom
  {
    std::vector<Basic_block *> post;
    post.reserve(func->bbs.size());
    std::set<Basic_block *> visited;
    inverse_dfs_walk(func->bbs.back(), post, visited);
    func->nearest_postdom.insert({post.back(), post.back()});
    for (size_t i = 1; i < post.size(); i++)
      {
	Basic_block *bb = post[post.size() - i - 1];
	assert(!bb->succs.empty());
	Basic_block *dom = bb->succs[0];
	for (;;)
	  {
	    bool found_dom = true;
	    for (auto succ : bb->succs)
	      {
		found_dom = found_dom && postdominates(dom, succ);
	      }
	    if (found_dom)
	      break;
	    dom = func->nearest_postdom.at(dom);
	  }
	func->nearest_postdom.insert({bb, dom});
      }
  }
}

Basic_block *nearest_dominator(const Basic_block *bb)
{
  assert(bb->func->has_dominance);
  return bb->func->nearest_dom.at(bb);
}

// Check if bb1 dominates bb2
bool dominates(const Basic_block *bb1, const Basic_block *bb2)
{
  assert(bb1->func->has_dominance);
  for (;;)
    {
      if (bb1 == bb2)
        return true;
      if (bb2 == bb2->func->bbs.front())
        return false;
      bb2 = bb2->func->nearest_dom.at(bb2);
    }
}

// Check if bb1 postdominates bb2
bool postdominates(const Basic_block *bb1, const Basic_block *bb2)
{
  assert(bb1->func->has_dominance);
  for (;;)
    {
      if (bb1 == bb2)
        return true;
      if (bb2 == bb2->func->bbs.back())
        return false;
      bb2 = bb2->func->nearest_postdom.at(bb2);
    }
}

void reverse_post_order(Function *func)
{
  auto it = std::find_if(func->bbs.begin(), func->bbs.end(),
			 [](const Basic_block *bb) {
			   return bb->last_inst->op == Op::RET;
			 });
  assert(it != func->bbs.end());
  Basic_block *exit_bb = *it;

  std::vector<Basic_block *> post;
  post.reserve(func->bbs.size());
  std::set<Basic_block *> visited;
  dfs_walk(func->bbs[0], post, visited);
  if (!visited.contains(exit_bb))
    throw Not_implemented("unreachable exit BB (infinite loop)");
  if (post.size() != func->bbs.size())
    {
      std::vector<Basic_block *> dead_bbs;
      for (auto bb : func->bbs)
	{
	  if (!visited.contains(bb))
	    dead_bbs.push_back(bb);
	}
      remove_dead_bbs(dead_bbs);
    }
  func->bbs.clear();
  std::reverse_copy(post.begin(), post.end(), std::back_inserter(func->bbs));
  if (func->bbs.back() != exit_bb)
    {
      auto it2 = std::find(func->bbs.begin(), func->bbs.end(), exit_bb);
      if (it2 != func->bbs.end())
	func->bbs.erase(it2);
      func->bbs.push_back(exit_bb);
    }
}

bool has_loops(Function *func)
{
  std::set<Basic_block*> visited;
  for (auto bb : func->bbs)
    {
      visited.insert(bb);
      for (auto succ : bb->succs)
	{
	  if (visited.contains(succ))
	    return true;
	}
    }
  return false;
}

bool simplify_cfg(Function *func)
{
  bool modified = false;

  for (auto bb : func->bbs)
    {
      // br 0, .1, .2  ->  br .2
      // br 1, .1, .2  ->  br .1
      if (bb->last_inst->op == Op::BR
	  && bb->last_inst->nof_args == 1)
	{
	  Inst *branch = bb->last_inst;
	  Inst *cond = bb->last_inst->args[0];
	  if (cond->op == Op::VALUE)
	    {
	      Basic_block *taken_bb =
		cond->value() ? branch->u.br3.true_bb : branch->u.br3.false_bb;
	      Basic_block *not_taken_bb =
		cond->value() ? branch->u.br3.false_bb : branch->u.br3.true_bb;
	      for (auto phi : not_taken_bb->phis)
		{
		  phi->remove_phi_arg(bb);
		}
	      destroy_instruction(branch);
	      bb->build_br_inst(taken_bb);
	      modified = true;
	    }
	}

      // If a BB with exactly one predecessor and successor follows a BB
      // with exactly one successor, then we may move all its instructions
      // to the predecessor BB (and the now empty BB will be removed later).
      if (bb->preds.size() == 1
	  && bb->preds[0] != bb
	  && bb->preds[0] != func->bbs[0]
	  && bb->preds[0]->succs.size() == 1
	  && bb->succs.size() == 1
	  && bb->phis.size() == 0)
	{
	  Inst *last_inst = bb->preds[0]->last_inst;
	  while (bb->first_inst->op != Op::BR)
	    {
	      bb->first_inst->move_before(last_inst);
	    }
	}

      // Eliminate conditional branches to always UB BBs. For example:
      //
      //   .1:
      //     br %10, .2, .3
      //
      //   .2:
      //     ub 1
      //     br .4
      //
      //   .3:
      //     ...
      //
      // is optimized to:
      //
      //   .1:
      //     ub %10
      //     br .3
      //
      //   .3:
      //     ...
      if (config.optimize_ub
	  && func->loop_unrolling_done
	  && bb->last_inst->op == Op::BR
	  && bb->last_inst->nof_args == 1)
	{
	  Inst *branch = bb->last_inst;
	  Basic_block *taken_bb = nullptr;
	  Basic_block *not_taken_bb = nullptr;
	  if (is_always_ub(branch->u.br3.true_bb))
	    {
	      bb->build_inst(Op::UB, branch->args[0]);
	      taken_bb = branch->u.br3.false_bb;
	      not_taken_bb = branch->u.br3.true_bb;
	    }
	  else if (is_always_ub(branch->u.br3.false_bb))
	    {
	      bb->build_inst(Op::UB, bb->build_inst(Op::NOT, branch->args[0]));
	      taken_bb = branch->u.br3.true_bb;
	      not_taken_bb = branch->u.br3.false_bb;
	    }
	  if (taken_bb)
	    {
	      for (auto phi : not_taken_bb->phis)
		{
		  phi->remove_phi_arg(bb);
		}
	      destroy_instruction(branch);
	      bb->build_br_inst(taken_bb);
	      modified = true;
	    }
	}

      // GCC sometimes needs an extra empty BB because a phi node in the
      // destination must be able to distinguish the two sources, which
      // in our IR looks like:
      //
      //   .4:
      //     %64 = flt %58, %22
      //     br %64, .5, .6
      //
      //   .5:
      //     br .6
      //
      //   .6:
      //     %67 = phi [ %39, .4 ], [ %45, .5 ]
      //
      // Change this to an Op::ITE instruction.
      if (bb->succs.size() == 2
	  && ((bb->succs[0]->succs.size() == 1
	       && bb->succs[0]->phis.empty()
	       && bb->succs[0]->first_inst->op == Op::BR
	       && bb->succs[0]->succs[0] == bb->succs[1]
	       && !bb->succs[1]->phis.empty())
	      || (bb->succs[1]->succs.size() == 1
		  && bb->succs[1]->phis.empty()
		  && bb->succs[1]->first_inst->op == Op::BR
		  && bb->succs[1]->succs[0] == bb->succs[0]
		  && !bb->succs[0]->phis.empty())))
	{
	  Basic_block *empty_bb;
	  Basic_block *dest_bb;
	  if (bb->succs[0]->phis.empty())
	    {
	      empty_bb = bb->succs[0];
	      dest_bb = bb->succs[1];
	    }
	  else
	    {
	      empty_bb = bb->succs[1];
	      dest_bb = bb->succs[0];
	    }
	  Basic_block *true_bb = bb->succs[0];
	  Inst *cond = bb->last_inst->args[0];

	  for (auto phi : dest_bb->phis)
	    {
	      Inst *inst1 = phi->get_phi_arg(empty_bb);
	      Inst *inst2 = phi->get_phi_arg(bb);
	      if (true_bb != empty_bb)
		std::swap(inst1, inst2);
	      Inst *ite = bb->build_inst(Op::ITE, cond, inst1, inst2);
	      phi->update_phi_arg(ite, bb);
	    }

	  destroy_instruction(bb->last_inst);
	  bb->build_br_inst(dest_bb);
	  modified = true;
	}

      // br (not x), .1, .2 -> br x, .2, .1
      if (bb->last_inst->op == Op::BR
	  && bb->last_inst->nof_args == 1
	  && bb->last_inst->args[0]->op == Op::NOT)
	{
	  Inst *branch = bb->last_inst;
	  Inst *cond = branch->args[0]->args[0];
	  Basic_block *true_bb = branch->u.br3.false_bb;
	  Basic_block *false_bb = branch->u.br3.true_bb;
	  destroy_instruction(branch);
	  bb->build_br_inst(cond, true_bb, false_bb);
	}

      // Remove empty BBs ending in unconditional branch by letting the
      // predecessors call the successor.
      remove_empty_bb(bb);
    }

  reverse_post_order(func);

  return modified;
}

bool simplify_cfg(Module *module)
{
  bool modified = false;
  for (auto func : module->functions)
    modified |= simplify_cfg(func);
  return modified;
}

} // end namespace smtgcc
