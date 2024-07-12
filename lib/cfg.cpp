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

void update_phi(Basic_block *dest_bb, Basic_block *orig_src_bb, Basic_block *new_src_bb)
{
  for (auto phi : dest_bb->phis)
    {
      Inst *inst = phi->get_phi_arg(orig_src_bb);
      phi->add_phi_arg(inst, new_src_bb);
    }
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

void simplify_cfg(Function *func)
{
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

      // Remove empty BBs ending in unconditional branch by letting the
      // predecessors call the successor.
      if (bb->first_inst->op == Op::BR
	  && bb->first_inst->nof_args == 0
	  && bb->phis.size() == 0
	  && bb->succs[0]->phis.size() == 0)
	{
	  Basic_block *dest_bb = bb->first_inst->u.br1.dest_bb;
	  std::vector<Basic_block *> preds = bb->preds;
	  for (auto pred : preds)
	    {
	      assert(pred->last_inst->op == Op::BR);
	      if (pred->last_inst->nof_args == 0)
		{
		  destroy_instruction(pred->last_inst);
		  pred->build_br_inst(dest_bb);
		  update_phi(dest_bb, bb, pred);
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
		      if (dest_bb->phis.size() == 0)
			{
			  destroy_instruction(pred->last_inst);
			  pred->build_br_inst(dest_bb);
			}
		    }
		  else
		    {
		      destroy_instruction(pred->last_inst);
		      pred->build_br_inst(cond, true_bb, false_bb);
		      update_phi(dest_bb, bb, pred);
		    }
		}
	    }

	  destroy_instruction(bb->first_inst);
	  bb->build_br_inst(bb);
	}
    }

  reverse_post_order(func);
}

void simplify_cfg(Module *module)
{
  for (auto func : module->functions)
    simplify_cfg(func);
}

} // end namespace smtgcc
