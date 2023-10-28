#include <algorithm>
#include <cassert>

#include "smtgcc.h"

namespace smtgcc {
namespace {

void rpo_walk(Basic_block *bb, std::vector<Basic_block *>& bbs, std::set<Basic_block *>& visited)
{
  assert(bb->last_inst->opcode == Op::BR || bb->last_inst->opcode == Op::RET);
  visited.insert(bb);
  if (bb->last_inst->opcode == Op::BR)
    {
      if (bb->last_inst->nof_args == 0)
	{
	  if (!visited.contains(bb->last_inst->u.br1.dest_bb))
	    rpo_walk(bb->last_inst->u.br1.dest_bb, bbs, visited);
	}
      else
	{
	  if (!visited.contains(bb->last_inst->u.br3.true_bb))
	    rpo_walk(bb->last_inst->u.br3.true_bb, bbs, visited);
	  if (!visited.contains(bb->last_inst->u.br3.false_bb))
	    rpo_walk(bb->last_inst->u.br3.false_bb, bbs, visited);
	}
      bbs.insert(bbs.begin(), bb);
    }
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
      for (Instruction *phi : bb->phis)
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
	  for (Instruction *inst = bb->last_inst; inst;)
	    {
	      Instruction *next_inst = inst->prev;
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

// We assume the CFG is loop-free and no dead BBs.
void calculate_dominance(Function *func)
{
  for (auto bb : func->bbs)
    {
      bb->dom.clear();
      bb->post_dom.clear();
    }

  int nof_bbs = func->bbs.size();

  // Dominators
  func->bbs[0]->dom.insert(func->bbs[0]);
  for (int i = 1; i < nof_bbs; i++)
    {
      Basic_block *bb = func->bbs[i];

      std::set<Basic_block*> intersection = bb->preds.at(0)->dom;
      for(size_t j = 1; j < bb->preds.size(); j++) {
        std::set<Basic_block*> temp;
	std::set<Basic_block*>& pred_dom = bb->preds.at(j)->dom;
        std::set_intersection(intersection.begin(), intersection.end(),
                              pred_dom.begin(), pred_dom.end(),
                              std::inserter(temp, temp.begin()));
        intersection = std::move(temp);
      }
      bb->dom = std::move(intersection);
      bb->dom.insert(bb);
    }

  // Post dominators
  func->bbs[nof_bbs - 1]->dom.insert(func->bbs[nof_bbs - 1]);
  for (int i = nof_bbs - 2; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];

      std::set<Basic_block*> intersection = bb->succs.at(0)->post_dom;
      for(size_t j = 1; j < bb->succs.size(); j++) {
        std::set<Basic_block*> temp;
	std::set<Basic_block*>& succ_post_dom = bb->succs.at(j)->post_dom;
        std::set_intersection(intersection.begin(), intersection.end(),
                              succ_post_dom.begin(), succ_post_dom.end(),
                              std::inserter(temp, temp.begin()));
        intersection = std::move(temp);
      }
      bb->post_dom = std::move(intersection);
      bb->post_dom.insert(bb);
    }
}

void update_phi(Basic_block *dest_bb, Basic_block *orig_src_bb, Basic_block *new_src_bb)
{
  for (auto phi : dest_bb->phis)
    {
      Instruction *inst = phi->get_phi_arg(orig_src_bb);
      phi->add_phi_arg(inst, new_src_bb);
    }
}

} // end anonymous namespace

Basic_block *nearest_dominator(const Basic_block *bb_in)
{
  if (bb_in->preds.size() == 0)
    return nullptr;

  Basic_block *bb = bb_in->preds.at(0);
  for (;;)
    {
      unsigned count = 0;
      for (auto pred : bb_in->preds)
	{
	  count += dominates(bb, pred);
	}
      if (count == bb_in->preds.size())
	return bb;
      if (!(bb->preds.size() > 0))
	abort();
      bb = bb->preds.at(0);
    }
}

// Check if bb1 dominates bb2
bool dominates(const Basic_block *bb1, const Basic_block *bb2)
{
  return bb2->dom.contains(const_cast<Basic_block *>(bb1));
}

// Check if bb1 post dominates bb2
bool post_dominates(const Basic_block *bb1, const Basic_block *bb2)
{
  return bb2->post_dom.contains(const_cast<Basic_block *>(bb1));
}

void reverse_post_order(Function *func)
{
  auto it = std::find_if(func->bbs.begin(), func->bbs.end(),
			 [](const Basic_block *bb) {
			   return bb->last_inst->opcode == Op::RET;
			 });
  assert(it != func->bbs.end());
  Basic_block *exit_bb = *it;

  std::vector<Basic_block *> bbs;
  std::set<Basic_block *> visited;
  rpo_walk(func->bbs[0], bbs, visited);
  bbs.push_back(exit_bb);
  if (!visited.contains(exit_bb))
    throw Not_implemented("unreachable exit BB (infinite loop)");
  if (bbs.size() != func->bbs.size())
    {
      std::vector<Basic_block *> dead_bbs;
      for (auto bb : func->bbs)
	{
	  if (!visited.contains(bb))
	    dead_bbs.push_back(bb);
	}
      remove_dead_bbs(dead_bbs);
    }
  func->bbs = bbs;

  if (!has_loops(func))
    calculate_dominance(func);
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
      if (bb->last_inst->opcode == Op::BR
	  && bb->last_inst->nof_args == 1)
	{
	  Instruction *branch = bb->last_inst;
	  Instruction *cond = bb->last_inst->arguments[0];
	  if (cond->opcode == Op::VALUE)
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

      // Remove empty BBs ending in unconditional branch by letting the
      // predecessors call the successor.
      if (bb->first_inst->opcode == Op::BR
	  && bb->first_inst->nof_args == 0
	  && bb->phis.size() == 0)
	{
	  Basic_block *dest_bb = bb->first_inst->u.br1.dest_bb;
	  std::vector<Basic_block *> preds = bb->preds;
	  for (auto pred : preds)
	    {
	      assert(pred->last_inst->opcode == Op::BR);
	      if (pred->last_inst->nof_args == 0)
		{
		  destroy_instruction(pred->last_inst);
		  pred->build_br_inst(dest_bb);
		  update_phi(dest_bb, bb, pred);
		}
	      else
		{
		  Instruction *cond = pred->last_inst->arguments[0];
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
