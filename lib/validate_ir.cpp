#include <algorithm>
#include <cassert>
#include <set>

#include "smtgcc.h"

// TODO: Change all assert to a validation_assert (that is not removed in
// NDEBUG builds, or maybe a more descriptive exception?

namespace smtgcc {

namespace {

void validate(Inst *inst)
{
  // Some instructions are required to be placed in the entry block.
  if (inst->op == Op::PARAM
      || inst->op == Op::MEMORY
      || inst->op == Op::VALUE)
    {
      assert(inst->bb == inst->bb->func->bbs[0]);
    }

  // RET and BR must be the last instruction in the basic block.
  // All other instructions must have a next instruction.
  if (inst->op == Op::RET || inst->op == Op::BR)
    {
      assert(inst == inst->bb->last_inst);
      assert(!inst->next);
    }
  else
    assert(inst->next);

  // The next and prev instructions (if any) must be in the same basic
  // block as the original instruction.
  if (inst->prev)
    assert(inst->bb == inst->prev->bb);
  if (inst->next)
    assert(inst->bb == inst->next->bb);
}

void validate(Basic_block *bb)
{
  // There must be instructions in the BB.
  assert(bb->first_inst);
  assert(bb->last_inst);

  // Check that the first and last instructions actually are the first
  // and last instructions in the basic block.
  assert(!bb->first_inst->prev);
  assert(!bb->last_inst->next);

  // Predecessors must not be in preds multiple times.
  std::set<Basic_block *> pred_set(bb->preds.begin(), bb->preds.end());
  assert(bb->preds.size() == pred_set.size());

  // Each predecessor must have this basic block as a successor.
  for (Basic_block *pred_bb : bb->preds)
    {
      auto it = std::find(pred_bb->succs.begin(), pred_bb->succs.end(), bb);
      assert(it != pred_bb->succs.end());
    }

  // The successors must agree with the last instruction, and the last BB
  // (and no other) must end by a RET instruction.
  assert(bb->succs.size() < 3);
  if (bb->succs.size() == 0)
    {
      assert(bb->last_inst->op == Op::RET);
    }
  else if (bb->succs.size() == 1)
    {
      assert(bb->last_inst->op == Op::BR);
      assert(bb->last_inst->u.br1.dest_bb == bb->succs[0]);
    }
  else
    {
      assert(bb->succs.size() == 2);
      assert(bb->last_inst->op == Op::BR);
      Basic_block *true_bb = bb->last_inst->u.br3.true_bb;
      Basic_block *false_bb = bb->last_inst->u.br3.false_bb;
      assert(bb->succs[0] == true_bb && bb->succs[1] == false_bb);
    }

  // Phi nodes must have one argument for each predecessor.
  for (Inst *phi : bb->phis)
    {
      assert(phi->phi_args.size() == bb->preds.size());
      for (auto [arg_inst, arg_bb] : phi->phi_args)
	{
	  auto it = std::find(bb->preds.begin(), bb->preds.end(), arg_bb);
	  assert(it != bb->preds.end());
	}
    }

  // TODO: For each inst, check that its used_by is correct.
  for (Inst *inst = bb->first_inst; inst; inst = inst->next)
    {
      assert(inst->bb == bb);
      validate(inst);
    }
}

} // end anonymous namespace

void validate(Function *func)
{
  assert(func->bbs.size() > 0);
  for (Basic_block *bb : func->bbs)
    {
      validate(bb);
    }

  // Check that each instruction has been defined before use.
  std::set<Inst *> defined;
  for (Basic_block *bb : func->bbs)
    {
      for (Inst *phi : bb->phis)
	{
	  assert(!defined.contains(phi));
	  defined.insert(phi);
	}
      for (Inst *inst = bb->first_inst; inst; inst = inst->next)
	{
	  assert(!defined.contains(inst));
	  for (unsigned i = 0 ; i < inst->nof_args; i++)
	    {
	      assert(defined.contains(inst->args[i]));
	    }
	  defined.insert(inst);
	}
    }
}

void validate(Module *module)
{
  for (auto func : module->functions)
    validate(func);
}

} // end namespace smtgcc
