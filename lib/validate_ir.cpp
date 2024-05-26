#include <algorithm>
#include <cassert>
#include <set>

#include "smtgcc.h"

// TODO: Change all assert to a validation_assert (that is not removed in
// NDEBUG builds, or maybe a more descriptive exception?

namespace smtgcc {

namespace {

void validate(Instruction *inst)
{
  if (inst->opcode == Op::PARAM
      || inst->opcode == Op::MEMORY
      || inst->opcode == Op::VALUE)
    {
      assert(inst->bb == inst->bb->func->bbs[0]);
    }
}

void validate(Basic_block *bb)
{
  // There must be instructions in the BB.
  assert(bb->first_inst);
  assert(bb->last_inst);

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
      assert(bb->last_inst->opcode == Op::RET);
    }
  else if (bb->succs.size() == 1)
    {
      assert(bb->last_inst->opcode == Op::BR);
      assert(bb->last_inst->u.br1.dest_bb == bb->succs[0]);
    }
  else
    {
      assert(bb->succs.size() == 2);
      assert(bb->last_inst->opcode == Op::BR);
      Basic_block *true_bb = bb->last_inst->u.br3.true_bb;
      Basic_block *false_bb = bb->last_inst->u.br3.false_bb;
      assert(bb->succs[0] == true_bb && bb->succs[1] == false_bb);
    }

  // Phi nodes must have one argument for each predecessor.
  for (Instruction *phi : bb->phis)
    {
      assert(phi->phi_args.size() == bb->preds.size());
      for (auto [arg_inst, arg_bb] : phi->phi_args)
	{
	  auto it = std::find(bb->preds.begin(), bb->preds.end(), arg_bb);
	  assert(it != bb->preds.end());
	}
    }

  // TODO: For each inst, check that its BB is correct.
  // TODO: For each inst, check that its used_by is correct.
  for (Instruction *inst = bb->first_inst; inst; inst = inst->next)
    {
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
}

void validate(Module *module)
{
  for (auto func : module->functions)
    validate(func);
}

} // end namespace smtgcc
