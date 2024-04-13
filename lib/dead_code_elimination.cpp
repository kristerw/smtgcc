// Remove trivially dead instructions.

#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

bool is_false(Instruction *inst)
{
  return inst->opcode == Op::VALUE && inst->value() == 0;
}

bool is_true(Instruction *inst)
{
  return inst->opcode == Op::VALUE && inst->value() == 1;
}

void destroy(Instruction *inst)
{
  // Memory instructions must be kept until the memory optimization passes.
  if (inst->opcode == Op::MEMORY)
    return;

  destroy_instruction(inst);
}

// Remove all instructions (except the "ub 1") for a basic block that
// is always UB.
void clear_ub_bb(Basic_block *bb, bool is_loopfree)
{
  if (is_loopfree
      && bb->last_inst->opcode == Op::BR && bb->last_inst->nof_args == 1)
    {
      // Change the conditional branch to an unconditional branch. It does
      // not matter which branch we take since the execution is UB anyway.
      Instruction *cond = bb->last_inst->arguments[0];
      cond->replace_use_with(bb->last_inst, bb->value_inst(1, 1));
    }

  Instruction *found_ub = nullptr;
  for (Instruction *inst = bb->last_inst->prev; inst;)
    {
      Instruction *next_inst = inst->prev;
      if (inst->opcode == Op::UB && is_true(inst->arguments[0]))
	{
	  if (found_ub)
	    destroy(inst);
	  else
	    found_ub = inst;
	}
      else if(inst->has_lhs() && !inst->used_by.empty())
	{
	  if (is_loopfree)
	    {
	      inst->replace_all_uses_with(bb->value_inst(0, inst->bitsize));
	      destroy(inst);
	    }
	}
      else
	destroy(inst);
      inst = next_inst;
    }
  assert(found_ub);

  while (!bb->phis.empty())
    {
      Instruction *inst = bb->phis.back();
      if (!inst->used_by.empty())
	inst->replace_all_uses_with(bb->value_inst(0, inst->bitsize));
      destroy(inst);
    }

  if (bb->first_inst != found_ub)
    found_ub->move_before(bb->first_inst);
}

} // end anonymous namespace

void dead_code_elimination(Function *func)
{
  bool is_loopfree = !has_loops(func);
  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];

      // Propagate "always UB" from successors (this BB is always UB if
      // all its successors are always UB).
      if (bb != func->bbs[0]
	  && !bb->succs.empty()
	  && (bb->first_inst->opcode != Op::UB
	      || !is_true(bb->first_inst->arguments[0])))
	{
	  bool succs_are_ub = true;
	  for (auto succ : bb->succs)
	    {
	      succs_are_ub =
		succs_are_ub && succ->first_inst->opcode == Op::UB
		&& is_true(succ->first_inst->arguments[0]);
	    }
	  if (succs_are_ub)
	    bb->build_inst(Op::UB, bb->value_inst(1, 1));
	}

      // Remove dead instructions.
      for (Instruction *inst = bb->last_inst; inst;)
	{
	  Instruction *next_inst = inst->prev;
	  if (inst->has_lhs() && inst->used_by.empty())
	    destroy(inst);
	  else if (bb != func->bbs[0]
		   && inst->opcode == Op::UB && is_true(inst->arguments[0]))
	    {
	      clear_ub_bb(bb, is_loopfree);
	      break;
	    }
	  else if (inst->opcode == Op::UB && is_false(inst->arguments[0]))
	    destroy(inst);
	  else if (inst->opcode == Op::ASSERT && is_true(inst->arguments[0]))
	    destroy(inst);
	  inst = next_inst;
	}

      // Remove dead phi-nodes.
      std::vector<Instruction *> dead_phis;
      for (auto phi : bb->phis)
	{
	  if (phi->used_by.empty())
	    dead_phis.push_back(phi);
	}
      for (auto phi : dead_phis)
	{
	  destroy(phi);
	}
    }
}

void dead_code_elimination(Module *module)
{
  for (auto func : module->functions)
    dead_code_elimination(func);
}

} // end namespace smtgcc
