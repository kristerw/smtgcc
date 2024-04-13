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
//
// Note: This should only be called for loop-free functions because
// we may otherwise create infinite loops where there is no path to the
// exit block if a removed instruction was used in a branch condition.
void clear_ub_bb(Basic_block *bb)
{
  if (bb->last_inst->opcode == Op::BR && bb->last_inst->nof_args == 1)
    {
      // Change the conditional branch to an unconditional branch. It does
      // not matter which branch we take since the execution is UB anyway.
      Instruction *cond = bb->last_inst->arguments[0];
      cond->replace_use_with(bb->last_inst, bb->value_inst(1, 1));
    }

  bool found_ub = false;
  for (Instruction *inst = bb->last_inst->prev; inst;)
    {
      Instruction *next_inst = inst->prev;
      if (inst->opcode == Op::UB && is_true(inst->arguments[0]))
	{
	  if (found_ub)
	    destroy(inst);
	  found_ub = true;
	}
      else if(inst->has_lhs())
	{
	  if (!inst->used_by.empty())
	    inst->replace_all_uses_with(bb->value_inst(0, inst->bitsize));
	  destroy(inst);
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
}

} // end anonymous namespace

void dead_code_elimination(Function *func)
{
  bool is_loopfree = !has_loops(func);
  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];

      // Remove dead instructions.
      for (Instruction *inst = bb->last_inst; inst;)
	{
	  Instruction *next_inst = inst->prev;
	  if (inst->has_lhs() && inst->used_by.empty())
	    destroy(inst);
	  else if (is_loopfree &&
		   bb != func->bbs[0]
		   && inst->opcode == Op::UB && is_true(inst->arguments[0]))
	    {
	      clear_ub_bb(bb);
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
	destroy(phi);
    }
}

void dead_code_elimination(Module *module)
{
  for (auto func : module->functions)
    dead_code_elimination(func);
}

} // end namespace smtgcc
