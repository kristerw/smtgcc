// Remove trivially dead instructions.

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

} // end anonymous namespace

void dead_code_elimination(Function *func)
{
  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];
      for (Instruction *inst = bb->last_inst; inst;)
	{
	  Instruction *next_inst = inst->prev;
	  if (inst->has_lhs() && inst->used_by.empty())
	    destroy(inst);
	  else if (inst->opcode == Op::UB && is_false(inst->arguments[0]))
	    destroy(inst);
	  else if (inst->opcode == Op::ASSERT && is_true(inst->arguments[0]))
	    destroy(inst);
	  inst = next_inst;
	}

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
