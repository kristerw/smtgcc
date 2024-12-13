// Remove trivially dead instructions.

#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

bool is_false(Inst *inst)
{
  return inst->op == Op::VALUE && inst->value() == 0;
}

bool is_true(Inst *inst)
{
  return inst->op == Op::VALUE && inst->value() == 1;
}

void destroy(Inst *inst)
{
  // Memory instructions must be kept until the memory optimization passes.
  if (inst->op == Op::MEMORY)
    return;

  destroy_instruction(inst);
}

} // end anonymous namespace

void dead_code_elimination(Function *func)
{
  uint32_t nof_inst = 0;
  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];

      // Remove dead instructions.
      for (Inst *inst = bb->last_inst; inst;)
	{
	  Inst *next_inst = inst->prev;
	  if (inst->has_lhs() && inst->used_by.empty())
	    destroy(inst);
	  else if (inst->op == Op::UB && is_false(inst->args[0]))
	    destroy(inst);
	  else if (inst->op == Op::ASSERT && is_true(inst->args[0]))
	    destroy(inst);
	  else
	    nof_inst++;
	  inst = next_inst;
	}

      // Remove dead phi-nodes.
      std::vector<Inst *> dead_phis;
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

  if (nof_inst > max_nof_inst)
    throw Not_implemented("too many instructions");
}

void dead_code_elimination(Module *module)
{
  for (auto func : module->functions)
    dead_code_elimination(func);
}

} // end namespace smtgcc
