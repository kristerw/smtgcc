#include "smtgcc.h"

namespace smtgcc {

void eliminate_registers(Function *func, int64_t& symbolic_id)
{
  // Collect all registers.
  std::vector<Inst *> registers;
  for (Inst *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->op == Op::REGISTER)
	registers.push_back(inst);
    }

  std::map<Basic_block *, std::map<Inst *, Inst *>> bb2reg_values;
  for (Basic_block *bb : func->bbs)
    {
      std::map<Inst *, Inst *>& reg_values = bb2reg_values[bb];

      // Set up the values at the top of the BB.
      if (bb->preds.size() == 0)
	{
	  // Create the initial register values.
	  for (auto reg : registers)
	    {
	      Inst *inst =
		bb->build_inst(Op::SYMBOLIC, symbolic_id++, reg->bitsize);
	      reg_values.insert({reg, inst});
	    }
	}
      else if (bb->preds.size() == 1)
	reg_values = bb2reg_values[bb->preds[0]];
      else
	{
	  for (auto reg : registers)
	    {
	      Inst *phi = bb->build_phi_inst(reg->bitsize);
	      reg_values.insert({reg, phi});
	      for (auto pred : bb->preds)
		{
		  Inst *reg_value;
		  if (pred == bb || bb2reg_values[pred].empty())
		    reg_value = pred->build_inst(Op::READ, reg);
		  else
		    reg_value = bb2reg_values[pred][reg];
		  phi->add_phi_arg(reg_value, pred);
		}
	    }
	}

      // Eliminate all Op::READ and Op::WRITE.
      for (Inst *inst = bb->first_inst; inst;)
	{
	  Inst *next_inst = inst->next;

	  if (inst->op == Op::WRITE)
	    {
	      reg_values[inst->args[0]] = inst->args[1];
	      destroy_instruction(inst);
	    }
	  else if (inst->op == Op::READ)
	    {
	      inst->replace_all_uses_with(reg_values[inst->args[0]]);
	      destroy_instruction(inst);
	    }

	  inst = next_inst;
	}
    }

  // Eliminate all Op::REGISTER.
  for (auto reg : registers)
    destroy_instruction(reg);
}

void eliminate_registers(Module *module)
{
  // TODO: Handle the case where the functions already contain Op::SYMBOLIC.
  int64_t symbolic_id = 0;
  for (auto func : module->functions)
    eliminate_registers(func, symbolic_id);
}

} // end namespace smtgcc
