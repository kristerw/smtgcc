// Reduce the bitsize of instructions when only part of the result is used.
// We assume the CFG is loop-free.

#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

Inst *build_trunc(Inst *current_inst, Inst *inst, uint64_t bitsize)
{
  Inst *new_inst = create_inst(Op::EXTRACT, inst, bitsize - 1, 0);
  new_inst->insert_before(current_inst);

  // It is possible that the previous instruction extended a
  // value. Simplify to ensure we get rid of such double bitsize
  // modifications, so we have correct usage information when
  // checking previous instructions.
  new_inst = simplify_inst(new_inst);

  return new_inst;
}

uint64_t get_needed_bitsize(Inst *inst)
{
  uint64_t bitsize = 0;
  for (auto use : inst->used_by)
    {
      if (use->op != Op::EXTRACT)
	return inst->bitsize;
      bitsize = std::max(bitsize, (uint64_t)use->args[1]->value() + 1);
    }
  assert(bitsize != 0);
  return bitsize;
}

void truncate_if_possible(Inst *inst)
{
  switch (inst->op)
    {
      case Op::NEG:
      case Op::NOT:
      case Op::ADD:
      case Op::AND:
      case Op::MUL:
      case Op::OR:
      case Op::SUB:
      case Op::XOR:
      case Op::ITE:
	break;
      default:
	return;
    }

  uint64_t needed_bitsize = get_needed_bitsize(inst);
  if (needed_bitsize == inst->bitsize)
    return;

  Inst *new_inst = nullptr;
  switch (inst->op)
    {
      case Op::NEG:
      case Op::NOT:
	{
	  Inst *arg1 = build_trunc(inst, inst->args[0], needed_bitsize);
	  new_inst = create_inst(inst->op, arg1);
	  new_inst->insert_before(inst);
	}
	break;
      case Op::ADD:
      case Op::AND:
      case Op::MUL:
      case Op::OR:
      case Op::SUB:
      case Op::XOR:
	{
	  Inst *arg1 = build_trunc(inst, inst->args[0], needed_bitsize);
	  Inst *arg2;
	  if (inst->args[1] == inst->args[0])
	    arg2 = arg1;
	  else
	    arg2 = build_trunc(inst, inst->args[1], needed_bitsize);
	  new_inst = create_inst(inst->op, arg1, arg2);
	  new_inst->insert_before(inst);
	}
	break;
      case Op::ITE:
	{
	  Inst *arg2 = build_trunc(inst, inst->args[1], needed_bitsize);
	  Inst *arg3;
	  if (inst->args[2] == inst->args[1])
	    arg3 = arg2;
	  else
	    arg3 = build_trunc(inst, inst->args[2], needed_bitsize);
	  new_inst = create_inst(Op::ITE, inst->args[0], arg2, arg3);
	  new_inst->insert_before(inst);
	}
	break;
      default:
	break;
    }

  new_inst = create_inst(Op::ZEXT, new_inst, inst->bitsize);
  new_inst->insert_before(inst);
  inst->replace_all_uses_with(new_inst);
}

void truncate_phi(Inst *phi, uint64_t bitsize)
{
  Inst *new_phi = phi->bb->build_phi_inst(bitsize);
  for (auto [inst, bb] : phi->phi_args)
    {
      Inst *new_inst = create_inst(Op::EXTRACT, inst, bitsize - 1, 0);
      new_inst->insert_before(bb->last_inst);

      // It is possible that the previous instruction extended a
      // value. Simplify to ensure we get rid of such double bitsize
      // modifications, so we have correct usage information when
      // checking previous instructions.
      new_inst = simplify_inst(new_inst);

      new_phi->add_phi_arg(new_inst, bb);
    }
  Inst *new_inst = create_inst(Op::ZEXT, new_phi, phi->bitsize);
  new_inst->insert_before(phi->bb->first_inst);
  phi->replace_all_uses_with(new_inst);
}

} // end anonymous namespace

void reduce_bitsize(Function *func)
{
  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];
      for (Inst *inst = bb->last_inst; inst; inst = inst->prev)
	{
	  if (inst->has_lhs())
	    {
	      if (!inst->used_by.empty())
		truncate_if_possible(inst);

	      // Dead instructions may prevent truncation if the dead
	      // use is the only use of the full value. Remove dead
	      // instructions to avoid this problem.
	      if (inst->used_by.empty() && inst->op != Op::MEMORY)
		{
		  inst = inst->next;
		  destroy_instruction(inst->prev);
		}
	    }
	}

      std::vector<std::pair<Inst *, uint64_t>> truncate_phis;
      for (auto phi : bb->phis)
	{
	  if (!phi->used_by.empty())
	    {
	      uint64_t needed_bitsize = get_needed_bitsize(phi);
	      if (needed_bitsize != phi->bitsize)
		truncate_phis.push_back({phi, needed_bitsize});
	    }
	}
      for (auto [phi, bitsize] : truncate_phis)
	{
	  truncate_phi(phi, bitsize);
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
	  destroy_instruction(phi);
	}
    }
}

void reduce_bitsize(Module *module)
{
  for (auto func : module->functions)
    reduce_bitsize(func);
}

} // end namespace smtgcc
