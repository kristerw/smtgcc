// Unroll loops.
//
// Only very simple loops (consisting of 1 or 2 basic blocks) are handled.

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <set>

#include "smtgcc.h"

using namespace std::string_literals;

namespace smtgcc {
namespace {

const int unroll_limit = 12;

class Unroller
{
  Function *func;
  Basic_block *loop_header;
  Basic_block *loop_exit;
  Basic_block *loop_body;
  Basic_block *orig_loop_exit;
  std::map<Instruction *, Instruction *> curr_inst;

  std::vector<Basic_block *> loop_bbs;

  void duplicate(Instruction *inst, Basic_block *bb);
  Instruction *translate(Instruction *inst);
  void ensure_lcssa(Instruction *inst);
  void create_lcssa();

public:
  Unroller(Basic_block *bb);
  void unroll();
};

Unroller::Unroller(Basic_block *bb)
{
  if (bb->succs.size() == 1)
    {
      assert(bb->succs.size() == 1 && bb->preds.size() == 1);
      assert(bb->succs[0] == bb->preds[0]);
      loop_body = bb;
      bb = loop_body->succs[0];
      orig_loop_exit = loop_body == bb->succs[0] ? bb->succs[1] : bb->succs[0];
      loop_bbs.push_back(loop_body);
    }
  else
    {
      loop_body = nullptr;
      orig_loop_exit = bb == bb->succs[0] ? bb->succs[1] : bb->succs[0];
    }
  assert(bb->succs.size() == 2);
  func = bb->func;
  loop_header = bb;
  loop_bbs.push_back(loop_header);
}

// Check that the instruction is used in LCSSA-safe way (i.e., all uses are
// either in the loop, or in a phi-node at a loop exit). If not, insert a new
// phi-node and make all loop-external uses use that.
void Unroller::ensure_lcssa(Instruction *inst)
{
  std::vector<Instruction *> invalid_use;
  for (auto use : inst->used_by)
    {
      if (use->bb == loop_exit && use->opcode == Op::PHI)
	continue;
      auto I = std::find(loop_bbs.begin(), loop_bbs.end(), use->bb);
      if (I == loop_bbs.end())
	invalid_use.push_back(use);
    }

  if (!invalid_use.empty())
    {
      Instruction *phi = loop_exit->build_phi_inst(inst->bitsize);
      phi->add_phi_arg(inst, loop_header);
      while (!invalid_use.empty())
	{
	  Instruction *use = invalid_use.back();
	  invalid_use.pop_back();
	  inst->replace_use_with(use, phi);
	}
    }
}

// Update all uses outside the loop to use a phi node in the loop exit block.
void Unroller::create_lcssa()
{
  for (auto bb : loop_bbs)
    {
      for (auto phi : bb->phis)
	{
	  ensure_lcssa(phi);
	}
      for (Instruction *inst = bb->first_inst; inst; inst = inst->next)
	{
	  ensure_lcssa(inst);
	}
    }
}

// Find a loop we can unroll. There are two cases:
// 1. The loop consist of one basic block.
//    If one such loop is found, then that BB is returned.
// 2. The loop consist of two basic blocks, where the loop body has
//    one predecessor and one successor.
//    If one such loop is found, then that loop body BB is returned.
// Otherwise nullptr is returned.
Basic_block *find_simple_loop(Function *func)
{
  for (auto bb : func->bbs)
    {
      if (bb->succs.size() == 2)
	{
	  if (bb->succs[0] == bb || bb->succs[1] == bb)
	    return bb;

	  for (auto succ : bb->succs)
	    {
	      if (succ->preds.size() != 1 || succ->succs.size() != 1)
		continue;
	      if (succ->preds[0] == bb && succ->succs[0] == bb)
		return succ;
	    }
	}
    }
  return nullptr;
}

// Get the SSA variable (i.e., instruction) corresponding to the input SSA
// variable for the use in this iteration.
Instruction *Unroller::translate(Instruction *inst)
{
  auto I = curr_inst.find(inst);
  if (I != curr_inst.end())
    return I->second;
  return inst;
}

void Unroller::duplicate(Instruction *inst, Basic_block *bb)
{
  Instruction *new_inst = nullptr;
  Inst_class iclass = inst->iclass();
  switch (iclass)
    {
    case Inst_class::iunary:
    case Inst_class::funary:
      {
	Instruction *arg = translate(inst->arguments[0]);
	new_inst = bb->build_inst(inst->opcode, arg);
      }
      break;
    case Inst_class::icomparison:
    case Inst_class::fcomparison:
    case Inst_class::ibinary:
    case Inst_class::fbinary:
    case Inst_class::conv:
      {
	Instruction *arg1 = translate(inst->arguments[0]);
	Instruction *arg2 = translate(inst->arguments[1]);
	new_inst = bb->build_inst(inst->opcode, arg1, arg2);
      }
      break;
    case Inst_class::ternary:
      {
	Instruction *arg1 = translate(inst->arguments[0]);
	Instruction *arg2 = translate(inst->arguments[1]);
	Instruction *arg3 = translate(inst->arguments[2]);
	new_inst = bb->build_inst(inst->opcode, arg1, arg2, arg3);
      }
      break;
    default:
      throw Not_implemented("unroller::duplicate: "s + inst->name());
    }
  assert(new_inst);
  curr_inst[inst] = new_inst;
}

void Unroller::unroll()
{
  // Add a new loop exit block that only have phi nodes. This is
  // not necessary, but it makes it easier to verify that we
  // create correct LCSSA.
  {
    loop_exit = func->build_bb();
    loop_exit->build_br_inst(orig_loop_exit);
    for (auto orig_phi : orig_loop_exit->phis)
      {
	Instruction *phi = loop_exit->build_phi_inst(orig_phi->bitsize);
	Instruction *arg = orig_phi->get_phi_arg(loop_header);
	phi->add_phi_arg(arg, loop_header);
	orig_phi->remove_phi_arg(loop_header);
	orig_phi->add_phi_arg(phi, loop_exit);
      }
    assert(loop_header->last_inst->opcode == Op::BR);
    assert(loop_header->last_inst->nof_args == 1);
    Instruction *cond = loop_header->last_inst->arguments[0];
    Basic_block *true_bb = loop_header->last_inst->u.br3.true_bb;
    if (true_bb == orig_loop_exit)
      true_bb = loop_exit;
    else
      assert(true_bb == loop_header || true_bb == loop_body);
    Basic_block *false_bb = loop_header->last_inst->u.br3.false_bb;
    if (false_bb == orig_loop_exit)
      false_bb = loop_exit;
    else
      assert(false_bb == loop_header || false_bb == loop_body);
    destroy_instruction(loop_header->last_inst);
    loop_header->build_br_inst(cond, true_bb, false_bb);
  }

  create_lcssa();

  std::vector<Basic_block *> bbs;
  std::vector<Basic_block *> body_bbs;
  for (int i = 0; i < unroll_limit; i++)
    bbs.push_back(func->build_bb());
  if (loop_body)
    {
      for (int i = 0; i < unroll_limit - 1; i++)
	{
	  body_bbs.push_back(func->build_bb());
	}
    }

  for (int i = 0; i < unroll_limit - 1; i++)
    {
      // We currently only support simple loops where we guarantee that
      // the phi nodes have exactly one argument for the looping case.
      // So the duplicated blocks will only have the looping case, and
      // we only update the translation table.
      //
      // We must translate phi nodes in two steps, because we may have
      //   .2:
      //     %10 = phi [ %7, .1 ], [ %5, .6 ], [ %49, .5 ]
      //     %12 = phi [ %5, .1 ], [ %10, .6 ], [ %10, .5 ]
      // where phi %12 uses the value of phi %10 from the previous iteration.
      // So we must translate all phi nodes before writing the new phi nodes
      // to the translation table.
      std::map<Instruction *, Instruction *> tmp_curr_inst;
      for (auto phi : loop_header->phis)
	{
	  if (loop_body)
	    tmp_curr_inst[phi] = translate(phi->get_phi_arg(loop_body));
	  else
	    tmp_curr_inst[phi] = translate(phi->get_phi_arg(loop_header));
	}
      for (auto [phi, translated_phi] : tmp_curr_inst)
	{
	  curr_inst[phi] = translated_phi;
	}

      for (Instruction *inst = loop_header->first_inst; inst; inst = inst->next)
	{
	  if (inst->opcode == Op::BR)
	    {
	      assert(inst->nof_args == 1);
	      Instruction *arg = translate(inst->arguments[0]);
	      Basic_block *true_bb;
	      if (loop_body)
		true_bb = body_bbs.at(i);
	      else
		true_bb = bbs.at(i + 1);
	      Basic_block *false_bb = loop_exit;
	      if (inst->u.br3.true_bb == loop_exit)
		std::swap(true_bb, false_bb);
	      bbs.at(i)->build_br_inst(arg, true_bb, false_bb);
	      for (auto phi : loop_exit->phis)
		{
		  Instruction *phi_arg = phi->get_phi_arg(loop_header);
		  phi->add_phi_arg(translate(phi_arg), bbs.at(i));
		}
	    }
	  else
	    duplicate(inst, bbs.at(i));
	}

      if (loop_body)
	{
	  for (Instruction *inst = loop_body->first_inst;
	       inst;
	       inst = inst->next)
	    {
	      if (inst->opcode == Op::BR)
		{
		  assert(inst->nof_args == 0);
		  body_bbs.at(i)->build_br_inst(bbs.at(i + 1));
		}
	      else
		duplicate(inst, body_bbs.at(i));
	    }
	}
    }

  // The last block is for cases the program loops more than our unroll limit.
  // This makes our analysis invalid, so we mark this as UB.
  Basic_block *last_bb = bbs.at(unroll_limit - 1);
  last_bb->build_inst(Op::UB, last_bb->value_inst(1, 1));
  last_bb->build_br_inst(loop_exit);
  for (auto phi : loop_exit->phis)
    {
      phi->add_phi_arg(last_bb->value_inst(0, phi->bitsize), last_bb);
    }

  // Update the original loop to only do the first iteration.
  if (loop_body)
    {
      for (auto phi : loop_header->phis)
	{
	  phi->remove_phi_arg(loop_body);
	}
      destroy_instruction(loop_body->last_inst);
      loop_body->build_br_inst(bbs.at(0));
   }
  else
    {
      for (auto phi : loop_header->phis)
	{
	  phi->remove_phi_arg(loop_header);
	}
      Instruction *cond = loop_header->last_inst->arguments[0];
      Basic_block *true_bb = loop_header->last_inst->u.br3.true_bb;
      Basic_block *false_bb = loop_header->last_inst->u.br3.false_bb;
      if (true_bb == loop_header)
	true_bb = bbs.at(0);
      else
	assert(true_bb == loop_exit);
      if (false_bb == loop_header)
	false_bb = bbs.at(0);
      else
	assert(false_bb == loop_exit);
      destroy_instruction(loop_header->last_inst);
      loop_header->build_br_inst(cond, true_bb, false_bb);
    }
}

} // end anonymous namespace

bool loop_unroll(Function *func)
{
  Basic_block *bb;
  bool unrolled = false;
  while ((bb = find_simple_loop(func)))
    {
      Unroller unroller(bb);
      unroller.unroll();
      reverse_post_order(func);
      unrolled = true;
    }

  // Report error if we could not unroll all loops.
  if (has_loops(func))
    throw Not_implemented("loops");

  return unrolled;
}

bool loop_unroll(Module *module)
{
  bool unrolled = false;
  for (auto func : module->functions)
    unrolled |= loop_unroll(func);
  return unrolled;
}

} // end namespace smtgcc
