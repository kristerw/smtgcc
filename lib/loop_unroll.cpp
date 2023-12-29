// Unroll loops.
//
// Only very simple loops (consisting of 1, 2 or 3 basic blocks) are handled.

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <optional>

#include "smtgcc.h"

#include "stdio.h"

using namespace std::string_literals;

namespace smtgcc {
namespace {

const int unroll_limit = 12;

struct Loop
{
  // The basic blocks of the loop, in reverse post order.
  std::vector<Basic_block *> bbs;

  // The basic block after the loop (we only allow one exit block for now).
  Basic_block *loop_exit;
};

class Unroller
{
  Loop& loop;
  Function *func;
  std::map<Instruction *, Instruction *> curr_inst;
  std::map<Basic_block *, Basic_block *> curr_bb;
  Basic_block *next_loop_header = nullptr;

  void build_new_loop_exit();
  void update_loop_exit(Basic_block *src_bb, Basic_block *dst_bb);
  void duplicate(Instruction *inst, Basic_block *bb);
  Instruction *translate(Instruction *inst);
  Basic_block *translate(Basic_block *bb);
  void ensure_lcssa(Instruction *inst);
  void create_lcssa();
  void unroll_one_iteration();

public:
  Unroller(Loop& loop);
  void unroll();
};

Unroller::Unroller(Loop& loop) : loop{loop}
{
  func = loop.bbs.back()->func;
}

// Check that the instruction is used in LCSSA-safe way (i.e., all uses are
// either in the loop, or in a phi-node at a loop exit). If not, insert a new
// phi-node and make all loop-external uses use that.
void Unroller::ensure_lcssa(Instruction *inst)
{
  std::vector<Instruction *> invalid_use;
  for (auto use : inst->used_by)
    {
      if (use->bb == loop.loop_exit && use->opcode == Op::PHI)
	continue;
      auto I = std::find(loop.bbs.begin(), loop.bbs.end(), use->bb);
      if (I == loop.bbs.end())
	invalid_use.push_back(use);
    }

  if (!invalid_use.empty())
    {
      Instruction *phi = loop.loop_exit->build_phi_inst(inst->bitsize);
      phi->add_phi_arg(inst, loop.loop_exit->preds[0]);
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
  build_new_loop_exit();

  for (auto bb : loop.bbs)
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
std::optional<Loop> find_simple_loop(Function *func)
{
  Loop loop;

  for (auto bb : func->bbs)
    {
      if (bb->succs.size() == 2)
	{
	  if (bb->succs[0] == bb || bb->succs[1] == bb)
	    {
	      loop.bbs.push_back(bb);
	      if (bb->succs[0] == bb)
		loop.loop_exit = bb->succs[1];
	      else
		loop.loop_exit = bb->succs[0];
	      return loop;
	    }

	  for (auto succ : bb->succs)
	    {
	      if (succ->preds.size() != 1 || succ->succs.size() != 1)
		continue;

	      if (succ->preds[0] == bb && succ->succs[0] == bb)
		{
		  loop.bbs.push_back(bb);
		  loop.bbs.push_back(succ);
		  if (bb->succs[0] == succ)
		    loop.loop_exit = bb->succs[1];
		  else
		    loop.loop_exit = bb->succs[0];
		  return loop;
		}
	    }
	}
    }
  return {};
}

// Find an 'advanced loop' -- a loop that looks like this:
//
//      \|/
//   +- HEAD<---+
//   |   |      |
//   |  MID     |
//   |   |      |
//   +->LATCH --+
//       |
//      exit
//
// Given a block L if these conditions hold, L is the latch of an advanced
// loop:
// 1 L has two predecessors H and M
// 2 L has two successors
// 3 H is the sole predecessor of M
// 4 H is a successor of L
// 5 M has exactly one successor
std::optional<Loop> find_advanced_loop(Function *func)
{
  for (auto l : func->bbs)
    {
      // Condition 1
      if (l->preds.size() != 2)
	continue;
      // Condition 2
      if (l->succs.size() != 2)
	continue;
      // Condition 3
      Basic_block *h = l->preds[0];
      Basic_block *m = l->preds[1];
      if (m->preds.size() != 1 || m->preds[0] != h)
	{
	  std::swap(h, m);
	  if (m->preds.size() != 1 || m->preds[0] != h)
	    continue;
	}
      // Condition 4
      if (h != l->succs[0] && h != l->succs[1])
	continue;
      // Condition 5
      if (m->succs.size() != 1)
	continue;

      Loop loop;
      loop.bbs.push_back(h);
      loop.bbs.push_back(m);
      loop.bbs.push_back(l);
      if (h == l->succs[0])
	loop.loop_exit = l->succs[1];
      else
	loop.loop_exit = l->succs[0];
      return loop;
    }
  return {};
}

std::optional<Loop> find_loop(Function *func)
{
  std::optional<Loop> loop = find_simple_loop(func);
  if (loop)
    return loop;
  return find_advanced_loop(func);
}

// Get the SSA variable (i.e., instruction) corresponding to the input SSA
// variable for use in this iteration.
Instruction *Unroller::translate(Instruction *inst)
{
  auto I = curr_inst.find(inst);
  if (I != curr_inst.end())
    return I->second;
  return inst;
}

Basic_block *Unroller::translate(Basic_block *bb)
{
  auto I = curr_bb.find(bb);
  if (I != curr_bb.end())
    return I->second;
  return bb;
}

void Unroller::build_new_loop_exit()
{
  Basic_block *orig_loop_exit = loop.loop_exit;

  loop.loop_exit = func->build_bb();
  loop.loop_exit->build_br_inst(orig_loop_exit);
  std::map<Instruction *, Instruction *> phi_map;
  for (auto phi : orig_loop_exit->phis)
    {
      Instruction *new_phi = loop.loop_exit->build_phi_inst(phi->bitsize);
      phi_map.insert({phi, new_phi});
      phi->add_phi_arg(new_phi, loop.loop_exit);
    }

  // Update exiting branches within the loop to point to the new
  // loop exit.
  for (auto bb : loop.bbs)
    {
      assert(bb->last_inst->opcode == Op::BR);
      bool updated = false;
      if (bb->last_inst->nof_args == 0)
	{
	  Basic_block *dest_bb = bb->last_inst->u.br1.dest_bb;
	  if (dest_bb == orig_loop_exit)
	    {
	      destroy_instruction(bb->last_inst);
	      bb->build_br_inst(loop.loop_exit);
	      updated = true;
	    }
	}
      else
	{
	  Instruction *arg = bb->last_inst->arguments[0];
	  Basic_block *true_bb = bb->last_inst->u.br3.true_bb;
	  Basic_block *false_bb = bb->last_inst->u.br3.false_bb;
	  if (true_bb == orig_loop_exit || false_bb == orig_loop_exit)
	    {
	      if (true_bb == orig_loop_exit)
		true_bb = loop.loop_exit;
	      if (false_bb == orig_loop_exit)
		false_bb = loop.loop_exit;
	      destroy_instruction(bb->last_inst);
	      bb->build_br_inst(arg, true_bb, false_bb);
	      updated = true;
	    }
	}
      if (updated)
	{
	  for (auto phi : orig_loop_exit->phis)
	    {
	      Instruction *phi_arg = phi->get_phi_arg(bb);
	      phi->remove_phi_arg(bb);
	      phi_map.at(phi)->add_phi_arg(phi_arg, bb);
	    }
	}
    }
}

void Unroller::update_loop_exit(Basic_block *src_bb, Basic_block *dst_bb)
{
  for (auto phi : loop.loop_exit->phis)
    {
      Instruction *inst = phi->get_phi_arg(src_bb);
      phi->add_phi_arg(translate(inst), dst_bb);
    }
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
      if (inst->opcode == Op::BR)
	{
	  if (inst->nof_args == 0)
	    {
	      Basic_block *dest_bb = translate(inst->u.br1.dest_bb);
	      bb->build_br_inst(dest_bb);
	      if (dest_bb == loop.loop_exit)
		update_loop_exit(inst->bb, bb);
	    }
	  else
	    {
	      Instruction *arg = translate(inst->arguments[0]);
	      Basic_block *true_bb = translate(inst->u.br3.true_bb);
	      Basic_block *false_bb = translate(inst->u.br3.false_bb);
	      bb->build_br_inst(arg, true_bb, false_bb);
	      if (true_bb == loop.loop_exit)
		update_loop_exit(inst->bb, bb);
	      if (false_bb == loop.loop_exit)
		update_loop_exit(inst->bb, bb);
	    }
	  return;
	}
      else if (inst->opcode == Op::PHI)
	{
	  new_inst = bb->build_phi_inst(inst->bitsize);
	  for (auto [arg_inst, arg_bb] : inst->phi_args)
	    {
	      new_inst->add_phi_arg(translate(arg_inst), translate(arg_bb));
	    }
	}
      else
	throw Not_implemented("unroller::duplicate: "s + inst->name());
    }
  assert(new_inst);
  curr_inst[inst] = new_inst;
}

void Unroller::unroll_one_iteration()
{
  Basic_block *current_loop_header = next_loop_header;

  // Copy the loop header.
  {
    Basic_block *src_bb = loop.bbs[0];
    Basic_block *dst_bb = current_loop_header;

    // We must translate phi nodes in two steps, because we may have
    //   .2:
    //     %10 = phi [ %7, .1 ], [ %5, .6 ], [ %49, .5 ]
    //     %12 = phi [ %5, .1 ], [ %10, .6 ], [ %10, .5 ]
    // where phi %12 uses the value of phi %10 from the previous iteration.
    // So we must translate all phi nodes before writing the new phi nodes
    // to the translation table.
    std::map<Instruction *, Instruction *> tmp_curr_inst;
    for (auto src_phi : src_bb->phis)
      {
	Instruction *dst_phi = dst_bb->build_phi_inst(src_phi->bitsize);
	tmp_curr_inst.insert({src_phi, dst_phi});
	for (auto [arg_inst, arg_bb] : src_phi->phi_args)
	  {
	    auto I = std::find(loop.bbs.begin(), loop.bbs.end(), arg_bb);
	    if (I != loop.bbs.end())
	      {
		dst_phi->add_phi_arg(translate(arg_inst), translate(arg_bb));
	      }
	  }
      }
    for (auto [phi, translated_phi] : tmp_curr_inst)
      {
	curr_inst[phi] = translated_phi;
      }

    for (Instruction *inst = src_bb->first_inst;
	 inst != src_bb->last_inst;
	 inst = inst->next)
      {
	duplicate(inst, dst_bb);
      }
  }

  curr_bb[loop.bbs[0]] = current_loop_header;
  for (size_t i = 1; i < loop.bbs.size(); i++)
    {
      Basic_block *src_bb = loop.bbs[i];
      curr_bb[src_bb] = func->build_bb();
    }

  // Copy the rest of the basic blocks for this iteration.
  for (size_t i = 1; i < loop.bbs.size(); i++)
    {
      Basic_block *src_bb = loop.bbs[i];
      Basic_block *dst_bb = translate(src_bb);
      for (auto phi : src_bb->phis)
	{
	  duplicate(phi, dst_bb);
	}
      for (Instruction *inst = src_bb->first_inst;
	   inst != src_bb->last_inst;
	   inst = inst->next)
	{
	  duplicate(inst, dst_bb);
	}
    }

  // Create the loop header for the next iteration, so that the back
  // edges will be created as branches to the next iteration.
  next_loop_header = func->build_bb();
  curr_bb[loop.bbs[0]] = next_loop_header;

  duplicate(loop.bbs[0]->last_inst, current_loop_header);
  for (size_t i = 1; i < loop.bbs.size(); i++)
    {
      Basic_block *src_bb = loop.bbs[i];
      Basic_block *dst_bb = translate(src_bb);
      duplicate(src_bb->last_inst, dst_bb);
    }

  curr_bb[loop.bbs[0]] = current_loop_header;
}

void Unroller::unroll()
{
  create_lcssa();

  Basic_block *first_unrolled = func->build_bb();
  next_loop_header = first_unrolled;

  for (int i = 0; i < unroll_limit - 1; i++)
    {
      unroll_one_iteration();
    }

  // The last block is for cases the program loops more than our unroll limit.
  // This makes our analysis invalid, so we mark this as UB.
  Basic_block *last_bb = next_loop_header;
  last_bb->build_inst(Op::UB, last_bb->value_inst(1, 1));
  last_bb->build_br_inst(loop.loop_exit);
  for (auto phi : loop.loop_exit->phis)
    {
      phi->add_phi_arg(last_bb->value_inst(0, phi->bitsize), last_bb);
    }

  // Update the original loop to only do the first iteration.
  Basic_block *loop_header = loop.bbs[0];
  std::vector<Basic_block *> deleted_branches;
  for (auto bb : loop.bbs)
    {
      assert(bb->last_inst->opcode == Op::BR);
      if (bb->last_inst->nof_args == 0)
	{
	  Basic_block *dest_bb = bb->last_inst->u.br1.dest_bb;
	  if (dest_bb == loop_header)
	    {
	      deleted_branches.push_back(bb);
	      dest_bb = first_unrolled;
	      destroy_instruction(bb->last_inst);
	      bb->build_br_inst(dest_bb);
	    }
	}
      else
	{
	  Basic_block *true_bb = bb->last_inst->u.br3.true_bb;
	  Basic_block *false_bb = bb->last_inst->u.br3.false_bb;
	  if (true_bb == loop_header || false_bb == loop_header)
	    {
	      deleted_branches.push_back(bb);
	      if (true_bb == loop_header)
		true_bb = first_unrolled;
	      if (false_bb == loop_header)
		false_bb = first_unrolled;
	      Instruction *cond = bb->last_inst->arguments[0];
	      destroy_instruction(bb->last_inst);
	      bb->build_br_inst(cond, true_bb, false_bb);
	    }
	}
    }
  for (auto bb : deleted_branches)
    {
      for (auto phi : loop_header->phis)
	{
	  phi->remove_phi_arg(bb);
	}
    }
}

} // end anonymous namespace

bool loop_unroll(Function *func)
{
  bool unrolled = false;

  while (std::optional<Loop> loop = find_loop(func))
    {
      Unroller unroller(*loop);
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
