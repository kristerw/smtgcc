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

class Loop_finder
{
  Function *func;

  // Map between BB and its index in the function.
  std::map<Basic_block *, size_t> idx;

  // State for Tarjan's algorithm for calculating strongly connected
  // components.
  std::map<Basic_block *, int> dfn;
  std::map<Basic_block *, int> lowlink;
  std::vector<Basic_block *> stack;
  std::vector<std::vector<Basic_block *>> sccs;
  int dfnum = 0;

  Basic_block *get_loop_exit(std::vector<Basic_block *>& scc);
  void strong_connect(Basic_block *bb);

public:
  Loop_finder(Function *func);
  std::optional<Loop> find_loop();
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
      for (auto pred : loop.loop_exit->preds)
	phi->add_phi_arg(inst, pred);
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

Loop_finder::Loop_finder(Function *func) : func{func}
{
  for (size_t i = 0; i < func->bbs.size(); i++)
    {
      idx.insert({func->bbs[i], i});
    }
  for (auto bb : func->bbs)
    {
      if (!dfn.contains(bb))
	strong_connect(bb);
    }
}

std::optional<Loop> Loop_finder::find_loop()
{
  for (auto& ssc : sccs)
    {
      Basic_block *loop_exit = get_loop_exit(ssc);
      if (loop_exit)
	{
	  Loop loop;
	  loop.bbs = ssc;
	  loop.loop_exit = loop_exit;
	  return loop;
	}
    }
  return {};
}

void Loop_finder::strong_connect(Basic_block *bb)
{
  dfn.insert({bb, dfnum});
  lowlink.insert({bb, dfnum});
  stack.push_back(bb);
  dfnum++;

  for (auto succ : bb->succs)
    {
      if (!dfn.contains(succ))
	{
	  strong_connect(succ);
	  lowlink[bb] = std::min(lowlink.at(bb), lowlink.at(succ));
	}
      else if (dfn.at(succ) < dfn.at(bb)
	       && std::find(stack.begin(), stack.end(), succ) != stack.end())
	{
	  lowlink[bb] = std::min(lowlink.at(bb), dfn.at(succ));
	}
    }

  if (lowlink.at(bb) == dfn.at(bb))
    {
      auto it = std::find(stack.begin(), stack.end(), bb);
      sccs.emplace_back(it, stack.end());
      stack.erase(it, stack.end());

      // Sort the blocks in reverse post order.
      std::vector<Basic_block*>& scc = sccs.back();
      std::sort(scc.begin(), scc.end(),
		[this](Basic_block *a, Basic_block *b) {
		  return idx.at(a) < idx.at(b);
		});
    }
}

Basic_block *Loop_finder::get_loop_exit(std::vector<Basic_block *>& scc)
{
  // The loop must have at least one back edge, and all back edges must
  // be to the loop header.
  Basic_block *loop_header = scc.front();
  bool found_back_edge = false;
  for (auto bb : scc)
    {
      for (auto succ : bb->succs)
	{
	  if (idx.at(succ) <= idx.at(bb))
	    {
	      found_back_edge = true;
	      if (succ != loop_header)
		return nullptr;
	    }
	}
    }
  if (!found_back_edge)
    return nullptr;

  // Only the loop header may have predecessors outside the loop.
  for (size_t i = 1; i < scc.size(); i++)
    {
      for (auto pred : scc[i]->preds)
	{
	  if (std::find(scc.begin(), scc.end(), pred) == scc.end())
	    return nullptr;
	}
    }

  // All successors must be in the loop, or the exit_block.
  Basic_block *loop_exit = nullptr;
  for (auto bb : scc)
    {
      for (auto succ : bb->succs)
	{
	  if (std::find(scc.begin(), scc.end(), succ) == scc.end())
	    {
	      if (!loop_exit)
		loop_exit = succ;
	      else if (succ != loop_exit)
		return nullptr;
	    }
	}
    }

  return loop_exit;
}

std::optional<Loop> find_loop(Function *func)
{
  Loop_finder loop_finder(func);
  return loop_finder.find_loop();
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
      simplify_insts(func);
      dead_code_elimination(func);
      simplify_cfg(func);
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
