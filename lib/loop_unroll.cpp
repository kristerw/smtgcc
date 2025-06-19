// Unroll loops.

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <optional>

#include "smtgcc.h"

#include "stdio.h"

using namespace std::string_literals;

namespace smtgcc {
namespace {

struct Loop
{
  // The basic blocks of the loop, in reverse post order.
  std::vector<Basic_block *> bbs;

  // The exit blocks for the original loop, in reverse post order.
  std::vector<Basic_block *> exit_blocks;
};

class Loop_finder
{
  Function *func;

  std::map<size_t, size_t> nbr2last;

  // Map between BB and its index in the function.
  std::map<Basic_block *, size_t> idx;

  bool is_ancestor(size_t, size_t);

public:
  Loop_finder(Function *func);
  std::optional<Loop> find_loop();
};

class Unroller
{
  Loop& loop;
  Function *func;
  std::map<Inst *, Inst *> curr_inst;
  std::map<Basic_block *, Basic_block *> curr_bb;
  Basic_block *next_loop_header = nullptr;

  void build_new_loop_exit();
  void update_loop_exit(Basic_block *orig_loop_bb,
			Basic_block *current_iter_bb, Basic_block *exit_bb);
  void duplicate(Inst *inst, Basic_block *bb);
  Inst *translate(Inst *inst);
  Basic_block *translate(Basic_block *bb);
  Inst *get_phi(std::map<Basic_block*, Inst *>& bb2phi,
		       Basic_block *bb, Inst *inst);
  void ensure_lcssa(Inst *inst);
  void create_lcssa();
  void unroll_one_iteration();

public:
  Unroller(Loop& loop);
  void unroll(int nof_unroll);
};

template <typename T>
bool contains(std::vector<T>& vec, const T& value)
{
  return std::find(vec.begin(), vec.end(), value) != vec.end();
}

template <typename T>
void push_unique(std::vector<T>& vec, const T& value)
{
  if (std::find(vec.begin(), vec.end(), value) == vec.end())
    vec.push_back(value);
}

Unroller::Unroller(Loop& loop) : loop{loop}
{
  func = loop.bbs.back()->func;
}

Inst *Unroller::get_phi(std::map<Basic_block*, Inst *>& bb2phi, Basic_block *bb, Inst *inst)
{
  if (bb2phi.contains(bb))
    return bb2phi[bb];
  assert(!contains(loop.exit_blocks, bb));

  Inst *phi = bb->build_phi_inst(inst->bitsize);
  bb2phi.insert({bb, phi});
  assert(bb->preds.size() > 0);
  for (auto pred : bb->preds)
    {
      phi->add_phi_arg(get_phi(bb2phi, pred, inst), pred);
    }

  return phi;
}

// Check that the instruction is used in LCSSA-safe way (i.e., all uses are
// either in the loop, or in a phi-node at a loop exit). If not, insert a new
// phi-node and make all loop-external uses use that.
void Unroller::ensure_lcssa(Inst *inst)
{
  std::vector<Inst *> invalid_use;
  for (auto use : inst->used_by)
    {
      if (!contains(loop.bbs, use->bb))
	push_unique(invalid_use, use);
    }

  if (invalid_use.empty())
    return;

  std::map<Basic_block*, Inst *> bb2phi;
  for (auto exit_block : loop.exit_blocks)
    {
      Inst *phi = exit_block->build_phi_inst(inst->bitsize);
      bb2phi.insert({exit_block, phi});
      for (auto pred : exit_block->preds)
	phi->add_phi_arg(inst, pred);
    }
  while (!invalid_use.empty())
    {
      Inst *use = invalid_use.back();
      invalid_use.pop_back();
      if (use->op == Op::PHI)
	{
	  for (auto phi_arg : use->phi_args)
	    {
	      if (phi_arg.inst == inst
		  && !contains(loop.bbs, phi_arg.bb))
		{
		  Inst *arg_inst = get_phi(bb2phi, phi_arg.bb, inst);
		  use->update_phi_arg(arg_inst, phi_arg.bb);
		}
	    }
	}
      else
	inst->replace_use_with(use, get_phi(bb2phi, use->bb, inst));
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
      for (Inst *inst = bb->first_inst; inst; inst = inst->next)
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
}

bool Loop_finder::is_ancestor(size_t w, size_t v)
{
  return w <= v && v <= nbr2last.at(w);
}

// Find a loop to unroll.
//
// This is an implementation of the algorithm described in the paper
// "Nesting of Reducible and Irreducible Loops" by Paul Havlak, but
// simplified to only return one loop (the innermost loop in case of
// nested loops).
std::optional<Loop> Loop_finder::find_loop()
{
  std::map<size_t, Basic_block *> nbr2bb;
  std::map<Basic_block *, size_t> bb2nbr;

  // Calculate preorder numbers.
  size_t nbr = 0;
  {
    std::set<Basic_block *> visited;
    std::vector<Basic_block *> worklist;
    worklist.push_back(func->bbs.front());
    while (!worklist.empty())
      {
	Basic_block *bb = worklist.back();
	if (visited.contains(bb))
	  {
	    worklist.pop_back();
	    nbr2last.insert({bb2nbr.at(bb), nbr - 1});
	  }
	else
	  {
	    visited.insert(bb);
	    nbr2bb.insert({nbr, bb});
	    bb2nbr.insert({bb, nbr});
	    nbr++;
	    for (auto succ : bb->succs)
	      {
		if (!visited.contains(succ))
		  worklist.push_back(succ);
	      }
	  }
      }
    assert(nbr == func->bbs.size());
  }

  // Make lists of predecessors (backedge and other).
  std::map<size_t, std::set<ssize_t>> back_preds;
  std::map<size_t, std::set<ssize_t>> nonback_preds;
  for (size_t w = 0; w < nbr; w++)
    {
      for (auto pred : nbr2bb.at(w)->preds)
	{
	  size_t v = bb2nbr.at(pred);
	  if (is_ancestor(w, v))
	    back_preds[w].insert(v);
	  else
	    nonback_preds[w].insert(v);
	}
    }

  for (ssize_t w = nbr - 1; w >= 0; w--)
    {
      std::set<size_t> P;

      for (auto v : back_preds[w])
	{
	  if (v != w)
	    P.insert(v);
	  else
	    {
	      Loop loop;
	      Basic_block *bb = nbr2bb.at(w);
	      loop.bbs.push_back(bb);
	      for (auto succ : bb->succs)
		{
		  if (succ != bb)
		    push_unique(loop.exit_blocks, succ);
		}
	      assert(loop.exit_blocks.size() < 2);
	      return loop;
	    }
	}

      std::vector<size_t> worklist(P.begin(), P.end());
      while (!worklist.empty())
	{
	  size_t x = worklist.back();
	  worklist.pop_back();
	  for (auto y : nonback_preds[x])
	    {
	      if (!is_ancestor(w, y))
		throw Not_implemented("irreducible loop");
	      if (y != w && !P.contains(y))
		{
		  P.insert(y);
		  worklist.push_back(y);
		}
	    }
	}

      if (!P.empty())
	{
	  Loop loop;
	  P.insert(w);
	  for (auto p : P)
	    {
	      Basic_block *bb = nbr2bb.at(p);
	      loop.bbs.push_back(bb);
	      for (auto succ : bb->succs)
		{
		  if (!P.contains(bb2nbr.at(succ)))
		    push_unique(loop.exit_blocks, succ);
		}
	    }

	  std::sort(loop.bbs.begin(), loop.bbs.end(),
		    [this](Basic_block *a, Basic_block *b) {
		      return idx.at(a) < idx.at(b);
		    });
	  std::sort(loop.exit_blocks.begin(), loop.exit_blocks.end(),
		    [this](Basic_block *a, Basic_block *b) {
		      return idx.at(a) < idx.at(b);
		    });

	  return loop;
	}
    }

  return {};
}

std::optional<Loop> find_loop(Function *func)
{
  Loop_finder loop_finder(func);
  std::optional<Loop> loop = loop_finder.find_loop();
  if (loop && loop->exit_blocks.empty())
    throw Not_implemented("infinite loop");
  return loop;
}

// Get the SSA variable (i.e., instruction) corresponding to the input SSA
// variable for use in this iteration.
Inst *Unroller::translate(Inst *inst)
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

// Insert new exit blocks to ensure that all predecessors in the exit block
// are within the loop and that no other basic block (except the loop header)
// has predecessors within the loop.
void Unroller::build_new_loop_exit()
{
  for (size_t i = 0; i < loop.exit_blocks.size(); i++)
    {
      Basic_block *orig_exit_block = loop.exit_blocks[i];
      Basic_block *exit_block = func->build_bb();
      exit_block->build_br_inst(orig_exit_block);
      std::map<Inst *, Inst *> phi_map;
      for (auto phi : orig_exit_block->phis)
	{
	  Inst *new_phi = exit_block->build_phi_inst(phi->bitsize);
	  phi_map.insert({phi, new_phi});
	  phi->add_phi_arg(new_phi, exit_block);
	}
      loop.exit_blocks[i] = exit_block;

      // Update the branches within the loop to use the new loop exit.
      for (auto bb : loop.bbs)
	{
	  assert(bb->last_inst->op == Op::BR);
	  bool updated = false;
	  if (bb->last_inst->nof_args == 0)
	    {
	      Basic_block *dest_bb = bb->last_inst->u.br1.dest_bb;
	      if (dest_bb == orig_exit_block)
		{
		  destroy_instruction(bb->last_inst);
		  bb->build_br_inst(exit_block);
		  updated = true;
		}
	    }
	  else
	    {
	      Inst *arg = bb->last_inst->args[0];
	      Basic_block *true_bb = bb->last_inst->u.br3.true_bb;
	      Basic_block *false_bb = bb->last_inst->u.br3.false_bb;
	      if (true_bb == orig_exit_block || false_bb == orig_exit_block)
		{
		  if (true_bb == orig_exit_block)
		    true_bb = exit_block;
		  if (false_bb == orig_exit_block)
		    false_bb = exit_block;
		  destroy_instruction(bb->last_inst);
		  bb->build_br_inst(arg, true_bb, false_bb);
		  updated = true;
		}
	    }
	  if (updated)
	    {
	      for (auto phi : orig_exit_block->phis)
		{
		  Inst *phi_arg = phi->get_phi_arg(bb);
		  phi->remove_phi_arg(bb);
		  phi_map.at(phi)->add_phi_arg(phi_arg, bb);
		}
	    }
	}
    }
}

// Update phi nodes in the exit block to handle a new predecessor for the
// current iteration of the loop.
void Unroller::update_loop_exit(Basic_block *orig_loop_bb, Basic_block *current_iter_bb, Basic_block *exit_bb)
{
  for (auto phi : exit_bb->phis)
    {
      Inst *inst = phi->get_phi_arg(orig_loop_bb);
      phi->add_phi_arg(translate(inst), current_iter_bb);
    }
}

void Unroller::duplicate(Inst *inst, Basic_block *bb)
{
  Inst *new_inst = nullptr;
  Inst_class iclass = inst->iclass();
  switch (iclass)
    {
    case Inst_class::iunary:
    case Inst_class::funary:
    case Inst_class::ls_unary:
    case Inst_class::reg_unary:
    case Inst_class::solver_unary:
      {
	Inst *arg = translate(inst->args[0]);
	new_inst = bb->build_inst(inst->op, arg);
      }
      break;
    case Inst_class::icomparison:
    case Inst_class::fcomparison:
    case Inst_class::ibinary:
    case Inst_class::fbinary:
    case Inst_class::conv:
    case Inst_class::ls_binary:
    case Inst_class::reg_binary:
    case Inst_class::solver_binary:
      {
	Inst *arg1 = translate(inst->args[0]);
	Inst *arg2 = translate(inst->args[1]);
	new_inst = bb->build_inst(inst->op, arg1, arg2);
      }
      break;
    case Inst_class::ternary:
    case Inst_class::ls_ternary:
    case Inst_class::solver_ternary:
      {
	Inst *arg1 = translate(inst->args[0]);
	Inst *arg2 = translate(inst->args[1]);
	Inst *arg3 = translate(inst->args[2]);
	new_inst = bb->build_inst(inst->op, arg1, arg2, arg3);
      }
      break;
    default:
      if (inst->op == Op::BR)
	{
	  if (inst->nof_args == 0)
	    {
	      Basic_block *dest_bb = translate(inst->u.br1.dest_bb);
	      bb->build_br_inst(dest_bb);
	      if (contains(loop.exit_blocks, dest_bb))
		update_loop_exit(inst->bb, bb, dest_bb);
	    }
	  else
	    {
	      Inst *arg = translate(inst->args[0]);
	      Basic_block *true_bb = translate(inst->u.br3.true_bb);
	      Basic_block *false_bb = translate(inst->u.br3.false_bb);
	      bb->build_br_inst(arg, true_bb, false_bb);
	      if (contains(loop.exit_blocks, true_bb))
		update_loop_exit(inst->bb, bb, true_bb);
	      if (contains(loop.exit_blocks, false_bb))
		update_loop_exit(inst->bb, bb, false_bb);
	    }
	  return;
	}
      else if (inst->op == Op::PHI)
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
    std::map<Inst *, Inst *> tmp_curr_inst;
    for (auto src_phi : src_bb->phis)
      {
	Inst *dst_phi = dst_bb->build_phi_inst(src_phi->bitsize);
	tmp_curr_inst.insert({src_phi, dst_phi});
	for (auto [arg_inst, arg_bb] : src_phi->phi_args)
	  {
	    if (contains(loop.bbs, arg_bb))
	      dst_phi->add_phi_arg(translate(arg_inst), translate(arg_bb));
	  }
      }
    for (auto [phi, translated_phi] : tmp_curr_inst)
      {
	curr_inst[phi] = translated_phi;
      }

    for (Inst *inst = src_bb->first_inst;
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
      for (Inst *inst = src_bb->first_inst;
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

void Unroller::unroll(int nof_unroll)
{
  create_lcssa();

  Basic_block *first_unrolled = func->build_bb();
  next_loop_header = first_unrolled;

  for (int i = 0; i < nof_unroll - 1; i++)
    {
      unroll_one_iteration();
    }

  // The last block is for cases the program loops more than our unroll limit.
  // This makes our analysis invalid, so we mark this as UB.
  // We must make it branch to an extit block. It does not matter which
  // exit block we use, but it seems likely that the last exit block is
  // the best.
  Basic_block *last_exit_block = loop.exit_blocks.back();
  Basic_block *last_bb = next_loop_header;
  last_bb->build_inst(Op::UB, last_bb->value_inst(1, 1));
  last_bb->build_br_inst(last_exit_block);
  for (auto phi : last_exit_block->phis)
    {
      phi->add_phi_arg(last_bb->value_inst(0, phi->bitsize), last_bb);
    }

  // Update the original loop to only do the first iteration.
  Basic_block *loop_header = loop.bbs[0];
  std::vector<Basic_block *> deleted_branches;
  for (auto bb : loop.bbs)
    {
      assert(bb->last_inst->op == Op::BR);
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
	      Inst *cond = bb->last_inst->args[0];
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

bool loop_unroll(Function *func, int nof_unroll)
{
  bool unrolled = false;

  while (std::optional<Loop> loop = find_loop(func))
    {
      Unroller unroller(*loop);
      unroller.unroll(nof_unroll);
      reverse_post_order(func);
      simplify_insts(func);
      dead_code_elimination(func);
      simplify_cfg(func);
      unrolled = true;
    }

  // Report error if we could not unroll all loops.
  if (has_loops(func))
    throw Not_implemented("loops");

  func->loop_unrolling_done = true;

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
