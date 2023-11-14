#include <algorithm>
#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

bool is_global_memory(Instruction *inst)
{
  if (inst->opcode != Op::MEMORY)
    return false;
  uint64_t id = inst->arguments[0]->value();
  return (id >> (inst->bb->func->module->ptr_id_bits - 1)) == 0;
}

Instruction *find_mem(const std::vector<Instruction *>& mem, Instruction *id)
{
  for (Instruction *inst : mem)
    {
      // Must compare the values as the instructions are from different
      // functions.
      if (inst->arguments[0]->value() == id->value())
	return inst;
    }
  return nullptr;
}

// Return true if the instruction is an unused memory instruction.
// Use in the entry BB is not counted -- those are only used for initialization
// and are therefore not relevant for the function if the memory does not
// have any other use.
bool is_unused_memory(Instruction *memory_inst)
{
  if (memory_inst->opcode != Op::MEMORY)
    return false;
  Basic_block *entry_bb = memory_inst->bb->func->bbs[0];
  assert(memory_inst->bb == entry_bb);
  if (memory_inst->used_by.empty())
    return true;

  // Check that all uses (and uses of uses) are in the entry block.
  // If not, then the memory_inst is not unused.
  std::set<Instruction *> visited;
  std::vector<Instruction *> sinks;
  uint64_t memory_size = memory_inst->arguments[1]->value();
  sinks.reserve(4 * memory_size);
  std::vector<Instruction *> worklist;
  worklist.reserve(4 * memory_size);
  worklist.insert(std::end(worklist), std::begin(memory_inst->used_by),
		  std::end(memory_inst->used_by));
  while (!worklist.empty())
    {
      Instruction *inst = worklist.back();
      worklist.pop_back();
      if (inst->bb != entry_bb)
	return false;
      if (visited.contains(inst))
	continue;

      visited.insert(inst);
      for (auto used_by : inst->used_by)
	{
	  if (used_by->opcode == Op::SET_MEM_UNDEF
	      || used_by->opcode == Op::STORE)
	    sinks.push_back(used_by);
	  else
	    worklist.push_back(used_by);
	}
    }

  // We have now verified that all use of memory_inst is in the entry block.
  // But this does not guarantee that the memory block is unused! We could
  // for example have code of the form
  //   int a;
  //   int *p = &a;
  // where the use of `a` in the entry block is used to initialize `p`
  // which may be used in the real function. We therefore must check that
  // all the store instructions (and other "sink" instructions) identified
  // above work on the memory_inst memory block.
  visited.clear();
  for (auto sink_inst : sinks)
    {
      assert(worklist.empty());

      worklist.push_back(sink_inst->arguments[0]);
      while (!worklist.empty())
	{
	  Instruction *inst = worklist.back();
	  worklist.pop_back();
	  if (visited.contains(inst))
	    continue;
	  visited.insert(inst);
	  if (inst->opcode == Op::VALUE)
	    {
	      // A constant is always a valid starting point. Nothing to do.
	    }
	  else if (inst->opcode == Op::MEMORY)
	    {
	      if (inst != memory_inst)
		{
		  // memory_inst is used when initializing a different
		  // memory block. I.e. memory_inst is not unused.
		  return false;
		}
	    }
	  else
	    {
	      assert(inst->nof_args > 0);
	      for (uint64_t i = 0; i < inst->nof_args; i++)
		worklist.push_back(inst->arguments[i]);
	    }
	}
    }

  return true;
}

void remove_unused_memory(Instruction *memory_inst)
{
  Basic_block *entry_bb = memory_inst->bb->func->bbs[0];
  assert(memory_inst->opcode == Op::MEMORY);
  assert(memory_inst->bb == entry_bb);

  std::vector<Instruction *> worklist;
  uint64_t memory_size = memory_inst->arguments[1]->value();
  worklist.reserve(4 * memory_size);
  worklist.push_back(memory_inst);
  while (!worklist.empty())
    {
      Instruction *inst = worklist.back();
      assert(inst->bb == entry_bb);
      if (inst->used_by.empty())
	{
	  worklist.pop_back();
	  destroy_instruction(inst);
	}
      else
	{
	  Instruction *used_by = *inst->used_by.begin();
	  worklist.push_back(used_by);
	}
    }
}

void dead_store_elim(Function *func)
{
  std::map<uint64_t, Instruction *> mem_undef;
  std::map<uint64_t, Instruction *> mem_flag;
  std::map<uint64_t, Instruction *> stores;
  Basic_block *prev_bb = nullptr;
  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];
      if (bb->succs.size() != 1 || bb->succs[0] != prev_bb)
	{
	  mem_undef.clear();
	  mem_flag.clear();
	  stores.clear();
	}

      for (Instruction *inst = bb->last_inst; inst;)
	{
	  Instruction *next_inst = inst->prev;

	  switch (inst->opcode)
	    {
	    case Op::SET_MEM_UNDEF:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (mem_undef.contains(ptr_val))
		      destroy_instruction(inst);
		    else
		      mem_undef[ptr_val] = inst;
		  }
	      }
	      break;
	    case Op::GET_MEM_UNDEF:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  mem_undef.erase(ptr->value());
		else
		  mem_undef.clear();
	      }
	      break;
	    case Op::SET_MEM_FLAG:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (mem_flag.contains(ptr_val))
		      destroy_instruction(inst);
		    else
		      mem_flag[ptr_val] = inst;
		  }
	      }
	      break;
	    case Op::GET_MEM_FLAG:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  mem_flag.erase(ptr->value());
		else
		  mem_flag.clear();
	      }
	      break;
	    case Op::STORE:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (stores.contains(ptr_val))
		      destroy_instruction(inst);
		    else
		      stores[ptr_val] = inst;
		  }
	      }
	      break;
	    case Op::LOAD:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  stores.erase(ptr->value());
		else
		  stores.clear();
	      }
	      break;
	    default:
	      break;
	    }

	  inst = next_inst;
	}
      prev_bb = bb;
    }
}

} // end anonymous namespace

// TODO: This pass is gimple_conv-specific. Move to gimple_conv.cpp?
void canonicalize_memory(Module *module)
{
  Function *src = module->functions[0];
  Function *tgt = module->functions[1];
  if (src->name != "src")
    std::swap(src, tgt);

  bool removed_mem = false;
  std::vector<Instruction *> src_mem;
  std::vector<Instruction *> tgt_mem;
  std::vector<Instruction *> remove_mem;
  for (Instruction *inst = src->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->opcode != Op::MEMORY)
	continue;

      uint32_t flags = inst->arguments[2]->value();
      if (is_global_memory(inst))
	src_mem.push_back(inst);
      else if (!(flags & MEM_KEEP) && is_unused_memory(inst))
	remove_mem.push_back(inst);
    }
  for (Instruction *inst = tgt->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->opcode != Op::MEMORY)
	continue;

      uint32_t flags = inst->arguments[2]->value();
      if (is_global_memory(inst))
	tgt_mem.push_back(inst);
      else if (!(flags & MEM_KEEP) && is_unused_memory(inst))
	remove_mem.push_back(inst);
    }
  for (Instruction *inst : remove_mem)
    {
      removed_mem = true;
      remove_unused_memory(inst);
    }

  for (Instruction *src_inst : src_mem)
    {
      Instruction *tgt_inst = find_mem(tgt_mem, src_inst->arguments[0]);
      if (tgt_inst)
	{
	  auto it = std::find(tgt_mem.begin(), tgt_mem.end(), tgt_inst);
	  tgt_mem.erase(it);

	  uint32_t src_flags = src_inst->arguments[2]->value();
	  uint32_t tgt_flags = tgt_inst->arguments[2]->value();
	  if (is_unused_memory(src_inst)
	      && !(src_flags & MEM_KEEP)
	      && is_unused_memory(tgt_inst)
	      && !(tgt_flags & MEM_KEEP))
	    {
	      remove_unused_memory(src_inst);
	      remove_unused_memory(tgt_inst);
	      removed_mem = true;
	    }
	}
      else
	{
	  uint32_t src_flags = src_inst->arguments[2]->value();
	  if (is_unused_memory(src_inst)
	      && !(src_flags & MEM_KEEP))
	    {
	      remove_unused_memory(src_inst);
	      removed_mem = true;
	    }
	  else
	    {
	      uint64_t id = src_inst->arguments[0]->value();
	      uint64_t size = src_inst->arguments[1]->value();
	      uint32_t flags = src_inst->arguments[2]->value();

	      Basic_block *bb = tgt->bbs[0];
	      uint32_t ptr_id_bits = bb->func->module->ptr_id_bits;
	      uint32_t ptr_offset_bits = bb->func->module->ptr_offset_bits;
	      Instruction *arg1 = bb->value_inst(id, ptr_id_bits);
	      Instruction *arg2 = bb->value_inst(size, ptr_offset_bits);
	      Instruction *arg3 = bb->value_inst(flags, 32);
	      bb->build_inst(Op::MEMORY, arg1, arg2, arg3);
	    }
	}
    }
  if (!tgt_mem.empty())
    {
      for (Instruction *tgt_inst : tgt_mem)
	{
	  uint32_t tgt_flags = tgt_inst->arguments[2]->value();
	  if (is_unused_memory(tgt_inst)
	      && !(tgt_flags & MEM_KEEP))
	    {
	      remove_unused_memory(tgt_inst);
	      removed_mem = true;
	    }
	  else
	    {
	      // TODO: Add missing src memory.
	      //       But this should not really happen as the memory added
	      //       by the compiler should be marked artificial, and
	      //       therefore not being treated as global mem by smtgcc.
	      throw smtgcc::Not_implemented("canonicalize_memory: missing src memory");
	    }
	}
    }

  // Removing memory may give us new opportunities. For example, for
  //   int b;
  //   int *p = &b;
  // b may be dead when we remove p. So we need to rerun the pass.
  if (removed_mem)
    canonicalize_memory(module);
}

void ls_elim(Function *func)
{
  dead_store_elim(func);
}

void ls_elim(Module *module)
{
  for (auto func : module->functions)
    ls_elim(func);
}

} // end namespace smtgcc
