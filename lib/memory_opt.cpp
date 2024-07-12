#include <algorithm>
#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

// Return a vector containing all the function's memory instructions.
std::vector<Instruction *> collect_mem(Function *func)
{
  std::vector<Instruction *> mem;
  for (Instruction *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->opcode == Op::MEMORY)
	mem.push_back(inst);
    }
  return mem;
}

// Ensure the memory instructions in the IR come in the same order as in the
// mem vector.
void reorder_mem(std::vector<Instruction *>& mem)
{
  if (mem.empty())
    return;

  Basic_block *bb = mem[0]->bb;
  Instruction *curr_inst = bb->first_inst;
  while (curr_inst->opcode == Op::VALUE)
    curr_inst = curr_inst->next;
  if (curr_inst != mem[0])
    {
      mem[0]->move_before(curr_inst);
      curr_inst = mem[0];
    }
  for (auto inst : mem)
    {
      if (inst != curr_inst)
	inst->move_after(curr_inst);
      curr_inst = inst;
    }
}

// Make a copy of inst in the first BB of the function func.
Instruction *clone_inst(Instruction *inst, Function *func)
{
  Basic_block *bb = func->bbs[0];
  if (inst->opcode == Op::VALUE)
    {
      return bb->value_inst(inst->value(), inst->bitsize);
    }

  if (inst->opcode == Op::MEMORY)
    {
      Instruction *arg1 = clone_inst(inst->arguments[0], func);
      Instruction *arg2 = clone_inst(inst->arguments[1], func);
      Instruction *arg3 = clone_inst(inst->arguments[2], func);
      return bb->build_inst(Op::MEMORY, arg1, arg2, arg3);
    }

  throw smtgcc::Not_implemented("clone_inst: unhandled instruction");
}

// Return true if the instruction is an unused memory instruction.
// Usage in the entry BB is not counted -- those are only used for
// initialization and are therefore not relevant for the function if
// the memory does not have any other use.
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
  std::vector<Instruction *> worklist;
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

  // We have now verified that all uses of memory_inst are in the entry block.
  // However, this does not guarantee that the memory block is unused! For
  // example, consider code of the form
  //   int a;
  //   int *p = &a;
  // where the use of `a` in the entry block is used to initialize `p`,
  // which may be used in the body of the function. We must therefore check
  // that all the store instructions (and other "sink" instructions) identified
  // above operate on the memory_inst memory block.
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

void store_load_forwarding(Function *func)
{
  std::map<Basic_block *, std::map<uint64_t, Instruction *>> bb2mem_undef;
  std::map<Basic_block *, std::map<uint64_t, Instruction *>> bb2mem_flag;
  std::map<Basic_block *, std::map<uint64_t, Instruction *>> bb2stores;

  for (auto bb : func->bbs)
    {
      std::map<uint64_t, Instruction *> mem_undef;
      std::map<uint64_t, Instruction *> mem_flag;
      std::map<uint64_t, Instruction *> stores;

      if (bb->preds.size() == 1)
	{
	  mem_undef = bb2mem_undef.at(bb->preds[0]);
	  mem_flag = bb2mem_flag.at(bb->preds[0]);
	  stores = bb2stores.at(bb->preds[0]);
	}
      else if (bb->preds.size() == 2)
	{
	  if (bb2mem_undef.at(bb->preds[0]) == bb2mem_undef.at(bb->preds[1]))
	    mem_undef = bb2mem_undef.at(bb->preds[0]);
	  if (bb2mem_flag.at(bb->preds[0]) == bb2mem_flag.at(bb->preds[1]))
	    mem_flag = bb2mem_flag.at(bb->preds[0]);
	  if (bb2stores.at(bb->preds[0]) == bb2stores.at(bb->preds[1]))
	    stores = bb2stores.at(bb->preds[0]);
	}

      for (Instruction *inst = bb->first_inst; inst;)
	{
	  Instruction *next_inst = inst->next;

	  switch (inst->opcode)
	    {
	    case Op::MEMORY:
	      {
		uint64_t id = inst->arguments[0]->value();
		uint64_t size = inst->arguments[1]->value();
		uint32_t flags = inst->arguments[2]->value();
		uint64_t addr = id << inst->bb->func->module->ptr_id_low;
		Instruction *undef;
		if (flags & MEM_UNINIT)
		  undef = bb->value_inst(255, 8);
		else
		  undef = bb->value_inst(0, 8);
		for (uint64_t i = 0; i < size; i++)
		  {
		    mem_undef[addr + i] = undef;
		  }
	      }
	      break;
	    case Op::SET_MEM_UNDEF:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  mem_undef[ptr->value()] = inst;
		else
		  mem_undef.clear();
	      }
	      break;
	    case Op::GET_MEM_UNDEF:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (mem_undef.contains(ptr_val))
		      {
			Instruction *value = mem_undef.at(ptr_val);
			if (value->opcode == Op::SET_MEM_UNDEF)
			  value = value->arguments[1];
			else
			  assert(value->opcode == Op::VALUE);
			inst->replace_all_uses_with(value);
			destroy_instruction(inst);
		      }
		  }
	      }
	      break;
	    case Op::SET_MEM_FLAG:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  mem_flag[ptr->value()] = inst;
		else
		  mem_flag.clear();
	      }
	      break;
	    case Op::GET_MEM_FLAG:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (mem_flag.contains(ptr_val))
		      {
			Instruction *set_mem_flag = mem_flag.at(ptr_val);
			Instruction *value = set_mem_flag->arguments[1];
			inst->replace_all_uses_with(value);
			destroy_instruction(inst);
		      }
		  }
	      }
	      break;
	    case Op::STORE:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  stores[ptr->value()] = inst;
		else
		  stores.clear();
	      }
	      break;
	    case Op::LOAD:
	      {
		Instruction *ptr = inst->arguments[0];
		if (ptr->opcode == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (stores.contains(ptr_val))
		      {
			Instruction *store = stores.at(ptr_val);
			Instruction *value = store->arguments[1];
			inst->replace_all_uses_with(value);
			destroy_instruction(inst);
		      }
		  }
	      }
	      break;
	    default:
	      break;
	    }

	  inst = next_inst;
	}

      bb2mem_undef[bb] = std::move(mem_undef);
      bb2mem_flag[bb] = std::move(mem_flag);
      bb2stores[bb] = std::move(stores);
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

// This function makes the memory instructions consistent between src and tgt:
//  * src and tgt have the same memory instructions
//  * memory that is unused in both src and tgt is removed
//  * The memory instructions are emitted in the same order
// This makes checking faster as more can be CSEd between src and tgt.
void canonicalize_memory(Module *module)
{
  Function *src = module->functions[0];
  Function *tgt = module->functions[1];
  if (src->name != "src")
    std::swap(src, tgt);

  struct {
    bool operator()(const Instruction *a, const Instruction *b) const {
      return a->arguments[0]->value() < b->arguments[0]->value();
    }
  } comp;

  std::vector<Instruction *> src_mem = collect_mem(src);
  std::vector<Instruction *> tgt_mem = collect_mem(tgt);
  std::sort(src_mem.begin(), src_mem.end(), comp);
  std::sort(tgt_mem.begin(), tgt_mem.end(), comp);

  // Add missing memory instructions.
  std::vector<Instruction *> missing_src;
  std::vector<Instruction *> missing_tgt;
  std::set_difference(tgt_mem.begin(), tgt_mem.end(),
		      src_mem.begin(), src_mem.end(),
		      std::back_inserter(missing_src), comp);
  std::set_difference(src_mem.begin(), src_mem.end(),
		      tgt_mem.begin(), tgt_mem.end(),
		      std::back_inserter(missing_tgt), comp);
  for (auto inst : missing_src)
    {
      src_mem.push_back(clone_inst(inst, src));
    }
  for (auto inst : missing_tgt)
    {
      tgt_mem.push_back(clone_inst(inst, tgt));
    }

  // Ensure that both src and tgt use the same instruction order.
  std::sort(src_mem.begin(), src_mem.end(), comp);
  std::sort(tgt_mem.begin(), tgt_mem.end(), comp);
  reorder_mem(src_mem);
  reorder_mem(tgt_mem);

  // Remove memory that is unused in both src and tgt.
  assert(src_mem.size() == tgt_mem.size());
  bool removed_mem = false;
  for (size_t i = 0; i < src_mem.size(); i++)
    {
      __int128 src_arg1 = src_mem[i]->arguments[0]->value();
      __int128 src_arg2 = src_mem[i]->arguments[1]->value();
      __int128 src_arg3 = src_mem[i]->arguments[2]->value();
      __int128 tgt_arg1 = tgt_mem[i]->arguments[0]->value();
      __int128 tgt_arg2 = tgt_mem[i]->arguments[1]->value();
      __int128 tgt_arg3 = tgt_mem[i]->arguments[2]->value();
      if (!(src_arg3 & MEM_KEEP)
	  && src_arg1 == tgt_arg1
	  && src_arg2 == tgt_arg2
	  && src_arg3 == tgt_arg3
	  && is_unused_memory(src_mem[i])
	  && is_unused_memory(tgt_mem[i]))
	{
	  removed_mem = true;
	  remove_unused_memory(src_mem[i]);
	  remove_unused_memory(tgt_mem[i]);
	}
    }

  // Removing memory may open up new opportunities. For example, consider:
  //   int b;
  //   int *p = &b;
  // b may become dead after we remove p. Therefore, we need to rerun the
  // pass.
  if (removed_mem)
    canonicalize_memory(module);
}

void ls_elim(Function *func)
{
  store_load_forwarding(func);
  dead_store_elim(func);
}

void ls_elim(Module *module)
{
  for (auto func : module->functions)
    ls_elim(func);
}

} // end namespace smtgcc
