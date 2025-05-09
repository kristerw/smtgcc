#include <algorithm>
#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

// Maximum number of bytes we track in an object.
uint64_t max_mem_unroll_limit = 10000;

bool is_local_memory(Inst *inst)
{
  if (inst->op != Op::MEMORY)
    return false;
  uint64_t id = inst->args[0]->value();
  return (id >> (inst->bb->func->module->ptr_id_bits - 1)) != 0;
}

// Return a vector containing all the function's memory instructions.
std::vector<Inst *> collect_mem(Function *func)
{
  std::vector<Inst *> mem;
  for (Inst *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->op == Op::MEMORY)
	mem.push_back(inst);
    }
  return mem;
}

// Ensure the memory instructions in the IR come in the same order as in the
// mem vector.
void reorder_mem(std::vector<Inst *>& mem)
{
  if (mem.empty())
    return;

  Basic_block *bb = mem[0]->bb;
  Inst *curr_inst = bb->first_inst;
  while (curr_inst->op == Op::VALUE)
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
Inst *clone_inst(Inst *inst, Function *func)
{
  Basic_block *bb = func->bbs[0];
  if (inst->op == Op::VALUE)
    {
      return bb->value_inst(inst->value(), inst->bitsize);
    }

  if (inst->op == Op::MEMORY)
    {
      Inst *arg1 = clone_inst(inst->args[0], func);
      Inst *arg2 = clone_inst(inst->args[1], func);
      Inst *arg3 = clone_inst(inst->args[2], func);
      return bb->build_inst(Op::MEMORY, arg1, arg2, arg3);
    }

  throw smtgcc::Not_implemented("clone_inst: unhandled instruction");
}

// Return true if the instruction is an unused memory instruction.
// Usage in the entry BB is not counted -- those are only used for
// initialization and are therefore not relevant for the function if
// the memory does not have any other use.
bool is_unused_memory(Inst *memory_inst)
{
  if (memory_inst->op != Op::MEMORY)
    return false;
  Basic_block *entry_bb = memory_inst->bb->func->bbs[0];
  assert(memory_inst->bb == entry_bb);
  if (memory_inst->used_by.empty())
    return true;

  // Check that all uses (and uses of uses) are in the entry block.
  // If not, then the memory_inst is not unused.
  std::set<Inst *> visited;
  std::vector<Inst *> sinks;
  std::vector<Inst *> worklist;
  worklist.insert(std::end(worklist), std::begin(memory_inst->used_by),
		  std::end(memory_inst->used_by));
  while (!worklist.empty())
    {
      Inst *inst = worklist.back();
      worklist.pop_back();
      if (inst->bb != entry_bb)
	return false;
      if (visited.contains(inst))
	continue;

      visited.insert(inst);
      for (auto used_by : inst->used_by)
	{
	  if (used_by->op == Op::SET_MEM_INDEF
	      || used_by->op == Op::STORE)
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

      worklist.push_back(sink_inst->args[0]);
      while (!worklist.empty())
	{
	  Inst *inst = worklist.back();
	  worklist.pop_back();
	  if (visited.contains(inst))
	    continue;
	  visited.insert(inst);
	  if (inst->op == Op::VALUE)
	    {
	      // A constant is always a valid starting point. Nothing to do.
	    }
	  else if (inst->op == Op::MEMORY)
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
		worklist.push_back(inst->args[i]);
	    }
	}
    }

  return true;
}

void remove_unused_memory(Inst *memory_inst)
{
  Basic_block *entry_bb = memory_inst->bb->func->bbs[0];
  assert(memory_inst->op == Op::MEMORY);
  assert(memory_inst->bb == entry_bb);

  std::vector<Inst *> worklist;
  worklist.push_back(memory_inst);
  while (!worklist.empty())
    {
      Inst *inst = worklist.back();
      assert(inst->bb == entry_bb);
      if (inst->used_by.empty())
	{
	  worklist.pop_back();
	  destroy_instruction(inst);
	}
      else
	{
	  Inst *used_by = *inst->used_by.begin();
	  worklist.push_back(used_by);
	}
    }
}

void store_load_forwarding(Function *func)
{
  std::map<Basic_block *, std::map<uint64_t, Inst *>> bb2mem_indef;
  std::map<Basic_block *, std::map<uint64_t, Inst *>> bb2mem_flag;
  std::map<Basic_block *, std::map<uint64_t, Inst *>> bb2stores;

  for (auto bb : func->bbs)
    {
      std::map<uint64_t, Inst *> mem_indef;
      std::map<uint64_t, Inst *> mem_flag;
      std::map<uint64_t, Inst *> stores;

      if (bb->preds.size() == 1)
	{
	  mem_indef = bb2mem_indef.at(bb->preds[0]);
	  mem_flag = bb2mem_flag.at(bb->preds[0]);
	  stores = bb2stores.at(bb->preds[0]);
	}
      else if (bb->preds.size() == 2)
	{
	  if (bb2mem_indef.at(bb->preds[0]) == bb2mem_indef.at(bb->preds[1]))
	    mem_indef = bb2mem_indef.at(bb->preds[0]);
	  if (bb2mem_flag.at(bb->preds[0]) == bb2mem_flag.at(bb->preds[1]))
	    mem_flag = bb2mem_flag.at(bb->preds[0]);
	  if (bb2stores.at(bb->preds[0]) == bb2stores.at(bb->preds[1]))
	    stores = bb2stores.at(bb->preds[0]);
	}

      for (Inst *inst = bb->first_inst; inst;)
	{
	  Inst *next_inst = inst->next;

	  switch (inst->op)
	    {
	    case Op::MEMORY:
	      {
		uint64_t id = inst->args[0]->value();
		uint64_t size = inst->args[1]->value();
		uint32_t flags = inst->args[2]->value();
		uint64_t addr = id << func->module->ptr_id_low;
		Inst *indef;
		if (flags & MEM_UNINIT)
		  indef = bb->value_inst(255, 8);
		else
		  indef = bb->value_inst(0, 8);
		size = std::min(size, max_mem_unroll_limit);
		for (uint64_t i = 0; i < size; i++)
		  {
		    mem_indef[addr + i] = indef;
		  }
	      }
	      break;
	    case Op::SET_MEM_INDEF:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  mem_indef[ptr->value()] = inst;
		else
		  mem_indef.clear();
	      }
	      break;
	    case Op::GET_MEM_INDEF:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (mem_indef.contains(ptr_val))
		      {
			Inst *value = mem_indef.at(ptr_val);
			if (value->op == Op::SET_MEM_INDEF)
			  value = value->args[1];
			else
			  assert(value->op == Op::VALUE);
			inst->replace_all_uses_with(value);
			destroy_instruction(inst);
		      }
		  }
	      }
	      break;
	    case Op::SET_MEM_FLAG:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  mem_flag[ptr->value()] = inst;
		else
		  mem_flag.clear();
	      }
	      break;
	    case Op::GET_MEM_FLAG:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (mem_flag.contains(ptr_val))
		      {
			Inst *set_mem_flag = mem_flag.at(ptr_val);
			Inst *value = set_mem_flag->args[1];
			inst->replace_all_uses_with(value);
			destroy_instruction(inst);
		      }
		  }
	      }
	      break;
	    case Op::STORE:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  stores[ptr->value()] = inst;
		else
		  stores.clear();
	      }
	      break;
	    case Op::LOAD:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (stores.contains(ptr_val))
		      {
			Inst *store = stores.at(ptr_val);
			Inst *value = store->args[1];
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

      bb2mem_indef[bb] = std::move(mem_indef);
      bb2mem_flag[bb] = std::move(mem_flag);
      bb2stores[bb] = std::move(stores);
    }
}

void dead_store_elim(Function *func)
{
  std::set<uint64_t> mem_indef;
  std::set<uint64_t> mem_flag;
  std::set<uint64_t> stores;

  // Seed the sets with the addresses of local memory, which will mark
  // earlier stores as dead if they are not read.
  for (Inst *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->op != Op::MEMORY)
	continue;
      if (!is_local_memory(inst))
	continue;

      uint64_t id = inst->args[0]->value();
      uint64_t mem_addr = id << func->module->ptr_id_low;
      uint64_t size = inst->args[1]->value();
      size = std::min(max_mem_unroll_limit, size);
      for (uint64_t i = 0; i < size; i++)
	{
	  uint64_t addr = mem_addr + i;
	  mem_indef.insert(addr);
	  mem_flag.insert(addr);
	  stores.insert(addr);
	}
    }

  Basic_block *prev_bb = nullptr;
  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];
      if (bb->succs.size() > 1
	  || (bb->succs.size() == 1 && bb->succs[0] != prev_bb))
	{
	  mem_indef.clear();
	  mem_flag.clear();
	  stores.clear();
	}

      for (Inst *inst = bb->last_inst; inst;)
	{
	  Inst *prev_inst = inst->prev;

	  switch (inst->op)
	    {
	    case Op::SET_MEM_INDEF:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (mem_indef.contains(ptr_val))
		      destroy_instruction(inst);
		    else
		      mem_indef.insert(ptr_val);
		  }
	      }
	      break;
	    case Op::GET_MEM_INDEF:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  mem_indef.erase(ptr->value());
		else
		  mem_indef.clear();
	      }
	      break;
	    case Op::SET_MEM_FLAG:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (mem_flag.contains(ptr_val))
		      destroy_instruction(inst);
		    else
		      mem_flag.insert(ptr_val);
		  }
	      }
	      break;
	    case Op::GET_MEM_FLAG:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  mem_flag.erase(ptr->value());
		else
		  mem_flag.clear();
	      }
	      break;
	    case Op::STORE:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  {
		    uint64_t ptr_val = ptr->value();
		    if (stores.contains(ptr_val))
		      destroy_instruction(inst);
		    else
		      stores.insert(ptr_val);
		  }
	      }
	      break;
	    case Op::LOAD:
	      {
		Inst *ptr = inst->args[0];
		if (ptr->op == Op::VALUE)
		  stores.erase(ptr->value());
		else
		  stores.clear();
	      }
	      break;
	    default:
	      break;
	    }

	  inst = prev_inst;
	}
      prev_bb = bb;
    }
}

Inst *get_value(Inst *inst)
{
  if (inst->op != Op::LOAD)
    return nullptr;

  Inst *ptr = inst->args[0];
  Inst *memory = nullptr;
  if (ptr->op == Op::MEMORY)
    memory = ptr;
  else if (ptr->op == Op::ADD
	   && ptr->args[0]->op == Op::MEMORY
	   && ptr->args[1]->op == Op::VALUE)
    memory = ptr->args[0];
  if (!memory)
    return nullptr;

  if (!(memory->args[2]->value() & MEM_CONST))
    return nullptr;

  Basic_block *entry_bb = inst->bb->func->bbs[0];
  for (Inst *inst = entry_bb->last_inst; inst; inst = inst->prev)
    {
      if (inst->op == Op::STORE
	  && (inst->args[0] == ptr
	      || (inst->args[0]->op == Op::ADD
		  && ptr->op == Op::ADD
		  && inst->args[0]->args[0] == ptr->args[0]
		  && inst->args[0]->args[1] == ptr->args[1])))
	return inst->args[1];
    }

  return nullptr;
}

void forward_const(Function *func)
{
  for (auto bb : func->bbs)
    {
      if (bb == func->bbs[0])
	{
	  // The entry block may, in some cases (such as for bit fields),
	  // create the value by a sequence where it loads/stores the
	  // value multiple times, and this naive forwarding implementation
	  // then creates values that are used before being defined. But
	  // there is no need to forward within the entry block, so just
	  // skip it.
	  continue;
	}

      for (Inst *inst = bb->first_inst; inst;)
	{
	  Inst *next_inst = inst->next;
	  if (Inst *value = get_value(inst))
	    {
	      inst->replace_all_uses_with(value);
	      destroy_instruction(inst);
	    }
	  inst = next_inst;
	}
    }
}

} // end anonymous namespace

// This function makes the memory instructions consistent between src and tgt:
//  * src and tgt have the same global memory instructions
//  * global memory that is unused in both src and tgt is removed
//  * unused local memory is removed
//  * the memory instructions are emitted in the same order
// This makes checking faster as more can be CSEd between src and tgt.
void canonicalize_memory(Module *module)
{
  Function *src = module->functions[0];
  Function *tgt = module->functions[1];
  if (src->name != "src")
    std::swap(src, tgt);

  struct {
    bool operator()(const Inst *a, const Inst *b) const {
      return a->args[0]->value() < b->args[0]->value();
    }
  } comp;

  // Substitute load from constant memory with the actual value. This is
  // also done by later load-to-store forwarding, but doing it here may
  // remove the need to set up the constant memory and therefore make
  // more code CSE and leave less memory for the SMT solver to track.
  if (config.optimize_ub)
    {
      forward_const(src);
      forward_const(tgt);
    }

  // Dead instructions using Op::MEMORY may make the code below treat the
  // memory as used. Run DCE first to ensure we get the intended result.
  dead_code_elimination(module);

  std::vector<Inst *> src_mem = collect_mem(src);
  std::vector<Inst *> tgt_mem = collect_mem(tgt);
  std::sort(src_mem.begin(), src_mem.end(), comp);
  std::sort(tgt_mem.begin(), tgt_mem.end(), comp);

  // Add missing global memory instructions.
  std::vector<Inst *> missing_src;
  std::vector<Inst *> missing_tgt;
  std::set_difference(tgt_mem.begin(), tgt_mem.end(),
		      src_mem.begin(), src_mem.end(),
		      std::back_inserter(missing_src), comp);
  std::set_difference(src_mem.begin(), src_mem.end(),
		      tgt_mem.begin(), tgt_mem.end(),
		      std::back_inserter(missing_tgt), comp);
  for (auto inst : missing_src)
    {
      if (!is_local_memory(inst))
	src_mem.push_back(clone_inst(inst, src));
    }
  for (auto inst : missing_tgt)
    {
      if (!is_local_memory(inst))
	tgt_mem.push_back(clone_inst(inst, tgt));
    }

  // Ensure that both src and tgt use the same instruction order.
  // The global memory is placed before the local memory.
  std::sort(src_mem.begin(), src_mem.end(), comp);
  std::sort(tgt_mem.begin(), tgt_mem.end(), comp);
  reorder_mem(src_mem);
  reorder_mem(tgt_mem);

  // Remove global memory that is unused in both src and tgt.
  std::vector<Inst *> remove;
  for (size_t i = 0; i < src_mem.size(); i++)
    {
      if (is_local_memory(src_mem[i]))
	break;
      __int128 src_arg1 = src_mem[i]->args[0]->value();
      __int128 src_arg2 = src_mem[i]->args[1]->value();
      __int128 src_arg3 = src_mem[i]->args[2]->value();
      __int128 tgt_arg1 = tgt_mem[i]->args[0]->value();
      __int128 tgt_arg2 = tgt_mem[i]->args[1]->value();
      __int128 tgt_arg3 = tgt_mem[i]->args[2]->value();
      if (!(src_arg3 & MEM_KEEP)
	  && src_arg1 == tgt_arg1
	  && src_arg2 == tgt_arg2
	  && src_arg3 == tgt_arg3
	  && is_unused_memory(src_mem[i])
	  && is_unused_memory(tgt_mem[i]))
	{
	  remove.push_back(src_mem[i]);
	  remove.push_back(tgt_mem[i]);
	}
    }

  // Remove unised local memory.
  for (auto inst : src_mem)
    {
      if (!(inst->args[2]->value() & MEM_KEEP)
	  && is_local_memory(inst)
	  && is_unused_memory(inst))
	remove.push_back(inst);
    }
  for (auto inst : tgt_mem)
    {
      if (!(inst->args[2]->value() & MEM_KEEP)
	  && is_local_memory(inst)
	  && is_unused_memory(inst))
	remove.push_back(inst);
    }

  if (!remove.empty())
    {
      for (auto inst : remove)
	{
	  remove_unused_memory(inst);
	}

      // Removing memory may open up new opportunities. For example, consider:
      //   int b;
      //   int *p = &b;
      // b may become dead after we remove p. Therefore, we need to rerun the
      // pass.
      canonicalize_memory(module);
    }
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
