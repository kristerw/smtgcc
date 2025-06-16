// CSE within each BB.
#include <unordered_map>

#include "smtgcc.h"

namespace smtgcc {

namespace {

struct Cse_key
{
  Op op;
  Inst *arg1 = nullptr;
  Inst *arg2 = nullptr;
  Inst *arg3 = nullptr;

  Cse_key(Inst *inst)
  {
    op = inst->op;
    if (inst->nof_args > 0)
      arg1 = inst->args[0];
    if (inst->nof_args > 1)
      arg2 = inst->args[1];
    if (inst->nof_args > 2)
      arg3 = inst->args[2];
  }

  friend bool operator<(const Cse_key& lhs, const Cse_key& rhs)
  {
    if (lhs.op < rhs.op) return true;
    if (lhs.op > rhs.op) return false;

    if (lhs.arg1 < rhs.arg1) return true;
    if (lhs.arg1 > rhs.arg1) return false;

    if (lhs.arg2 < rhs.arg2) return true;
    if (lhs.arg2 > rhs.arg2) return false;

    return lhs.arg3 < rhs.arg3;
  }

  bool operator==(const Cse_key &other) const
  {
    return op == other.op
      && arg1 == other.arg1
      && arg2 == other.arg2
      && arg3 == other.arg3;
  }
};

struct Cse_key_hash {
  std::size_t operator()(const Cse_key& key) const
  {
    size_t h = (size_t)key.op;
    h ^= 0x9e3779b9 + (h << 6) + (h >> 2) + (size_t)key.arg1;
    h ^= 0x9e3779b9 + (h << 6) + (h >> 2) + (size_t)key.arg2;
    h ^= 0x9e3779b9 + (h << 6) + (h >> 2) + (size_t)key.arg3;
    return h;
  }
};

} // end anonymous namespace

void cse(Function *func)
{
  for (auto bb : func->bbs)
    {
      std::unordered_map<Cse_key, Inst *, Cse_key_hash> key2inst;

      for (Inst *inst = bb->first_inst; inst;)
	{
	  Inst *next_inst = inst->next;
	  switch (inst->iclass())
	    {
	    case Inst_class::iunary:
	    case Inst_class::funary:
	    case Inst_class::ibinary:
	    case Inst_class::fbinary:
	    case Inst_class::icomparison:
	    case Inst_class::fcomparison:
	    case Inst_class::conv:
	    case Inst_class::ternary:
	      {
		Cse_key key(inst);
		Inst *cse_inst = nullptr;
		auto I = key2inst.find(key);
		if (I != key2inst.end())
		  cse_inst = I->second;
		else if (inst_info[(int)key.op].is_commutative)
		  {
		    Cse_key tmp_key = key;
		    std::swap(tmp_key.arg1, tmp_key.arg2);
		    I = key2inst.find(tmp_key);
		    if (I != key2inst.end())
		      cse_inst = I->second;
		  }
		if (cse_inst)
		  {
		    inst->replace_all_uses_with(cse_inst);
		    destroy_instruction(inst);
		  }
		else
		  key2inst.insert({key, inst});
	      }
	      break;
	    default:
	      break;
	    }
	  inst = next_inst;
	}
    }
}

void cse(Module *module)
{
  for (auto func : module->functions)
    cse(func);
}

} // end namespace smtgcc
