// This file contains peephole optimizations and constant folding.
// We do not want to optimize "everything" in this optimization pass as
// that risks introducing new bugs/hiding GCC bugs. Instead, we aim to
// just eliminate common cases where our translations from GIMPLE introduce
// lots of extra instructions. For example, the UB checks for constant
// shift amount, or constant pointer arithmetic.

#include <algorithm>
#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

bool is_boolean_sext(Inst *inst)
{
  return inst->op == Op::SEXT && inst->args[0]->bitsize == 1;
}

bool is_ite_min(Inst *inst)
{
  if (inst->op != Op::ITE)
    return false;

  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];
  Inst *const arg3 = inst->args[2];
  if ((arg1->op == Op::SLT
       || arg1->op == Op::SLE
       || arg1->op == Op::ULT
       || arg1->op == Op::ULE)
      && arg1->args[0] == arg2
      && arg1->args[1] == arg3)
    return true;
  if ((arg1->op == Op::SGT
       || arg1->op == Op::SGE
       || arg1->op == Op::UGT
       || arg1->op == Op::UGE)
      && arg1->args[0] == arg3
      && arg1->args[1] == arg2)
    return true;

  return false;
}

bool is_ite_max(Inst *inst)
{
  if (inst->op != Op::ITE)
    return false;

  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];
  Inst *const arg3 = inst->args[2];
  if ((arg1->op == Op::SGT
       || arg1->op == Op::SGE
       || arg1->op == Op::UGT
       || arg1->op == Op::UGE)
      && arg1->args[0] == arg2
      && arg1->args[1] == arg3)
    return true;
  if ((arg1->op == Op::SLT
       || arg1->op == Op::SLE
       || arg1->op == Op::ULT
       || arg1->op == Op::ULE)
      && arg1->args[0] == arg3
      && arg1->args[1] == arg2)
    return true;

  return false;
}

void flatten(Inst *inst, std::vector<Inst *>& elems)
{
  Op op = inst->op;
  if (inst->args[1]->op == op)
    flatten(inst->args[1], elems);
  else
    elems.push_back(inst->args[1]);
  if (inst->args[0]->op == op)
    flatten(inst->args[0], elems);
  else
    elems.push_back(inst->args[0]);
}

Inst *cfold_ashr(Inst *inst)
{
  __int128 arg1_val = inst->args[0]->signed_value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  if (arg2_val > 127)
    arg2_val = 127;
  return inst->bb->value_inst(arg1_val >> arg2_val, inst->bitsize);
}

Inst *cfold_lshr(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  if (arg2_val > 127)
    return inst->bb->value_inst(0, inst->bitsize);
  return inst->bb->value_inst(arg1_val >> arg2_val, inst->bitsize);
}

Inst *cfold_shl(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  if (arg2_val > 127)
    return inst->bb->value_inst(0, inst->bitsize);
  return inst->bb->value_inst(arg1_val << arg2_val, inst->bitsize);
}

Inst *cfold_add(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val + arg2_val, inst->bitsize);
}

Inst *cfold_sub(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val - arg2_val, inst->bitsize);
}

Inst *cfold_and(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val & arg2_val, inst->bitsize);
}

Inst *cfold_or(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val | arg2_val, inst->bitsize);
}

Inst *cfold_xor(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val ^ arg2_val, inst->bitsize);
}

Inst *cfold_concat(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  unsigned __int128 val = (arg1_val << inst->args[1]->bitsize) | arg2_val;
  return inst->bb->value_inst(val, inst->bitsize);
}

Inst *cfold_mul(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val * arg2_val, inst->bitsize);
}

Inst *cfold_extract(Inst *inst)
{
  unsigned __int128 arg_val = inst->args[0]->value();
  arg_val >>= inst->args[2]->value();
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Inst *cfold_neg(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  return inst->bb->value_inst(-arg1_val, inst->bitsize);
}

Inst *cfold_not(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  return inst->bb->value_inst(~arg1_val, inst->bitsize);
}

Inst *cfold_ne(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val != arg2_val, 1);
}

Inst *cfold_eq(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val == arg2_val, 1);
}

Inst *cfold_uge(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val >= arg2_val, 1);
}

Inst *cfold_ugt(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val > arg2_val, 1);
}

Inst *cfold_ule(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val <= arg2_val, 1);
}

Inst *cfold_ult(Inst *inst)
{
  unsigned __int128 arg1_val = inst->args[0]->value();
  unsigned __int128 arg2_val = inst->args[1]->value();
  return inst->bb->value_inst(arg1_val < arg2_val, 1);
}

Inst *cfold_sge(Inst *inst)
{
  uint32_t shift = 128 - inst->args[0]->bitsize;
  __int128 arg1_val = inst->args[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->args[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val >= arg2_val, 1);
}

Inst *cfold_sgt(Inst *inst)
{
  uint32_t shift = 128 - inst->args[0]->bitsize;
  __int128 arg1_val = inst->args[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->args[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val > arg2_val, 1);
}

Inst *cfold_sle(Inst *inst)
{
  uint32_t shift = 128 - inst->args[0]->bitsize;
  __int128 arg1_val = inst->args[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->args[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val <= arg2_val, 1);
}

Inst *cfold_slt(Inst *inst)
{
  uint32_t shift = 128 - inst->args[0]->bitsize;
  __int128 arg1_val = inst->args[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->args[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val < arg2_val, 1);
}

Inst *cfold_zext(Inst *inst)
{
  unsigned __int128 arg_val = inst->args[0]->value();
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Inst *cfold_sext(Inst *inst)
{
  __int128 arg_val = inst->args[0]->value();
  uint32_t shift = 128 - inst->args[0]->bitsize;
  arg_val <<= shift;
  arg_val >>= shift;
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Inst *simplify_mem_size(Inst *inst, const std::map<uint64_t,uint64_t>& id2size)
{
  if (inst->args[0]->op == Op::VALUE)
    {
      uint64_t id = inst->args[0]->value();
      auto it = id2size.find(id);
      if (it != id2size.end())
	return inst->bb->value_inst(it->second, inst->bitsize);
    }
  return inst;
}

Inst *simplify_add(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // add x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // add (add, x, c2), c1 -> add x, (c1 + c2)
  if (arg2->op == Op::VALUE &&
      arg1->op == Op::ADD &&
      arg1->args[1]->op == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->args[1]->value();
      Inst *val = inst->bb->value_inst(c1 + c2, inst->bitsize);
      Inst *new_inst = create_inst(Op::ADD, arg1->args[0], val);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // add x, x -> shl x, 1
  if (arg1 == arg2)
    {
      Inst *shift = inst->bb->value_inst(1, inst->bitsize);
      Inst *new_inst = create_inst(Op::SHL, arg1, shift);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Inst *simplify_and(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // and x, 0 -> 0
  if (is_value_zero(arg2))
    return arg2;

  // and x, -1 -> x
  if (is_value_m1(arg2))
    return arg1;

  // and x, x -> x
  if (arg1 == arg2)
    return arg1;

  // and (and, x, c2), c1 -> and x, (c1 & c2)
  if (arg2->op == Op::VALUE &&
      arg1->op == Op::AND &&
      arg1->args[1]->op == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->args[1]->value();
      Inst *val = inst->bb->value_inst(c1 & c2, inst->bitsize);
      Inst *new_inst = create_inst(Op::AND, arg1->args[0], val);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // and (sext x) (sext y) -> sext (and x, y)
  // and (zext x) (zext y) -> zext (and x, y)
  if (((arg1->op == Op::SEXT && arg2->op == Op::SEXT)
       || (arg1->op == Op::ZEXT && arg2->op == Op::ZEXT))
      && arg1->args[0]->bitsize == arg2->args[0]->bitsize)
    {
      Inst *new_inst1 = create_inst(Op::AND, arg1->args[0], arg2->args[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(arg1->op, new_inst1, arg1->args[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // and (eq x, y), (ne x, y) -> 0
  // and (ne x, y), (eq x, y) -> 0
  if (((arg1->op == Op::EQ && arg2->op == Op::NE)
       || (arg1->op == Op::NE && arg2->op == Op::EQ))
      && ((arg1->args[0] == arg2->args[0]
	   && arg1->args[1] == arg2->args[1])
	  || (arg1->args[0] == arg2->args[1]
	      && arg1->args[1] == arg2->args[0])))
    return inst->bb->value_inst(0, 1);

  // and (not x), x -> 0
  // and x, (not x) -> 0
  if (arg1->op == Op::NOT && arg1->args[0] == arg2)
    return inst->bb->value_inst(0, inst->bitsize);
  if (arg2->op == Op::NOT && arg2->args[0] == arg1)
    return inst->bb->value_inst(0, inst->bitsize);

  // For Boolean x: and (sext x), 1 -> zext x
  if (is_boolean_sext(arg1) && is_value_one(arg2))
    {
      Inst *new_inst = create_inst(Op::ZEXT, arg1->args[0], arg1->args[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // and x, (-1 << c) -> shl (lshr x, c), c
  if (arg2->op == Op::VALUE
      && popcount(arg2->value()) + ctz(arg2->value()) == inst->bitsize)
    {
      Inst *c = inst->bb->value_inst(ctz(arg2->value()), inst->bitsize);
      Inst *new_inst1 = create_inst(Op::LSHR, arg1, c);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::SHL, new_inst1, c);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Inst *simplify_concat(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // concat (extract x, hi1, lo1), (extract x, hi2, lo2)
  //   -> extract x, hi1, lo2) if lo1 = hi2 + 1
  if (arg1->op == Op::EXTRACT
      && arg2->op == Op::EXTRACT
      && arg1->args[0] == arg2->args[0]
      && arg1->args[2]->value() == arg2->args[1]->value() + 1)
    {
      Inst *x = arg1->args[0];
      Inst *hi1 = arg1->args[1];
      Inst *lo2 = arg2->args[2];
      Inst *new_inst = create_inst(Op::EXTRACT, x, hi1, lo2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // concat (extract x, hi1, lo1), (concat (extract x, hi2, lo2), y)
  //   -> (concat (extract x, hi1, lo2)), y) if lo1 = hi2 + 1
  if (arg1->op == Op::EXTRACT
      && arg2->op == Op::CONCAT
      && arg2->args[0]->op == Op::EXTRACT
      && arg1->args[0] == arg2->args[0]->args[0]
      && (arg1->args[2]->value()
	  == arg2->args[0]->args[1]->value() + 1))
    {
      Inst *x = arg1->args[0];
      Inst *y = arg2->args[1];
      Inst *hi1 = arg1->args[1];
      Inst *lo2 = arg2->args[0]->args[2];
      Inst *new_inst1 = create_inst(Op::EXTRACT, x, hi1, lo2);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::CONCAT, new_inst1, y);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // concat c1, (concat c2, x)) -> concat c, x
  if (arg1->op == Op::VALUE
      && arg2->op == Op::CONCAT
      && arg2->args[0]->op == Op::VALUE
      && arg1->bitsize + arg2->args[0]->bitsize <= 128)
    {
      Inst *new_const = create_inst(Op::CONCAT, arg1, arg2->args[0]);
      new_const->insert_before(inst);
      new_const = simplify_inst(new_const);
      Inst *new_inst = create_inst(Op::CONCAT, new_const, arg2->args[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // concat (concat x, c2), c1 -> concat x, c
  if (arg2->op == Op::VALUE
      && arg1->op == Op::CONCAT
      && arg1->args[1]->op == Op::VALUE
      && arg2->bitsize + arg1->args[1]->bitsize <= 128)
    {
      Inst *new_const = create_inst(Op::CONCAT, arg1->args[1], arg2);
      new_const->insert_before(inst);
      new_const = simplify_inst(new_const);
      Inst *new_inst = create_inst(Op::CONCAT, arg1->args[0], new_const);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // concat 0, x -> zext x
  if (is_value_zero(arg1))
    {
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst1 = create_inst(Op::ZEXT, arg2, bs);
      new_inst1->insert_before(inst);
      return new_inst1;
    }

  // concat (sext (extract x, x->bitsize-1, x->bitsize-1)), x -> sext x
  if (arg1->op == Op::SEXT
      && arg1->args[0]->op == Op::EXTRACT
      && arg1->args[0]->args[0] == arg2
      && arg1->args[0]->args[1] == arg1->args[0]->args[2]
      && arg1->args[0]->args[1]->value() == arg2->bitsize - 1)
    {
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst1 = create_inst(Op::SEXT, arg2, bs);
      new_inst1->insert_before(inst);
      return new_inst1;
    }

  // concat (sext (extract x, x->bitsize-1, x->bitsize-1)), (sext x) -> sext x
  if (arg1->op == Op::SEXT
      && arg2->op == Op::SEXT
      && arg1->args[0]->op == Op::EXTRACT
      && arg1->args[0]->args[0] == arg2->args[0]
      && arg1->args[0]->args[1] == arg1->args[0]->args[2]
      && arg1->args[0]->args[1]->value() == arg2->args[0]->bitsize - 1)
    {
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst1 = create_inst(Op::SEXT, arg2->args[0], bs);
      new_inst1->insert_before(inst);
      return new_inst1;
    }

  // We canonicalize concat so sequences of concat are always expressed
  // as "concat x, (concat y, concat ...)". The intention is that we should
  // get the same IR if we load an 8-byte value as if we concatenate two
  // 4-byte values.
  if (arg1->op == Op::CONCAT)
    {
      std::vector<Inst *> elems;
      flatten(arg1, elems);
      Inst *new_inst = arg2;
      for (auto elem : elems)
	{
	  new_inst = create_inst(Op::CONCAT, elem, new_inst);
	  new_inst->insert_before(inst);
	  new_inst = simplify_inst(new_inst);
	}
      return new_inst;
    }

  return inst;
}

Inst *simplify_eq(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // Comparing MININT with a sign extended value is always false.
  // This is common in UB checks when code is negating a promoted char/short.
  if (arg1->op == Op::SEXT
      && arg2->op == Op::VALUE
      && is_value_signed_min(arg2))
    return inst->bb->value_inst(0, 1);

  // Comparing a negative value with a zero extended value is always false.
  if (arg1->op == Op::ZEXT
      && arg2->op == Op::VALUE
      && arg2->signed_value() < 0)
    return inst->bb->value_inst(0, 1);

  // For Boolean x: x == 1 -> x
  if (arg1->bitsize == 1 && is_value_one(arg2))
    return arg1;

  // For Boolean x: x == 0 -> not x
  if (arg1->bitsize == 1 && is_value_zero(arg2))
    {
      Inst *new_inst = create_inst(Op::NOT, arg1);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (sext/zext x) == 0 -> not x
  if ((arg1->op == Op::SEXT || arg1->op == Op::ZEXT)
      && arg1->args[0]->bitsize == 1
      && is_value_zero(arg2))
    {
      Inst *new_inst = create_inst(Op::NOT, arg1->args[0]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (sext x) == -1 -> x
  if (arg1->op == Op::SEXT
      && arg1->args[0]->bitsize == 1
      && is_value_m1(arg2))
    return arg1->args[0];

  // For Boolean x: (zext x) == 1 -> x
  if (arg1->op == Op::ZEXT
      && arg1->args[0]->bitsize == 1
      && is_value_one(arg2))
    return arg1->args[0];

  // (zext x) == c -> x == c
  if (arg1->op == Op::ZEXT && arg2->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      unsigned __int128 c = arg2->value();
      unsigned shift = 128 - x->bitsize;
      unsigned __int128 trunc_c = (c << shift) >> shift;
      if (c == trunc_c)
	{
	  Inst *new_c = inst->bb->value_inst(trunc_c, x->bitsize);
	  Inst *new_inst = create_inst(Op::EQ, x, new_c);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // (sext x) == c -> x == c
  if (arg1->op == Op::SEXT && arg2->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      __int128 c = arg2->value();
      int shift = 128 - x->bitsize;
      __int128 trunc_c = (c << shift) >> shift;
      if (c == trunc_c)
	{
	  Inst *new_c = inst->bb->value_inst(trunc_c, x->bitsize);
	  Inst *new_inst = create_inst(Op::EQ, x, new_c);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // x == x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  // (x - y) == 0 -> x == y
  if (arg1->op == Op::SUB && is_value_zero(arg2))
    {
      Inst *new_inst = create_inst(Op::EQ, arg1->args[0], arg1->args[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Inst *simplify_ne(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // For Boolean x: x != 0 -> x
  if (arg1->bitsize == 1 && is_value_zero(arg2))
    return arg1;

  // For Boolean x: x != 1 -> not x
  if (arg1->bitsize == 1 && is_value_one(arg2))
    {
      Inst *new_inst = create_inst(Op::NOT, arg1);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (sext/zext x) != 0 -> x
  if ((arg1->op == Op::SEXT || arg1->op == Op::ZEXT)
      && arg1->args[0]->bitsize == 1
      && is_value_zero(arg2))
    return arg1->args[0];

  // For Boolean x: (sext x) != -1 -> not x
  if (arg1->op == Op::SEXT
      && arg1->args[0]->bitsize == 1
      && is_value_m1(arg2))
    {
      Inst *new_inst = create_inst(Op::NOT, arg1->args[0]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (zext x) != 1 -> not x
  if (arg1->op == Op::ZEXT
      && arg1->args[0]->bitsize == 1
      && is_value_one(arg2))
    {
      Inst *new_inst = create_inst(Op::NOT, arg1->args[0]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // x != x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  // (zext x) != c -> x == c
  if (arg1->op == Op::ZEXT && arg2->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      unsigned __int128 c = arg2->value();
      unsigned shift = 128 - x->bitsize;
      unsigned __int128 trunc_c = (c << shift) >> shift;
      if (c == trunc_c)
	{
	  Inst *new_c = inst->bb->value_inst(trunc_c, x->bitsize);
	  Inst *new_inst = create_inst(Op::NE, x, new_c);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // (sext x) != c -> x == c
  if (arg1->op == Op::SEXT && arg2->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      __int128 c = arg2->value();
      int shift = 128 - x->bitsize;
      __int128 trunc_c = (c << shift) >> shift;
      if (c == trunc_c)
	{
	  Inst *new_c = inst->bb->value_inst(trunc_c, x->bitsize);
	  Inst *new_inst = create_inst(Op::NE, x, new_c);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // Comparing chains of identical elements by 0 is changed to only
  // compare one element with 0. For example,
  //   (concat (concat x, x), x) != 0 -> x != 0
  // This is common in checks for uninitialized memory.
  if (arg1->op == Op::CONCAT && is_value_zero(arg2))
    {
      std::vector<Inst *> elems;
      flatten(arg1, elems);
      bool are_identical =
	std::all_of(elems.begin(), elems.end(),
		    [&](auto elem) { return elem == elems.front(); });
      if (are_identical)
	{
	  Inst *elem = elems.front();
	  Inst *zero = inst->bb->value_inst(0, elem->bitsize);
	  Inst *new_inst = create_inst(Op::NE, elem, zero);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // (x - y) != 0 -> x != y
  if (arg1->op == Op::SUB && is_value_zero(arg2))
    {
      Inst *new_inst = create_inst(Op::NE, arg1->args[0], arg1->args[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Inst *simplify_ashr(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // ashr x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // ashr x, c -> sext (extract x (bitsize-1) (bitsize-1)) if c >= (bitsize-1)
  if (arg2->op == Op::VALUE && arg2->value() >= (inst->bitsize - 1))
    {
      Inst *idx = inst->bb->value_inst(inst->bitsize - 1, 32);
      Inst *new_inst1 = create_inst(Op::EXTRACT, arg1, idx, idx);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst2 = create_inst(Op::SEXT, new_inst1, bs);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // ashr (ashr x, c1), c2 -> ashr x, (c1 + c2)
  if (arg2->op == Op::VALUE &&
      arg1->op == Op::ASHR &&
      arg1->args[1]->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      unsigned __int128 c1 = arg1->args[1]->value();
      unsigned __int128 c2 = arg2->value();
      assert(c1 > 0 && c1 < inst->bitsize);
      assert(c2 > 0 && c2 < inst->bitsize);
      Inst *c = inst->bb->value_inst(c1 + c2, inst->bitsize);
      Inst *new_inst = create_inst(Op::ASHR, x, c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ashr (lshr x, c1), c2 -> lshr x, (c1 + c2)
  if (arg2->op == Op::VALUE &&
      arg1->op == Op::LSHR &&
      arg1->args[1]->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      unsigned __int128 c1 = arg1->args[1]->value();
      unsigned __int128 c2 = arg2->value();
      assert(c1 > 0 && c1 < inst->bitsize);
      assert(c2 > 0 && c2 < inst->bitsize);
      Inst *c = inst->bb->value_inst(c1 + c2, inst->bitsize);
      Inst *new_inst = create_inst(Op::LSHR, x, c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ashr (zext x), y -> lshr (zext x), y
  if (arg1->op == Op::ZEXT)
    {
      Inst *new_inst = create_inst(Op::LSHR, arg1, arg2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ashr x, c -> sext (extract x)
  //
  // We only do this if x is a "concat", "sext", "zext", or "extract"
  // instruction, as it is only then that the transformation has any
  // real possibility of improving the result.
  if (arg2->op == Op::VALUE
      && (arg1->op == Op::CONCAT
	  || arg1->op == Op::SEXT
	  || arg1->op == Op::ZEXT
	  || arg1->op == Op::EXTRACT))
    {
      uint64_t c = arg2->value();
      assert(c > 0 && c < arg1->bitsize);
      Inst *high = inst->bb->value_inst(arg1->bitsize - 1, 32);
      Inst *low = inst->bb->value_inst(c, 32);
      Inst *new_inst1 = create_inst(Op::EXTRACT, arg1, high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst2 = create_inst(Op::SEXT, new_inst1, bs);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Inst *simplify_lshr(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // lshr x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // lshr x, c -> 0 if c >= bitsize
  if (arg2->op == Op::VALUE && arg2->value() >= inst->bitsize)
    return inst->bb->value_inst(0, inst->bitsize);

  // lshr (zext x), c -> 0 if c >= x->bitsize
  if (arg1->op == Op::ZEXT
      && arg2->op == Op::VALUE
      && arg2->value() >= arg1->args[0]->bitsize)
    return inst->bb->value_inst(0, inst->bitsize);

  // lshr (zext x), c -> zext (lshr x, c)
  if (arg1->op == Op::ZEXT && arg2->op == Op::VALUE)
    {
      Inst *shift = inst->bb->value_inst(arg2->value(), arg1->args[0]->bitsize);
      Inst *new_inst1 = create_inst(Op::LSHR, arg1->args[0], shift);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::ZEXT, new_inst1, arg1->args[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // lshr (lshr x, c1), c2 -> lshr x, (c1 + c2)
  if (inst->args[1]->op == Op::VALUE
      && inst->args[0]->op == Op::LSHR
      && inst->args[0]->args[1]->op == Op::VALUE)
    {
      Inst *x = inst->args[0]->args[0];
      unsigned __int128 c1 = inst->args[0]->args[1]->value();
      unsigned __int128 c2 = inst->args[1]->value();
      assert(c1 < inst->bitsize);
      assert(c2 < inst->bitsize);
      Inst *c = inst->bb->value_inst(c1 + c2, inst->bitsize);
      Inst *new_inst = create_inst(Op::LSHR, x, c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // lshr (shl (sext x), c), c -> zext x if c is the same as the number of
  // extended bits.
  if (inst->args[1]->op == Op::VALUE
      && inst->args[0]->op == Op::SHL
      && inst->args[0]->args[1] == inst->args[1]
      && inst->args[0]->args[0]->op == Op::SEXT)
    {
      unsigned __int128 c = inst->args[1]->value();
      Inst *x = inst->args[0]->args[0]->args[0];
      if (c == inst->bitsize - x->bitsize)
	{
	  Inst *new_inst = create_inst(Op::ZEXT, x,
				       inst->args[0]->args[0]->args[1]);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // lshr (shl (zext x), c), c -> zext x if c <= the number of
  // extended bits.
  if (inst->args[1]->op == Op::VALUE
      && inst->args[0]->op == Op::SHL
      && inst->args[0]->args[1] == inst->args[1]
      && inst->args[0]->args[0]->op == Op::ZEXT)
    {
      unsigned __int128 c = inst->args[1]->value();
      Inst *x = inst->args[0]->args[0]->args[0];
      if (c <= inst->bitsize - x->bitsize)
	return inst->args[0]->args[0];
    }

  return inst;
}

Inst *simplify_or(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // or x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // or x, -1 -> -1
  if (is_value_m1(arg2))
    return arg2;

  // or x, x -> x
  if (arg1 == arg2)
    return arg1;

  // or (sext x) (sext y) -> sext (or x, y)
  // or (zext x) (zext y) -> zext (or x, y)
  if (((arg1->op == Op::SEXT && arg2->op == Op::SEXT)
       || (arg1->op == Op::ZEXT && arg2->op == Op::ZEXT))
      && arg1->args[0]->bitsize == arg2->args[0]->bitsize)
    {
      Inst *new_inst1 = create_inst(Op::OR, arg1->args[0], arg2->args[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(arg1->op, new_inst1, arg1->args[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // For Boolean x: or (zext x), 1 -> 1
  if (arg1->op == Op::ZEXT
      && arg1->args[0]->bitsize == 1
      && is_value_one(arg2))
    return arg2;

  return inst;
}

Inst *simplify_xor(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // xor x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // xor x, -1 -> not x
  if (is_value_m1(arg2))
    {
      Inst *new_inst = create_inst(Op::NOT, arg1);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // xor x, x -> 0
  if (arg1 == arg2)
    return inst->bb->value_inst(0, inst->bitsize);

  // xor (sext x) (sext y) -> sext (xor x, y)
  // xor (zext x) (zext y) -> zext (xor x, y)
  if (((arg1->op == Op::SEXT && arg2->op == Op::SEXT)
       || (arg1->op == Op::ZEXT && arg2->op == Op::ZEXT))
      && arg1->args[0]->bitsize == arg2->args[0]->bitsize)
    {
      Inst *new_inst1 = create_inst(Op::XOR, arg1->args[0], arg2->args[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(arg1->op, new_inst1, arg1->args[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // For Boolean x: xor (zext x), 1 -> zext (not x)
  if (arg1->op == Op::ZEXT
      && arg1->args[0]->bitsize == 1
      && is_value_one(arg2))
    {
      Inst *new_inst1 = create_inst(Op::NOT, arg1->args[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::ZEXT, new_inst1, arg1->args[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Inst *simplify_sext(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // sext (sext x) -> sext x
  if (arg1->op == Op::SEXT)
    {
      Inst *new_inst = create_inst(Op::SEXT, arg1->args[0], arg2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // sext (zext x) -> zext x
  if (arg1->op == Op::ZEXT)
    {
      Inst *new_inst = create_inst(Op::ZEXT, arg1->args[0], arg2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // sext (extract (sext x)) -> sext (extract x)
  if (arg1->op == Op::EXTRACT && arg1->args[0]->op == Op::SEXT)
    {
      Inst *x = arg1->args[0]->args[0];

      // Extraction from only the original instruction or only the extended
      // bits should have been simplified by simplify_extract.
      assert(arg1->args[2]->value() < x->bitsize);
      assert(arg1->args[1]->value() >= x->bitsize);

      Inst *high = inst->bb->value_inst(x->bitsize - 1, 32);
      Inst *low = arg1->args[2];
      Inst *new_inst1 = create_inst(Op::EXTRACT, x, high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::SEXT, new_inst1, arg2);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Inst *simplify_zext(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // zext (zext x) -> zext x
  if (arg1->op == Op::ZEXT)
    {
      Inst *new_inst = create_inst(Op::ZEXT, arg1->args[0], arg2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // zext (extract (zext x)) -> zext (extract x)
  if (arg1->op == Op::EXTRACT && arg1->args[0]->op == Op::ZEXT)
    {
      Inst *x = arg1->args[0]->args[0];

      // Extraction from only the original instruction or only the extended
      // bits should have been simplified by simplify_extract.
      assert(arg1->args[2]->value() < x->bitsize);
      assert(arg1->args[1]->value() >= x->bitsize);

      Inst *high = inst->bb->value_inst(x->bitsize - 1, 32);
      Inst *low = arg1->args[2];
      Inst *new_inst1 = create_inst(Op::EXTRACT, x, high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::ZEXT, new_inst1, arg2);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Inst *simplify_mov(Inst *inst)
{
  return inst->args[0];
}

Inst *simplify_mul(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // mul x, 0 -> 0
  if (is_value_zero(arg2))
    return arg2;

  // mul x, 1 -> x
  if (is_value_one(arg2))
    return arg1;

  // mul x, (1 << c) -> shl x, c
  if (is_value_pow2(arg2))
    {
      Inst *c = inst->bb->value_inst(ctz(arg2->value()), arg1->bitsize);
      Inst *new_inst = create_inst(Op::SHL, arg1, c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // mul (add x, c2), c1 -> add (mul x, c1), (c1 * c2)
  if (arg2->op == Op::VALUE &&
      arg1->op == Op::ADD &&
      arg1->args[1]->op == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->args[1]->value();
      Inst *new_inst1 = create_inst(Op::MUL, arg1->args[0], arg2);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *val = inst->bb->value_inst(c1 * c2, inst->bitsize);
      Inst *new_inst2 = create_inst(Op::ADD, new_inst1, val);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // mul (mul x, c2), c1 -> mul x, (c1 * c2)
  if (arg2->op == Op::VALUE
      && arg1->op == Op::MUL
      && arg1->args[1]->op == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->args[1]->value();
      Inst *c = inst->bb->value_inst(c1 * c2, inst->bitsize);
      Inst *new_inst = create_inst(Op::MUL, arg1->args[0], c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // mul (shl x, c2), c1 -> mul x, (c1 * (1 << c2))
  if (arg2->op == Op::VALUE
      && arg1->op == Op::SHL
      && arg1->args[1]->op == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->args[1]->value();
      assert(c2 < inst->bitsize);
      Inst *c = inst->bb->value_inst(c1 * (((unsigned __int128)1) << c2),
				     inst->bitsize);
      Inst *new_inst = create_inst(Op::MUL, arg1->args[0], c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Inst *simplify_neg(Inst *inst)
{
  Inst *const arg1 = inst->args[0];

  // neg (neg x) -> x
  if (arg1->op == Op::NEG)
    return arg1->args[0];

  // For Boolean x: neg (zext x) -> sext x
  if (arg1->op == Op::ZEXT && arg1->args[0]->bitsize == 1)
    {
      Inst *new_inst = create_inst(Op::SEXT, arg1->args[0], arg1->args[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: neg (sext x) -> zext x
  if (arg1->op == Op::SEXT && arg1->args[0]->bitsize == 1)
    {
      Inst *new_inst = create_inst(Op::ZEXT, arg1->args[0], arg1->args[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Inst *simplify_not(Inst *inst)
{
  Inst *const arg1 = inst->args[0];

  // not (not x) -> x
  if (arg1->op == Op::NOT)
    return arg1->args[0];

  // For Boolean x: not (sext x) -> sext (not x)
  if (is_boolean_sext(arg1))
    {
      Inst *new_inst1 = create_inst(Op::NOT, arg1->args[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::SEXT, new_inst1, arg1->args[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

// Helper function for simplify_sadd_wraps and simplify_ssub_wraps.
// Check if inst is an instruction that extends more than one bit, or a
// VALUE where the two most significant bits are `00` or `11`.
bool is_ext(Inst *inst)
{
  Inst *const arg1 = inst->args[0];

  if (inst->op == Op::SEXT && arg1->bitsize < inst->bitsize - 1)
    return true;

  if (inst->op == Op::CONCAT
      && is_value_zero(arg1)
      && arg1->bitsize < inst->bitsize - 1)
    return true;

  if (inst->op == Op::VALUE && inst->bitsize >= 3)
    {
      unsigned __int128 top_bits = inst->value() >> (inst->bitsize - 2);
      if (top_bits == 0 || top_bits == 3)
	return true;
    }

  return false;
}

Inst *simplify_sge(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // sge x, x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Inst *simplify_sgt(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // sgt signed_min_val, x -> false
  if (is_value_signed_min(arg1))
    return inst->bb->value_inst(0, 1);

  // sgt x, signed_max_val -> false
  if (is_value_signed_max(arg2))
    return inst->bb->value_inst(0, 1);

  // For Boolean x: sgt (zext x), c -> false if c is a constant > 0.
  // This is rather common in UB checks of range information where a Boolean
  // has been extended to an integer.
  if (arg1->op == Op::ZEXT
      && arg1->args[0]->bitsize == 1
      && arg2->op == Op::VALUE
      && arg2->signed_value() > 0)
    return inst->bb->value_inst(0, 1);

  // For Boolean x: sgt c, (zext x) -> false if c is a constant <= 0.
  if (arg1->op == Op::VALUE
      && arg1->signed_value() <= 0
      && arg2->op == Op::ZEXT
      && arg2->args[0]->bitsize == 1)
    return inst->bb->value_inst(0, 1);

  // sgt c, (sext x) -> false if c <= the minimal possible value of x
  if (arg1->op == Op::VALUE && arg2->op == Op::SEXT)
    {
      __int128 min_val = 1;
      min_val = min_val << 127;
      min_val = min_val >> (128 - arg2->args[0]->bitsize);
      if (arg1->signed_value() <= min_val)
	return inst->bb->value_inst(0, 1);
    }

  // sgt (sext x), c -> false if c >= the maximal possible value of x
  if (arg2->op == Op::VALUE
      && arg1->op == Op::SEXT
      && arg1->args[0]->bitsize > 1)
    {
      unsigned __int128 max_val = -1;
      max_val = max_val >> (129 - arg1->args[0]->bitsize);
      if (arg2->signed_value() >= (__int128)max_val)
	return inst->bb->value_inst(0, 1);
    }

  // sgt 0, (zext x) -> false
  if (is_value_zero(arg1) && arg2->op == Op::ZEXT)
    return inst->bb->value_inst(0, 1);

  // sgt (zext x), c -> false if c >= (zext -1)
  if (arg1->bitsize <= 128
      && arg1->op == Op::ZEXT
      && arg2->op == Op::VALUE
      && arg2->signed_value() >= (((__int128)1 << arg1->args[0]->bitsize) - 1))
    return inst->bb->value_inst(0, 1);

  // sgt x, x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Inst *simplify_sle(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // sle x, x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Inst *simplify_slt(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // slt x, x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Inst *simplify_sadd_wraps(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // sadd_wraps 0, x -> false
  if (is_value_zero(arg1))
    return inst->bb->value_inst(0, 1);

  // sadd_wraps x, 0 -> false
  if (is_value_zero(arg2))
    return inst->bb->value_inst(0, 1);

  // sadd_wraps x, y is always false if x and y are zext/sext that expand
  // more than one bit, or constant that could have been extended in that
  // way. This is a common case for e.g. char/short arithmetic that is
  // promoted to int.
  if (is_ext(arg1) && is_ext(arg2))
    return inst->bb->value_inst(0, 1);

  return inst;
 }

Inst *simplify_ssub_wraps(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // ssub_wraps 0, x -> x == minint
  if (is_value_zero(arg1))
    {
      unsigned __int128 minint = ((unsigned __int128)1) << (arg2->bitsize - 1);
      Inst *minint_inst = inst->bb->value_inst(minint, arg2->bitsize);
      Inst *new_inst = create_inst(Op::EQ, arg2, minint_inst);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ssub_wraps x, 0 -> false
  if (is_value_zero(arg2))
    return inst->bb->value_inst(0, 1);

  // ssub_wraps x, y is always false if x and y are zext/sext that expand
  // more than one bit, or constant that could have been extended in that
  // way. This is a common case for e.g. char/short arithmetic that is
  // promoted to int.
  if (is_ext(arg1) && is_ext(arg2))
    return inst->bb->value_inst(0, 1);

  // For Boolean y: ssub_wraps x, (sext y) -> sadd_wraps x, (zext y)
  if (is_boolean_sext(arg2))
    {
      Inst *new_inst1 = create_inst(Op::ZEXT, arg2->args[0], arg2->args[1]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::SADD_WRAPS, arg1, new_inst1);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Inst *simplify_sub(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // sub x, x -> 0
  if (arg1 == arg2)
    return inst->bb->value_inst(0, inst->bitsize);

  // sub x, c -> add x, -c
  if (arg2->op == Op::VALUE)
    {
      Inst *val = inst->bb->value_inst(-arg2->value(), inst->bitsize);
      Inst *new_inst = create_inst(Op::ADD, arg1, val);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean y: sub x, (sext y) -> add x, (zext y)
  if (is_boolean_sext(arg2))
    {
      Inst *new_inst1 = create_inst(Op::ZEXT, arg2->args[0], arg2->args[1]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::ADD, arg1, new_inst1);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Inst *simplify_ite(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];
  Inst *const arg3 = inst->args[2];

  // ite 0, a, b -> b
  if (is_value_zero(arg1))
    return arg3;

  // ite 1, a, b -> a
  if (is_value_one(arg1))
    return arg2;

  // ite a, b, b -> b
  if (arg2 == arg3)
    return arg2;

  // ite a, 1, 0 -> zext a
  if (is_value_one(arg2) && is_value_zero(arg3))
    {
      if (inst->bitsize == 1)
	return arg1;
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst = create_inst(Op::ZEXT, arg1, bs);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ite a, 0, 1 -> zext (not a)
  if (is_value_one(arg3) && is_value_zero(arg2))
    {
      Inst *cond = create_inst(Op::NOT, arg1);
      cond->insert_before(inst);
      cond = simplify_inst(cond);
      if (inst->bitsize == 1)
	return cond;
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst = create_inst(Op::ZEXT, cond, bs);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ite a, -1, 0 -> sext a
  if (is_value_m1(arg2) && is_value_zero(arg3))
    {
      if (inst->bitsize == 1)
	return arg1;
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst = create_inst(Op::SEXT, arg1, bs);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ite a, 0, -1 -> sext (not a)
  if (is_value_m1(arg3) && is_value_zero(arg2))
    {
      Inst *cond = create_inst(Op::NOT, arg1);
      cond->insert_before(inst);
      cond = simplify_inst(cond);
      if (inst->bitsize == 1)
	return cond;
      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
      Inst *new_inst = create_inst(Op::SEXT, cond, bs);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ite (not c), a, b -> ite c, b, a
  if (arg1->op == Op::NOT)
    {
      Inst *new_cond = arg1->args[0];
      Inst *new_inst = create_inst(Op::ITE, new_cond, arg3, arg2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // Canonicalize ite doing min/max as
  //   ite (x <= y) ? x : y   ; min
  //   ite (x <= y) ? y : x   ; max
  if (is_ite_min(inst) && arg1->op != Op::SLE && arg1->op != Op::ULE)
    {
      Op op = arg1->op;
      Inst *x = arg1->args[0];
      Inst *y = arg1->args[1];
      switch (op)
	{
	case Op::SLT:
	  op = Op::SLE;
	  break;
	case Op::ULT:
	  op = Op::ULE;
	  break;
	case Op::SGT:
	case Op::SGE:
	  op = Op::SLE;
	  std::swap(x, y);
	  break;
	case Op::UGT:
	case Op::UGE:
	  op = Op::ULE;
	  std::swap(x, y);
	  break;
	default:
	  assert(0);
	  break;
	}
      Inst *new_inst1 = create_inst(op, x, y);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::ITE, new_inst1, x, y);
      new_inst2->insert_before(inst);
      return new_inst2;
    }
  if (is_ite_max(inst) && arg1->op != Op::SLE && arg1->op != Op::ULE)
    {
      Op op = arg1->op;
      Inst *x = arg1->args[0];
      Inst *y = arg1->args[1];
      switch (op)
	{
	case Op::SLT:
	case Op::SLE:
	  op = Op::SLE;
	  break;
	case Op::ULT:
	case Op::ULE:
	  op = Op::ULE;
	  break;
	case Op::FLT:
	case Op::FLE:
	  op = Op::FLE;
	  break;
	case Op::SGT:
	case Op::SGE:
	  op = Op::SLE;
	  std::swap(x, y);
	  break;
	case Op::UGT:
	case Op::UGE:
	  op = Op::ULE;
	  std::swap(x, y);
	  break;
	case Op::FGT:
	case Op::FGE:
	  op = Op::FLE;
	  std::swap(x, y);
	  break;
	default:
	  assert(0);
	  break;
	}
      Inst *new_inst1 = create_inst(op, x, y);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::ITE, new_inst1, y, x);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Inst *simplify_uge(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // uge x, x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Inst *simplify_ugt(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // ugt 0, x -> false
  if (is_value_zero(arg1))
    return inst->bb->value_inst(0, 1);

  // ugt x, -1 -> false
  if (is_value_m1(arg2))
    return inst->bb->value_inst(0, 1);

  // ugt (zext x), c -> false if c >= the maximal possible value of x
  if (arg1->op == Op::ZEXT
      && arg2->op == Op::VALUE)
    {
      unsigned __int128 max_val = -1;
      max_val = max_val >> (128 - arg1->args[0]->bitsize);
      if (arg2->value() >= max_val)
	return inst->bb->value_inst(0, 1);
    }

  // ugt (and x, y), x -> false
  // ugt (and x, y), y -> false
  if (arg1->op == Op::AND
      && (arg1->args[0] == arg2 || arg1->args[1] == arg2))
    return inst->bb->value_inst(0, 1);

  // ugt x, x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Inst *simplify_ule(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // ule x, x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Inst *simplify_ult(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // ult x, x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Inst *simplify_shl(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];

  // shl x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // shl x, c -> 0 if c >= bitsize
  if (arg2->op == Op::VALUE && arg2->value() >= inst->bitsize)
    return inst->bb->value_inst(0, inst->bitsize);

  // shl (ashr x, c), c -> shl (lshr x, c), c
  if (arg2->op == Op::VALUE
      && arg1->op == Op::ASHR
      && arg1->args[1] == arg2)
    {
      Inst *new_inst1 = create_inst(Op::LSHR, arg1->args[0], arg2);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Inst *new_inst2 = create_inst(Op::SHL, new_inst1, arg2);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // shl (shl x, c1), c2 -> shl x, (c1 + c2)
  if (inst->args[1]->op == Op::VALUE
      && inst->args[0]->op == Op::SHL
      && inst->args[0]->args[1]->op == Op::VALUE)
    {
      Inst *x = inst->args[0]->args[0];
      unsigned __int128 c1 = inst->args[0]->args[1]->value();
      unsigned __int128 c2 = inst->args[1]->value();
      assert(c1 < inst->bitsize);
      assert(c2 < inst->bitsize);
      Inst *c = inst->bb->value_inst(c1 + c2, inst->bitsize);
      Inst *new_inst = create_inst(Op::SHL, x, c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // shl (mul x, c2), c1 -> mul x, ((1 << c1) * c2)
  if (arg2->op == Op::VALUE
      && arg1->op == Op::MUL
      && arg1->args[1]->op == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->args[1]->value();
      assert(c1 < inst->bitsize);
      Inst *c = inst->bb->value_inst((((unsigned __int128)1) << c1) * c2,
				     inst->bitsize);
      Inst *new_inst = create_inst(Op::MUL, arg1->args[0], c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Inst *simplify_memory(Inst *inst)
{
  uint64_t id = inst->args[0]->value();
  uint64_t addr = id << inst->bb->func->module->ptr_id_low;
  return inst->bb->value_inst(addr, inst->bb->func->module->ptr_bits);
}

Inst *simplify_phi(Inst *phi)
{
  // If phi only references itself or one other value it can be replaced by
  // that value, e.g. %2 = phi [ %1, .1] [ %2, .2] [%1, .3]
  Inst *inst = nullptr;
  for (auto phi_arg : phi->phi_args)
    {
      if (phi_arg.inst != phi)
	{
	  if (!inst)
	    inst = phi_arg.inst;
	  else if (phi_arg.inst != inst)
	    return phi;
	}
    }

  return inst;
}

Inst *simplify_extract(Inst *inst)
{
  Inst *const arg1 = inst->args[0];
  Inst *const arg2 = inst->args[1];
  Inst *const arg3 = inst->args[2];

  const uint32_t high_val = arg2->value();
  const uint32_t low_val = arg3->value();

  // extract x -> x if the range completely cover x.
  if (low_val == 0 && high_val == arg1->bitsize - 1)
    return arg1;

  // "extract (extract x)" is changed to "extract x".
  if (arg1->op == Op::EXTRACT)
    {
      uint32_t arg_low_val = arg1->args[2]->value();
      Inst *high = inst->bb->value_inst(high_val + arg_low_val, 32);
      Inst *low = inst->bb->value_inst(low_val + arg_low_val, 32);
      Inst *new_inst = create_inst(Op::EXTRACT, arg1->args[0], high, low);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // Simplify "extract (sext/zext x)":
  //  * If it is only extracting from x, it is changed to "extract x".
  //  * If it is only extracting from the extended bits, it is changed
  //    to a sext of the most significant bit of x, or 0 if the instruction
  //    is zext.
  //  * If it is truncating the value, but still using bits from both x and
  //    the extended bits, then it is changed to "zext/sext x" with a smaller
  //    bitwidth.
  if (arg1->op == Op::SEXT || arg1->op == Op::ZEXT)
    {
      Inst *ext_arg = arg1->args[0];
      if (low_val == 0 && high_val == ext_arg->bitsize - 1)
	return ext_arg;
      if (high_val < ext_arg->bitsize)
	{
	  Inst *high = arg2;
	  Inst *low = arg3;
	  Inst *new_inst = create_inst(Op::EXTRACT, ext_arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
      if (low_val >= ext_arg->bitsize)
	{
	  if (arg1->op == Op::ZEXT)
	    return inst->bb->value_inst(0, inst->bitsize);
	  else
	    {
	      Inst *idx = inst->bb->value_inst(ext_arg->bitsize - 1, 32);
	      Inst *new_inst = create_inst(Op::EXTRACT, ext_arg, idx, idx);
	      new_inst->insert_before(inst);
	      new_inst = simplify_inst(new_inst);
	      if (new_inst->bitsize < inst->bitsize)
		{
		  Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
		  new_inst = create_inst(Op::SEXT, new_inst, bs);
		  new_inst->insert_before(inst);
		}
	      return new_inst;
	    }
	}
      if (low_val == 0)
	{
	  assert(high_val >= ext_arg->bitsize);
	  Inst *bs = arg1->bb->value_inst(high_val + 1, 32);
	  Inst *new_inst = create_inst(arg1->op, ext_arg, bs);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // Simplify "extract (shl x, c)":
  //  * If it is only extracting from x, it is changed to "extract x".
  //  * If it is only extracting from the extended bits, it is changed to 0.
  if (arg1->op == Op::SHL && arg1->args[1]->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      uint64_t c = arg1->args[1]->value();
      assert(c > 0 && c < x->bitsize);
      if (high_val < c)
	return inst->bb->value_inst(0, inst->bitsize);
      else if (low_val >= c)
	{
	  Inst *high = inst->bb->value_inst(high_val - c, 32);
	  Inst *low = inst->bb->value_inst(low_val - c, 32);
	  Inst *new_inst = create_inst(Op::EXTRACT, x, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // Simplify "extract (lshr x, c)":
  //  * If it is only extracting from x, it is changed to "extract x".
  //  * If it is only extracting from the extended bits, it is changed to 0.
  if (arg1->op == Op::LSHR && arg1->args[1]->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      uint64_t c = arg1->args[1]->value();
      assert(c > 0 && c < x->bitsize);
      uint32_t hi_val = high_val + c;
      uint32_t lo_val = low_val + c;
      if (hi_val < x->bitsize)
	{
	  Inst *high = inst->bb->value_inst(hi_val, 32);
	  Inst *low = inst->bb->value_inst(lo_val, 32);
	  Inst *new_inst = create_inst(Op::EXTRACT, x, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
      else if (lo_val >= x->bitsize)
	return inst->bb->value_inst(0, inst->bitsize);
    }

  // Simplify "extract (ashr x, c)":
  //  * If it is only extracting from x, it is changed to "extract x".
  //  * If it is only extracting from the extended bits, including the
  //    most significant bit of x, it is changed to a sext of the most
  //    significant bit of x.
  if (arg1->op == Op::ASHR && arg1->args[1]->op == Op::VALUE)
    {
      Inst *x = arg1->args[0];
      uint64_t c = arg1->args[1]->value();
      assert(c > 0 && c < x->bitsize);
      uint32_t hi_val = high_val + c;
      uint32_t lo_val = low_val + c;
      if (hi_val < x->bitsize)
	{
	  Inst *high = inst->bb->value_inst(hi_val, 32);
	  Inst *low = inst->bb->value_inst(lo_val, 32);
	  Inst *new_inst = create_inst(Op::EXTRACT, x, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
      else if (lo_val >= x->bitsize - 1)
	{
	  Inst *idx = inst->bb->value_inst(x->bitsize - 1, 32);
	  Inst *new_inst = create_inst(Op::EXTRACT, x, idx, idx);
	  new_inst->insert_before(inst);
	  new_inst = simplify_inst(new_inst);
	  if (new_inst->bitsize < inst->bitsize)
	    {
	      Inst *bs = inst->bb->value_inst(inst->bitsize, 32);
	      new_inst = create_inst(Op::SEXT, new_inst, bs);
	      new_inst->insert_before(inst);
	    }
	  return new_inst;
	}
    }

  // "extract (concat x, y)" is changed to "extract x" or "extract y" if the
  // range only accesses bits from one of the arguments.
  if (arg1->op == Op::CONCAT)
    {
      // We often have chains of concat for loads and vectors, so we iterate
      // to find the final element instead of needing to recursively simplify
      // the new instruction.
      Inst *arg = arg1;
      uint32_t hi_val = high_val;
      uint32_t lo_val = low_val;
      while (arg->op == Op::CONCAT)
	{
	  uint32_t low_bitsize = arg->args[1]->bitsize;
	  if (hi_val < low_bitsize)
	    arg = arg->args[1];
	  else if (lo_val >= low_bitsize)
	    {
	      hi_val -= low_bitsize;
	      lo_val -= low_bitsize;
	      arg = arg->args[0];
	    }
	  else
	    break;
	}
      if (arg != arg1 || hi_val != high_val || lo_val != low_val)
	{
	  if (low_val == 0 && high_val == arg->bitsize - 1)
	    return arg;
	  Inst *high = inst->bb->value_inst(hi_val, 32);
	  Inst *low = inst->bb->value_inst(lo_val, 32);
	  Inst *new_inst = create_inst(Op::EXTRACT, arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // We often have chains of concat (for vectors, structures, etc.), and
  // the extract only needs a few elements in the middle, which are not
  // handled by the previous "extract (concat x, y)" optimization.
  if (arg1->op == Op::CONCAT)
    {
      std::vector<Inst *> elems;
      flatten(arg1, elems);

      if (low_val >= elems[0]->bitsize
	  || high_val < (arg1->bitsize - elems.back()->bitsize))
	{
	  uint32_t hi_val = high_val;
	  uint32_t lo_val = low_val;

	  int i;
	  for (i = 0; lo_val >= elems[i]->bitsize; i++)
	    {
	      hi_val -= elems[i]->bitsize;
	      lo_val -= elems[i]->bitsize;
	    }
	  Inst *arg = elems[i++];
	  for (; hi_val >= arg->bitsize; i++)
	    {
	      arg = create_inst(Op::CONCAT, elems[i], arg);
	      arg->insert_before(inst);
	      arg = simplify_inst(arg);
	    }

	  if (lo_val == 0 && hi_val == arg->bitsize - 1)
	    return arg;
	  Inst *high = inst->bb->value_inst(hi_val, 32);
	  Inst *low = inst->bb->value_inst(lo_val, 32);
	  Inst *new_inst = create_inst(Op::EXTRACT, arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // Create a smaller concat where we have extracted the elements.
  if (arg1->op == Op::CONCAT)
    {
      Inst *low_elem = arg1->args[1];
      Inst *high_elem = arg1->args[0];
      assert(low_val < low_elem->bitsize);
      assert(high_val >= low_elem->bitsize);
      if (high_val != arg1->bitsize - 1)
	{
	  Inst *hi = inst->bb->value_inst(high_val - low_elem->bitsize, 32);
	  Inst *lo = inst->bb->value_inst(0, 32);
	  high_elem = create_inst(Op::EXTRACT, high_elem, hi, lo);
	  high_elem->insert_before(inst);
	  high_elem = simplify_inst(high_elem);
	}
      if (low_val != 0)
	{
	  Inst *hi = inst->bb->value_inst(low_elem->bitsize - 1, 32);
	  Inst *lo = arg3;
	  low_elem = create_inst(Op::EXTRACT, low_elem, hi, lo);
	  low_elem->insert_before(inst);
	  low_elem = simplify_inst(low_elem);
	}
      Inst *new_inst = create_inst(Op::CONCAT, high_elem, low_elem);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // extract (add x, c) -> extract x if the high_val least significant bits
  // of c are 0.
  if (arg1->op == Op::ADD
      && arg1->args[1]->op == Op::VALUE
      && (arg1->args[1]->value() << (127 - high_val)) == 0)
    {
      Inst *high = arg2;
      Inst *low = arg3;
      Inst *new_inst = create_inst(Op::EXTRACT, arg1->args[0], high, low);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Inst *simplify_is_const_mem(Inst *inst, const std::map<uint64_t,Inst *>& id2mem_inst, bool has_const_mem)
{
  // We know the memory is not const if the function does not have any const
  // memory.
  if (!has_const_mem)
    return inst->bb->value_inst(0, 1);

  if (inst->args[0]->op == Op::VALUE)
    {
      uint64_t id = inst->args[0]->value();
      if (id2mem_inst.contains(id))
	{
	  Inst *mem_inst = id2mem_inst.at(id);
	  uint32_t flags = mem_inst->args[2]->value();
	  bool is_const_mem = (flags & MEM_CONST) != 0;
	  return inst->bb->value_inst(is_const_mem, 1);
	}
    }
  return inst;
}

void destroy(Inst *inst)
{
  // Memory removal is done in memory-specific passes.
  if (inst->op == Op::MEMORY)
    return;

  destroy_instruction(inst);
}

} // end anonymous namespace

Inst *constant_fold_inst(Inst *inst)
{
  if (inst->bitsize > 128)
    return inst;
  for (uint64_t i = 0; i < inst->nof_args; i++)
    {
      Inst *arg = inst->args[i];
      if (arg->op != Op::VALUE)
	return inst;
    }

  switch (inst->op)
    {
    case Op::ASHR:
      inst = cfold_ashr(inst);
      break;
    case Op::LSHR:
      inst = cfold_lshr(inst);
      break;
    case Op::SHL:
      inst = cfold_shl(inst);
      break;
    case Op::ADD:
      inst = cfold_add(inst);
      break;
    case Op::SUB:
      inst = cfold_sub(inst);
      break;
    case Op::AND:
      inst = cfold_and(inst);
      break;
    case Op::OR:
      inst = cfold_or(inst);
      break;
    case Op::XOR:
      inst = cfold_xor(inst);
      break;
    case Op::CONCAT:
      inst = cfold_concat(inst);
      break;
    case Op::EXTRACT:
      inst = cfold_extract(inst);
      break;
    case Op::MUL:
      inst = cfold_mul(inst);
      break;
    case Op::NE:
      inst = cfold_ne(inst);
      break;
    case Op::EQ:
      inst = cfold_eq(inst);
      break;
    case Op::NEG:
      inst = cfold_neg(inst);
      break;
    case Op::NOT:
      inst = cfold_not(inst);
      break;
    case Op::SEXT:
      inst = cfold_sext(inst);
      break;
    case Op::SGE:
      inst = cfold_sge(inst);
      break;
    case Op::SGT:
      inst = cfold_sgt(inst);
      break;
    case Op::SLE:
      inst = cfold_sle(inst);
      break;
    case Op::SLT:
      inst = cfold_slt(inst);
      break;
    case Op::UGE:
      inst = cfold_uge(inst);
      break;
    case Op::UGT:
      inst = cfold_ugt(inst);
      break;
    case Op::ULE:
      inst = cfold_ule(inst);
      break;
    case Op::ULT:
      inst = cfold_ult(inst);
      break;
    case Op::ZEXT:
      inst = cfold_zext(inst);
      break;
    default:
      break;
    }

  return inst;
}

Inst *simplify_inst(Inst *inst)
{
  Inst *original_inst = inst;

  inst = constant_fold_inst(inst);
  if (inst != original_inst)
    return inst;

  // Commutative instructions should have constants as the 2nd argument.
  // This is enforced when the instruction is created, but this may change
  // when optimization passes modify the instructions.
  if (inst->is_commutative()
      && inst->args[0]->op == Op::VALUE
      && inst->args[1]->op != Op::VALUE)
    std::swap(inst->args[0], inst->args[1]);

  switch (inst->op)
    {
    case Op::ADD:
      inst = simplify_add(inst);
      break;
    case Op::AND:
      inst = simplify_and(inst);
      break;
    case Op::ASHR:
      inst = simplify_ashr(inst);
      break;
    case Op::CONCAT:
      inst = simplify_concat(inst);
      break;
    case Op::EQ:
      inst = simplify_eq(inst);
      break;
    case Op::NE:
      inst = simplify_ne(inst);
      break;
    case Op::EXTRACT:
      inst = simplify_extract(inst);
      break;
    case Op::LSHR:
      inst = simplify_lshr(inst);
      break;
    case Op::MOV:
      inst = simplify_mov(inst);
      break;
    case Op::MUL:
      inst = simplify_mul(inst);
      break;
    case Op::NEG:
      inst = simplify_neg(inst);
      break;
    case Op::NOT:
      inst = simplify_not(inst);
      break;
    case Op::OR:
      inst = simplify_or(inst);
      break;
    case Op::SADD_WRAPS:
      inst = simplify_sadd_wraps(inst);
      break;
    case Op:: SEXT:
      inst = simplify_sext(inst);
      break;
    case Op::SGE:
      inst = simplify_sge(inst);
      break;
    case Op::SGT:
      inst = simplify_sgt(inst);
      break;
    case Op::SLE:
      inst = simplify_sle(inst);
      break;
    case Op::SLT:
      inst = simplify_slt(inst);
      break;
    case Op::ITE:
      inst = simplify_ite(inst);
      break;
    case Op::SHL:
      inst = simplify_shl(inst);
      break;
    case Op::SSUB_WRAPS:
      inst = simplify_ssub_wraps(inst);
      break;
    case Op::SUB:
      inst = simplify_sub(inst);
      break;
    case Op::UGE:
      inst = simplify_uge(inst);
      break;
    case Op::UGT:
      inst = simplify_ugt(inst);
      break;
    case Op::ULE:
      inst = simplify_ule(inst);
      break;
    case Op::ULT:
      inst = simplify_ult(inst);
      break;
    case Op::XOR:
      inst = simplify_xor(inst);
      break;
    case Op:: ZEXT:
      inst = simplify_zext(inst);
      break;
    default:
      break;
    }

  if (inst != original_inst)
    inst = simplify_inst(inst);

  return inst;
}

void simplify_insts(Function *func)
{
  for (Basic_block *bb : func->bbs)
    {
      std::vector<Inst*> dead_phis;
      for (auto phi : bb->phis)
	{
	  Inst *res = simplify_phi(phi);
	  if (res != phi)
	    phi->replace_all_uses_with(res);
	  if (phi->used_by.empty())
	    dead_phis.push_back(phi);
	}
      for (auto phi : dead_phis)
	{
	  destroy(phi);
	}
      for (Inst *inst = bb->first_inst; inst;)
	{
	  Inst *next_inst = inst->next;
	  if (inst->has_lhs())
	    {
	      if (!inst->used_by.empty())
		{
		  Inst *res = simplify_inst(inst);
		  if (res != inst)
		    {
		      inst->replace_all_uses_with(res);
		      destroy(inst);
		    }
		}
	      else
		destroy(inst);
	    }
	  inst = next_inst;
	}
    }
}

void simplify_insts(Module *module)
{
  for (auto func : module->functions)
    simplify_insts(func);
}

void simplify_mem(Function *func)
{
  std::map<uint64_t,Inst *> id2mem_inst;
  std::map<uint64_t,uint64_t> id2size;
  bool has_const_mem = false;
  for (Inst *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->op == Op::MEMORY)
	{
	  uint64_t id = inst->args[0]->value();
	  uint64_t size = inst->args[1]->value();
	  uint32_t flags = inst->args[2]->value();
	  assert(!id2mem_inst.contains(id));
	  id2mem_inst[id] = inst;
	  id2size[id] = size;

	  has_const_mem |= (flags & MEM_CONST) != 0;
	}
    }

  for (Basic_block *bb : func->bbs)
    {
      for (auto phi : bb->phis)
	{
	  Inst *res = simplify_phi(phi);
	  if (res != phi)
	    phi->replace_all_uses_with(res);
	}
      for (Inst *inst = bb->first_inst; inst;)
	{
	  Inst *next_inst = inst->next;

	  if (inst->has_lhs() && inst->used_by.empty())
	    {
	      destroy(inst);
	      inst = next_inst;
	      continue;
	    }

	  Inst *res;
	  switch (inst->op)
	    {
	    case Op::IS_CONST_MEM:
	      res = simplify_is_const_mem(inst, id2mem_inst, has_const_mem);
	      break;
	    case Op::GET_MEM_SIZE:
	      res = simplify_mem_size(inst, id2size);
	      break;
	    case Op::FREE:
	      // Remove the size value for an ID when the size changes -- we
	      // are iterating in reverse post order, so this prevents use
	      // of incorrect values.
	      // Note: We must remove the value (i.e. not update it) as
	      // some future blocks may use the old value if they are later
	      // in the iteration order, but not dominated by this BB.
	      if (inst->args[0]->op == Op::VALUE)
		{
		  uint64_t id = inst->args[0]->value();
		  id2size.erase(id);
		}
	      else
		id2size.clear();
	      res = inst;
	      break;
	    case Op::MEMORY:
	      res = simplify_memory(inst);
	      break;
	    default:
	      res = simplify_inst(inst);
	      break;
	    }
	  if (res != inst)
	    {
	      inst->replace_all_uses_with(res);
	      destroy(inst);
	    }

	  inst = next_inst;
	}
    }
}

void simplify_mem(Module *module)
{
  for (auto func : module->functions)
    simplify_mem(func);
}

} // end namespace smtgcc
