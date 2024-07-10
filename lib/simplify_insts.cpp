// This file contains peephole optimizations and constant folding.
// We do not want to optimize "everything" in this optimization pass as
// that risks introducing new bugs/hiding GCC bugs. Instead, we aim to
// just eliminate common cases where our translations from GIMPLE introduce
// lots of extra instructions. For example, the UB checks for constant
// shift amount, or constant pointer arithmetic.

#include <algorithm>
#include <cassert>

#include "smtgcc.h"
#include "util.h"

namespace smtgcc {

namespace {

bool is_boolean_sext(Instruction *inst)
{
  return inst->opcode == Op::SEXT && inst->arguments[0]->bitsize == 1;
}

void flatten(Instruction *inst, std::vector<Instruction *>& elems)
{
  Op op = inst->opcode;
  if (inst->arguments[1]->opcode == op)
    flatten(inst->arguments[1], elems);
  else
    elems.push_back(inst->arguments[1]);
  if (inst->arguments[0]->opcode == op)
    flatten(inst->arguments[0], elems);
  else
    elems.push_back(inst->arguments[0]);
}

Instruction *cfold_add(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val + arg2_val, inst->bitsize);
}

Instruction *cfold_sub(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val - arg2_val, inst->bitsize);
}

Instruction *cfold_and(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val & arg2_val, inst->bitsize);
}

Instruction *cfold_or(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val | arg2_val, inst->bitsize);
}

Instruction *cfold_xor(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val ^ arg2_val, inst->bitsize);
}

Instruction *cfold_concat(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  unsigned __int128 val = (arg1_val << inst->arguments[1]->bitsize) | arg2_val;
  return inst->bb->value_inst(val, inst->bitsize);
}

Instruction *cfold_mul(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val * arg2_val, inst->bitsize);
}

Instruction *cfold_extract(Instruction *inst)
{
  unsigned __int128 arg_val = inst->arguments[0]->value();
  arg_val >>= inst->arguments[2]->value();
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Instruction *cfold_neg(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  return inst->bb->value_inst(-arg1_val, inst->bitsize);
}

Instruction *cfold_not(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  return inst->bb->value_inst(~arg1_val, inst->bitsize);
}

Instruction *cfold_ne(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val != arg2_val, 1);
}

Instruction *cfold_eq(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val == arg2_val, 1);
}

Instruction *cfold_uge(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val >= arg2_val, 1);
}

Instruction *cfold_ugt(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val > arg2_val, 1);
}

Instruction *cfold_ule(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val <= arg2_val, 1);
}

Instruction *cfold_ult(Instruction *inst)
{
  unsigned __int128 arg1_val = inst->arguments[0]->value();
  unsigned __int128 arg2_val = inst->arguments[1]->value();
  return inst->bb->value_inst(arg1_val < arg2_val, 1);
}

Instruction *cfold_sge(Instruction *inst)
{
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  __int128 arg1_val = inst->arguments[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->arguments[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val >= arg2_val, 1);
}

Instruction *cfold_sgt(Instruction *inst)
{
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  __int128 arg1_val = inst->arguments[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->arguments[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val > arg2_val, 1);
}

Instruction *cfold_sle(Instruction *inst)
{
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  __int128 arg1_val = inst->arguments[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->arguments[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val <= arg2_val, 1);
}

Instruction *cfold_slt(Instruction *inst)
{
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  __int128 arg1_val = inst->arguments[0]->value();
  arg1_val = (arg1_val << shift) >> shift;
  __int128 arg2_val = inst->arguments[1]->value();
  arg2_val = (arg2_val << shift) >> shift;
  return inst->bb->value_inst(arg1_val < arg2_val, 1);
}

Instruction *cfold_zext(Instruction *inst)
{
  unsigned __int128 arg_val = inst->arguments[0]->value();
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Instruction *cfold_sext(Instruction *inst)
{
  __int128 arg_val = inst->arguments[0]->value();
  uint32_t shift = 128 - inst->arguments[0]->bitsize;
  arg_val <<= shift;
  arg_val >>= shift;
  return inst->bb->value_inst(arg_val, inst->bitsize);
}

Instruction *constant_fold_inst(Instruction *inst)
{
  if (inst->bitsize > 128)
    return inst;
  for (uint64_t i = 0; i < inst->nof_args; i++)
    {
      Instruction *arg = inst->arguments[i];
      if (arg->opcode != Op::VALUE)
	return inst;
    }

  switch (inst->opcode)
    {
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

Instruction *simplify_mem_size(Instruction *inst, const std::map<uint64_t,uint64_t>& id2size)
{
  if (inst->arguments[0]->opcode == Op::VALUE)
    {
      uint64_t id = inst->arguments[0]->value();
      auto it = id2size.find(id);
      if (it != id2size.end())
	return inst->bb->value_inst(it->second, inst->bitsize);
    }
  return inst;
}

Instruction *simplify_add(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // add 0, x -> x
  if (is_value_zero(arg1))
    return arg2;

  // add x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // add (add, x, c2), c1 -> add x, (c1 + c2)
  if (arg2->opcode == Op::VALUE &&
      arg1->opcode == Op::ADD &&
      arg1->arguments[1]->opcode == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->arguments[1]->value();
      Instruction *val = inst->bb->value_inst(c1 + c2, inst->bitsize);
      Instruction *new_inst = create_inst(Op::ADD, arg1->arguments[0], val);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Instruction *simplify_and(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // and x, 0 -> 0
  if (is_value_zero(arg2))
    return arg2;

  // and x, -1 -> x
  if (is_value_m1(arg2))
    return arg1;

  // and (and, x, c2), c1 -> and x, (c1 & c2)
  if (arg2->opcode == Op::VALUE &&
      arg1->opcode == Op::AND &&
      arg1->arguments[1]->opcode == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->arguments[1]->value();
      Instruction *val = inst->bb->value_inst(c1 & c2, inst->bitsize);
      Instruction *new_inst = create_inst(Op::AND, arg1->arguments[0], val);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // and (sext x) (sext y) -> sext (and x, y)
  if (arg1->opcode == Op::SEXT
      && arg2->opcode == Op::SEXT
      && arg1->arguments[0]->bitsize == arg2->arguments[0]->bitsize)
    {
      Instruction *new_inst1 =
	create_inst(Op::AND, arg1->arguments[0], arg2->arguments[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *new_inst2 =
	create_inst(Op::SEXT, new_inst1, arg1->arguments[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // and (concat x1, x2) (concat y1, y2) -> concat (and x1, y1) (and x2, y2)
  // if x1 and y1 have the same size.
  //
  // This optimization is not obviously beneficial -- there are cases
  // where store-to-load forwarding, etc., will later make it possible to
  // eliminate the "concat" instructions and perform the "and" instruction
  // on an original, wider value. We therefore only do this when at least
  // one of the elements is 0, as we then know that we reduce the number
  // of instructions.
  //
  // Similarly, we change
  //   and (concat x1, x2) y
  //     -> concat (and x1, (extract y)) (and x2, (extract y))
  //   and y (concat x1, x2)
  //     -> concat (and x1, (extract y)) (and x2, (extract y))
  // when x1 or x2 is 0.
  if (arg1->opcode == Op::CONCAT
      && arg2->opcode == Op::CONCAT
      && arg1->arguments[0]->bitsize == arg2->arguments[0]->bitsize
      && (is_value_zero(arg1->arguments[0])
	  || is_value_zero(arg1->arguments[1])
	  || is_value_zero(arg2->arguments[0])
	  || is_value_zero(arg2->arguments[1])))
    {
      Instruction *r1 =
	create_inst(Op::AND, arg1->arguments[0], arg2->arguments[0]);
      r1->insert_before(inst);
      r1 = simplify_inst(r1);
      Instruction *r2 =
	create_inst(Op::AND, arg1->arguments[1], arg2->arguments[1]);
      r2->insert_before(inst);
      r2 = simplify_inst(r2);
      Instruction *new_inst = create_inst(Op::CONCAT, r1, r2);
      new_inst->insert_before(inst);
      return new_inst;
    }
  if ((arg1->opcode == Op::CONCAT
       && arg2->opcode != Op::CONCAT
       && (is_value_zero(arg1->arguments[0])
	   || is_value_zero(arg1->arguments[1])))
      || (arg1->opcode != Op::CONCAT
	  && arg2->opcode == Op::CONCAT
	  && (is_value_zero(arg2->arguments[0])
	      || is_value_zero(arg2->arguments[1]))))
    {
      Instruction *x = arg1->opcode == Op::CONCAT ? arg1 : arg2;
      Instruction *y = arg1->opcode == Op::CONCAT ? arg2 : arg1;
      Instruction *hi = inst->bb->value_inst(y->bitsize - 1, 32);
      Instruction *lo = inst->bb->value_inst(x->arguments[1]->bitsize, 32);
      Instruction *y1 = create_inst(Op::EXTRACT, y, hi, lo);
      y1->insert_before(inst);
      y1 = simplify_inst(y1);
      hi = inst->bb->value_inst(x->arguments[1]->bitsize - 1, 32);
      lo = inst->bb->value_inst(0, 32);
      Instruction *y2 = create_inst(Op::EXTRACT, y, hi, lo);
      y2->insert_before(inst);
      y2 = simplify_inst(y2);
      Instruction *r1 = create_inst(Op::AND, x->arguments[0], y1);
      r1->insert_before(inst);
      r1 = simplify_inst(r1);
      Instruction *r2 = create_inst(Op::AND, x->arguments[1], y2);
      r2->insert_before(inst);
      r2 = simplify_inst(r2);
      Instruction *new_inst = create_inst(Op::CONCAT, r1, r2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // and (eq x, y), (ne x, y) -> 0
  // and (ne x, y), (eq x, y) -> 0
  if (((arg1->opcode == Op::EQ && arg2->opcode == Op::NE)
       || (arg1->opcode == Op::NE && arg2->opcode == Op::EQ))
      && ((arg1->arguments[0] == arg2->arguments[0]
	   && arg1->arguments[1] == arg2->arguments[1])
	  || (arg1->arguments[0] == arg2->arguments[1]
	      && arg1->arguments[1] == arg2->arguments[0])))
    return inst->bb->value_inst(0, 1);

  // and (not x), x -> 0
  // and x, (not x) -> 0
  if (arg1->opcode == Op::NOT && arg1->arguments[0] == arg2)
    return inst->bb->value_inst(0, inst->bitsize);
  if (arg2->opcode == Op::NOT && arg2->arguments[0] == arg1)
    return inst->bb->value_inst(0, inst->bitsize);

  return inst;
}

Instruction *simplify_concat(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // concat (extract x, hi1, lo1), (extract x, hi2, lo2)
  //   -> extract x, hi1, lo2) if lo1 = hi2 + 1
  if (arg1->opcode == Op::EXTRACT
      && arg2->opcode == Op::EXTRACT
      && arg1->arguments[0] == arg2->arguments[0]
      && arg1->arguments[2]->value() == arg2->arguments[1]->value() + 1)
    {
      Instruction *x = arg1->arguments[0];
      Instruction *hi1 = arg1->arguments[1];
      Instruction *lo2 = arg2->arguments[2];
      Instruction *new_inst = create_inst(Op::EXTRACT, x, hi1, lo2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // concat (extract x, hi1, lo1), (concat (extract x, hi2, lo2), y)
  //   -> (concat (extract x, hi1, lo2)), y) if lo1 = hi2 + 1
  if (arg1->opcode == Op::EXTRACT
      && arg2->opcode == Op::CONCAT
      && arg2->arguments[0]->opcode == Op::EXTRACT
      && arg1->arguments[0] == arg2->arguments[0]->arguments[0]
      && (arg1->arguments[2]->value()
	  == arg2->arguments[0]->arguments[1]->value() + 1))
    {
      Instruction *x = arg1->arguments[0];
      Instruction *y = arg2->arguments[1];
      Instruction *hi1 = arg1->arguments[1];
      Instruction *lo2 = arg2->arguments[0]->arguments[2];
      Instruction *new_inst1 = create_inst(Op::EXTRACT, x, hi1, lo2);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *new_inst2 = create_inst(Op::CONCAT, new_inst1, y);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // concat c1, (concat c2, x)) -> concat c, x
  if (arg1->opcode == Op::VALUE
      && arg2->opcode == Op::CONCAT
      && arg2->arguments[0]->opcode == Op::VALUE
      && arg1->bitsize + arg2->arguments[0]->bitsize <= 128)
    {
      Instruction *new_const =
	create_inst(Op::CONCAT, arg1, arg2->arguments[0]);
      new_const->insert_before(inst);
      new_const = simplify_inst(new_const);
      Instruction *new_inst =
	create_inst(Op::CONCAT, new_const, arg2->arguments[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // concat (concat x, c2), c1 -> concat x, c
  if (arg2->opcode == Op::VALUE
      && arg1->opcode == Op::CONCAT
      && arg1->arguments[1]->opcode == Op::VALUE
      && arg2->bitsize + arg1->arguments[1]->bitsize <= 128)
    {
      Instruction *new_const =
	create_inst(Op::CONCAT, arg1->arguments[1], arg2);
      new_const->insert_before(inst);
      new_const = simplify_inst(new_const);
      Instruction *new_inst =
	create_inst(Op::CONCAT, arg1->arguments[0], new_const);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Instruction *simplify_eq(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // Comparing MININT with a sign extended value is always false.
  // This is common in UB checks when code is negating a promoted char/short.
  if (arg1->opcode == Op::SEXT
      && arg2->opcode == Op::VALUE
      && is_value_signed_min(arg2))
    return inst->bb->value_inst(0, 1);

  // Comparing MININT with "concat 0, x" is always false.
  if (arg1->opcode == Op::CONCAT
      && (is_value_zero(arg1->arguments[0])
	  || is_value_zero(arg1->arguments[1]))
      && is_value_signed_min(arg2))
    return inst->bb->value_inst(0, 1);

  // For Boolean x: x == 1 -> x
  if (arg1->bitsize == 1 && is_value_one(arg2))
    return arg1;

  // For Boolean x: x == 0 -> not x
  if (arg1->bitsize == 1 && is_value_zero(arg2))
    {
      Instruction *new_inst = create_inst(Op::NOT, arg1);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (sext x) == 0 -> not x
  if (arg1->opcode == Op::SEXT
      && arg1->arguments[0]->bitsize == 1
      && is_value_zero(arg2))
    {
      Instruction *new_inst = create_inst(Op::NOT, arg1->arguments[0]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (sext x) == -1 -> x
  if (arg1->opcode == Op::SEXT
      && arg1->arguments[0]->bitsize == 1
      && is_value_m1(arg2))
    return arg1->arguments[0];

  // For Boolean x: (concat 0, x) == 0 -> not x
  if (arg1->opcode == Op::CONCAT
      && arg1->arguments[1]->bitsize == 1
      && is_value_zero(arg1->arguments[0])
      && is_value_zero(arg2))
    {
      Instruction *new_inst = create_inst(Op::NOT, arg1->arguments[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (concat 0, x) == 1 -> x
  if (arg1->opcode == Op::CONCAT
      && arg1->arguments[1]->bitsize == 1
      && is_value_zero(arg1->arguments[0])
      && is_value_one(arg2))
    return arg1->arguments[1];

  // x == x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  // (x - y) == 0 -> x == y
  if (arg1->opcode == Op::SUB && is_value_zero(arg2))
    {
      Instruction *new_inst =
	create_inst(Op::EQ, arg1->arguments[0], arg1->arguments[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Instruction *simplify_ne(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // For Boolean x: x != 0 -> x
  if (arg1->bitsize == 1 && is_value_zero(arg2))
    return arg1;

  // For Boolean x: x != 1 -> not x
  if (arg1->bitsize == 1 && is_value_one(arg2))
    {
      Instruction *new_inst = create_inst(Op::NOT, arg1);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (sext x) != 0 -> x
  if (arg1->opcode == Op::SEXT
      && arg1->arguments[0]->bitsize == 1
      && is_value_zero(arg2))
    return arg1->arguments[0];

  // For Boolean x: (sext x) != -1 -> not x
  if (arg1->opcode == Op::SEXT
      && arg1->arguments[0]->bitsize == 1
      && is_value_m1(arg2))
    {
      Instruction *new_inst = create_inst(Op::NOT, arg1->arguments[0]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: (concat 0, x) != 0 -> x
  if (arg1->opcode == Op::CONCAT
      && arg1->arguments[1]->bitsize == 1
      && is_value_zero(arg1->arguments[0])
      && is_value_zero(arg2))
    return arg1->arguments[1];

  // For Boolean x: (concat 0, x) != 1 -> not x
  if (arg1->opcode == Op::CONCAT
      && arg1->arguments[1]->bitsize == 1
      && is_value_zero(arg1->arguments[0])
      && is_value_one(arg2))
    {
      Instruction *new_inst = create_inst(Op::NOT, arg1->arguments[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // x != x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  // Comparing chains of identical elements by 0 is changed to only
  // compare one element with 0. For example,
  //   (concat (concat x, x), x) != 0 -> x != 0
  // This is common in checks for uninitialized memory.
  if (arg1->opcode == Op::CONCAT && is_value_zero(arg2))
    {
      std::vector<Instruction *> elems;
      flatten(arg1, elems);
      bool are_identical =
	std::all_of(elems.begin(), elems.end(),
		    [&](auto elem) { return elem == elems.front(); });
      if (are_identical)
	{
	  Instruction *elem = elems.front();
	  Instruction *zero = inst->bb->value_inst(0, elem->bitsize);
	  Instruction *new_inst = create_inst(Op::NE, elem, zero);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // (x - y) != 0 -> x != y
  if (arg1->opcode == Op::SUB && is_value_zero(arg2))
    {
      Instruction *new_inst =
	create_inst(Op::NE, arg1->arguments[0], arg1->arguments[1]);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Instruction *simplify_ashr(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // ashr x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // ashr x, c -> sext (extract x (bitsize-1) (bitsize-1)) if c >= (bitsize-1)
  if (arg2->opcode == Op::VALUE && arg2->value() >= (inst->bitsize - 1))
    {
      Instruction *idx = inst->bb->value_inst(inst->bitsize - 1, 32);
      Instruction *new_inst1 = create_inst(Op::EXTRACT, arg1, idx, idx);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *bs = inst->bb->value_inst(inst->bitsize, 32);
      Instruction *new_inst2 = create_inst(Op::SEXT, new_inst1, bs);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // ashr (ashr x, c1), c2 -> ashr x, (c1 + c2)
  if (arg2->opcode == Op::VALUE &&
      arg1->opcode == Op::ASHR &&
      arg1->arguments[1]->opcode == Op::VALUE)
    {
      Instruction *x = arg1->arguments[0];
      unsigned __int128 c1 = arg1->arguments[1]->value();
      unsigned __int128 c2 = arg2->value();
      assert(c1 > 0 && c1 < inst->bitsize);
      assert(c2 > 0 && c2 < inst->bitsize);
      Instruction *c = inst->bb->value_inst(c1 + c2, inst->bitsize);
      Instruction *new_inst = create_inst(Op::ASHR, x, c);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ashr x, c -> sext (extract x)
  //
  // We only do this if x is a "concat", "sext", or "extract" instruction,
  // as it is only then that the transformation has any real possibility of
  // improving the result.
  if (arg2->opcode == Op::VALUE
      && (arg1->opcode == Op::CONCAT
	  || arg1->opcode == Op::SEXT
	  || arg1->opcode == Op::EXTRACT))
    {
      uint64_t c = arg2->value();
      assert(c > 0 && c < arg1->bitsize);
      Instruction *high = inst->bb->value_inst(arg1->bitsize - 1, 32);
      Instruction *low = inst->bb->value_inst(c, 32);
      Instruction *new_inst1 = create_inst(Op::EXTRACT, arg1, high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *bs = inst->bb->value_inst(inst->bitsize, 32);
      Instruction *new_inst2 = create_inst(Op::SEXT, new_inst1, bs);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Instruction *simplify_lshr(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // lshr x, 0 -> x
  if (is_value_zero(arg2))
    return inst->arguments[0];

  // lshr x, c -> 0 if c >= bitsize
  if (arg2->opcode == Op::VALUE && arg2->value() >= inst->bitsize)
    return inst->bb->value_inst(0, inst->bitsize);

  // lshr x, c -> concat 0, (extract x)
  if (arg2->opcode == Op::VALUE)
    {
      uint64_t c = arg2->value();
      assert(c > 0 && c < arg1->bitsize);
      Instruction *high = inst->bb->value_inst(arg1->bitsize - 1, 32);
      Instruction *low = inst->bb->value_inst(c, 32);
      Instruction *new_inst1 = create_inst(Op::EXTRACT, arg1, high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *zero = inst->bb->value_inst(0, c);
      Instruction *new_inst2 = create_inst(Op::CONCAT, zero, new_inst1);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Instruction *simplify_or(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // or x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // or x, -1 -> -1
  if (is_value_m1(arg2))
    return arg2;

  // For Boolean x, y: or (sext x) (sext y) -> sext (or x, y)
  if (is_boolean_sext(arg1) && is_boolean_sext(arg2))
    {
      Instruction *new_inst1 =
	create_inst(Op::OR, arg1->arguments[0], arg2->arguments[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *new_inst2 =
	create_inst(Op::SEXT, new_inst1, arg1->arguments[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // or (concat x1, x2) (concat y1, y2) -> concat (or x1, y1) (or x2, y2)
  // if x1 and y1 have the same size.
  //
  // This optimization is not obviously beneficial -- there are cases
  // where store-to-load forwarding, etc., will later make it possible to
  // eliminate the "concat" instructions and perform the "or" instruction
  // on an original, wider value. We therefore only do this when at least
  // one of the elements is 0, as we then know that we reduce the number
  // of instructions.
  if (arg1->opcode == Op::CONCAT
      && arg2->opcode == Op::CONCAT
      && arg1->arguments[0]->bitsize == arg2->arguments[0]->bitsize
      && (is_value_zero(arg1->arguments[0])
	  || is_value_zero(arg1->arguments[1])
	  || is_value_zero(arg2->arguments[0])
	  || is_value_zero(arg2->arguments[1])))
    {
      Instruction *r1 =
	create_inst(Op::OR, arg1->arguments[0], arg2->arguments[0]);
      r1->insert_before(inst);
      r1 = simplify_inst(r1);
      Instruction *r2 =
	create_inst(Op::OR, arg1->arguments[1], arg2->arguments[1]);
      r2->insert_before(inst);
      r2 = simplify_inst(r2);
      Instruction *new_inst = create_inst(Op::CONCAT, r1, r2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: or (concat 0, x), 1 -> 1
  if (arg1->opcode == Op::CONCAT
      && is_value_zero(arg1->arguments[0])
      && arg1->arguments[1]->bitsize == 1
      && is_value_one(arg2))
    return arg2;

  return inst;
}

Instruction *simplify_xor(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // xor x, 0 -> x
  if (is_value_zero(arg2))
    return arg1;

  // xor x, -1 -> not x
  if (is_value_m1(arg2))
    {
      Instruction *new_inst = create_inst(Op::NOT, arg1);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x, y: xor (sext x) (sext y) -> sext (xor x, y)
  if (is_boolean_sext(arg1) && is_boolean_sext(arg2))
    {
      Instruction *new_inst1 =
	create_inst(Op::XOR, arg1->arguments[0], arg2->arguments[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *new_inst2 =
	create_inst(Op::SEXT, new_inst1, arg1->arguments[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // xor (concat x1, x2) (concat y1, y2) -> concat (xor x1, y1) (xor x2, y2)
  // if x1 and y1 have the same size.
  //
  // This optimization is not obviously beneficial -- there are cases
  // where store-to-load forwarding, etc., will later make it possible to
  // eliminate the "concat" instructions and perform the "xor" instruction
  // on an original, wider value. We therefore only do this when at least
  // one of the elements is 0, as we then know that we reduce the number
  // of instructions.
  if (arg1->opcode == Op::CONCAT
      && arg2->opcode == Op::CONCAT
      && arg1->arguments[0]->bitsize == arg2->arguments[0]->bitsize
      && (is_value_zero(arg1->arguments[0])
	  || is_value_zero(arg1->arguments[1])
	  || is_value_zero(arg2->arguments[0])
	  || is_value_zero(arg2->arguments[1])))
    {
      Instruction *r1 =
	create_inst(Op::XOR, arg1->arguments[0], arg2->arguments[0]);
      r1->insert_before(inst);
      r1 = simplify_inst(r1);
      Instruction *r2 =
	create_inst(Op::XOR, arg1->arguments[1], arg2->arguments[1]);
      r2->insert_before(inst);
      r2 = simplify_inst(r2);
      Instruction *new_inst = create_inst(Op::CONCAT, r1, r2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // For Boolean x: xor (concat 0, x), 1 -> concat 0, (not x)
  if (arg1->opcode == Op::CONCAT
      && is_value_zero(arg1->arguments[0])
      && arg1->arguments[1]->bitsize == 1
      && is_value_one(arg2))
    {
      Instruction *new_inst1 = create_inst(Op::NOT, arg1->arguments[1]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *new_inst2 =
	create_inst(Op::CONCAT, arg1->arguments[0], new_inst1);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Instruction *simplify_sext(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // sext (sext x) -> sext x
  if (arg1->opcode == Op::SEXT)
    {
      Instruction *new_inst = create_inst(Op::SEXT, arg1->arguments[0], arg2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // sext (extract (sext x)) -> sext (extract x)
  if (arg1->opcode == Op::EXTRACT && arg1->arguments[0]->opcode == Op::SEXT)
    {
      Instruction *x = arg1->arguments[0]->arguments[0];

      // Extraction from only the original instruction or only the extended
      // bits should have been simplified by simplify_extract.
      assert(arg1->arguments[2]->value() < x->bitsize);
      assert(arg1->arguments[1]->value() >= x->bitsize);

      Instruction *high = inst->bb->value_inst(x->bitsize - 1, 32);
      Instruction *low = arg1->arguments[2];
      Instruction *new_inst1 = create_inst(Op::EXTRACT, x, high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *new_inst2 = create_inst(Op::SEXT, new_inst1, arg2);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // sext (concat x, y) -> concat (sext x), y
  if (arg1->opcode == Op::CONCAT)
    {
      Instruction *x = arg1->arguments[0];
      Instruction *y = arg1->arguments[1];
      Instruction *bs = inst->bb->value_inst(inst->bitsize - y->bitsize, 32);
      Instruction *new_inst1 = create_inst(Op::SEXT, x, bs);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *new_inst2 = create_inst(Op::CONCAT, new_inst1, y);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Instruction *simplify_zext(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];

  // zext x -> concat 0, x
  Instruction *zero = inst->bb->value_inst(0, inst->bitsize - arg1->bitsize);
  Instruction *new_inst = create_inst(Op::CONCAT, zero, arg1);
  new_inst->insert_before(inst);
  return new_inst;
}

Instruction *simplify_mov(Instruction *inst)
{
  return inst->arguments[0];
}

Instruction *simplify_mul(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // mul 0, x -> 0
  if (is_value_zero(inst->arguments[0]))
    return inst->arguments[0];

  // mul x, 0 -> 0
  if (is_value_zero(inst->arguments[1]))
    return inst->arguments[1];

  // mul 1, x -> x
  if (is_value_one(inst->arguments[0]))
    return inst->arguments[1];

  // mul x, 1 -> x
  if (is_value_one(inst->arguments[1]))
    return inst->arguments[0];

  // mul x, (1 << c) -> concat (extract x), 0
  if (is_value_pow2(arg2))
    {
      uint64_t c = ctz(arg2->value());
      Instruction *high = inst->bb->value_inst(arg1->bitsize - 1 - c, 32);
      Instruction *low = inst->bb->value_inst(0, 32);
      Instruction *new_inst1 = create_inst(Op::EXTRACT, arg1, high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *zero = inst->bb->value_inst(0, c);
      Instruction *new_inst2 = create_inst(Op::CONCAT, new_inst1, zero);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  // mul (concat x, 0), (concat y, 0)
  //    -> concat (mul (extract x), (extract y)), 0
  if (arg1->opcode == Op::CONCAT && is_value_zero(arg1->arguments[1])
      && arg2->opcode == Op::CONCAT && is_value_zero(arg2->arguments[1]))
    {
      uint64_t a1_zero_bits = arg1->arguments[1]->bitsize;
      uint64_t a2_zero_bits = arg2->arguments[1]->bitsize;
      if (a1_zero_bits + a2_zero_bits >= inst->bitsize)
	return inst->bb->value_inst(0, inst->bitsize);
      uint64_t mul_bits = inst->bitsize - (a1_zero_bits + a2_zero_bits);
      Instruction *high = inst->bb->value_inst(mul_bits - 1, 32);
      Instruction *low = inst->bb->value_inst(0, 32);
      Instruction *a1 = create_inst(Op::EXTRACT, arg1->arguments[0], high, low);
      a1->insert_before(inst);
      a1 = simplify_inst(a1);
      Instruction *a2 = create_inst(Op::EXTRACT, arg2->arguments[0], high, low);
      a2->insert_before(inst);
      a2 = simplify_inst(a2);
      Instruction *mul = create_inst(Op::MUL, a1, a2);
      mul->insert_before(inst);
      mul = simplify_inst(mul);
      Instruction *zero = inst->bb->value_inst(0, inst->bitsize - mul_bits);
      Instruction *concat = create_inst(Op::CONCAT, mul, zero);
      concat->insert_before(inst);
      return concat;
    }

  // mul (concat x, 0), y -> concat (mul x, (extract y)), 0
  // mul y, (concat x, 0) -> concat (mul x, (extract y)), 0
  if ((arg1->opcode == Op::CONCAT && is_value_zero(arg1->arguments[1]))
      || (arg2->opcode == Op::CONCAT && is_value_zero(arg2->arguments[1])))
    {
      Instruction *x, *y;
      Instruction *zero;
      if (arg1->opcode == Op::CONCAT && is_value_zero(arg1->arguments[1]))
	{
	  x = arg1->arguments[0];
	  y = arg2;
	  zero = arg1->arguments[1];
	}
      else
	{
	  x = arg2->arguments[0];
	  y = arg1;
	  zero = arg2->arguments[1];
	}
      uint64_t mul_bits = inst->bitsize - zero->bitsize;
      Instruction *high = inst->bb->value_inst(mul_bits - 1, 32);
      Instruction *low = inst->bb->value_inst(0, 32);
      y = create_inst(Op::EXTRACT, y, high, low);
      y->insert_before(inst);
      y = simplify_inst(y);
      Instruction *mul = create_inst(Op::MUL, x, y);
      mul->insert_before(inst);
      mul = simplify_inst(mul);
      Instruction *concat = create_inst(Op::CONCAT, mul, zero);
      concat->insert_before(inst);
      return concat;
    }

  // mul (add, x, c2), c1 -> add (mul x, c1), (c1 * c2)
  if (arg2->opcode == Op::VALUE &&
      arg1->opcode == Op::ADD &&
      arg1->arguments[1]->opcode == Op::VALUE)
    {
      unsigned __int128 c1 = arg2->value();
      unsigned __int128 c2 = arg1->arguments[1]->value();
      Instruction *new_inst1 = create_inst(Op::MUL, arg1->arguments[0], arg2);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *val = inst->bb->value_inst(c1 * c2, inst->bitsize);
      Instruction *new_inst2 = create_inst(Op::ADD, new_inst1, val);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Instruction *simplify_not(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];

  // not (not x) -> x
  if (arg1->opcode == Op::NOT)
    return arg1->arguments[0];

  // For Boolean x: not (sext x) -> sext (not x)
  if (is_boolean_sext(arg1))
    {
      Instruction *new_inst1 = create_inst(Op::NOT, arg1->arguments[0]);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *new_inst2 =
	create_inst(Op::SEXT, new_inst1, arg1->arguments[1]);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

// Helper function for simplify_sadd_wraps and simplify_ssub_wraps.
// Check if inst is an instruction that extends more than one bit, or a
// VALUE where the two most significant bits are `00` or `11`.
bool is_ext(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];

  if (inst->opcode == Op::SEXT && arg1->bitsize < inst->bitsize - 1)
    return true;

  if (inst->opcode == Op::CONCAT
      && is_value_zero(inst->arguments[0])
      && inst->arguments[0]->bitsize < inst->bitsize - 1)
    return true;

  if (inst->opcode == Op::VALUE && inst->bitsize >= 3)
    {
      unsigned __int128 top_bits = inst->value() >> (inst->bitsize - 2);
      if (top_bits == 0 || top_bits == 3)
	return true;
    }

  return false;
}

Instruction *simplify_sge(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // sge x, x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Instruction *simplify_sgt(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // sgt signed_min_val, x -> false
  if (is_value_signed_min(arg1))
    return inst->bb->value_inst(0, 1);

  // sgt x, signed_max_val -> false
  if (is_value_signed_max(arg2))
    return inst->bb->value_inst(0, 1);

  // For Boolean x: sgt (concat 0, x), c -> false if c is a constant > 0.
  // This is rather common in UB checks of range information where a Boolean
  // has been extended to an integer.
  if (arg1->opcode == Op::CONCAT
      && arg1->arguments[1]->bitsize == 1
      && is_value_zero(arg1->arguments[0])
      && arg2->opcode == Op::VALUE
      && arg2->signed_value() > 0)
    return inst->bb->value_inst(0, 1);

  // For Boolean x: sgt c, (concat 0, x) -> false if c is a constant <= 0.
  if (arg1->opcode == Op::VALUE
      && arg1->signed_value() <= 0
      && arg2->opcode == Op::CONCAT
      && arg2->arguments[1]->bitsize == 1
      && is_value_zero(arg2->arguments[0]))
    return inst->bb->value_inst(0, 1);

  // sgt 0, (concat 0, x) -> false
  if (is_value_zero(arg1)
      && arg2->opcode == Op::CONCAT
      && is_value_zero(arg2->arguments[0]))
    return inst->bb->value_inst(0, 1);

  // sgt (concat 0, x), c -> false if c >= (zext -1)
  if (arg1->bitsize <= 128
      && arg1->opcode == Op::CONCAT
      &&is_value_zero(arg1->arguments[0])
      && arg2->opcode == Op::VALUE
      && arg2->signed_value() >= (((__int128)1 << arg1->arguments[1]->bitsize) - 1))
    return inst->bb->value_inst(0, 1);

  // sgt x, x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Instruction *simplify_sle(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // sle x, x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Instruction *simplify_slt(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // slt x, x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Instruction *simplify_sadd_wraps(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

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

Instruction *simplify_ssub_wraps(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // ssub_wraps 0, x -> x == minint
  if (is_value_zero(arg1))
    {
      unsigned __int128 minint = ((unsigned __int128)1) << (arg2->bitsize - 1);
      Instruction *minint_inst = inst->bb->value_inst(minint, arg2->bitsize);
      Instruction *new_inst = create_inst(Op::EQ, arg2, minint_inst);
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

  return inst;
}

Instruction *simplify_sub(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // sub x, c -> add x, -c
  if (arg2->opcode == Op::VALUE)
    {
      Instruction *val = inst->bb->value_inst(-arg2->value(), inst->bitsize);
      Instruction *new_inst = create_inst(Op::ADD, arg1, val);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Instruction *simplify_ite(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];
  Instruction *const arg3 = inst->arguments[2];

  // ite 0, a, b -> b
  if (is_value_zero(arg1))
    return arg3;

  // ite 1, a, b -> a
  if (is_value_one(arg1))
    return arg2;

  // ite a, b, b -> b
  if (arg2 == arg3)
    return arg2;

  // ite a, 1, 0 -> concat 0, a
  if (is_value_one(arg2) && is_value_zero(arg3))
    {
      if (inst->bitsize == 1)
	return arg1;
      Instruction *zero = inst->bb->value_inst(0, inst->bitsize - 1);
      Instruction *new_inst = create_inst(Op::CONCAT, zero, arg1);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ite a, 0, 1 -> concat 0, (not a)
  if (is_value_one(arg3) && is_value_zero(arg2))
    {
      Instruction *cond = create_inst(Op::NOT, arg1);
      cond->insert_before(inst);
      cond = simplify_inst(cond);
      if (inst->bitsize == 1)
	return cond;
      Instruction *zero = inst->bb->value_inst(0, inst->bitsize - 1);
      Instruction *new_inst = create_inst(Op::CONCAT, zero, cond);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ite a, -1, 0 -> sext a
  if (is_value_m1(arg2) && is_value_zero(arg3))
    {
      if (inst->bitsize == 1)
	return arg1;
      Instruction *bs = inst->bb->value_inst(inst->bitsize, 32);
      Instruction *new_inst = create_inst(Op::SEXT, arg1, bs);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ite a, 0, -1 -> sext (not a)
  if (is_value_m1(arg3) && is_value_zero(arg2))
    {
      Instruction *cond = create_inst(Op::NOT, arg1);
      cond->insert_before(inst);
      cond = simplify_inst(cond);
      if (inst->bitsize == 1)
	return cond;
      Instruction *bs = inst->bb->value_inst(inst->bitsize, 32);
      Instruction *new_inst = create_inst(Op::SEXT, cond, bs);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // ite (not c), a, b -> ite c, b, a
  if (arg1->opcode == Op::NOT)
    {
      Instruction *new_cond = arg1->arguments[0];
      Instruction *new_inst = create_inst(Op::ITE, new_cond, arg3, arg2);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Instruction *simplify_uge(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // uge x, x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Instruction *simplify_ugt(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // ugt 0, x -> false
  if (is_value_zero(arg1))
    return inst->bb->value_inst(0, 1);

  // ugt x, -1 -> false
  if (is_value_m1(arg2))
    return inst->bb->value_inst(0, 1);

  // For Boolean x: ugt (concat 0, x), c -> false if c is a constant > 0.
  // This is rather common in UB checks of range information where a Boolean
  // has been extended to an integer.
  if (arg1->opcode == Op::CONCAT
      && arg1->arguments[1]->bitsize == 1
      && is_value_zero(arg1->arguments[0])
      && arg2->opcode == Op::VALUE
      && arg2->value() > 0)
    return inst->bb->value_inst(0, 1);

  // ugt (and x, y), x -> false
  // ugt (and x, y), y -> false
  if (arg1->opcode == Op::AND
      && (arg1->arguments[0] == arg2 || arg1->arguments[1] == arg2))
    return inst->bb->value_inst(0, 1);

  // ugt x, x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Instruction *simplify_ule(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // ule x, x -> true
  if (arg1 == arg2)
    return inst->bb->value_inst(1, 1);

  return inst;
}

Instruction *simplify_ult(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // ult x, x -> false
  if (arg1 == arg2)
    return inst->bb->value_inst(0, 1);

  return inst;
}

Instruction *simplify_shl(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];

  // shl x, 0 -> x
  if (is_value_zero(arg2))
    return inst->arguments[0];

  // shl x, c -> 0 if c >= bitsize
  if (arg2->opcode == Op::VALUE && arg2->value() >= inst->bitsize)
    return inst->bb->value_inst(0, inst->bitsize);

  // shl x, c -> concat (extract x), 0
  if (arg2->opcode == Op::VALUE)
    {
      uint64_t c = arg2->value();
      assert(c > 0 && c < arg1->bitsize);
      Instruction *high = inst->bb->value_inst(arg1->bitsize - 1 - c, 32);
      Instruction *low = inst->bb->value_inst(0, 32);
      Instruction *new_inst1 = create_inst(Op::EXTRACT, arg1, high, low);
      new_inst1->insert_before(inst);
      new_inst1 = simplify_inst(new_inst1);
      Instruction *zero = inst->bb->value_inst(0, c);
      Instruction *new_inst2 = create_inst(Op::CONCAT, new_inst1, zero);
      new_inst2->insert_before(inst);
      return new_inst2;
    }

  return inst;
}

Instruction *simplify_memory(Instruction *inst)
{
  uint64_t id = inst->arguments[0]->value();
  uint64_t addr = id << inst->bb->func->module->ptr_id_low;
  return inst->bb->value_inst(addr, inst->bb->func->module->ptr_bits);
}

Instruction *simplify_phi(Instruction *phi)
{
  // If phi only references itself or one other value it can be replaced by
  // that value, e.g. %2 = phi [ %1, .1] [ %2, .2] [%1, .3]

  Instruction *inst = nullptr;
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

Instruction *simplify_extract(Instruction *inst)
{
  Instruction *const arg1 = inst->arguments[0];
  Instruction *const arg2 = inst->arguments[1];
  Instruction *const arg3 = inst->arguments[2];

  const uint32_t high_val = arg2->value();
  const uint32_t low_val = arg3->value();

  // extract x -> x if the range completely cover x.
  if (low_val == 0 && high_val == arg1->bitsize - 1)
    return arg1;

  // "extract (extract x)" is changed to "extract x".
  if (arg1->opcode == Op::EXTRACT)
    {
      uint32_t arg_low_val = arg1->arguments[2]->value();
      Instruction *high = inst->bb->value_inst(high_val + arg_low_val, 32);
      Instruction *low = inst->bb->value_inst(low_val + arg_low_val, 32);
      Instruction *new_inst =
	create_inst(Op::EXTRACT, arg1->arguments[0], high, low);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // Simplify "extract (sext x)":
  //  * If it is only extracting from x, it is changed to "extract x".
  //  * If it is only extracting from the extended bits, it is changed
  //    to a sext of the most significant bit of x.
  //  * If it is truncating the value, but still using bits from both x and
  //    the extended bits, then it is changed to "sext x" with a smaller
  //    bitwidth.
  if (arg1->opcode == Op::SEXT)
    {
      Instruction *ext_arg = arg1->arguments[0];
      if (low_val == 0 && high_val == ext_arg->bitsize - 1)
	return ext_arg;
      if (high_val < ext_arg->bitsize)
	{
	  Instruction *high = arg2;
	  Instruction *low = arg3;
	  Instruction *new_inst = create_inst(Op::EXTRACT, ext_arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
      if (low_val >= ext_arg->bitsize)
	{
	  Instruction *idx = inst->bb->value_inst(ext_arg->bitsize - 1, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, ext_arg, idx, idx);
	  new_inst->insert_before(inst);
	  new_inst = simplify_inst(new_inst);
	  if (new_inst->bitsize < inst->bitsize)
	    {
	      Instruction *bs = inst->bb->value_inst(inst->bitsize, 32);
	      new_inst = create_inst(Op::SEXT, new_inst, bs);
	      new_inst->insert_before(inst);
	    }
	  return new_inst;
	}
      if (low_val == 0)
	{
	  assert(high_val >= ext_arg->bitsize);
	  Instruction *bs = arg1->bb->value_inst(high_val + 1, 32);
	  Instruction *new_inst = create_inst(Op::SEXT, ext_arg, bs);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // Simplify "extract (ashr x, c)":
  //  * If it is only extracting from x, it is changed to "extract x".
  //  * If it is only extracting from the extended bits, it is changed
  //    to a sext of the most significant bit of x.
  if (arg1->opcode == Op::ASHR && arg1->arguments[1]->opcode == Op::VALUE)
    {
      Instruction *x = arg1->arguments[0];
      uint64_t c = arg1->arguments[1]->value();
      assert(c > 0 && c < x->bitsize);
      uint32_t hi_val = high_val + c;
      uint32_t lo_val = low_val + c;
      if (hi_val < x->bitsize)
	{
	  Instruction *high = inst->bb->value_inst(hi_val, 32);
	  Instruction *low = inst->bb->value_inst(lo_val, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, x, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
      else if (lo_val >= x->bitsize)
	{
	  Instruction *idx = inst->bb->value_inst(x->bitsize - 1, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, x, idx, idx);
	  new_inst->insert_before(inst);
	  new_inst = simplify_inst(new_inst);
	  if (new_inst->bitsize < inst->bitsize)
	    {
	      Instruction *bs = inst->bb->value_inst(inst->bitsize, 32);
	      new_inst = create_inst(Op::SEXT, new_inst, bs);
	      new_inst->insert_before(inst);
	    }
	  return new_inst;
	}
    }

  // "extract (concat x, y)" is changed to "extract x" or "extract y" if the
  // range only accesses bits from one of the arguments.
  if (arg1->opcode == Op::CONCAT)
    {
      // We often have chains of concat for loads and vectors, so we iterate
      // to find the final element instead of needing to recursively simplify
      // the new instruction.
      Instruction *arg = arg1;
      uint32_t hi_val = high_val;
      uint32_t lo_val = low_val;
      while (arg->opcode == Op::CONCAT)
	{
	  uint32_t low_bitsize = arg->arguments[1]->bitsize;
	  if (hi_val < low_bitsize)
	    arg = arg->arguments[1];
	  else if (lo_val >= low_bitsize)
	    {
	      hi_val -= low_bitsize;
	      lo_val -= low_bitsize;
	      arg = arg->arguments[0];
	    }
	  else
	    break;
	}
      if (arg != arg1 || hi_val != high_val || lo_val != low_val)
	{
	  if (low_val == 0 && high_val == arg->bitsize - 1)
	    return arg;
	  Instruction *high = inst->bb->value_inst(hi_val, 32);
	  Instruction *low = inst->bb->value_inst(lo_val, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // We often have chains of concat (for vectors, structures, etc.), and
  // the extract only needs a few elements in the middle, which are not
  // handled by the previous "extract (concat x, y)" optimization.
  if (arg1->opcode == Op::CONCAT)
    {
      std::vector<Instruction *> elems;
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
	  Instruction *arg = elems[i++];
	  for (; hi_val >= arg->bitsize; i++)
	    {
	      arg = create_inst(Op::CONCAT, elems[i], arg);
	      arg->insert_before(inst);
	      arg = simplify_inst(arg);
	    }

	  if (lo_val == 0 && hi_val == arg->bitsize - 1)
	    return arg;
	  Instruction *high = inst->bb->value_inst(hi_val, 32);
	  Instruction *low = inst->bb->value_inst(lo_val, 32);
	  Instruction *new_inst = create_inst(Op::EXTRACT, arg, high, low);
	  new_inst->insert_before(inst);
	  return new_inst;
	}
    }

  // Create a smaller concat where we have extracted the elements.
  if (arg1->opcode == Op::CONCAT)
    {
      Instruction *low_elem = arg1->arguments[1];
      Instruction *high_elem = arg1->arguments[0];
      assert(low_val < low_elem->bitsize);
      assert(high_val >= low_elem->bitsize);
      if (high_val != arg1->bitsize - 1)
	{
	  Instruction *hi =
	    inst->bb->value_inst(high_val - low_elem->bitsize, 32);
	  Instruction *lo = inst->bb->value_inst(0, 32);
	  high_elem = create_inst(Op::EXTRACT, high_elem, hi, lo);
	  high_elem->insert_before(inst);
	  high_elem = simplify_inst(high_elem);
	}
      if (low_val != 0)
	{
	  Instruction *hi = inst->bb->value_inst(low_elem->bitsize - 1, 32);
	  Instruction *lo = arg3;
	  low_elem = create_inst(Op::EXTRACT, low_elem, hi, lo);
	  low_elem->insert_before(inst);
	  low_elem = simplify_inst(low_elem);
	}
      Instruction *new_inst = create_inst(Op::CONCAT, high_elem, low_elem);
      new_inst->insert_before(inst);
      return new_inst;
    }

  // extract (add x, c) -> extract x if the high_val least significant bits
  // of c are 0.
  if (arg1->opcode == Op::ADD
      && arg1->arguments[1]->opcode == Op::VALUE
      && (arg1->arguments[1]->value() << (127 - high_val)) == 0)
    {
      Instruction *high = arg2;
      Instruction *low = arg3;
      Instruction *new_inst =
	create_inst(Op::EXTRACT, arg1->arguments[0], high, low);
      new_inst->insert_before(inst);
      return new_inst;
    }

  return inst;
}

Instruction *simplify_is_const_mem(Instruction *inst, const std::map<uint64_t,Instruction *>& id2mem_inst, bool has_const_mem)
{
  // We know the memory is not const if the function does not have any const
  // memory.
  if (!has_const_mem)
    return inst->bb->value_inst(0, 1);

  if (inst->arguments[0]->opcode == Op::VALUE)
    {
      uint64_t id = inst->arguments[0]->value();
      if (id2mem_inst.contains(id))
	{
	  Instruction *mem_inst = id2mem_inst.at(id);
	  uint32_t flags = mem_inst->arguments[2]->value();
	  bool is_const_mem = (flags & MEM_CONST) != 0;
	  return inst->bb->value_inst(is_const_mem, 1);
	}
    }
  return inst;
}

void destroy(Instruction *inst)
{
  // Memory removal is done in memory-specific passes.
  if (inst->opcode == Op::MEMORY)
    return;

  destroy_instruction(inst);
}

} // end anonymous namespace

Instruction *simplify_inst(Instruction *inst)
{
  Instruction *original_inst = inst;

  inst = constant_fold_inst(inst);
  if (inst != original_inst)
    return inst;

  // Commutative instructions should have constants as the 2nd argument.
  // This is enforced when the instruction is created, but this may change
  // when optimization passes modify the instructions.
  if (inst->is_commutative()
      && inst->arguments[0]->opcode == Op::VALUE
      && inst->arguments[1]->opcode != Op::VALUE)
    std::swap(inst->arguments[0], inst->arguments[1]);

  switch (inst->opcode)
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
      for (auto phi : bb->phis)
	{
	  Instruction *res = simplify_phi(phi);
	  if (res != phi)
	    phi->replace_all_uses_with(res);
	}
      for (Instruction *inst = bb->first_inst; inst;)
	{
	  Instruction *next_inst = inst->next;
	  if (inst->has_lhs())
	    {
	      if (!inst->used_by.empty())
		{
		  Instruction *res = simplify_inst(inst);
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
  std::map<uint64_t,Instruction *> id2mem_inst;
  std::map<uint64_t,uint64_t> id2size;
  bool has_const_mem = false;
  for (Instruction *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->opcode == Op::MEMORY)
	{
	  uint64_t id = inst->arguments[0]->value();
	  uint64_t size = inst->arguments[1]->value();
	  uint32_t flags = inst->arguments[2]->value();
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
	  Instruction *res = simplify_phi(phi);
	  if (res != phi)
	    phi->replace_all_uses_with(res);
	}
      for (Instruction *inst = bb->first_inst; inst;)
	{
	  Instruction *next_inst = inst->next;

	  if (inst->has_lhs() && inst->used_by.empty())
	    {
	      destroy(inst);
	      inst = next_inst;
	      continue;
	    }

	  Instruction *res;
	  switch (inst->opcode)
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
	      if (inst->arguments[0]->opcode == Op::VALUE)
		{
		  uint64_t id = inst->arguments[0]->value();
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
