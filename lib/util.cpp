#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

namespace {

Inst *gen_fmin_fmax(Basic_block *bb, Inst *elem1, Inst *elem2, bool is_min)
{
  Inst *is_nan = bb->build_inst(Op::IS_NAN, elem2);
  Inst *cmp;
  if (is_min)
    cmp = bb->build_inst(Op::FLT, elem1, elem2);
  else
    cmp = bb->build_inst(Op::FLT, elem2, elem1);
  Inst *res1 = bb->build_inst(Op::ITE, cmp, elem1, elem2);
  Inst *res2 = bb->build_inst(Op::ITE, is_nan, elem1, res1);
  // 0.0 and -0.0 is equal as floating-point values, and fmin(0.0, -0.0)
  // may return eiter of them. But we treat them as 0.0 > -0.0 here,
  // otherwise we will report miscompilations when GCC switch the order
  // of the arguments.
  Inst *zero = bb->value_inst(0, elem1->bitsize);
  Inst *is_zero1 = bb->build_inst(Op::FEQ, elem1, zero);
  Inst *is_zero2 = bb->build_inst(Op::FEQ, elem2, zero);
  Inst *is_zero = bb->build_inst(Op::AND, is_zero1, is_zero2);
  Inst *cmp2;
  if (is_min)
    cmp2 = bb->build_inst(Op::SLT, elem1, elem2);
  else
    cmp2 = bb->build_inst(Op::SLT, elem2, elem1);
  Inst *res3 = bb->build_inst(Op::ITE, cmp2, elem1, elem2);
  Inst *res = bb->build_inst(Op::ITE, is_zero, res3, res2);
  return res;
}

} // end anonymous namespace

uint32_t popcount(unsigned __int128 x)
{
  uint32_t result = 0;
  for (int i = 0; i < 4; i++)
    {
      uint32_t t = x >> (i* 32);
      result += __builtin_popcount(t);
    }
  return result;
}

uint32_t clz(unsigned __int128 x)
{
  uint32_t result = 0;
  for (int i = 0; i < 4; i++)
    {
      uint32_t t = x >> ((3 - i) * 32);
      if (t)
	return result + __builtin_clz(t);
      result += 32;
    }
  return result;
}

uint32_t ctz(unsigned __int128 x)
{
  uint32_t result = 0;
  for (int i = 0; i < 4; i++)
    {
      uint32_t t = x >> (i * 32);
      if (t)
	return result + __builtin_ctz(t);
      result += 32;
    }
  return result;
}

bool is_pow2(unsigned __int128 x)
{
  return x != 0 && (x & (x - 1)) == 0;
}

bool is_value_zero(Inst *inst)
{
  return inst->op == Op::VALUE && inst->value() == 0;
}

bool is_value_one(Inst *inst)
{
  return inst->op == Op::VALUE && inst->value() == 1;
}

bool is_value_signed_min(Inst *inst)
{
  if (inst->op != Op::VALUE)
    return false;
  unsigned __int128 smin = ((unsigned __int128)1) << (inst->bitsize - 1);
  return inst->value() == smin;
}

bool is_value_signed_min(Inst *inst, uint32_t bitsize)
{
  if (inst->op != Op::VALUE)
    return false;
  __int128 smin = ((unsigned __int128)1) << (bitsize - 1);
  smin = (smin << (128 - bitsize)) >> (128 - bitsize);
  return inst->signed_value() == smin;
}

bool is_value_signed_max(Inst *inst)
{
  if (inst->op != Op::VALUE)
    return false;
  unsigned __int128 smax = (((unsigned __int128)1) << (inst->bitsize - 1)) - 1;
  return inst->value() == smax;
}

bool is_value_signed_max(Inst *inst, uint32_t bitsize)
{
  if (inst->op != Op::VALUE)
    return false;
  unsigned __int128 smax = (((unsigned __int128)1) << (bitsize - 1)) - 1;
  return inst->value() == smax;
}

bool is_value_m1(Inst *inst)
{
  if (inst->op != Op::VALUE)
    return false;
  unsigned __int128 m1 = ~((unsigned __int128)0);
  m1 = (m1 << (128 - inst->bitsize)) >> (128 - inst->bitsize);
  return inst->value() == m1;
}

bool is_value_pow2(Inst *inst)
{
  if (inst->op != Op::VALUE)
    return false;
  return is_pow2(inst->value());
}

Inst *gen_fmin(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return gen_fmin_fmax(bb, elem1, elem2, true);
}

Inst *gen_fmax(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return gen_fmin_fmax(bb, elem1, elem2, false);
}

Inst *gen_bitreverse(Basic_block *bb, Inst *arg)
{
  Inst *inst = bb->build_trunc(arg, 1);
  for (uint32_t i = 1; i < arg->bitsize; i += 1)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      inst = bb->build_inst(Op::CONCAT, inst, bit);
    }
  return inst;
}

Inst *gen_clz(Basic_block *bb, Inst *arg)
{
  Inst *inst = bb->value_inst(arg->bitsize, arg->bitsize);
  for (unsigned i = 0; i < arg->bitsize; i++)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *val = bb->value_inst(arg->bitsize - i - 1, arg->bitsize);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  return inst;
}

Inst *gen_ctz(Basic_block *bb, Inst *arg)
{
  Inst *inst = bb->value_inst(arg->bitsize, arg->bitsize);
  for (int i = arg->bitsize - 1; i >= 0; i--)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *val = bb->value_inst(i, arg->bitsize);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  return inst;
}

Inst *gen_clrsb(Basic_block *bb, Inst *arg)
{
  Inst *signbit = bb->build_extract_bit(arg, arg->bitsize - 1);
  Inst *inst = bb->value_inst(arg->bitsize - 1, arg->bitsize);
  for (unsigned i = 0; i < arg->bitsize - 1; i++)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *cmp = bb->build_inst(Op::NE, bit, signbit);
      Inst *val = bb->value_inst(arg->bitsize - i - 2, arg->bitsize);
      inst = bb->build_inst(Op::ITE, cmp, val, inst);
    }
  return inst;
}

Inst *gen_popcount(Basic_block *bb, Inst *arg)
{
  Inst *bit = bb->build_extract_bit(arg, 0);
  Inst *inst = bb->build_inst(Op::ZEXT, bit, arg->bitsize);
  for (uint32_t i = 1; i < arg->bitsize; i++)
    {
      bit = bb->build_extract_bit(arg, i);
      Inst *ext = bb->build_inst(Op::ZEXT, bit, arg->bitsize);
      inst = bb->build_inst(Op::ADD, inst, ext);
    }
  return inst;
}

Inst *gen_bswap(Basic_block *bb, Inst *arg)
{
  Inst *inst = bb->build_trunc(arg, 8);
  for (uint32_t i = 8; i < arg->bitsize; i += 8)
    {
      Inst *byte = bb->build_inst(Op::EXTRACT, arg, i + 7, i);
      inst = bb->build_inst(Op::CONCAT, inst, byte);
    }
  return inst;
}

}  // end namespace smtgcc
