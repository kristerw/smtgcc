#include <cassert>

#include "smtgcc.h"

namespace smtgcc {

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
  unsigned __int128 smin = ((unsigned __int128)1) << (inst->bitsize - 1);
  return inst->op == Op::VALUE && inst->value() == smin;
}

bool is_value_signed_min(Inst *inst, uint32_t bitsize)
{
  assert(bitsize < 128);
  __int128 smin = ((unsigned __int128)1) << (bitsize - 1);
  smin = (smin << (128 - bitsize)) >> (128 - bitsize);
  return inst->op == Op::VALUE && inst->signed_value() == smin;
}

bool is_value_signed_max(Inst *inst)
{
  unsigned __int128 smax = (((unsigned __int128)1) << (inst->bitsize - 1)) - 1;
  return inst->op == Op::VALUE && inst->value() == smax;
}

bool is_value_signed_max(Inst *inst, uint32_t bitsize)
{
  unsigned __int128 smax = (((unsigned __int128)1) << (bitsize - 1)) - 1;
  return inst->op == Op::VALUE && inst->value() == smax;
}

bool is_value_m1(Inst *inst)
{
  unsigned __int128 m1 = ~((unsigned __int128)0);
  m1 = (m1 << (128 - inst->bitsize)) >> (128 - inst->bitsize);
  return inst->op == Op::VALUE && inst->value() == m1;
}

bool is_value_pow2(Inst *inst)
{
  if (inst->op != Op::VALUE)
    return false;
  unsigned __int128 value = inst->value();
  return value != 0 && (value & (value - 1)) == 0;
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
  Inst *inst = bb->value_inst(arg->bitsize, 32);
  for (unsigned i = 0; i < arg->bitsize; i++)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *val = bb->value_inst(arg->bitsize - i - 1, 32);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  return inst;
}

Inst *gen_clrsb(Basic_block *bb, Inst *arg)
{
  Inst *signbit = bb->build_extract_bit(arg, arg->bitsize - 1);
  Inst *inst = bb->value_inst(arg->bitsize - 1, 32);
  for (unsigned i = 0; i < arg->bitsize - 1; i++)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *cmp = bb->build_inst(Op::NE, bit, signbit);
      Inst *val = bb->value_inst(arg->bitsize - i - 2, 32);
      inst = bb->build_inst(Op::ITE, cmp, val, inst);
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
