#include <cstdint>

#include "util.h"

namespace smtgcc {

int popcount(unsigned __int128 x)
{
  int result = 0;
  for (int i = 0; i < 4; i++)
    {
      uint32_t t = x >> (i* 32);
      result += __builtin_popcount(t);
    }
  return result;
}

int clz(unsigned __int128 x)
{
  int result = 0;
  for (int i = 0; i < 4; i++)
    {
      uint32_t t = x >> ((3 - i) * 32);
      if (t)
	return result + __builtin_clz(t);
      result += 32;
    }
  return result;
}

int ctz(unsigned __int128 x)
{
  int result = 0;
  for (int i = 0; i < 4; i++)
    {
      uint32_t t = x >> (i * 32);
      if (t)
	return result + __builtin_ctz(t);
      result += 32;
    }
  return result;
}

bool is_value_zero(Instruction *inst)
{
  return inst->opcode == Op::VALUE && inst->value() == 0;
}

bool is_value_one(Instruction *inst)
{
  return inst->opcode == Op::VALUE && inst->value() == 1;
}

bool is_value_signed_min(Instruction *inst)
{
  unsigned __int128 smin = ((unsigned __int128)1) << (inst->bitsize - 1);
  return inst->opcode == Op::VALUE && inst->value() == smin;
}

bool is_value_signed_max(Instruction *inst)
{
  unsigned __int128 smax = (((unsigned __int128)1) << (inst->bitsize - 1)) - 1;
  return inst->opcode == Op::VALUE && inst->value() == smax;
}

bool is_value_m1(Instruction *inst)
{
  unsigned __int128 m1 = ~((unsigned __int128)0);
  m1 = (m1 << (128 - inst->bitsize)) >> (128 - inst->bitsize);
  return inst->opcode == Op::VALUE && inst->value() == m1;
}

bool is_value_pow2(Instruction *inst)
{
  if (inst->opcode != Op::VALUE)
    return false;
  unsigned __int128 value = inst->value();
  return value != 0 && (value & (value - 1)) == 0;
}

}  // end namespace smtgcc
