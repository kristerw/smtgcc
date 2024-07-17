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

bool is_value_signed_max(Inst *inst)
{
  unsigned __int128 smax = (((unsigned __int128)1) << (inst->bitsize - 1)) - 1;
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

}  // end namespace smtgcc
