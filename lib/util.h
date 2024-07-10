#ifndef UTIL_H
#define UTIL_H

#include "smtgcc.h"

namespace smtgcc {

int popcount(unsigned __int128 x);
int clz(unsigned __int128 x);
int ctz(unsigned __int128 x);

bool is_value_zero(Instruction *inst);
bool is_value_one(Instruction *inst);
bool is_value_signed_min(Instruction *inst);
bool is_value_signed_max(Instruction *inst);
bool is_value_m1(Instruction *inst);
bool is_value_pow2(Instruction *inst);

} // end namespace smtgcc

#endif
