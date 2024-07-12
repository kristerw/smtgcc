#ifndef UTIL_H
#define UTIL_H

#include "smtgcc.h"

namespace smtgcc {

int popcount(unsigned __int128 x);
int clz(unsigned __int128 x);
int ctz(unsigned __int128 x);

bool is_value_zero(Inst *inst);
bool is_value_one(Inst *inst);
bool is_value_signed_min(Inst *inst);
bool is_value_signed_max(Inst *inst);
bool is_value_m1(Inst *inst);
bool is_value_pow2(Inst *inst);

} // end namespace smtgcc

#endif
