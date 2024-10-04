#ifndef SMTGCC_GIMPLE_CONV_H
#define SMTGCC_GIMPLE_CONV_H

#include <map>
#include <vector>

#include "gcc-plugin.h"
#include "tree.h"

#include "smtgcc.h"

enum class Arch {
  generic,
  riscv
};

struct CommonState {
  // ID 0 - reserved for NULL
  //    1 - reserved for anonymous memory
  int64_t id_local = 0;
  int64_t id_global = 2;
  std::map<tree, int64_t> decl2id;

  std::vector<smtgcc::MemoryObject> memory_objects;
};

smtgcc::Function *process_function(smtgcc::Module *module, CommonState *, function *fun, bool is_tgt_func, Arch arch = Arch::generic);
void unroll_and_optimize(smtgcc::Function *func);
void unroll_and_optimize(smtgcc::Module *module);
smtgcc::Module *create_module(Arch arch = Arch::generic);
uint64_t bitsize_for_type(tree type);
unsigned __int128 get_int_cst_val(tree expr);

smtgcc::riscv_state setup_riscv_function(CommonState *state, smtgcc::Function *src_func, function *fun);

#endif
