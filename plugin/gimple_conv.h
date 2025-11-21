#ifndef SMTGCC_GIMPLE_CONV_H
#define SMTGCC_GIMPLE_CONV_H

#include <map>
#include <vector>

#include "smtgcc.h"

enum class Arch {
  gimple,
  aarch64,
  bpf,
  riscv,
  sh
};

struct CommonState {
  CommonState(Arch arch = Arch::gimple);

  // ID 0 - reserved for NULL
  //    1 - reserved for anonymous memory
  int64_t id_local = 0;
  int64_t id_global = 2;
  std::map<tree, int64_t> decl2id;
  std::vector<smtgcc::Inst*> restrict_ids;

  int64_t ptr_id_min;
  int64_t ptr_id_max;

  int64_t symbolic_id = 0;

  Arch arch;

  std::vector<smtgcc::MemoryObject> memory_objects;
};

smtgcc::Function *process_function(smtgcc::Module *module, CommonState *, function *fun, bool is_tgt_func);
void unroll_and_optimize(smtgcc::Function *func);
void unroll_and_optimize(smtgcc::Module *module);
smtgcc::Module *create_module(Arch arch = Arch::gimple);
uint64_t bitsize_for_type(tree type);
unsigned __int128 get_int_cst_val(tree expr);

smtgcc::aarch64_state setup_aarch64_function(CommonState *state, smtgcc::Function *src_func, function *fun);
smtgcc::bpf_state setup_bpf_function(CommonState *state, smtgcc::Function *src_func, function *fun);
smtgcc::riscv_state setup_riscv_function(CommonState *state, smtgcc::Function *src_func, function *fun);
smtgcc::sh_state setup_sh_function(CommonState *state, smtgcc::Function *src_func, function *fun);

#endif
