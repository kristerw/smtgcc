#ifndef SMTGCC_GIMPLE_CONV_H
#define SMTGCC_GIMPLE_CONV_H

#include <map>
#include <vector>

// The symbolic instruction index used for CFN_LOOP_VECTORIZED.
#define LOOP_VECT_SYM_IDX  0

struct CommonState {
  // ID 0 - reserved for NULL
  //    1 - reserved for anonymous memory
  int64_t id_local = 0;
  int64_t id_global = 2;
  std::map<tree, int64_t> decl2id;

  std::vector<smtgcc::Param_info> params;
};

smtgcc::Function *process_function(smtgcc::Module *module, CommonState *, function *fun, bool is_tgt_func = false);
smtgcc::Module *create_module();
void adjust_loop_vectorized(smtgcc::Module *module);

#endif
