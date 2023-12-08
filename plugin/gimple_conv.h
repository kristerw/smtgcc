#ifndef SMTGCC_GIMPLE_CONV_H
#define SMTGCC_GIMPLE_CONV_H

#include <map>
#include <vector>

struct CommonState {
  // ID 0 - reserved for NULL
  //    1 - reserved for anonymous memory
  int64_t id_local = 0;
  int64_t id_global = 2;
  std::map<tree, int64_t> decl2id;

  std::vector<smtgcc::Param_info> params;

  // The next free index for a symbolic instruction.
  uint32_t symbolic_idx = 0;
};

smtgcc::Function *process_function(smtgcc::Module *module, CommonState *, function *fun);
smtgcc::Module *create_module();

#endif
