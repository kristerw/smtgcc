#include <cassert>

#include "gcc-plugin.h"
#include "tree-pass.h"
#include "context.h"
#include "tree.h"
#include "diagnostic.h"

#include "smtgcc.h"
#include "gimple_conv.h"

using namespace std::string_literals;
using namespace smtgcc;

int plugin_is_GPL_compatible;

const pass_data tv_pass_data =
{
  GIMPLE_PASS,
  "smtgcc-refine-check",
  OPTGROUP_NONE,
  TV_NONE,
  PROP_gimple_any,
  0,
  0,
  0,
  0
};

struct tv_pass : gimple_opt_pass
{
  tv_pass(gcc::context *ctx)
    : gimple_opt_pass(tv_pass_data, ctx)
  {
    module = create_module();
  }
  unsigned int execute(function *fun) final override;
  Module *module;
  bool error_has_been_reported = false;
};

unsigned int tv_pass::execute(function *fun)
{
  if (error_has_been_reported)
    return 0;

  try
    {
      CommonState state;
      process_function(module, &state, fun);
      if (module->functions.size() == 2)
	{
	  // TODO: Is canonicalize memory needed? It should obviously be
	  // the same globals in both.
	  canonicalize_memory(module);
	  simplify_mem(module);
	  ls_elim(module);
	  simplify_insts(module);
	  simplify_cfg(module);
	  dead_code_elimination(module);

	  Solver_result result = check_refine(module);
	  if (result.status != Result_status::correct)
	    {
	      assert(result.message);
	      std::string msg = *result.message;
	      msg.pop_back();
	      inform(DECL_SOURCE_LOCATION(cfun->decl), "%s", msg.c_str());
	    }
	}
    }
  catch (Not_implemented& error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
      error_has_been_reported = true;
    }
  return 0;
}

int
plugin_init(struct plugin_name_args *plugin_info,
	    [[maybe_unused]] struct plugin_gcc_version *version)
{
  const char * const plugin_name = plugin_info->base_name;

  struct register_pass_info tv_pass_info;
  tv_pass_info.pass = new tv_pass(g);
  tv_pass_info.reference_pass_name = "ssa";
  tv_pass_info.ref_pass_instance_number = 1;
  tv_pass_info.pos_op = PASS_POS_INSERT_AFTER;
  register_callback(plugin_name, PLUGIN_PASS_MANAGER_SETUP, NULL,
		    &tv_pass_info);

  return 0;
}
