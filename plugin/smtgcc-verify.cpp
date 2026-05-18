#include <cassert>
#include <set>

#include "gcc-plugin.h"
#include "plugin-version.h"
#include "tree-pass.h"
#include "tree.h"
#include "cgraph.h"
#include "tree.h"
#include "diagnostic-core.h"

#include "smtgcc.h"
#include "gimple_conv.h"

using namespace std::string_literals;
using namespace smtgcc;

int plugin_is_GPL_compatible;

struct my_plugin;

struct my_plugin {
  bool done = false;
  bool error_occurred = false;
};

static Function *convert_function(Module *module, CommonState *state)
{
  try
    {
      return process_function(module, state, cfun, Function_role::ver);
    }
  catch (Not_implemented& error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
    }
  return nullptr;
}

static void verify(my_plugin *plugin_data)
{
  Module *module = create_module();
  CommonState *state = new CommonState();
  Function *func = convert_function(module, state);
  if (!func)
    {
      destroy_module(module);
      delete state;
      return;
    }

  if (config.verbose > 0)
    fprintf(stderr, "SMTGCC: Verifying: %s\n", function_name(cfun));

  try
    {
      int64_t symbolic_id = 0;
      unroll_and_optimize(func, symbolic_id);
      canonicalize_memory(func);
      cse(func);
      simplify_mem(func);
      dead_code_elimination(func);
      bool cfg_modified = simplify_cfg(func);
      while (cfg_modified)
	{
	  simplify_insts(func);
	  dead_code_elimination(func);
	  cfg_modified = simplify_cfg(func);
	}
      sort_stores(func);

      validate(func);

      Solver_result result = verify(func);
      if (result.status != Result_status::correct)
	{
	  assert(result.message);
	  std::string warning = *result.message;
	  warning.pop_back();
	  inform(DECL_SOURCE_LOCATION(cfun->decl), "%s", warning.c_str());
	}
    }
  catch (Not_implemented& error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
    }
  catch (Error& error)
    {
      fprintf(stderr, "Error: %s\n", error.msg.c_str());

      // The library may be in a bad state. For example, if the SMT
      // solver has thrown an out of memory error (because the solver
      // ignored the memory limit), it may not have freed the memory,
      // and all subsequent solver runs may return "unknown" even if
      // they could have been analyzed. This pollutes the cache, so
      // future compilations will get an incorrectly cached timeout
      // message even if we compile them separately without the
      // function that ran out of memory. So we stop here instead
      // of (incorrectly) processing the subsequent functions.
      plugin_data->error_occurred = true;
    }

  destroy_module(module);
  delete state;
}

static void ipa_pass(my_plugin *plugin_data)
{
  struct cgraph_node *node;
  FOR_EACH_FUNCTION_WITH_GIMPLE_BODY(node)
    {
      push_cfun(node->get_fun());
      verify(plugin_data);
      pop_cfun();
      if (plugin_data->error_occurred)
	return;
    }
}

static void pass_execution(void *event_data, void *data)
{
  opt_pass *pass = (opt_pass *)event_data;
  my_plugin *plugin_data = (my_plugin *)data;

  if (pass->name[0] == '*')
    return;

  if (plugin_data->done || plugin_data->error_occurred)
    return;

  // We must run the verification after the ubsan pass so we can verify
  // sanitizer properties. We also want to do this as an IPA pass so
  // we can see all functions (to be able to use other functions in
  // contracts).
  if (pass->type == SIMPLE_IPA_PASS && !strcmp(pass->name, "targetclone"))
    {
      plugin_data->done = true;
      ipa_pass(plugin_data);
    }
}

static void finish(void *, void *data)
{
  my_plugin *plugin_data = (my_plugin *)data;
  delete plugin_data;
}

int
plugin_init(struct plugin_name_args *plugin_info,
	    struct plugin_gcc_version *version)
{
  if (!plugin_default_version_check(version, &gcc_version))
    return 1;

  const char * const plugin_name = plugin_info->base_name;
  my_plugin *mp = new my_plugin;
  register_callback(plugin_name, PLUGIN_PASS_EXECUTION, pass_execution,
		    (void*)mp);
  register_callback(plugin_name, PLUGIN_FINISH, finish, (void*)mp);

  return 0;
}
