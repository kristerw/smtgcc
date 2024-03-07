#include <cassert>
#include <set>

#include "gcc-plugin.h"
#include "tree-pass.h"
#include "tree.h"
#include "cgraph.h"
#include "tree.h"
#include "diagnostic.h"

#include "smtgcc.h"
#include "gimple_conv.h"

using namespace std::string_literals;
using namespace smtgcc;

int plugin_is_GPL_compatible;

// Function keeping track of the translation validation information for
// a function.
struct tv_function
{
  std::string prev_pass_name;
  std::string pass_name;
  bool in_ssa_form = false;
  std::set<std::string> errors;
  Module *module = nullptr;
  CommonState *state = nullptr;

  void check();
  void delete_ir();
};

struct my_plugin {
  bool has_run_ssa_pass = false;
  bool new_functions_are_ssa = false;

  std::map<unsigned int, tv_function *> fun2tvfun;
};

// Delete the IR (if any) from the previous pass.
void tv_function::delete_ir()
{
  if (module)
    {
      destroy_module(module);
      module = nullptr;
    }
  if (state)
    {
      delete state;
      state = nullptr;
    }
  prev_pass_name = "";
}

static Function *convert_function(tv_function *tv_fun, bool is_tgt_func = false)
{
  const char *name = is_tgt_func ? "tgt" : "src";
  try
    {
      Function *func =
	process_function(tv_fun->module, tv_fun->state, cfun, is_tgt_func);
      func->rename(name);
      return func;
    }
  catch (Not_implemented& error)
    {
      if (!tv_fun->errors.contains(error.msg))
	{
	  fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
	  tv_fun->errors.insert(error.msg);
	}
    }
  return nullptr;
}

void tv_function::check()
{
  try
    {
      if (config.verbose > 0)
	fprintf(stderr, "SMTGCC: Checking %s -> %s : %s\n",
		prev_pass_name.c_str(), pass_name.c_str(),
		function_name(cfun));

      adjust_loop_vectorized(module);
      canonicalize_memory(module);
      simplify_mem(module);
      ls_elim(module);
      simplify_insts(module);
      simplify_cfg(module);
      dead_code_elimination(module);

      validate(module);

      Solver_result result = check_refine(module);
      if (result.status != Result_status::correct)
	{
	  assert(result.message);
	  std::string warning =
	    prev_pass_name + " -> " + pass_name + ": " + *result.message;
	  warning.pop_back();
	  inform(DECL_SOURCE_LOCATION(cfun->decl), "%s", warning.c_str());
	}
    }
  catch (Not_implemented& error)
    {
      if (!errors.contains(error.msg))
	{
	  fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
	  errors.insert(error.msg);
	}
    }
}

static void ipa_pass(opt_pass *pass, my_plugin *plugin_data)
{
  if (plugin_data->has_run_ssa_pass)
    plugin_data->new_functions_are_ssa = true;

  struct cgraph_node *node;
  FOR_EACH_FUNCTION_WITH_GIMPLE_BODY(node)
    {
      function *fun = node->get_fun();
      if (!plugin_data->fun2tvfun.contains(DECL_UID(fun->decl)))
	continue;
      tv_function *tv_fun = plugin_data->fun2tvfun.at(DECL_UID(fun->decl));
      if (!tv_fun->in_ssa_form)
	continue;
      if (!tv_fun->module)
	continue;

      push_cfun(fun);
      Function *func = convert_function(tv_fun, true);
      if (func)
	tv_fun->check();
      pop_cfun();

      tv_fun->delete_ir();
      tv_fun->pass_name = pass->name;
    }
}

static void gimple_pass(opt_pass *pass, my_plugin *plugin_data)
{
  tv_function *tv_fun;
  if (!plugin_data->fun2tvfun.contains(DECL_UID(cfun->decl)))
    {
      tv_fun = new tv_function;
      plugin_data->fun2tvfun[DECL_UID(cfun->decl)] = tv_fun;
      if (plugin_data->new_functions_are_ssa)
	tv_fun->in_ssa_form = true;
    }
  else
    tv_fun = plugin_data->fun2tvfun.at(DECL_UID(cfun->decl));

  if (tv_fun->pass_name == "ssa")
    {
      plugin_data->has_run_ssa_pass = true;
      tv_fun->in_ssa_form = true;
    }

  if (!tv_fun->in_ssa_form)
    {
      tv_fun->delete_ir();
      tv_fun->pass_name = pass->name;
      return;
    }

  if (tv_fun->module)
    {
      // The vectorizer modifies a copy of the scalar loop in-place
      // and relies on dce to remove unused calculations. Some of the
      // unused instruction may start to overflow from the vectorization
      // (see PR 111257), so we must wait for the following dce pass
      // before checking the IR.
      if (tv_fun->pass_name == "vect")
	{
	  tv_fun->pass_name = pass->name;
	  return;
	}

      Function *func = convert_function(tv_fun, true);
      if (!func)
	{
	  tv_fun->delete_ir();
	  tv_fun->pass_name = pass->name;
	  return;
	}
      tv_fun->check();
      tv_fun->delete_ir();
    }

  assert(!tv_fun->module);
  tv_fun->module = create_module();
  tv_fun->state = new CommonState();
  Function *func = convert_function(tv_fun);
  if (!func)
    {
      tv_fun->delete_ir();
      tv_fun->pass_name = pass->name;
      return;
    }
  tv_fun->prev_pass_name = tv_fun->pass_name;
  tv_fun->pass_name = pass->name;
}

static void pass_execution(void *event_data, void *data)
{
  opt_pass *pass = (opt_pass *)event_data;
  my_plugin *plugin_data = (my_plugin *)data;

  if (pass->name[0] == '*')
    return;

  if (pass->type == GIMPLE_PASS)
    gimple_pass(pass, plugin_data);
  else if (pass->type == IPA_PASS || pass->type == SIMPLE_IPA_PASS)
    ipa_pass(pass, plugin_data);
}

int
plugin_init(struct plugin_name_args *plugin_info,
	    [[maybe_unused]] struct plugin_gcc_version *version)
{
  const char * const plugin_name = plugin_info->base_name;

  my_plugin *mp = new my_plugin;
  register_callback(plugin_name, PLUGIN_PASS_EXECUTION, pass_execution, (void*)mp);

  return 0;
}
