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
  "smtgcc-tv-backend",
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
  riscv_state *rstate = new riscv_state;
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
      rstate->module = module;
      rstate->param_is_unsigned = state.param_is_unsigned;
    }
  catch (Not_implemented& error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
      error_has_been_reported = true;
    }
  return 0;
}

// Note: This assumes that we do not have any loops.
static void eliminate_registers(Function *func)
{
  std::map<Basic_block *, std::map<Instruction *, Instruction *>> bb2reg_values;

  // Collect all registers. This is not completely necessary, but we want
  // to iterate over the registers in a consisten order when we create
  // phi-nodes etc. and iterating over the maps could change order between
  // different runs.
  std::vector<Instruction*> registers;
  for (Instruction *inst = func->bbs[0]->first_inst;
       inst;
       inst = inst->next)
    {
      if (inst->opcode == Op::REGISTER)
	{
	  Basic_block *bb = func->bbs[0];
	  registers.push_back(inst);
	  // TODO: Should be undef instead of 0.
	  bb2reg_values[bb][inst] = bb->value_inst(0, inst->bitsize);
	}
    }

  for (Basic_block *bb : func->bbs)
    {
      std::map<Instruction *, Instruction *>& reg_values =
	bb2reg_values[bb];
      if (bb->preds.size() == 1)
	{
	  reg_values = bb2reg_values.at(bb->preds[0]);
	}
      else if (bb->preds.size() > 1)
	{
	  for (auto reg : registers)
	    {
	      Instruction *phi = bb->build_phi_inst(reg->bitsize);
	      reg_values[reg] = phi;
	      for (auto pred_bb : bb->preds)
		{
		  if (!bb2reg_values.at(pred_bb).contains(reg))
		    throw Not_implemented("eliminate_registers: Read of uninit register");
		  Instruction *arg = bb2reg_values.at(pred_bb).at(reg);
		  phi->add_phi_arg(arg, pred_bb);
		}
	    }
	}

      Instruction *inst = bb->first_inst;
      while (inst)
	{
	  Instruction *next_inst = inst->next;
	  if (inst->opcode == Op::READ)
	    {
	      if (!reg_values.contains(inst->arguments[0]))
		throw Not_implemented("eliminate_registers: Read of uninit register");
	      inst->replace_all_uses_with(reg_values.at(inst->arguments[0]));
	      destroy_instruction(inst);
	    }
	  else if (inst->opcode == Op::WRITE)
	    {
	      reg_values[inst->arguments[0]] = inst->arguments[1];
	      destroy_instruction(inst);
	    }
	  inst = next_inst;
	}
    }

  for (auto inst : registers)
    {
      assert(inst->used_by.size() == 0);
      destroy_instruction(inst);
    }
}

static void finish(void *, void *data)
{
  // TODO: Check that we are generating a .s file, and use the name of the
  // newly generated .s file.
  const char *file_name = "k.s";

  struct tv_pass *my_pass = (struct tv_pass *)data;
  if (my_pass->error_has_been_reported)
    return;

  try
    {
      riscv_state *state = my_pass->rstate;
      Module *module = state->module;
      module->functions[0]->name = "src";
      Function *func = parse_riscv("k.s", state);
      reverse_post_order(func);
      eliminate_registers(func);
      simplify_insts(func);
      dead_code_elimination(func);

      Solver_result result = check_refine(module);
      if (result.status != Result_status::correct)
	{
	  assert(result.message);
	  std::string msg = *result.message;
	  msg.pop_back();
	  inform(UNKNOWN_LOCATION, "%s", msg.c_str());
	}
    }
 catch (Parse_error error)
    {
      fprintf(stderr, "%s:%d: Parse error: %s\n", file_name, error.line,
	      error.msg.c_str());
    }
  catch (Not_implemented& error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
    }
}

int
plugin_init(struct plugin_name_args *plugin_info,
	    [[maybe_unused]] struct plugin_gcc_version *version)
{
  const char * const plugin_name = plugin_info->base_name;

  struct register_pass_info tv_pass_info;
  struct tv_pass *my_pass = new tv_pass(g);
  tv_pass_info.pass = my_pass;
  tv_pass_info.reference_pass_name = "optimized";
  tv_pass_info.ref_pass_instance_number = 1;
  tv_pass_info.pos_op = PASS_POS_INSERT_AFTER;
  register_callback(plugin_name, PLUGIN_PASS_MANAGER_SETUP, NULL,
		    &tv_pass_info);

  register_callback(plugin_name, PLUGIN_FINISH, finish, (void*)my_pass);

  return 0;
}
