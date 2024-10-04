#include <cassert>

#include "gcc-plugin.h"
#include "tree-pass.h"
#include "context.h"
#include "tree.h"
#include "diagnostic-core.h"

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
  PROP_cfg,
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
  }
  unsigned int execute(function *fun) final override;
  std::vector<riscv_state> functions;
};

unsigned int tv_pass::execute(function *fun)
{
  try
    {
      CommonState state;
      Module *module = create_module(Arch::riscv);
      Function *src = process_function(module, &state, fun, false, Arch::riscv);
      src->name = "src";
      unroll_and_optimize(src);

      riscv_state rstate = setup_riscv_function(&state, src, fun);
      functions.push_back(rstate);
    }
  catch (Not_implemented& error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
    }
  return 0;
}

// Note: This assumes that we do not have any loops.
static void eliminate_registers(Function *func)
{
  std::map<Basic_block *, std::map<Inst *, Inst *>> bb2reg_values;

  // Collect all registers. This is not completely necessary, but we want
  // to iterate over the registers in a consisten order when we create
  // phi-nodes etc. and iterating over the maps could change order between
  // different runs.
  std::vector<Inst *> registers;
  for (Inst *inst = func->bbs[0]->first_inst;
       inst;
       inst = inst->next)
    {
      if (inst->op == Op::REGISTER)
	{
	  Basic_block *bb = func->bbs[0];
	  registers.push_back(inst);
	  // TODO: Should be an arbitrary value (i.e., a symbolic value)
	  // instead of 0.
	  bb2reg_values[bb][inst] = bb->value_inst(0, inst->bitsize);
	}
    }

  for (Basic_block *bb : func->bbs)
    {
      std::map<Inst *, Inst *>& reg_values =
	bb2reg_values[bb];
      if (bb->preds.size() == 1)
	{
	  reg_values = bb2reg_values.at(bb->preds[0]);
	}
      else if (bb->preds.size() > 1)
	{
	  for (auto reg : registers)
	    {
	      Inst *phi = bb->build_phi_inst(reg->bitsize);
	      reg_values[reg] = phi;
	      for (auto pred_bb : bb->preds)
		{
		  if (!bb2reg_values.at(pred_bb).contains(reg))
		    throw Not_implemented("eliminate_registers: Read of uninit register");
		  Inst *arg = bb2reg_values.at(pred_bb).at(reg);
		  phi->add_phi_arg(arg, pred_bb);
		}
	    }
	}

      Inst *inst = bb->first_inst;
      while (inst)
	{
	  Inst *next_inst = inst->next;
	  if (inst->op == Op::READ)
	    {
	      if (!reg_values.contains(inst->args[0]))
		throw Not_implemented("eliminate_registers: Read of uninit register");
	      inst->replace_all_uses_with(reg_values.at(inst->args[0]));
	      destroy_instruction(inst);
	    }
	  else if (inst->op == Op::WRITE)
	    {
	      reg_values[inst->args[0]] = inst->args[1];
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
  if (seen_error())
    return;

  struct tv_pass *my_pass = (struct tv_pass *)data;
  for (auto& state : my_pass->functions)
    {
      if (config.verbose > 0)
	fprintf(stderr, "SMTGCC: Checking %s\n", state.func_name.c_str());

      try
	{
	  Module *module = state.module;
	  Function *func = parse_riscv(asm_file_name, &state);
	  validate(func);

	  simplify_cfg(func);
	  if (loop_unroll(func))
	    {
	      simplify_insts(func);
	      dead_code_elimination(func);
	      simplify_cfg(func);
	    }

	  eliminate_registers(func);
	  validate(func);

	  // Simplify the code several times -- this is often necessary
	  // as instruction simplification enables new CFG simplifications
	  // that then enable new instruction simplifications.
	  // This is handled during unrolling for the GIMPLE passes, but
	  // it does not work here because we must do unrolling before
	  // eliminating the register instructions.
	  simplify_insts(func);
	  dead_code_elimination(func);
	  simplify_cfg(func);
	  vrp(func);
	  for (int i = 0; i < 2; i++)
	    {
	      simplify_insts(func);
	      dead_code_elimination(func);
	      simplify_cfg(func);
	    }

	  canonicalize_memory(module);
	  simplify_mem(module);
	  ls_elim(module);
	  simplify_insts(module);
	  dead_code_elimination(module);
	  simplify_cfg(module);

	  Solver_result result = check_refine(module);
	  if (result.status != Result_status::correct)
	    {
	      assert(result.message);
	      std::string msg = *result.message;
	      msg.pop_back();
	      fprintf(stderr, "%s:%s: %s\n", state.file_name.c_str(),
		      state.func_name.c_str(), msg.c_str());
	    }
	}
      catch (Parse_error error)
	{
	  fprintf(stderr, "%s:%d: Parse error: %s\n", asm_file_name, error.line,
		  error.msg.c_str());
	}
      catch (Not_implemented& error)
	{
	  fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
	}
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
