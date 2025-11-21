#include <cassert>

#include "gcc-plugin.h"
#include "plugin-version.h"
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
#ifdef SMTGCC_AARCH64
  std::vector<aarch64_state> functions;
#endif
#ifdef SMTGCC_BPF
  std::vector<bpf_state> functions;
#endif
#ifdef SMTGCC_RISCV
  std::vector<riscv_state> functions;
#endif
#ifdef SMTGCC_SH
  std::vector<sh_state> functions;
#endif
};

unsigned int tv_pass::execute(function *fun)
{
  try
    {
#if defined(SMTGCC_AARCH64)
      CommonState state(Arch::aarch64);
      Module *module = create_module(Arch::aarch64);
#elif defined(SMTGCC_BPF)
      CommonState state(Arch::bpf);
      Module *module = create_module(Arch::bpf);
#elif defined(SMTGCC_RISCV)
      CommonState state(Arch::riscv);
      Module *module = create_module(Arch::riscv);
#elif defined(SMTGCC_SH)
      CommonState state(Arch::sh);
      Module *module = create_module(Arch::sh);
#endif
      Function *src = process_function(module, &state, fun, false);
      src->name = "src";
      unroll_and_optimize(src);

#if defined(SMTGCC_AARCH64)
      aarch64_state rstate = setup_aarch64_function(&state, src, fun);
#elif defined(SMTGCC_BPF)
      bpf_state rstate = setup_bpf_function(&state, src, fun);
#elif defined(SMTGCC_RISCV)
      riscv_state rstate = setup_riscv_function(&state, src, fun);
#elif defined(SMTGCC_SH)
      sh_state rstate = setup_sh_function(&state, src, fun);
#endif
      functions.push_back(rstate);
    }
  catch (Not_implemented& error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
    }
  return 0;
}

static void eliminate_registers(Function *func, int64_t& symbolic_id)
{
  // Collect all registers.
  std::vector<Inst *> registers;
  for (Inst *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->op == Op::REGISTER)
	registers.push_back(inst);
    }

  std::map<Basic_block *, std::map<Inst *, Inst *>> bb2reg_values;
  for (Basic_block *bb : func->bbs)
    {
      std::map<Inst *, Inst *>& reg_values = bb2reg_values[bb];

      // Set up the values at the top of the BB.
      if (bb->preds.size() == 0)
	{
	  // Create the initial register values.
	  for (auto reg : registers)
	    {
	      Inst *inst =
		bb->build_inst(Op::SYMBOLIC, symbolic_id++, reg->bitsize);
	      reg_values.insert({reg, inst});
	    }
	}
      else if (bb->preds.size() == 1)
	reg_values = bb2reg_values[bb->preds[0]];
      else
	{
	  for (auto reg : registers)
	    {
	      Inst *phi = bb->build_phi_inst(reg->bitsize);
	      reg_values.insert({reg, phi});
	      for (auto pred : bb->preds)
		{
		  Inst *reg_value;
		  if (pred == bb || bb2reg_values[pred].empty())
		    reg_value = pred->build_inst(Op::READ, reg);
		  else
		    reg_value = bb2reg_values[pred][reg];
		  phi->add_phi_arg(reg_value, pred);
		}
	    }
	}

      // Eliminate all Op::READ and Op::WRITE.
      for (Inst *inst = bb->first_inst; inst;)
	{
	  Inst *next_inst = inst->next;

	  if (inst->op == Op::WRITE)
	    {
	      reg_values[inst->args[0]] = inst->args[1];
	      destroy_instruction(inst);
	    }
	  else if (inst->op == Op::READ)
	    {
	      inst->replace_all_uses_with(reg_values[inst->args[0]]);
	      destroy_instruction(inst);
	    }

	  inst = next_inst;
	}
    }

  // Eliminate all Op::REGISTER.
  for (auto reg : registers)
    destroy_instruction(reg);
}

static void finish(void *, void *data)
{
  if (seen_error())
    return;

  const char *file_name = getenv("SMTGCC_ASM");
  if (!file_name)
    file_name = asm_file_name;

  struct tv_pass *my_pass = (struct tv_pass *)data;
  for (auto& state : my_pass->functions)
    {
      if (config.verbose > 0)
	fprintf(stderr, "SMTGCC: Checking %s\n", state.func_name.c_str());

      try
	{
	  Module *module = state.module;
#if defined(SMTGCC_AARCH64)
	  Function *func = parse_aarch64(file_name, &state);
#elif defined(SMTGCC_BPF)
	  Function *func = parse_bpf(file_name, &state);
#elif defined(SMTGCC_RISCV)
	  Function *func = parse_riscv(file_name, &state);
#elif defined(SMTGCC_SH)
	  Function *func = parse_sh(file_name, &state);
#endif
	  validate(func);

	  eliminate_registers(func, state.symbolic_id);
	  unroll_and_optimize(func);

	  canonicalize_memory(module);
	  cse(module);
	  simplify_mem(module);
	  ls_elim(module);
	  reduce_bitsize(module);
	  bool cfg_modified;
	  do
	    {
	      simplify_insts(module);
	      dead_code_elimination(module);
	      cfg_modified = simplify_cfg(module);
	    }
	  while (cfg_modified);
	  sort_stores(module);

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
	  fprintf(stderr, "%s:%d: Parse error: %s\n", file_name, error.line,
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
	    struct plugin_gcc_version *version)
{
  if (!plugin_default_version_check(version, &gcc_version))
    return 1;

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
