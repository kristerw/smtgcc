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
      rstate->params = state.params;
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

Instruction *extract(Instruction *inst, uint32_t reg_bitsize, uint32_t idx)
{
  Instruction *high = inst->bb->value_inst((idx + 1) * reg_bitsize - 1, 32);
  Instruction *low = inst->bb->value_inst(idx * reg_bitsize, 32);
  return create_inst(Op::EXTRACT, inst, high, low);
}

static void adjust_abi(Function *func, Function *src_func, riscv_state *state)
{
  Basic_block *src_last_bb = src_func->bbs.back();
  Basic_block *entry_bb = func->bbs[0];

  // Find the first instruction that is not a VALUE instruction.
  Instruction *first_inst = entry_bb->first_inst;
  while (first_inst->opcode == Op::VALUE)
    first_inst = first_inst->next;

  // Determine the register to use for each function parameter.
  int reg_nbr = 10;
  for (auto& param_info : state->params)
    {
      param_info.reg_nbr = reg_nbr;
      param_info.num_regs =
	(param_info.bitsize + state->reg_bitsize - 1) / state->reg_bitsize;
      reg_nbr += param_info.num_regs;
    }

  // Create an Op::PARAM instruction for each function parameter and store
  // it in the registers.
  Basic_block *src_entry_bb = src_func->bbs[0];
  Instruction *reg_bitsize_inst = entry_bb->value_inst(state->reg_bitsize, 32);
  for (Instruction *inst = src_entry_bb->first_inst; inst; inst = inst->next)
    {
      if (inst->opcode != Op::PARAM)
	continue;

      // Create a copy of the source function's Op::PARAM instruction.
      int param_number = inst->arguments[0]->value();
      Param_info& param_info = state->params.at(param_number);
      Instruction *param_nbr = entry_bb->value_inst(param_number, 32);
      Instruction *param_bitsize = entry_bb->value_inst(inst->bitsize, 32);
      Instruction *param = create_inst(Op::PARAM, param_nbr, param_bitsize);
      param->insert_before(first_inst);

      // Pad it out to a multiple of the register size.
      if (param->bitsize < state->reg_bitsize * param_info.num_regs)
	{
	  if (param_info.is_unsigned && param->bitsize != 32)
	    param = create_inst(Op::ZEXT, param, reg_bitsize_inst);
	  else
	    param = create_inst(Op::SEXT, param, reg_bitsize_inst);
	  param->insert_before(first_inst);
	}

      // Write the parameter value to the registers.
      for (uint32_t i = 0; i < param_info.num_regs; i++)
	{
	  Instruction *reg_value = extract(param, state->reg_bitsize, i);
	  reg_value->insert_before(first_inst);
	  Instruction *reg = state->registers[param_info.reg_nbr + i];
	  Instruction *write = create_inst(Op::WRITE, reg, reg_value);
	  write->insert_before(first_inst);
	}
    }

  // Generate the return value from the registers.
  assert(src_last_bb->last_inst->opcode == Op::RET);
  if (src_last_bb->last_inst->nof_args > 0)
    {
      Basic_block *exit_bb = func->bbs.back();
      uint32_t ret_size = src_last_bb->last_inst->arguments[0]->bitsize;
      Instruction *retval =
	exit_bb->build_inst(Op::READ, state->registers[10]);
      for (int reg_nbr = 11; retval->bitsize < ret_size; reg_nbr++)
	{
	  Instruction *inst =
	    exit_bb->build_inst(Op::READ, state->registers[reg_nbr]);
	  retval = exit_bb->build_inst(Op::CONCAT, inst, retval);
	}
      if (ret_size < retval->bitsize)
	retval = exit_bb->build_trunc(retval, ret_size);
      destroy_instruction(exit_bb->last_inst);
      exit_bb->build_ret_inst(retval);
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
      state->reg_bitsize = TARGET_64BIT ? 64 : 32;
      Function *func = parse_riscv("k.s", state);
      reverse_post_order(func);
      adjust_abi(func, module->functions[0], state);
      eliminate_registers(func);
      simplify_insts(func);
      dead_code_elimination(func);

      canonicalize_memory(module);
      simplify_mem(module);
      ls_elim(module);
      simplify_insts(module);
      dead_code_elimination(module);

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
