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
  bool error_has_been_reported = false;
  std::vector<riscv_state> functions;
};

struct Regs
{
  Instruction *regs[2] = {nullptr, nullptr};
  Instruction *fregs[2] = {nullptr, nullptr};
};

static std::optional<Regs> regs_for_value(riscv_state *rstate, Instruction *value, tree type)
{
  Basic_block *bb = value->bb;

  if (COMPLEX_FLOAT_TYPE_P(type) && value->bitsize <= 2 * rstate->freg_bitsize)
    {
      Regs regs;
      uint64_t elt_bitsize = value->bitsize / 2;
      assert(elt_bitsize == 16 || elt_bitsize == 32 || elt_bitsize == 64);
      Instruction *reg_value = bb->build_trunc(value, elt_bitsize);
      if (reg_value->bitsize < rstate->freg_bitsize)
	{
	  uint32_t padding_bitsize =
	    rstate->freg_bitsize - reg_value->bitsize;
	  Instruction *m1 = bb->value_m1_inst(padding_bitsize);
	  reg_value = bb->build_inst(Op::CONCAT, m1, reg_value);
	}
      regs.fregs[0] = reg_value;
      Instruction *high = bb->value_inst(value->bitsize - 1, 32);
      Instruction *low = bb->value_inst(elt_bitsize, 32);
      reg_value = bb->build_inst(Op::EXTRACT, value, high, low);
      if (reg_value->bitsize < rstate->freg_bitsize)
	{
	  uint32_t padding_bitsize =
	    rstate->freg_bitsize - reg_value->bitsize;
	  Instruction *m1 = bb->value_m1_inst(padding_bitsize);
	  reg_value = bb->build_inst(Op::CONCAT, m1, reg_value);
	}
      regs.fregs[1] = reg_value;
      return regs;
    }

  if (SCALAR_FLOAT_TYPE_P(type) && value->bitsize <= rstate->freg_bitsize)
    {
      Instruction *reg_value = value;
      if (reg_value->bitsize < rstate->freg_bitsize)
	{
	  uint32_t padding_bitsize =
	    rstate->freg_bitsize - reg_value->bitsize;
	  Instruction *m1 = bb->value_m1_inst(padding_bitsize);
	  reg_value = bb->build_inst(Op::CONCAT, m1, reg_value);
	}
      Regs regs;
      regs.fregs[0] = reg_value;
      return regs;
    }

  if (value->bitsize <= 2 * rstate->reg_bitsize)
    {
      // Pad it out to a multiple of the register size.
      uint32_t num_regs = value->bitsize <= rstate->reg_bitsize ? 1 : 2;
      if (value->bitsize < rstate->reg_bitsize * num_regs)
	{
	  bool is_unsigned = INTEGRAL_TYPE_P(type) && TYPE_UNSIGNED(type);
	  Instruction *bs_inst =
	    bb->value_inst(rstate->reg_bitsize * num_regs, 32);
	  if (is_unsigned && value->bitsize != 32)
	    value = bb->build_inst(Op::ZEXT, value, bs_inst);
	  else
	    value = bb->build_inst(Op::SEXT, value, bs_inst);
	}

      Regs regs;
      regs.regs[0] = bb->build_trunc(value, rstate->reg_bitsize);
      if (num_regs > 1)
	{
	  Instruction *high = bb->value_inst(value->bitsize - 1, 32);
	  Instruction *low = bb->value_inst(rstate->reg_bitsize, 32);
	  regs.regs[1] = bb->build_inst(Op::EXTRACT, value, high, low);
	}
      return regs;
    }

  return {};
}

static void setup_riscv_function(riscv_state *rstate, Function *src_func, function *fun)
{
  assert(rstate->module->functions.size() == 2);
  Function *tgt = rstate->module->functions[1];
  Basic_block *entry_bb = rstate->entry_bb;
  Basic_block *exit_bb = rstate->exit_bb;

  // Registers x0-x31.
  for (int i = 0; i < 32; i++)
    {
      Instruction *bitsize = entry_bb->value_inst(rstate->reg_bitsize, 32);
      Instruction *reg = entry_bb->build_inst(Op::REGISTER, bitsize);
      rstate->registers.push_back(reg);
    }

  // Registers f0-f31.
  for (int i = 0; i < 32; i++)
    {
      Instruction *bitsize = entry_bb->value_inst(rstate->freg_bitsize, 32);
      Instruction *reg = entry_bb->build_inst(Op::REGISTER, bitsize);
      rstate->fregisters.push_back(reg);
    }

  // Create MEMORY instructions for the global variables we saw in the
  // GIMPLE IR.
  for (const auto& mem_obj : rstate->memory_objects)
    {
      Instruction *id =
	entry_bb->value_inst(mem_obj.id, rstate->module->ptr_id_bits);
      Instruction *size =
	entry_bb->value_inst(mem_obj.size, rstate->module->ptr_offset_bits);
      Instruction *flags = entry_bb->value_inst(mem_obj.flags, 32);
      Instruction *mem = entry_bb->build_inst(Op::MEMORY, id, size, flags);
      rstate->sym_name2mem.insert({mem_obj.sym_name, mem});
    }

  // Determine the register to use for each function parameter.
  uint32_t reg_nbr = 10;
  uint32_t freg_nbr = 10;

  Basic_block *src_last_bb = src_func->bbs.back();
  assert(src_last_bb->last_inst->opcode == Op::RET);
  uint64_t ret_bitsize = 0;
  if (src_last_bb->last_inst->nof_args > 0)
    ret_bitsize = src_last_bb->last_inst->arguments[0]->bitsize;
  if (ret_bitsize > 0)
    {
      Instruction *retval = nullptr;
      tree ret_type = TREE_TYPE(DECL_RESULT(fun->decl));
      if (COMPLEX_FLOAT_TYPE_P(ret_type)
	  && ret_bitsize <= 2 * rstate->freg_bitsize)
	{
	  // Generate the return value from the registers.
	  uint64_t elt_bitsize = ret_bitsize / 2;
	  assert(elt_bitsize == 16 || elt_bitsize == 32 || elt_bitsize == 64);
	  Instruction *real =
	    exit_bb->build_inst(Op::READ, rstate->fregisters[10]);
	  real = exit_bb->build_trunc(real, elt_bitsize);
	  Instruction *imag =
	    exit_bb->build_inst(Op::READ, rstate->fregisters[11]);
	  imag = exit_bb->build_trunc(imag, elt_bitsize);
	  retval = exit_bb->build_inst(Op::CONCAT, imag, real);
	}
      else if (SCALAR_FLOAT_TYPE_P(ret_type)
	       && ret_bitsize <= rstate->freg_bitsize)
	{
	  // Generate the return value from the registers.
	  retval = exit_bb->build_inst(Op::READ, rstate->fregisters[10]);
	  if (ret_bitsize < retval->bitsize)
	    retval = exit_bb->build_trunc(retval, ret_bitsize);
	}
      else if (ret_bitsize <= 2 * rstate->reg_bitsize)
	{
	  // Generate the return value from the registers.
	  retval = exit_bb->build_inst(Op::READ, rstate->registers[10]);
	  if (retval->bitsize < ret_bitsize)
	    {
	      Instruction *inst =
		exit_bb->build_inst(Op::READ, rstate->registers[11]);
	      retval = exit_bb->build_inst(Op::CONCAT, inst, retval);
	    }
	  if (ret_bitsize < retval->bitsize)
	    retval = exit_bb->build_trunc(retval, ret_bitsize);
	}
      else
	{
	  // Return of values wider than 2*reg_bitsize are passed in memory,
	  // where the address is specified by an implicit first parameter.
	  assert((ret_bitsize & 7) == 0);
	  Instruction *id = tgt->value_inst(-127, tgt->module->ptr_id_bits);
	  Instruction *mem_size =
	    tgt->value_inst(ret_bitsize / 8, tgt->module->ptr_offset_bits);
	  Instruction *flags = tgt->value_inst(0, 32);

	  Instruction *ret_mem =
	    entry_bb->build_inst(Op::MEMORY, id, mem_size, flags);
	  Instruction *reg = rstate->registers[reg_nbr++];
	  entry_bb->build_inst(Op::WRITE, reg, ret_mem);

	  // Generate the return value from the value returned in memory.
	  uint64_t size = ret_bitsize / 8;
	  for (uint64_t i = 0; i < size; i++)
	    {
	      Instruction *offset = exit_bb->value_inst(i, ret_mem->bitsize);
	      Instruction *ptr = exit_bb->build_inst(Op::ADD, ret_mem, offset);
	      Instruction *data_byte = exit_bb->build_inst(Op::LOAD, ptr);
	      if (retval)
		retval = exit_bb->build_inst(Op::CONCAT, data_byte, retval);
	      else
		retval = data_byte;
	    }
	}
      exit_bb->build_ret_inst(retval);
    }
  else
    exit_bb->build_ret_inst();

  // Set up the PARAM instructions and copy the result to the correct
  // register or memory as required by the ABI.
  int param_number = 0;
  for (tree decl = DECL_ARGUMENTS(fun->decl); decl; decl = DECL_CHAIN(decl))
    {
      uint32_t bitsize = bitsize_for_type(TREE_TYPE(decl));
      if (bitsize <= 0)
	throw Not_implemented("Parameter size == 0");

      Instruction *param_nbr = entry_bb->value_inst(param_number, 32);
      Instruction *param_bitsize = entry_bb->value_inst(bitsize, 32);
      Instruction *param =
	entry_bb->build_inst(Op::PARAM, param_nbr, param_bitsize);

      tree type = TREE_TYPE(decl);
      if (param_number == 0
	  && !strcmp(IDENTIFIER_POINTER(DECL_NAME(fun->decl)), "__ct_base "))
	{
	  // TODO: The "this" pointer in C++ constructors needs to be handled
	  // as a special case in the same way as in gimple_conv.cpp when
	  // setting up the parameters.
	  throw Not_implemented("setup_riscv_function: C++ constructors");
	}

      std::optional<Regs> arg_regs = regs_for_value(rstate, param, type);
      if (arg_regs)
	{
	  if ((*arg_regs).regs[0])
	    {
	      // TODO: Values are passed on the stack when all registers
	      // are used.
	      if (reg_nbr > 17)
		throw Not_implemented("riscv: too many arguments");

	      Instruction *reg = rstate->registers[reg_nbr++];
	      entry_bb->build_inst(Op::WRITE, reg, (*arg_regs).regs[0]);
	    }
	  if ((*arg_regs).regs[1])
	    {
	      // TODO: Values are passed on the stack when all registers
	      // are used.
	      if (reg_nbr > 17)
		throw Not_implemented("riscv: too many arguments");

	      Instruction *reg = rstate->registers[reg_nbr++];
	      entry_bb->build_inst(Op::WRITE, reg, (*arg_regs).regs[1]);
	    }
	  if ((*arg_regs).fregs[0])
	    {
	      // TODO: Values are passed on the stack when all registers
	      // are used.
	      if (freg_nbr > 17)
		throw Not_implemented("riscv: too many arguments");

	      Instruction *reg = rstate->fregisters[freg_nbr++];
	      entry_bb->build_inst(Op::WRITE, reg, (*arg_regs).fregs[0]);
	    }
	  if ((*arg_regs).fregs[1])
	    {
	      // TODO: Values are passed on the stack when all registers
	      // are used.
	      if (freg_nbr > 17)
		throw Not_implemented("riscv: too many arguments");

	      Instruction *reg = rstate->fregisters[freg_nbr++];
	      entry_bb->build_inst(Op::WRITE, reg, (*arg_regs).fregs[1]);
	    }
	}
      else
	{
	  // TODO: Implement passing of large params in memory.
	  throw Not_implemented("setup_riscv_function: too wide param type");
	}

      param_number++;
    }
}

unsigned int tv_pass::execute(function *fun)
{
  if (error_has_been_reported)
    return 0;

  try
    {
      CommonState state;
      Module *module = create_module();
      Function *src = process_function(module, &state, fun, false);
      src->name = "src";
      unroll_and_optimize(src);

      riscv_state rstate;
      rstate.reg_bitsize = TARGET_64BIT ? 64 : 32;
      rstate.freg_bitsize = 64;
      rstate.module = module;
      rstate.memory_objects = state.memory_objects;
      rstate.func_name = IDENTIFIER_POINTER(DECL_ASSEMBLER_NAME(fun->decl));

      Function *tgt = module->build_function("tgt");
      rstate.entry_bb = tgt->build_bb();
      rstate.exit_bb = tgt->build_bb();

      setup_riscv_function(&rstate, src, fun);

      functions.push_back(rstate);
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
  if (seen_error())
    return;

  struct tv_pass *my_pass = (struct tv_pass *)data;
  if (my_pass->error_has_been_reported)
    return;

  for (auto& state : my_pass->functions)
    {
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
	  for (int i = 0; i < 3; i++)
	    {
	      simplify_insts(func);
	      dead_code_elimination(func);
	      simplify_cfg(func);
	    }

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
	      inform(UNKNOWN_LOCATION, "%s", msg.c_str());
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
