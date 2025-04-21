#include "gcc-plugin.h"
#include "tree.h"

#include <cassert>

#include "gimple_conv.h"

using namespace smtgcc;

namespace {

const int stack_size = 1024 * 100;

void build_return(aarch64_state *rstate, Function *src_func, function *fun)
{
  Basic_block *bb = rstate->exit_bb;
  tree ret_type = TREE_TYPE(DECL_RESULT(fun->decl));

  Basic_block *src_last_bb = src_func->bbs.back();
  assert(src_last_bb->last_inst->op == Op::RET);
  uint64_t ret_bitsize = 0;
  if (src_last_bb->last_inst->nof_args > 0)
    ret_bitsize = src_last_bb->last_inst->args[0]->bitsize;
  if (ret_bitsize == 0)
    {
      bb->build_ret_inst();
      return;
    }

  if (SCALAR_FLOAT_TYPE_P(ret_type) && ret_bitsize <= rstate->freg_bitsize)
    {
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z0]);
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  if ((INTEGRAL_TYPE_P(ret_type) || POINTER_TYPE_P(ret_type))
      && ret_bitsize <= rstate->reg_bitsize)
    {
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::x0]);
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  throw Not_implemented("aarch64: Unhandled return type");
}

void write_reg(Basic_block *bb, Inst *reg, Inst *value)
{
  assert(reg->op == Op::REGISTER);
  if (reg->bitsize > value->bitsize)
    value = bb->build_inst(Op::ZEXT, value, reg->bitsize);
  bb->build_inst(Op::WRITE, reg, value);
}

} // end anonymous namespace

aarch64_state setup_aarch64_function(CommonState *state, Function *src_func, function *fun)
{
  Module *module = src_func->module;

  aarch64_state rstate;
  rstate.reg_bitsize = 64;
  rstate.freg_bitsize = 128;
  rstate.module = module;
  rstate.memory_objects = state->memory_objects;
  rstate.func_name = IDENTIFIER_POINTER(DECL_ASSEMBLER_NAME(fun->decl));
  rstate.file_name = DECL_SOURCE_FILE(fun->decl);

  Function *tgt = module->build_function("tgt");
  rstate.entry_bb = tgt->build_bb();
  rstate.exit_bb = tgt->build_bb();

  assert(module->functions.size() == 2);
  Basic_block *bb = rstate.entry_bb;

  // Registers x0-x31.
  for (int i = 0; i < 32; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 64));

  // Registers z0-z31.
  for (int i = 0; i < 32; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 128));

  // Registers p0-p16.
  for (int i = 0; i < 16; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 16));

  // SP
  rstate.registers.push_back(bb->build_inst(Op::REGISTER, 64));

  // Condition flags: N, Z, C, V
  for (int i = 0; i < 4; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 1));

  // Pseudo condition flags
  for (int i = 0; i < 3; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 1));

  // Pseudo registers tracking abort/exit
  rstate.registers.push_back(bb->build_inst(Op::REGISTER, 1));
  rstate.registers.push_back(bb->build_inst(Op::REGISTER, 1));
  rstate.registers.push_back(bb->build_inst(Op::REGISTER, 32));

  // Create MEMORY instructions for the global variables we saw in the
  // GIMPLE IR.
  for (const auto& mem_obj : rstate.memory_objects)
    {
      Inst *id = bb->value_inst(mem_obj.id, module->ptr_id_bits);
      Inst *size = bb->value_inst(mem_obj.size, module->ptr_offset_bits);
      Inst *flags = bb->value_inst(mem_obj.flags, 32);
      Inst *mem = bb->build_inst(Op::MEMORY, id, size, flags);
      rstate.sym_name2mem.insert({mem_obj.sym_name, mem});
    }

  // Set up the stack.
  assert(stack_size < (((uint64_t)1) << module->ptr_offset_bits));
  uint64_t id_val = ((uint64_t)1) << (module->ptr_id_bits - 1);
  Inst *id = bb->value_inst(id_val, module->ptr_id_bits);
  Inst *mem_size = bb->value_inst(stack_size, module->ptr_offset_bits);
  Inst *flags = bb->value_inst(0, 32);
  Inst *stack = bb->build_inst(Op::MEMORY, id, mem_size, flags);
  Inst *size = bb->value_inst(stack_size, stack->bitsize);
  stack = bb->build_inst(Op::ADD, stack, size);
  bb->build_inst(Op::WRITE, rstate.registers[Aarch64RegIdx::sp], stack);

  uint32_t reg_nbr = 0;
  uint32_t freg_nbr = 0;

  build_return(&rstate, src_func, fun);

  // Set up the PARAM instructions and copy the result to the correct
  // register or memory as required by the ABI.
  int param_number = 0;
  std::vector<Inst*> stack_values;
  for (tree decl = DECL_ARGUMENTS(fun->decl); decl; decl = DECL_CHAIN(decl))
    {
      uint32_t bitsize = bitsize_for_type(TREE_TYPE(decl));
      if (bitsize <= 0)
	throw Not_implemented("Parameter size == 0");

      Inst *param_nbr = bb->value_inst(param_number, 32);
      Inst *param_bitsize = bb->value_inst(bitsize, 32);
      Inst *param = bb->build_inst(Op::PARAM, param_nbr, param_bitsize);

      tree type = TREE_TYPE(decl);
      if (param_number == 0
	  && !strcmp(IDENTIFIER_POINTER(DECL_NAME(fun->decl)), "__ct_base "))
	{
	  // TODO: The "this" pointer in C++ constructors needs to be handled
	  // as a special case in the same way as in gimple_conv.cpp when
	  // setting up the parameters.
	  throw Not_implemented("setup_aarch64_function: C++ constructors");
	}

      if (SCALAR_FLOAT_TYPE_P(type) && param->bitsize <= rstate.freg_bitsize)
	{
	  if (freg_nbr >= 8)
	    throw Not_implemented("setup_aarch64_function: too many params");
	  write_reg(bb, rstate.registers[Aarch64RegIdx::z0 + freg_nbr], param);
	  freg_nbr++;
	}
      else if ((INTEGRAL_TYPE_P(type) || POINTER_TYPE_P(type))
	       && param->bitsize <= rstate.reg_bitsize)
	{
	  if (reg_nbr >= 8)
	    throw Not_implemented("setup_aarch64_function: too many params");
	  write_reg(bb, rstate.registers[Aarch64RegIdx::x0 + reg_nbr], param);
	  reg_nbr++;
	}
      else
	throw Not_implemented("setup_aarch64_function: param type not handled");

      param_number++;
    }

  if (!stack_values.empty())
    {
      uint32_t reg_size = rstate.reg_bitsize / 8;
      uint32_t size = stack_values.size() * reg_size;
      size = (size + 15) & ~15;   // Keep the stack 16-bytes aligned.
      Inst *size_inst = bb->value_inst(size, rstate.reg_bitsize);
      Inst *sp_reg = rstate.registers[2];
      Inst *sp = bb->build_inst(Op::READ, sp_reg);
      sp = bb->build_inst(Op::SUB, sp, size_inst);
      bb->build_inst(Op::WRITE, sp_reg, sp);

      for (auto value : stack_values)
	{
	  if (!value)
	    {
	      Inst *size_inst = bb->value_inst(reg_size, sp->bitsize);
	      sp = bb->build_inst(Op::ADD, sp, size_inst);
	      continue;
	    }

	  for (uint32_t i = 0; i < reg_size; i++)
	    {
	      Inst *high = bb->value_inst(i * 8 + 7, 32);
	      Inst *low = bb->value_inst(i * 8, 32);
	      Inst *byte = bb->build_inst(Op::EXTRACT, value, high, low);
	      bb->build_inst(Op::STORE, sp, byte);
	      Inst *one = bb->value_inst(1, sp->bitsize);
	      sp = bb->build_inst(Op::ADD, sp, one);
	    }
	}
    }

  // Functionality for abort/exit.
  {
    Inst *b0 = bb->value_inst(0, 1);
    Inst *zero = bb->value_inst(0, 32);
    bb->build_inst(Op::WRITE, rstate.registers[Aarch64RegIdx::abort], b0);
    bb->build_inst(Op::WRITE, rstate.registers[Aarch64RegIdx::exit], b0);
    bb->build_inst(Op::WRITE, rstate.registers[Aarch64RegIdx::exit_val], zero);

    Inst *abort_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[Aarch64RegIdx::abort]);
    Inst *exit_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[Aarch64RegIdx::exit]);
    Inst *exit_val =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[Aarch64RegIdx::exit_val]);
    rstate.exit_bb->build_inst(Op::EXIT, abort_cond, exit_cond, exit_val);
  }

  return rstate;
}
