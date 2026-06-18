#include "gcc-plugin.h"
#include "tree.h"
#include "gimple.h"
#include "ssa.h"
#include "attribs.h"

#include <cassert>

#include "gimple_conv.h"

using namespace smtgcc;

namespace {

const int stack_size = 1024 * 100;

void build_return(m68k_state *rstate, Function *src_func, function *fun)
{
  Basic_block *bb = rstate->exit_bb;
  Basic_block *src_last_bb = src_func->bbs.back();
  assert(src_last_bb->last_inst->op == Op::RET);
  if (src_last_bb->last_inst->nof_args == 0)
    {
      bb->build_ret_inst();
      return;
    }
  tree ret_expr = DECL_RESULT(fun->decl);
  tree ret_type = TREE_TYPE(ret_expr);
  uint64_t ret_bitsize = src_last_bb->last_inst->args[0]->bitsize;
  if (SCALAR_FLOAT_TYPE_P(ret_type) && ret_bitsize == 64)
    {
      Inst *ret =
	bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::fp0_64]);
      bb->build_ret_inst(bb->build_trunc(ret, ret_bitsize));
      return;
    }
  if (SCALAR_FLOAT_TYPE_P(ret_type) && ret_bitsize == 32)
    {
      Inst *ret =
	bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::fp0_32]);
      bb->build_ret_inst(bb->build_trunc(ret, ret_bitsize));
      return;
    }
  if ((INTEGRAL_TYPE_P(ret_type)
       || POINTER_TYPE_P(ret_type)
       || TREE_CODE(ret_type) == NULLPTR_TYPE)
      && ret_bitsize <= 64)
    {
      Inst *ret = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::d0]);
      if (ret_bitsize > 32)
	{
	  Inst *d1 =
	    bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::d1]);
	  ret = bb->build_inst(Op::CONCAT, ret, d1);
	}
      bb->build_ret_inst(bb->build_trunc(ret, ret_bitsize));
      return;
    }

  throw Not_implemented("m68k: Unhandled return type");
}

} // end anonymous namespace

m68k_state setup_m68k_function(CommonState *state, Function *src_func, function *fun)
{
  Module *module = src_func->module;

  m68k_state rstate;
  rstate.module = module;
  rstate.memory_objects = state->memory_objects;
  rstate.func_name = IDENTIFIER_POINTER(DECL_ASSEMBLER_NAME(fun->decl));
  rstate.file_name = DECL_SOURCE_FILE(fun->decl);
  rstate.symbolic_id = state->symbolic_id;

  Function *tgt = module->build_function("tgt");
  rstate.entry_bb = tgt->build_bb();
  rstate.exit_bb = tgt->build_bb();

  assert(module->functions.size() == 2);
  Basic_block *bb = rstate.entry_bb;

  // Registers a0-a7.
  for (int i = 0; i < 8; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 32));

  // Registers d0-d7.
  for (int i = 0; i < 8; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 32));

  // Registers fp0-fp7.
  for (int i = 0; i < 8; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 64));
  for (int i = 0; i < 8; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 32));

  // Condition flags: X, N, Z, V, C
  for (int i = 0; i < 5; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 1));

  // Pseudo registers tracking abort/exit
  rstate.registers.push_back(bb->build_inst(Op::REGISTER, 1));
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
  rstate.next_local_id = ((int64_t)-1) << (module->ptr_id_bits - 1);
  assert(stack_size < (((uint64_t)1) << module->ptr_offset_bits));
  Inst *id = bb->value_inst(rstate.next_local_id++, module->ptr_id_bits);
  Inst *mem_size = bb->value_inst(stack_size, module->ptr_offset_bits);
  Inst *flags = bb->value_inst(0, 32);
  Inst *stack_mem = bb->build_inst(Op::MEMORY, id, mem_size, flags);
  Inst *size = bb->value_inst(stack_size, stack_mem->bitsize);
  Inst *stack = bb->build_inst(Op::ADD, stack_mem, size);
  bb->build_inst(Op::WRITE, rstate.registers[M68kRegIdx::a7], stack);

  build_return(&rstate, src_func, fun);

  // Set up the PARAM instructions and copy the result to the correct
  // register or memory as required by the ABI.
  int param_number = 0;
  uint32_t param_offset = 4;
  for (tree decl = DECL_ARGUMENTS(fun->decl); decl; decl = DECL_CHAIN(decl))
    {
      tree type = TREE_TYPE(decl);
      uint32_t bitsize = bitsize_for_type(type);
      if (bitsize == 0)
	throw Not_implemented("setup_sh_function: Parameter size == 0");
      Inst *param_nbr = bb->value_inst(param_number, 32);
      Inst *param_bitsize = bb->value_inst(bitsize, 32);
      Inst *param = bb->build_inst(Op::PARAM, param_nbr, param_bitsize);
      if (INTEGRAL_TYPE_P(type) && (bitsize & 7) != 0)
	{
	  bitsize = (bitsize + 7) & ~7;;
	  Op op = TYPE_UNSIGNED(type) ? Op::ZEXT : Op::SEXT;
	  param = bb->build_inst(op, param, bitsize);
	}
      if ((bitsize & 7) == 0)
	{
	  Inst *off = bb->value_inst(param_offset, stack->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, stack, off);
	  if ((bitsize & 31) != 0)
	    {
	      Inst *uninit_padding_bits =
		bb->build_inst(Op::SYMBOLIC, rstate.symbolic_id++,
			       32 - (bitsize & 31));
	      if (bitsize < 32)
		param = bb->build_inst(Op::CONCAT, uninit_padding_bits, param);
	      else
		param = bb->build_inst(Op::CONCAT, param, uninit_padding_bits);
	      bitsize += 32 - (bitsize & 31);
	    }
	  bb->build_inst(Op::STORE_BE, ptr, param);
	  param_offset += bitsize / 8;
	}
      else
	throw Not_implemented("setup_m68k_function: param type not handled");

      param_number++;
    }
  if (param_number > 0)
    {
      Inst *new_mem_size =
	bb->value_inst(stack_size + param_offset, module->ptr_offset_bits);
      stack_mem->update_arg(1, new_mem_size);
    }

  // Functionality for abort/exit.
  {
    Inst *b0 = bb->value_inst(0, 1);
    Inst *zero = bb->value_inst(0, 32);
    bb->build_inst(Op::WRITE, rstate.registers[M68kRegIdx::abort], b0);
    bb->build_inst(Op::WRITE, rstate.registers[M68kRegIdx::abort_san], b0);
    bb->build_inst(Op::WRITE, rstate.registers[M68kRegIdx::exit], b0);
    bb->build_inst(Op::WRITE, rstate.registers[M68kRegIdx::exit_val], zero);

    Inst *abort_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[M68kRegIdx::abort]);
    Inst *abort_san_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[M68kRegIdx::abort_san]);
    rstate.exit_bb->build_inst(Op::ABORT, abort_cond, abort_san_cond);

    Inst *exit_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[M68kRegIdx::exit]);
    Inst *exit_val =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[M68kRegIdx::exit_val]);
    rstate.exit_bb->build_inst(Op::EXIT, exit_cond, exit_val);
  }

  return rstate;
}
