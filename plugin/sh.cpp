#include "gcc-plugin.h"
#include "tree.h"

#include <cassert>

#include "gimple_conv.h"

unsigned int get_object_alignment(tree exp);

using namespace smtgcc;

namespace {

const int stack_size = 1024 * 100;

bool is_returned_in_fregs(tree expr)
{
  tree type = TREE_TYPE(expr);
  uint64_t bitsize = bitsize_for_type(type);
  if (SCALAR_FLOAT_TYPE_P(type) && (bitsize == 32 || bitsize == 64))
    return true;

  if (TREE_CODE(type) == COMPLEX_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      uint64_t elem_bitsize = bitsize_for_type(elem_type);
      if (SCALAR_FLOAT_TYPE_P(elem_type)
	  && (elem_bitsize == 32 || elem_bitsize == 64))
	return true;
    }

  return false;
}

bool is_returned_in_regs(tree expr)
{
  tree type = TREE_TYPE(expr);
  uint64_t bitsize = bitsize_for_type(type);
  if (!(bitsize == 8
	|| bitsize == 16
	|| bitsize == 32
	|| bitsize == 64))
    return false;

  if (INTEGRAL_TYPE_P(type) || POINTER_TYPE_P(type))
    return true;

  if (TREE_CODE(type) == RECORD_TYPE)
    {
      uint64_t alignment = get_object_alignment(expr);
      if ((bitsize == 8 && alignment == 8)
	  || (bitsize == 16 && alignment == 16)
	  || (bitsize == 32 && alignment == 32)
	  || (bitsize == 64 && alignment == 32))
	return true;
    }

  return false;
}

void build_return(sh_state *rstate, Function *src_func, function *fun)
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
  uint64_t ret_bitsize = src_last_bb->last_inst->args[0]->bitsize;

  if (is_returned_in_fregs(ret_expr))
    {
      Inst *retval = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::fr0]);
      for (int i = 1; retval->bitsize < ret_bitsize; i++)
	{
	  Inst *inst =
	    bb->build_inst(Op::READ, rstate->registers[ShRegIdx::fr0 + i]);
	  retval = bb->build_inst(Op::CONCAT, inst, retval);
	}
      bb->build_ret_inst(retval);
      return;
    }
  if (is_returned_in_regs(ret_expr))
    {
      Inst *retval = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::r0]);
      if (retval->bitsize < ret_bitsize)
	{
	  Inst *inst =
	    bb->build_inst(Op::READ, rstate->registers[ShRegIdx::r1]);
	  retval = bb->build_inst(Op::CONCAT, inst, retval);
	}
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  throw Not_implemented("sh: Unhandled return type");
}

Inst *extract_vec_elem(Basic_block *bb, Inst *inst, uint32_t elem_bitsize, uint32_t idx)
{
  if (idx == 0 && inst->bitsize == elem_bitsize)
    return inst;
  assert(inst->bitsize % elem_bitsize == 0);
  Inst *high = bb->value_inst(idx * elem_bitsize + elem_bitsize - 1, 32);
  Inst *low = bb->value_inst(idx * elem_bitsize, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
}

} // end anonymous namespace

sh_state setup_sh_function(CommonState *state, Function *src_func, function *fun)
{
  Module *module = src_func->module;

  sh_state rstate;
  rstate.module = module;
  rstate.memory_objects = state->memory_objects;
  rstate.func_name = IDENTIFIER_POINTER(DECL_ASSEMBLER_NAME(fun->decl));
  rstate.file_name = DECL_SOURCE_FILE(fun->decl);

  Function *tgt = module->build_function("tgt");
  rstate.entry_bb = tgt->build_bb();
  rstate.exit_bb = tgt->build_bb();

  assert(module->functions.size() == 2);
  Basic_block *bb = rstate.entry_bb;

  // Registers r0-r15.
  for (int i = 0; i < 16; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 32));

  // Registers fr0-fr15.
  for (int i = 0; i < 16; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 32));

  // System registers
  for (int i = 0; i < 3; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 32));

  // FPU system registers
  for (int i = 0; i < 2; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 32));

  // Control registers
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
  rstate.next_local_id = ((int64_t)-1) << (module->ptr_id_bits - 1);
  assert(stack_size < (((uint64_t)1) << module->ptr_offset_bits));
  Inst *id = bb->value_inst(rstate.next_local_id++, module->ptr_id_bits);
  Inst *mem_size = bb->value_inst(stack_size, module->ptr_offset_bits);
  Inst *flags = bb->value_inst(0, 32);
  Inst *stack = bb->build_inst(Op::MEMORY, id, mem_size, flags);
  Inst *size = bb->value_inst(stack_size, stack->bitsize);
  stack = bb->build_inst(Op::ADD, stack, size);
  bb->build_inst(Op::WRITE, rstate.registers[ShRegIdx::r15], stack);

  build_return(&rstate, src_func, fun);

  // Set up the PARAM instructions and copy the result to the correct
  // register or memory as required by the ABI.
  int param_number = 0;
  int reg_nr = 0;
  int freg_nr = 0;
  for (tree decl = DECL_ARGUMENTS(fun->decl); decl; decl = DECL_CHAIN(decl))
    {
      uint32_t bitsize = bitsize_for_type(TREE_TYPE(decl));
      tree type = TREE_TYPE(decl);
      if (bitsize == 0)
	throw Not_implemented("setup_sh_function: Parameter size == 0");
      Inst *param_nbr = bb->value_inst(param_number, 32);
      Inst *param_bitsize = bb->value_inst(bitsize, 32);
      Inst *param = bb->build_inst(Op::PARAM, param_nbr, param_bitsize);

      if ((SCALAR_FLOAT_TYPE_P(type) && (bitsize == 32 || bitsize == 64))
	  || (TREE_CODE(type) == COMPLEX_TYPE
	      && SCALAR_FLOAT_TYPE_P(TREE_TYPE(type))
	      && (bitsize == 64 || bitsize == 128)))
	{
	  if (bitsize > 32 && (freg_nr & 1) != 0)
	    freg_nr++;

	  uint32_t nof_regs = bitsize / 32;
	  if (freg_nr + nof_regs > 8)
	    throw Not_implemented("setup_sh_function: too many params");

	  for (uint32_t i = 0; i < nof_regs; i++)
	    {
	      Inst *reg = rstate.registers[ShRegIdx::fr4 + freg_nr++];
	      Inst *value = extract_vec_elem(bb, param, 32, i);
	      bb->build_inst(Op::WRITE, reg, value);
	    }
	}
      else if ((INTEGRAL_TYPE_P(type)
		|| POINTER_TYPE_P(type)
		|| TREE_CODE(type) == RECORD_TYPE)
	       && bitsize <= 128)
	{
	  uint32_t expanded_bitsize = (bitsize + 31) & ~31;
	  uint32_t nof_regs = expanded_bitsize / 32;
	  if (reg_nr + nof_regs > 4)
	    throw Not_implemented("setup_sh_function: too many params");

	  if (bitsize < expanded_bitsize)
	    {
	      Op op = TYPE_UNSIGNED(TREE_TYPE(decl)) ? Op::ZEXT : Op::SEXT;
	      param = bb->build_inst(op, param, expanded_bitsize);
	    }
	  for (uint32_t i = 0; i < nof_regs; i++)
	    {
	      Inst *reg = rstate.registers[ShRegIdx::r4 + reg_nr++];
	      Inst *value = extract_vec_elem(bb, param, 32, i);
	      bb->build_inst(Op::WRITE, reg, value);
	    }
	}
      else
	throw Not_implemented("setup_sh_function: param type not handled");

      param_number++;
    }

  // Functionality for abort/exit.
  {
    Inst *b0 = bb->value_inst(0, 1);
    Inst *zero = bb->value_inst(0, 32);
    bb->build_inst(Op::WRITE, rstate.registers[ShRegIdx::abort], b0);
    bb->build_inst(Op::WRITE, rstate.registers[ShRegIdx::exit], b0);
    bb->build_inst(Op::WRITE, rstate.registers[ShRegIdx::exit_val], zero);

    Inst *abort_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[ShRegIdx::abort]);
    Inst *exit_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[ShRegIdx::exit]);
    Inst *exit_val =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[ShRegIdx::exit_val]);
    rstate.exit_bb->build_inst(Op::EXIT, abort_cond, exit_cond, exit_val);
  }

  return rstate;
}
