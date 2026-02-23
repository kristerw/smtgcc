#include "gcc-plugin.h"
#include "tree.h"
#include "gimple.h"
#include "ssa.h"
#include "attribs.h"

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

  if (VECTOR_INTEGER_TYPE_P(type) && bitsize <= 128)
    return true;

  if (INTEGRAL_TYPE_P(type)
      || POINTER_TYPE_P(type)
      || TREE_CODE(type) == NULLPTR_TYPE)
    {
      return (bitsize == 1
	      || bitsize == 8
	      || bitsize == 16
	      || bitsize == 32
	      || bitsize == 64);
    }

  if (TREE_CODE(type) == COMPLEX_TYPE && INTEGRAL_TYPE_P(TREE_TYPE(type)))
    return true;

  if (TREE_CODE(type) == RECORD_TYPE || TREE_CODE(type) == UNION_TYPE)
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

Inst *extract_vec_elem(Basic_block *bb, Inst *inst, uint32_t elem_bitsize, uint32_t idx)
{
  if (idx == 0 && inst->bitsize == elem_bitsize)
    return inst;
  assert(inst->bitsize % elem_bitsize == 0);
  Inst *high = bb->value_inst(idx * elem_bitsize + elem_bitsize - 1, 32);
  Inst *low = bb->value_inst(idx * elem_bitsize, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
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

  if (TREE_CODE(TREE_TYPE(ret_expr)) == COMPLEX_TYPE
      && is_returned_in_fregs(ret_expr))
    {
      assert(ret_bitsize == 64 || ret_bitsize == 128);

      uint64_t reg_idx = ShRegIdx::fr0;
      Inst *real = bb->build_inst(Op::READ, rstate->registers[reg_idx++]);
      if (ret_bitsize == 128)
	{
	  Inst *inst = bb->build_inst(Op::READ, rstate->registers[reg_idx++]);
	  if (TARGET_LITTLE_ENDIAN)
	    real = bb->build_inst(Op::CONCAT, real, inst);
	  else
	    real = bb->build_inst(Op::CONCAT, inst, real);
	}
      Inst *imag = bb->build_inst(Op::READ, rstate->registers[reg_idx++]);
      if (ret_bitsize == 128)
	{
	  Inst *inst = bb->build_inst(Op::READ, rstate->registers[reg_idx++]);
	  if (TARGET_LITTLE_ENDIAN)
	    imag = bb->build_inst(Op::CONCAT, imag, inst);
	  else
	    imag = bb->build_inst(Op::CONCAT, inst, imag);
	}
      Inst *retval = bb->build_inst(Op::CONCAT, imag, real);

      bb->build_ret_inst(retval);
      return;
    }
  if (is_returned_in_fregs(ret_expr))
    {
      Inst *retval = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::fr0]);
      for (int i = 1; retval->bitsize < ret_bitsize; i++)
	{
	  Inst *inst =
	    bb->build_inst(Op::READ, rstate->registers[ShRegIdx::fr0 + i]);
	  if (TARGET_LITTLE_ENDIAN)
	    retval = bb->build_inst(Op::CONCAT, retval, inst);
	  else
	    retval = bb->build_inst(Op::CONCAT, inst, retval);
	}
      bb->build_ret_inst(retval);
      return;
    }
  if (is_returned_in_regs(ret_expr))
    {
      Inst *retval = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::r0]);
      for (int i = 1; retval->bitsize < ret_bitsize; i++)
	{
	  Inst *inst =
	    bb->build_inst(Op::READ, rstate->registers[ShRegIdx::r0 + i]);
	  retval = bb->build_inst(Op::CONCAT, inst, retval);
	}
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }
  if (TREE_CODE(TREE_TYPE(ret_expr)) == RECORD_TYPE
      || TREE_CODE(TREE_TYPE(ret_expr)) == UNION_TYPE
      || VECTOR_TYPE_P(TREE_TYPE(ret_expr)))
    {
      // Return value is passed in memory, where the address is specified
      // by R2.
      assert((ret_bitsize & 7) == 0);
      Inst *id =
	bb->value_inst(rstate->next_local_id++, rstate->module->ptr_id_bits);
      Inst *mem_size =
	bb->value_inst(ret_bitsize / 8, rstate->module->ptr_offset_bits);
      Inst *flags = bb->value_inst(0, 32);

      Basic_block *entry_bb = rstate->entry_bb;
      Inst *ret_mem =
	entry_bb->build_inst(Op::MEMORY, id, mem_size, flags);
      Inst *reg = rstate->registers[ShRegIdx::r2];
      entry_bb->build_inst(Op::WRITE, reg, ret_mem);

      // Generate the return value from the value returned in memory.
      uint64_t size = ret_bitsize / 8;
      Inst *retval = 0;
      for (uint64_t i = 0; i < size; i++)
	{
	  Inst *offset = bb->value_inst(i, ret_mem->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, ret_mem, offset);
	  Inst *data_byte = bb->build_inst(Op::LOAD, ptr);
	  if (retval)
	    retval = bb->build_inst(Op::CONCAT, data_byte, retval);
	  else
	    retval = data_byte;
	}
      bb->build_ret_inst(retval);
      return;
    }

  throw Not_implemented("sh: Unhandled return type");
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
  rstate.symbolic_id = state->symbolic_id;

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

  // Set up the initial value of FPSCR.
  //   SZ = 0
  //   PR = 1 (unless TARGET_FPU_SINGLE)
  // TODO: Check that the other bits are initialized appropriately.
  Inst *fpscr;
  if (TARGET_FPU_SINGLE)
    fpscr = bb->value_inst(0x00000000, 32);
  else
    fpscr = bb->value_inst(0x00080000, 32);
  bb->build_inst(Op::WRITE, rstate.registers[ShRegIdx::fpscr], fpscr);

  // TODO: Implement interrupt_handler ABI.
  if (lookup_attribute("interrupt_handler", DECL_ATTRIBUTES(fun->decl)))
    throw Not_implemented("setup_sh_function: interrupt_handler");

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

      if (SCALAR_FLOAT_TYPE_P(type) && (bitsize == 32 || bitsize == 64))
	{
	  if (bitsize > 32 && (freg_nr & 1) != 0)
	    freg_nr++;

	  uint32_t nof_regs = bitsize / 32;
	  if (freg_nr + nof_regs > 8)
	    throw Not_implemented("setup_sh_function: too many params");

	  for (uint32_t i = 0; i < nof_regs; i++)
	    {
	      int idx = freg_nr++;
	      if (TARGET_LITTLE_ENDIAN)
		{
		  const int order[] = {1, 0, 3, 2, 5, 4, 7, 6};
		  idx = order[idx];
		}
	      Inst *reg = rstate.registers[ShRegIdx::fr4 + idx];
	      Inst *value = extract_vec_elem(bb, param, 32, i);
	      bb->build_inst(Op::WRITE, reg, value);
	    }
	}
      else if (TREE_CODE(type) == COMPLEX_TYPE
	       && SCALAR_FLOAT_TYPE_P(TREE_TYPE(type))
	       && (bitsize == 64 || bitsize == 128))
	{
	  if (bitsize > 32 && (freg_nr & 1) != 0)
	    freg_nr++;

	  uint32_t nof_regs = bitsize / 32;
	  if (freg_nr + nof_regs > 8)
	    throw Not_implemented("setup_sh_function: too many params");

	  Inst *real = extract_vec_elem(bb, param, bitsize / 2, 0);
	  for (uint32_t i = 0; i < nof_regs / 2; i++)
	    {
	      int idx = freg_nr++;
	      if (TARGET_LITTLE_ENDIAN && bitsize == 128)
		{
		  const int order[] = {1, 0, 3, 2, 5, 4, 7, 6};
		  idx = order[idx];
		}
	      Inst *reg = rstate.registers[ShRegIdx::fr4 + idx];
	      Inst *value = extract_vec_elem(bb, real, 32, i);
	      bb->build_inst(Op::WRITE, reg, value);
	    }
	  Inst *imag = extract_vec_elem(bb, param, bitsize / 2, 1);
	  for (uint32_t i = 0; i < nof_regs / 2; i++)
	    {
	      int idx = freg_nr++;
	      if (TARGET_LITTLE_ENDIAN && bitsize == 128)
		{
		  const int order[] = {1, 0, 3, 2, 5, 4, 7, 6};
		  idx = order[idx];
		}
	      Inst *reg = rstate.registers[ShRegIdx::fr4 + idx];
	      Inst *value = extract_vec_elem(bb, imag, 32, i);
	      bb->build_inst(Op::WRITE, reg, value);
	    }
	}
      else if ((INTEGRAL_TYPE_P(type)
		|| POINTER_TYPE_P(type)
		|| TREE_CODE(type) == NULLPTR_TYPE
		|| TREE_CODE(type) == RECORD_TYPE
		|| VECTOR_INTEGER_TYPE_P(type)
		|| (TREE_CODE(type) == COMPLEX_TYPE
		    && INTEGRAL_TYPE_P(TREE_TYPE(type))))
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
