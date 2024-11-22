#include "gcc-plugin.h"
#include "tree.h"

#include <cassert>

#include "gimple_conv.h"

using namespace smtgcc;

namespace {

const int stack_size = 1024 * 100;

struct Regs
{
  Inst *regs[2] = {nullptr, nullptr};
  Inst *fregs[2] = {nullptr, nullptr};
};

Inst *pad_to_freg_size(riscv_state *rstate, Inst *inst)
{
  assert(inst->bitsize <= rstate->freg_bitsize);
  if (inst->bitsize < rstate->freg_bitsize)
    {
      uint32_t padding_bitsize = rstate->freg_bitsize - inst->bitsize;
      Inst *m1 = inst->bb->value_m1_inst(padding_bitsize);
      inst = inst->bb->build_inst(Op::CONCAT, m1, inst);
    }
  return inst;
}

Inst *pad_to_reg_size(riscv_state *rstate, Inst *inst, tree type)
{
  assert(inst->bitsize <= rstate->reg_bitsize);
  if (inst->bitsize < rstate->reg_bitsize)
    {
      if (INTEGRAL_TYPE_P(type) && TYPE_UNSIGNED(type))
	inst = inst->bb->build_inst(Op::ZEXT, inst, rstate->reg_bitsize);
      else
	inst = inst->bb->build_inst(Op::SEXT, inst, rstate->reg_bitsize);
    }
  return inst;
}

struct struct_elem {
  tree fld;
  uint64_t bit_offset;
};

// Flatten structure fields as described in the hardware floating-point
// calling convention.
// Returns false if the structure cannot be handled by the calling convention
// (and therefore must fall back to the integer calling convention).
bool flatten_struct(riscv_state *rstate, tree struct_type, std::vector<struct_elem>& elems, int& nof_r, int& nof_f, uint64_t bitoffset = 0)
{
  for (tree fld = TYPE_FIELDS(struct_type); fld; fld = DECL_CHAIN(fld))
    {
      if (TREE_CODE(fld) != FIELD_DECL)
	continue;
      tree type = TREE_TYPE(fld);
      uint64_t bitsize = bitsize_for_type(type);
      if (bitsize == 0)
	continue;
      uint64_t off = get_int_cst_val(DECL_FIELD_OFFSET(fld));
      uint64_t bitoff = get_int_cst_val(DECL_FIELD_BIT_OFFSET(fld));
      uint64_t fld_bitoffset = 8 * off + bitoff + bitoffset;
      if (SCALAR_FLOAT_TYPE_P(type) && bitsize <= rstate->freg_bitsize)
	{
	  elems.push_back({fld, fld_bitoffset});
	  nof_f++;
	}
      else if (COMPLEX_FLOAT_TYPE_P(type) &&
	       bitsize <= 2 * rstate->freg_bitsize)
	{
	  elems.push_back({fld, fld_bitoffset});
	  nof_f += 2;
	}
      else if (INTEGRAL_TYPE_P(type) && bitsize <= rstate->reg_bitsize)
	{
	  elems.push_back({fld, fld_bitoffset});
	  nof_r++;
	}
      else if (TREE_CODE(type) == RECORD_TYPE)
	{
	  if (!flatten_struct(rstate, type, elems, nof_r, nof_f, fld_bitoffset))
	    return false;
	}
      else
	return false;
    }

  if (nof_f == 0)
    return false;
  if (nof_r + nof_f > 2)
    return false;

  return true;
}

// Determines if the structure can be handled by the hardware floating-point
// calling convention, and in that case, splits the structure into
// instructions for each register the structure will be passed in.
std::optional<Regs> regs_for_fp_struct(riscv_state *rstate, Inst *value, tree struct_type)
{
  std::vector<struct_elem> elems;
  int nof_r = 0;
  int nof_f = 0;
  if (!flatten_struct(rstate, struct_type, elems, nof_r, nof_f))
    return {};

  Basic_block *bb = value->bb;
  Regs regs;
  int reg_nbr = 0;
  int freg_nbr = 0;
  for (auto [fld, fld_bitoffset] : elems)
    {
      tree type = TREE_TYPE(fld);
      uint64_t low_val = fld_bitoffset;
      uint64_t high_val = low_val + bitsize_for_type(type) - 1;
      Inst *inst = bb->build_inst(Op::EXTRACT, value, high_val, low_val);
      if (SCALAR_FLOAT_TYPE_P(type))
	{
	  assert(freg_nbr < 2);
	  regs.fregs[freg_nbr++] = pad_to_freg_size(rstate, inst);
	}
      else if (COMPLEX_FLOAT_TYPE_P(type))
	{
	  assert(freg_nbr == 0);
	  uint64_t elt_bitsize = value->bitsize / 2;
	  assert(elt_bitsize == 16 || elt_bitsize == 32 || elt_bitsize == 64);
	  Inst *reg_value = bb->build_trunc(value, elt_bitsize);
	  regs.fregs[freg_nbr++] = pad_to_freg_size(rstate, reg_value);
	  Inst *high = bb->value_inst(value->bitsize - 1, 32);
	  Inst *low = bb->value_inst(elt_bitsize, 32);
	  reg_value = bb->build_inst(Op::EXTRACT, value, high, low);
	  regs.fregs[freg_nbr++] = pad_to_freg_size(rstate, reg_value);
	}
      else if (INTEGRAL_TYPE_P(type))
	{
	  assert(reg_nbr < 2);
	  regs.regs[reg_nbr++] = pad_to_reg_size(rstate, inst, type);
	}
      else
	assert(0);
    }

  return regs;
}

// Determines if the value can be passed in registers, and in that case,
// splits the structure into instructions for each register the structure
// will be passed in.
std::optional<Regs> regs_for_value(riscv_state *rstate, Inst *value, tree type)
{
  Basic_block *bb = value->bb;

  // Handle the hardware floating-point calling convention.
  if (COMPLEX_FLOAT_TYPE_P(type) && value->bitsize <= 2 * rstate->freg_bitsize)
    {
      Regs regs;
      uint64_t elt_bitsize = value->bitsize / 2;
      assert(elt_bitsize == 16 || elt_bitsize == 32 || elt_bitsize == 64);
      Inst *reg_value = bb->build_trunc(value, elt_bitsize);
      regs.fregs[0] = pad_to_freg_size(rstate, reg_value);
      Inst *high = bb->value_inst(value->bitsize - 1, 32);
      Inst *low = bb->value_inst(elt_bitsize, 32);
      reg_value = bb->build_inst(Op::EXTRACT, value, high, low);
      regs.fregs[1] = pad_to_freg_size(rstate, reg_value);
      return regs;
    }
  if (SCALAR_FLOAT_TYPE_P(type) && value->bitsize <= rstate->freg_bitsize)
    {
      Regs regs;
      regs.fregs[0] = pad_to_freg_size(rstate, value);
      return regs;
    }
  if (TREE_CODE(type) == RECORD_TYPE)
    {
      std::optional<Regs> res = regs_for_fp_struct(rstate, value, type);
      if (res)
	return res;
    }

  // Handle the integer calling convention.
  if (value->bitsize <= 2 * rstate->reg_bitsize)
    {
      // Pad it out to a multiple of the register size.
      uint32_t num_regs = value->bitsize <= rstate->reg_bitsize ? 1 : 2;
      if (value->bitsize < rstate->reg_bitsize * num_regs)
	{
	  bool is_unsigned = INTEGRAL_TYPE_P(type) && TYPE_UNSIGNED(type);
	  Inst *bs_inst =
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
	  Inst *high = bb->value_inst(value->bitsize - 1, 32);
	  Inst *low = bb->value_inst(rstate->reg_bitsize, 32);
	  regs.regs[1] = bb->build_inst(Op::EXTRACT, value, high, low);
	}
      return regs;
    }

  return {};
}

void build_return(riscv_state *rstate, Function *src_func, function *fun, uint32_t *reg_nbr)
{
  Function *tgt = rstate->module->functions[1];
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

  // Handle the hardware floating-point calling convention.
  std::vector<struct_elem> elems;
  int nof_r = 0;
  int nof_f = 0;
  if (TREE_CODE(ret_type) == RECORD_TYPE
      && flatten_struct(rstate, ret_type, elems, nof_r, nof_f))
    {
      uint32_t reg_nbr = RiscvRegIdx::x10;
      uint32_t freg_nbr = RiscvRegIdx::f10;
      Inst *retval = nullptr;
      for (auto [fld, fld_bitoffset] : elems)
	{
	  tree type = TREE_TYPE(fld);
	  uint64_t type_bitsize = bitsize_for_type(type);
	  Inst *value = nullptr;
	  if (SCALAR_FLOAT_TYPE_P(type))
	    value = bb->build_inst(Op::READ, rstate->registers[freg_nbr++]);
	  else if (INTEGRAL_TYPE_P(type))
	    value = bb->build_inst(Op::READ, rstate->registers[reg_nbr++]);
	  else if (COMPLEX_FLOAT_TYPE_P(type))
	    {
	      assert(elems.size() == 1);
	      uint64_t elt_bitsize = type_bitsize / 2;
	      assert(elt_bitsize == 16
		     || elt_bitsize == 32
		     || elt_bitsize == 64);
	      Inst *real =
		bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f10]);
	      real = bb->build_trunc(real, elt_bitsize);
	      Inst *imag =
		bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f11]);
	      imag = bb->build_trunc(imag, elt_bitsize);
	      value =
		bb->build_ret_inst(bb->build_inst(Op::CONCAT, imag, real));
	      return;
	    }
	  if (fld_bitoffset > 0 && (!retval || retval->bitsize < fld_bitoffset))
	    {
	      uint64_t nof_pad =
		retval ? fld_bitoffset - retval->bitsize : fld_bitoffset;
	      Inst *pad = bb->value_inst(0, nof_pad);
	      if (retval)
		retval = bb->build_inst(Op::CONCAT, pad, retval);
	      else
		retval = pad;
	    }
	  value = bb->build_trunc(value, type_bitsize);
	  if (retval)
	    retval = bb->build_inst(Op::CONCAT, value, retval);
	  else
	    retval = value;
	}
      if (retval->bitsize != ret_bitsize)
	{
	  Inst *pad = bb->value_inst(0, ret_bitsize - retval->bitsize);
	  retval = bb->build_inst(Op::CONCAT, pad, retval);
	}
      bb->build_ret_inst(retval);
      return;
    }
  if (COMPLEX_FLOAT_TYPE_P(ret_type)
      && ret_bitsize <= 2 * rstate->freg_bitsize)
    {
      uint64_t elt_bitsize = ret_bitsize / 2;
      assert(elt_bitsize == 16 || elt_bitsize == 32 || elt_bitsize == 64);
      Inst *real =
	bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f10]);
      real = bb->build_trunc(real, elt_bitsize);
      Inst *imag =
	bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f11]);
      imag = bb->build_trunc(imag, elt_bitsize);
      bb->build_ret_inst(bb->build_inst(Op::CONCAT, imag, real));
      return;
    }
  if (SCALAR_FLOAT_TYPE_P(ret_type) && ret_bitsize <= rstate->freg_bitsize)
    {
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f10]);
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  // Handle the integer calling convention.
  if (ret_bitsize <= 2 * rstate->reg_bitsize)
    {
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::x10]);
      if (retval->bitsize < ret_bitsize)
	{
	  Inst *inst =
	    bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::x11]);
	  retval = bb->build_inst(Op::CONCAT, inst, retval);
	}
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
    }
  else
    {
      // Return of values wider than 2*reg_bitsize are passed in memory,
      // where the address is specified by an implicit first parameter.
      assert((ret_bitsize & 7) == 0);
      Inst *id = tgt->value_inst(-127, tgt->module->ptr_id_bits);
      Inst *mem_size =
	tgt->value_inst(ret_bitsize / 8, tgt->module->ptr_offset_bits);
      Inst *flags = tgt->value_inst(0, 32);

      Basic_block *entry_bb = rstate->entry_bb;
      Inst *ret_mem =
	entry_bb->build_inst(Op::MEMORY, id, mem_size, flags);
      Inst *reg = rstate->registers[(*reg_nbr)++];
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
    }
}

} // end anonymous namespace

riscv_state setup_riscv_function(CommonState *state, Function *src_func, function *fun)
{
  Module *module = src_func->module;

  riscv_state rstate;
  rstate.reg_bitsize = TARGET_64BIT ? 64 : 32;
  rstate.freg_bitsize = 64;
  rstate.vreg_bitsize = 128;
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
    {
      Inst *reg = bb->build_inst(Op::REGISTER, rstate.reg_bitsize);
      rstate.registers.push_back(reg);
    }

  // Registers f0-f31.
  for (int i = 0; i < 32; i++)
    {
      Inst *reg = bb->build_inst(Op::REGISTER, rstate.freg_bitsize);
      rstate.registers.push_back(reg);
    }

  // Registers v0-v31.
  for (int i = 0; i < 32; i++)
    {
      Inst *reg = bb->build_inst(Op::REGISTER, rstate.vreg_bitsize);
      rstate.registers.push_back(reg);
    }

  // vtype
  rstate.registers.push_back(bb->build_inst(Op::REGISTER, 3));

  // vl
  rstate.registers.push_back(bb->build_inst(Op::REGISTER, rstate.reg_bitsize));

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
  Inst *id = bb->value_inst(-128, module->ptr_id_bits);
  Inst *mem_size = bb->value_inst(stack_size, module->ptr_offset_bits);
  Inst *flags = bb->value_inst(0, 32);
  Inst *stack = bb->build_inst(Op::MEMORY, id, mem_size, flags);
  Inst *size = bb->value_inst(stack_size, stack->bitsize);
  stack = bb->build_inst(Op::ADD, stack, size);
  bb->build_inst(Op::WRITE, rstate.registers[RiscvRegIdx::x2], stack);

  uint32_t reg_nbr = RiscvRegIdx::x10;
  uint32_t freg_nbr = RiscvRegIdx::f10;

  build_return(&rstate, src_func, fun, &reg_nbr);

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
	  throw Not_implemented("setup_riscv_function: C++ constructors");
	}

      std::optional<Regs> arg_regs = regs_for_value(&rstate, param, type);
      if (arg_regs)
	{
	  // Ensure the stack is aligned when writing values wider than
	  // one register.
	  if (reg_nbr > RiscvRegIdx::x17
	       && (*arg_regs).regs[0]
	       && (*arg_regs).regs[1]
	       && (stack_values.size() & 1))
	    stack_values.push_back(nullptr);

	  if ((*arg_regs).regs[0])
	    {
	      if (reg_nbr > RiscvRegIdx::x17)
		stack_values.push_back((*arg_regs).regs[0]);
	      else
		{
		  Inst *reg = rstate.registers[reg_nbr++];
		  bb->build_inst(Op::WRITE, reg, (*arg_regs).regs[0]);
		}
	    }
	  if ((*arg_regs).regs[1])
	    {
	      if (reg_nbr > RiscvRegIdx::x17)
		stack_values.push_back((*arg_regs).regs[1]);
	      else
		{
		  Inst *reg = rstate.registers[reg_nbr++];
		  bb->build_inst(Op::WRITE, reg, (*arg_regs).regs[1]);
		}
	    }

	  // Ensure the stack is aligned when writing floating-point values
	  // wider than one register.
	  if (reg_nbr > RiscvRegIdx::x17
	       && (*arg_regs).fregs[0]
	       && (*arg_regs).fregs[1]
	       && (stack_values.size() & 1))
	    stack_values.push_back(nullptr);

	  if ((*arg_regs).fregs[0])
	    {
	      if (freg_nbr > RiscvRegIdx::f17)
		stack_values.push_back((*arg_regs).fregs[0]);
	      else
		{
		  Inst *reg = rstate.registers[freg_nbr++];
		  bb->build_inst(Op::WRITE, reg, (*arg_regs).fregs[0]);
		}
	    }
	  if ((*arg_regs).fregs[1])
	    {
	      if (freg_nbr > RiscvRegIdx::f17)
		stack_values.push_back((*arg_regs).fregs[1]);
	      else
		{
		  Inst *reg = rstate.registers[freg_nbr++];
		  bb->build_inst(Op::WRITE, reg, (*arg_regs).fregs[1]);
		}
	    }
	}
      else
	{
	  // TODO: Implement passing of large params in memory.
	  throw Not_implemented("setup_riscv_function: too wide param type");
	}

      param_number++;
    }

  if (!stack_values.empty())
    {
      uint32_t reg_size = rstate.reg_bitsize / 8;
      uint32_t size = stack_values.size() * reg_size;
      size = (size + 15) & ~15;   // Keep the stack 16-bytes aligned.
      Inst *size_inst = bb->value_inst(size, rstate.reg_bitsize);
      Inst *sp_reg = rstate.registers[RiscvRegIdx::x2];
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

  return rstate;
}
