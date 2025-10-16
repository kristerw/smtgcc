#include "gcc-plugin.h"
#include "tree.h"
#include "stringpool.h"
#include "attribs.h"

#include <cassert>

#include "gimple_conv.h"

using namespace smtgcc;

namespace {

const int stack_size = 1024 * 100;

bool is_sve_type(tree type)
{
  return POLY_INT_CST_P(TYPE_SIZE(type));
}

bool is_short_vector(tree type)
{
  if (is_sve_type(type))
    return false;
  uint64_t bitsize = bitsize_for_type(type);
  return VECTOR_TYPE_P(type) && (bitsize == 128 || bitsize == 64);
}

std::optional<std::pair<unsigned, unsigned>> hfa_size(aarch64_state *rstate, tree type)
{
  if (TREE_CODE(type) == COMPLEX_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      if (!SCALAR_FLOAT_TYPE_P(elem_type))
	return {};
      unsigned elem_bitsize = bitsize_for_type(elem_type);
      if (elem_bitsize > rstate->freg_bitsize)
	return {};
      return {{2u, elem_bitsize}};
    }

  if (TREE_CODE(type) == ARRAY_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      if (!SCALAR_FLOAT_TYPE_P(elem_type))
	return {};
      unsigned elem_bitsize = bitsize_for_type(elem_type);
      if (elem_bitsize > rstate->freg_bitsize)
	return {};
      unsigned nof_elem = bitsize_for_type(type) / elem_bitsize;
      if (nof_elem > 4)
	return {};
      return {{nof_elem, elem_bitsize}};
    }

  if (TREE_CODE(type) != RECORD_TYPE)
    return {};

  unsigned nof_elem = 0;
  unsigned bitsize = 0;

  for (tree fld = TYPE_FIELDS(type); fld; fld = DECL_CHAIN(fld))
    {
      if (TREE_CODE(fld) != FIELD_DECL)
	continue;
      tree elem_type = TREE_TYPE(fld);
      if (!SCALAR_FLOAT_TYPE_P(elem_type))
	{
	  auto elem_res = hfa_size(rstate, elem_type);
	  if (!elem_res)
	    return {};
	  auto [elem_nof_elem, elem_bitsize] = *elem_res;
	  if (nof_elem == 0)
	    {
	      nof_elem = elem_nof_elem;
	      bitsize = elem_bitsize;
	    }
	  else
	    {
	      if (bitsize != elem_bitsize)
		return {};
	      nof_elem += elem_nof_elem;
	      if (nof_elem > 4)
		return {};
	    }
	}
      else
	{
	  if (++nof_elem > 4)
	    return {};
	  unsigned elem_bitsize = bitsize_for_type(elem_type);
	  if (nof_elem == 1)
	    {
	      bitsize = elem_bitsize;
	      if (bitsize > rstate->freg_bitsize)
		return {};
	    }
	  if (bitsize != elem_bitsize)
	    return {};
	}
    }

  return {{nof_elem, bitsize}};
}

std::optional<std::pair<unsigned, unsigned>> hva_size(aarch64_state *rstate, tree type)
{
  if (TREE_CODE(type) == ARRAY_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      if (!is_short_vector(elem_type))
	return {};
      unsigned elem_bitsize = bitsize_for_type(elem_type);
      unsigned nof_elem = bitsize_for_type(type) / elem_bitsize;
      if (nof_elem > 4)
	return {};
      return {{nof_elem, elem_bitsize}};
    }

  if (TREE_CODE(type) != RECORD_TYPE)
    return {};

  unsigned nof_elem = 0;
  unsigned bitsize = 0;

  for (tree fld = TYPE_FIELDS(type); fld; fld = DECL_CHAIN(fld))
    {
      if (TREE_CODE(fld) != FIELD_DECL)
	continue;
      tree elem_type = TREE_TYPE(fld);
      if (!is_short_vector(elem_type))
	{
	  auto elem_res = hva_size(rstate, elem_type);
	  if (!elem_res)
	    return {};
	  auto [elem_nof_elem, elem_bitsize] = *elem_res;
	  if (nof_elem == 0)
	    {
	      nof_elem = elem_nof_elem;
	      bitsize = elem_bitsize;
	    }
	  else
	    {
	      if (bitsize != elem_bitsize)
		return {};
	      nof_elem += elem_nof_elem;
	      if (nof_elem > 4)
		return {};
	    }
	}
      else
	{
	  if (++nof_elem > 4)
	    return {};
	  unsigned elem_bitsize = bitsize_for_type(elem_type);
	  if (nof_elem == 1)
	    {
	      bitsize = elem_bitsize;
	      if (bitsize > rstate->freg_bitsize)
		return {};
	    }
	  if (bitsize != elem_bitsize)
	    return {};
	}
    }

  return {{nof_elem, bitsize}};
}

std::optional<std::pair<unsigned, unsigned>> hfa_hva_size(aarch64_state *rstate, tree type)
{
  if (auto hfa = hfa_size(rstate, type); hfa)
    return hfa;
  return hva_size(rstate, type);
}

void build_return(aarch64_state *rstate, Function *src_func, function *fun)
{
  Basic_block *bb = rstate->exit_bb;
  Basic_block *src_last_bb = src_func->bbs.back();
  assert(src_last_bb->last_inst->op == Op::RET);
  if (src_last_bb->last_inst->nof_args == 0)
    {
      bb->build_ret_inst();
      return;
    }
  tree ret_type = TREE_TYPE(DECL_RESULT(fun->decl));
  uint64_t ret_bitsize = src_last_bb->last_inst->args[0]->bitsize;

  if (is_sve_type(ret_type))
    {
      Inst *z0 = rstate->registers[Aarch64RegIdx::z0];
      if (ret_bitsize >= z0->bitsize)
	{
	  assert((ret_bitsize % z0->bitsize) == 0);
	  unsigned nof_regs = ret_bitsize / z0->bitsize;
	  assert(nof_regs <= 8);

	  Inst *retval =
	    bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z0]);
	  for (unsigned i = 1; i < nof_regs; i++)
	    {
	      Inst *reg = rstate->registers[Aarch64RegIdx::z0 + i];
	      Inst *inst = bb->build_inst(Op::READ, reg);
	      retval = bb->build_inst(Op::CONCAT, inst, retval);
	    }
	  bb->build_ret_inst(retval);
	  return;
	}
      else
	{
	  Inst *p0 = rstate->registers[Aarch64RegIdx::p0];
	  assert((ret_bitsize % p0->bitsize) == 0);
	  unsigned nof_regs = ret_bitsize / p0->bitsize;
	  assert(nof_regs <= 4);
	  Inst *retval =
	    bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::p0]);
	  for (unsigned i = 1; i < nof_regs; i++)
	    {
	      Inst *reg = rstate->registers[Aarch64RegIdx::p0 + i];
	      Inst *inst = bb->build_inst(Op::READ, reg);
	      retval = bb->build_inst(Op::CONCAT, inst, retval);
	    }
	  bb->build_ret_inst(retval);
	  return;
	}
    }

  if ((SCALAR_FLOAT_TYPE_P(ret_type) && ret_bitsize <= rstate->freg_bitsize)
      || TYPE_MAIN_VARIANT(ret_type) == aarch64_mfp8_type_node)
    {
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z0]);
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  if ((INTEGRAL_TYPE_P(ret_type)
       || POINTER_TYPE_P(ret_type)
       || TREE_CODE(ret_type) == NULLPTR_TYPE)
      && ret_bitsize <= rstate->reg_bitsize)
    {
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::x0]);
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  if (is_short_vector(ret_type))
    {
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z0]);
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  if (auto hfa = hfa_hva_size(rstate, ret_type); hfa)
    {
      auto [nof_regs, elem_bitsize] = *hfa;
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z0]);
      retval = bb->build_trunc(retval, elem_bitsize);
      for (unsigned i = 1; i < nof_regs; i++)
	{
	  Inst *inst =
	    bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z0 + i]);
	  inst = bb->build_trunc(inst, elem_bitsize);
	  retval = bb->build_inst(Op::CONCAT, inst, retval);
	}
      bb->build_ret_inst(retval);
      return;
    }

  if (ret_bitsize <= 2 * rstate->reg_bitsize
      && (TREE_CODE(ret_type) == RECORD_TYPE
	  || TREE_CODE(ret_type) == UNION_TYPE
	  || TREE_CODE(ret_type) == COMPLEX_TYPE
	  || INTEGRAL_TYPE_P(ret_type)
	  || (VECTOR_TYPE_P(ret_type) && !VECTOR_FLOAT_TYPE_P(ret_type))))
    {
      unsigned nof_regs =
	(ret_bitsize + rstate->reg_bitsize - 1) / rstate->reg_bitsize;
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::x0]);
      for (unsigned i = 1; i < nof_regs; i++)
	{
	  Inst *inst =
	    bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::x0 + i]);
	  retval = bb->build_inst(Op::CONCAT, inst, retval);
	}
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  if (ret_bitsize > 2 * rstate->reg_bitsize
      && (TREE_CODE(ret_type) == RECORD_TYPE
	  || TREE_CODE(ret_type) == UNION_TYPE
	  || TREE_CODE(ret_type) == COMPLEX_TYPE
	  || INTEGRAL_TYPE_P(ret_type)
	  || VECTOR_TYPE_P(ret_type)))
    {
      // The the value is passed in memory, where the address is specified
      // by x8.
      Module *module = rstate->module;
      Inst *id = bb->value_inst(rstate->next_local_id++, module->ptr_id_bits);
      Inst *mem_size = bb->value_inst(ret_bitsize / 8, module->ptr_offset_bits);
      Inst *flags = bb->value_inst(0, 32);
      Basic_block *entry_bb = rstate->entry_bb;
      Inst *ret_mem = entry_bb->build_inst(Op::MEMORY, id, mem_size, flags);
      Inst *reg = rstate->registers[Aarch64RegIdx::x8];
      entry_bb->build_inst(Op::WRITE, reg, ret_mem);

      // Generate the return value from the value returned in memory.
      uint64_t size = (ret_bitsize + 7) / 8;
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

Inst *extract_vec_elem(Basic_block *bb, Inst *inst, uint32_t elem_bitsize, uint32_t idx)
{
  if (idx == 0 && inst->bitsize == elem_bitsize)
    return inst;
  assert(inst->bitsize % elem_bitsize == 0);
  Inst *high = bb->value_inst(idx * elem_bitsize + elem_bitsize - 1, 32);
  Inst *low = bb->value_inst(idx * elem_bitsize, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
}

Inst *param_in_mem(Basic_block *bb, aarch64_state *rstate, Inst *value)
{
  if ((value->bitsize & 7) != 0)
    value = bb->build_inst(Op::ZEXT, value, (value->bitsize + 7) & ~7);
  Module *module = rstate->module;
  Inst *id = bb->value_inst(rstate->next_local_id++, module->ptr_id_bits);
  Inst *mem_size = bb->value_inst(value->bitsize / 8, module->ptr_offset_bits);
  Inst *flags = bb->value_inst(0, 32);
  Inst *mem = bb->build_inst(Op::MEMORY, id, mem_size, flags);
  for (uint64_t i = 0; i < value->bitsize / 8; i++)
    {
      Inst *offset = bb->value_inst(i, mem->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, mem, offset);
      Inst *data_byte = extract_vec_elem(bb, value, 8, i);
      bb->build_inst(Op::STORE, addr, data_byte);
    }
  return mem;
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
  rstate.next_local_id = ((int64_t)-1) << (module->ptr_id_bits - 1);
  assert(stack_size < (((uint64_t)1) << module->ptr_offset_bits));
  Inst *id = bb->value_inst(rstate.next_local_id++, module->ptr_id_bits);
  Inst *mem_size = bb->value_inst(stack_size, module->ptr_offset_bits);
  Inst *flags = bb->value_inst(0, 32);
  Inst *stack = bb->build_inst(Op::MEMORY, id, mem_size, flags);
  Inst *size = bb->value_inst(stack_size, stack->bitsize);
  stack = bb->build_inst(Op::ADD, stack, size);
  bb->build_inst(Op::WRITE, rstate.registers[Aarch64RegIdx::sp], stack);

  // Next General-purpose Register Number
  uint32_t ngrn = 0;

  // Next SIMD and Floating-point Register Number
  uint32_t nsrn = 0;

  // Next Scalable Predicate Register Number
  uint32_t nprn = 0;

  build_return(&rstate, src_func, fun);

  // Set up the PARAM instructions and copy the result to the correct
  // register or memory as required by the ABI.
  int param_number = 0;
  std::vector<Inst*> stack_values;
  for (tree decl = DECL_ARGUMENTS(fun->decl); decl; decl = DECL_CHAIN(decl))
    {
      tree type = TREE_TYPE(decl);
      uint32_t bitsize = bitsize_for_type(type);
      if (bitsize <= 0)
	throw Not_implemented("setup_aarch64_function: Parameter size == 0");
      Inst *param_nbr = bb->value_inst(param_number, 32);
      Inst *param_bitsize = bb->value_inst(bitsize, 32);
      Inst *param = bb->build_inst(Op::PARAM, param_nbr, param_bitsize);

      if (param_number == 0
	  && !strcmp(IDENTIFIER_POINTER(DECL_NAME(fun->decl)), "__ct_base "))
	{
	  // TODO: The "this" pointer in C++ constructors needs to be handled
	  // as a special case in the same way as in gimple_conv.cpp when
	  // setting up the parameters.
	  throw Not_implemented("setup_aarch64_function: C++ constructors");
	}

      if ((SCALAR_FLOAT_TYPE_P(type) && param->bitsize <= rstate.freg_bitsize)
	  || TYPE_MAIN_VARIANT(type) == aarch64_mfp8_type_node)
	{
	  if (nsrn < 8)
	    {
	      write_reg(bb, rstate.registers[Aarch64RegIdx::z0 + nsrn], param);
	      nsrn++;
	    }
	  else
	    throw Not_implemented("setup_aarch64_function: too many params");
	}
      else if (is_short_vector(type))
	{
	  if (nsrn < 8)
	    {
	      write_reg(bb, rstate.registers[Aarch64RegIdx::z0 + nsrn], param);
	      nsrn++;
	    }
	  else
	    throw Not_implemented("setup_aarch64_function: too many params");
	}
      else if (auto hfa = hfa_hva_size(&rstate, type); hfa)
	{
	  auto [nof_regs, elem_bitsize] = *hfa;
	  if (nsrn + nof_regs < 8)
	    {
	      for (unsigned i = 0; i < nof_regs; i++)
		{
		  Inst *value = extract_vec_elem(bb, param, elem_bitsize, i);
		  write_reg(bb, rstate.registers[Aarch64RegIdx::z0 + nsrn],
			    value);
		  nsrn++;
		}
	    }
	  else
	    throw Not_implemented("setup_aarch64_function: too many params");
	}
      else if (is_sve_type(type))
	{
	  Inst *z0 = rstate.registers[Aarch64RegIdx::z0];
	  if (param->bitsize >= z0->bitsize)
	    {
	      assert((param->bitsize % z0->bitsize) == 0);
	      unsigned nof_regs = param->bitsize / z0->bitsize;
	      if (nsrn + nof_regs <= 8)
		{
		  for (unsigned i = 0; i < nof_regs; i++)
		    {
		      Inst *reg = rstate.registers[Aarch64RegIdx::z0 + nsrn++];
		      Inst *value = extract_vec_elem(bb, param, z0->bitsize, i);
		      write_reg(bb, reg, value);
		    }
		}
	      else
		throw Not_implemented("setup_aarch64_function: too many params");
	    }
	  else
	    {
	      Inst *p0 = rstate.registers[Aarch64RegIdx::p0];
	      assert((param->bitsize % p0->bitsize) == 0);
	      unsigned nof_regs = param->bitsize / p0->bitsize;
	      if (nprn + nof_regs <= 4)
		{
		  for (unsigned i = 0; i < nof_regs; i++)
		    {
		      Inst *reg = rstate.registers[Aarch64RegIdx::p0 + nprn++];
		      Inst *value = extract_vec_elem(bb, param, p0->bitsize, i);
		      write_reg(bb, reg, value);
		    }
		}
	      else
		throw Not_implemented("setup_aarch64_function: too many params");
	    }
	}
      else if ((INTEGRAL_TYPE_P(type)
		|| POINTER_TYPE_P(type)
		|| TREE_CODE(type) == NULLPTR_TYPE)
	       && param->bitsize <= rstate.reg_bitsize)
	{
	  if (ngrn < 8)
	    {
	      write_reg(bb, rstate.registers[Aarch64RegIdx::x0 + ngrn], param);
	      ngrn++;
	    }
	  else
	    throw Not_implemented("setup_aarch64_function: too many params");
	}
      else if (param->bitsize <= 2 * rstate.reg_bitsize
	       && (TREE_CODE(type) == RECORD_TYPE
		   || TREE_CODE(type) == UNION_TYPE
		   || TREE_CODE(type) == COMPLEX_TYPE
		   || INTEGRAL_TYPE_P(type)
		   || (VECTOR_TYPE_P(type) && !VECTOR_FLOAT_TYPE_P(type))))
	{
	  if (ngrn < 8
	      && TYPE_ALIGN(type) > rstate.reg_bitsize
	      && (ngrn & 1) != 0)
	    ngrn++;

	  if (ngrn < 8)
	    {
	      Inst *value = param;
	      if (value->bitsize > rstate.reg_bitsize)
		value = bb->build_trunc(value, rstate.reg_bitsize);
	      write_reg(bb, rstate.registers[Aarch64RegIdx::x0 + ngrn], value);
	      ngrn++;

	      if (param->bitsize > rstate.reg_bitsize)
		{
		  Inst *high = bb->value_inst(param->bitsize - 1, 32);
		  Inst *low = bb->value_inst(rstate.reg_bitsize, 32);
		  value = bb->build_inst(Op::EXTRACT, param, high, low);
		  if (ngrn >= 8)
		    throw Not_implemented("setup_aarch64_function: "
					  "too many params");
		  write_reg(bb, rstate.registers[Aarch64RegIdx::x0 + ngrn],
			    value);
		  ngrn++;
		}
	    }
	  else
	    throw Not_implemented("setup_aarch64_function: too many params");
	}
      else if (param->bitsize > 2 * rstate.reg_bitsize
	       && (TREE_CODE(type) == RECORD_TYPE
		   || TREE_CODE(type) == UNION_TYPE
		   || TREE_CODE(type) == COMPLEX_TYPE
		   || INTEGRAL_TYPE_P(type)
		   || VECTOR_TYPE_P(type)))
	{
	  Inst *value = param_in_mem(bb, &rstate, param);
	  if (ngrn < 8)
	    {
	      write_reg(bb, rstate.registers[Aarch64RegIdx::x0 + ngrn], value);
	      ngrn++;
	    }
	  else
	    throw Not_implemented("setup_aarch64_function: too many params");
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
