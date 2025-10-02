#include "gcc-plugin.h"
#include "tree.h"

#include <cassert>

#include "gimple_conv.h"

using namespace smtgcc;

namespace {

const int stack_size = 1024 * 100;

void build_return(bpf_state *rstate, Function *src_func, function *fun)
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

  if ((INTEGRAL_TYPE_P(ret_type) || POINTER_TYPE_P(ret_type))
      && ret_bitsize <= 64)
    {
      Inst *retval =
	bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::x0]);
      if (ret_bitsize < retval->bitsize)
	retval = bb->build_trunc(retval, ret_bitsize);
      bb->build_ret_inst(retval);
      return;
    }

  throw Not_implemented("bpf: Unhandled return type");
}

} // end anonymous namespace

bpf_state setup_bpf_function(CommonState *state, Function *src_func, function *fun)
{
  Module *module = src_func->module;

  bpf_state rstate;
  rstate.module = module;
  rstate.memory_objects = state->memory_objects;
  rstate.func_name = IDENTIFIER_POINTER(DECL_ASSEMBLER_NAME(fun->decl));
  rstate.file_name = DECL_SOURCE_FILE(fun->decl);

  Function *tgt = module->build_function("tgt");
  rstate.entry_bb = tgt->build_bb();
  rstate.exit_bb = tgt->build_bb();

  assert(module->functions.size() == 2);
  Basic_block *bb = rstate.entry_bb;

  // Registers r0-r9.
  for (int i = 0; i < 10; i++)
    rstate.registers.push_back(bb->build_inst(Op::REGISTER, 64));

  // fp register.
  rstate.registers.push_back(bb->build_inst(Op::REGISTER, 64));

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
  bb->build_inst(Op::WRITE, rstate.registers[BpfRegIdx::fp], stack);

  build_return(&rstate, src_func, fun);

  // Set up the PARAM instructions and copy the result to the correct
  // register or memory as required by the ABI.
  int param_number = 0;
  for (tree decl = DECL_ARGUMENTS(fun->decl); decl; decl = DECL_CHAIN(decl))
    {
      if (param_number >= 5)
	throw Not_implemented("setup_bpf_function: too many params");
      uint32_t bitsize = bitsize_for_type(TREE_TYPE(decl));
      tree type = TREE_TYPE(decl);
      if ((!INTEGRAL_TYPE_P(type) && !POINTER_TYPE_P(type)) || bitsize > 64)
	throw Not_implemented("setup_bpf_function: param type not handled");
      if (bitsize <= 0)
	throw Not_implemented("Parameter size == 0");

      Inst *param_nbr = bb->value_inst(param_number, 32);
      Inst *param_bitsize = bb->value_inst(bitsize, 32);
      Inst *param = bb->build_inst(Op::PARAM, param_nbr, param_bitsize);

      Inst *reg = rstate.registers[BpfRegIdx::r1 + param_number];
      if (bitsize < 64)
	param = bb->build_inst(Op::ZEXT, param, 64);
      bb->build_inst(Op::WRITE, reg, param);

      param_number++;
    }

  // Functionality for abort/exit.
  {
    Inst *b0 = bb->value_inst(0, 1);
    Inst *zero = bb->value_inst(0, 32);
    bb->build_inst(Op::WRITE, rstate.registers[BpfRegIdx::abort], b0);
    bb->build_inst(Op::WRITE, rstate.registers[BpfRegIdx::exit], b0);
    bb->build_inst(Op::WRITE, rstate.registers[BpfRegIdx::exit_val], zero);

    Inst *abort_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[BpfRegIdx::abort]);
    Inst *exit_cond =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[BpfRegIdx::exit]);
    Inst *exit_val =
      rstate.exit_bb->build_inst(Op::READ,
				 rstate.registers[BpfRegIdx::exit_val]);
    rstate.exit_bb->build_inst(Op::EXIT, abort_cond, exit_cond, exit_val);
  }

  return rstate;
}
