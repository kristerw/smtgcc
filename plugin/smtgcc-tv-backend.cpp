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
#ifdef SMTGCC_RISCV
  std::vector<riscv_state> functions;
#endif
};

unsigned int tv_pass::execute(function *fun)
{
  try
    {
#if defined(SMTGCC_AARCH64)
      CommonState state(Arch::aarch64);
      Module *module = create_module(Arch::aarch64);
#elif defined(SMTGCC_RISCV)
      CommonState state(Arch::riscv);
      Module *module = create_module(Arch::riscv);
#endif
      Function *src = process_function(module, &state, fun, false);
      src->name = "src";
      unroll_and_optimize(src);

#if defined(SMTGCC_AARCH64)
      aarch64_state rstate = setup_aarch64_function(&state, src, fun);
#elif defined(SMTGCC_RISCV)
      riscv_state rstate = setup_riscv_function(&state, src, fun);
#endif
      functions.push_back(rstate);
    }
  catch (Not_implemented& error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
    }
  return 0;
}

static Inst *extract_vec_elem(Basic_block *bb, Inst *inst, uint32_t elem_bitsize, uint32_t idx)
{
  if (idx == 0 && inst->bitsize == elem_bitsize)
    return inst;
  assert(inst->bitsize % elem_bitsize == 0);
  Inst *high = bb->value_inst(idx * elem_bitsize + elem_bitsize - 1, 32);
  Inst *low = bb->value_inst(idx * elem_bitsize, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
}

// Return the element size to use for a vector phi if this is a vector phi.
// For this to be considered a vector phi, all uses must be an Op::EXTRACT
// of a size that divides the phi bitsize, and the extraction must be
// extraced from a correctly aligned position. We then returns the smallest
// size of the extract.
// TODO: We currently skip 8 as this is to agressive when the phi is stored.
// This need a better heuristic.

// Return the element size to use for a vector phi if this is a vector phi.
// For this to be considered a vector phi, all uses must be an Op::EXTRACT
// of a size that divides the phi bitsize, and the extraction must be
// extracted from a correctly aligned position. We then return the smallest
// size of the extractions.
// TODO: We currently skip element size <= 8 as this is too aggressive when
// the phi is stored. This needs a better heuristic.
static std::optional<uint32_t> get_phi_elem_bitsize(Inst *phi)
{
  assert(phi->op == Op::PHI);

  // Vectors are at lest 128 bits.
  if (phi->bitsize < 128)
    return {};

  uint32_t elem_bitsize = phi->bitsize;
  for (auto use : phi->used_by)
    {
      if (use->op != Op::EXTRACT)
	return {};
      if (use->bitsize <= 8)
	return {};
      elem_bitsize = std::min(elem_bitsize, use->bitsize);
      if (use->args[2]->value() % elem_bitsize != 0)
	return {};
    }

  return elem_bitsize;
}

static Inst *split_phi(Inst *phi, uint64_t elem_bitsize, std::map<std::pair<Inst *, uint64_t>, std::vector<Inst *>>& cache)
{
  assert(phi->op == Op::PHI);
  assert(phi->bitsize % elem_bitsize == 0);
  if (phi->bitsize == elem_bitsize)
    return phi;
  Inst *res = nullptr;
  uint32_t nof_elem = phi->bitsize / elem_bitsize;
  std::vector<Inst *> phis;
  phis.reserve(nof_elem);
  for (uint64_t i = 0; i < nof_elem; i++)
    {
      Inst *inst = phi->bb->build_phi_inst(elem_bitsize);
      phis.push_back(inst);
      if (res)
	{
	  Inst *concat = create_inst(Op::CONCAT, inst, res);
	  if (res->op == Op::PHI)
	    {
	      if (phi->bb->first_inst)
		concat->insert_before(phi->bb->first_inst);
	      else
		phi->bb->insert_last(concat);
	    }
	  else
	    concat->insert_after(res);
	  res = concat;
	}
      else
	res = inst;
    }
  phi->replace_all_uses_with(res);

  for (auto [arg_inst, arg_bb] : phi->phi_args)
    {
      std::vector<Inst *>& split = cache[{arg_inst, elem_bitsize}];
      if (split.empty())
	{
	  for (uint64_t i = 0; i < nof_elem; i++)
	    {
	      Inst *inst =
		extract_vec_elem(arg_inst->bb, arg_inst, elem_bitsize, i);
	      split.push_back(inst);
	    }
	}
      for (uint64_t i = 0; i < nof_elem; i++)
	{
	  phis[i]->add_phi_arg(split[i], arg_bb);
	}
    }

  return res;
}

// Note: This assumes that we do not have any loops.
static void eliminate_registers(Function *func)
{
  std::map<Basic_block *, std::map<Inst *, Inst *>> bb2reg_values;

  // Collect all registers. This is not completely necessary, but we want
  // to iterate over the registers in a consisten order when we create
  // phi-nodes etc. and iterating over the maps could change order between
  // different runs.
  std::vector<Inst *> registers;
  for (Inst *inst = func->bbs[0]->first_inst;
       inst;
       inst = inst->next)
    {
      if (inst->op == Op::REGISTER)
	{
	  Basic_block *bb = func->bbs[0];
	  registers.push_back(inst);
	  // TODO: Should be an arbitrary value (i.e., a symbolic value)
	  // instead of 0.
	  bb2reg_values[bb][inst] = bb->value_inst(0, inst->bitsize);
	}
    }

  for (Basic_block *bb : func->bbs)
    {
      std::map<Inst *, Inst *>& reg_values =
	bb2reg_values[bb];
      if (bb->preds.size() == 1)
	{
	  reg_values = bb2reg_values.at(bb->preds[0]);
	}
      else if (bb->preds.size() > 1)
	{
	  for (auto reg : registers)
	    {
	      Inst *phi = bb->build_phi_inst(reg->bitsize);
	      reg_values[reg] = phi;
	      for (auto pred_bb : bb->preds)
		{
		  if (!bb2reg_values.at(pred_bb).contains(reg))
		    throw Not_implemented("eliminate_registers: Read of uninit register");
		  Inst *arg = bb2reg_values.at(pred_bb).at(reg);
		  phi->add_phi_arg(arg, pred_bb);
		}
	    }
	}

      Inst *inst = bb->first_inst;
      while (inst)
	{
	  Inst *next_inst = inst->next;
	  if (inst->op == Op::READ)
	    {
	      if (!reg_values.contains(inst->args[0]))
		throw Not_implemented("eliminate_registers: Read of uninit register");
	      inst->replace_all_uses_with(reg_values.at(inst->args[0]));
	      destroy_instruction(inst);
	    }
	  else if (inst->op == Op::WRITE)
	    {
	      reg_values[inst->args[0]] = inst->args[1];
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

  for (int i = func->bbs.size() - 1; i >= 0; i--)
    {
      Basic_block *bb = func->bbs[i];
      std::vector<std::pair<Inst *, uint32_t>> phis;
      for (auto phi : bb->phis)
	{
	  if (std::optional<uint32_t> elem_bitsize = get_phi_elem_bitsize(phi))
	    phis.push_back({phi, *elem_bitsize});
	}
      std::map<std::pair<Inst *, uint64_t>, std::vector<Inst *>> cache;
      for (auto [phi, elem_bitsize] : phis)
	split_phi(phi, elem_bitsize, cache);
    }
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
#elif defined(SMTGCC_RISCV)
	  Function *func = parse_riscv(file_name, &state);
#endif
	  validate(func);

	  simplify_cfg(func);
	  if (loop_unroll(func))
	    {
	      bool cfg_modified;
	      do
		{
		  simplify_insts(func);
		  dead_code_elimination(func);
		  cfg_modified = simplify_cfg(func);
		}
	      while (cfg_modified);
	    }

	  eliminate_registers(func);
	  validate(func);

	  // Simplify the code several times -- this is often necessary
	  // as instruction simplification enables new CFG simplifications
	  // that then enable new instruction simplifications.
	  // This is handled during unrolling for the GIMPLE passes, but
	  // it does not work here because we must do unrolling before
	  // eliminating the register instructions.
	  simplify_insts(func);
	  dead_code_elimination(func);
	  simplify_cfg(func);
	  vrp(func);
	  bool cfg_modified;
	  do
	    {
	      simplify_insts(func);
	      dead_code_elimination(func);
	      cfg_modified = simplify_cfg(func);
	    }
	  while (cfg_modified);

	  canonicalize_memory(module);
	  simplify_mem(module);
	  ls_elim(module);
	  reduce_bitsize(module);
	  do
	    {
	      simplify_insts(module);
	      dead_code_elimination(module);
	      cfg_modified = simplify_cfg(module);
	    }
	  while (cfg_modified);

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
