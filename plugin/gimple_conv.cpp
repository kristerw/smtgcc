#include "gcc-plugin.h"
#include "tree-pass.h"
#include "tree.h"
#include "tree-cfg.h"
#include "gimple.h"
#include "gimple-iterator.h"
#include "print-tree.h"
#include "internal-fn.h"
#include "tree-ssa-operands.h"
#include "ssa.h"
#include "cgraph.h"
#include "alloc-pool.h"
#include "symbol-summary.h"
#include "sreal.h"
#include "ipa-cp.h"
#include "ipa-prop.h"
#include "value-query.h"
#include "attribs.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "smtgcc.h"
#include "gimple_conv.h"

static_assert(sizeof(HOST_WIDE_INT) == 8);

// TODO: This is declared in builtins.h, but I get some problems with missing
// includes when I include it.
unsigned int get_object_alignment(tree exp);

// How many bytes load, store, __builtin_memset, etc. can expand.
#define MAX_MEMORY_UNROLL_LIMIT  10000

// Size of anonymous memory size blocks we may need to introduce (for example,
// so that function pointer arguments have memory to point to).
#define ANON_MEM_SIZE 136

using namespace std::string_literals;
using namespace smtgcc;

namespace {

struct Addr {
  Inst *ptr;
  uint64_t bitoffset;
  Inst *prov;
};

struct Converter {
  Converter(Module *module, CommonState *state, function *fun, bool is_tgt_func)
    : module{module}
    , state{state}
    , fun{fun}
    , is_tgt_func{is_tgt_func}
  {}
  ~Converter()
  {
    if (func)
      destroy_function(func);
  }
  Module *module;
  CommonState *state;
  function *fun;
  Function *func = nullptr;
  Basic_block *bb = nullptr;
  Inst *loop_vect_sym = nullptr;
  std::string pass_name;
  std::map<Basic_block *, std::set<Basic_block *>> switch_bbs;
  std::map<basic_block, Basic_block *> gccbb_top2bb;
  std::map<basic_block, Basic_block *> gccbb_bottom2bb;
  std::map<Basic_block *, std::pair<Inst *, Inst *> > bb2retval;
  std::set<Basic_block *> bb_abort;
  std::map<Basic_block *, Inst *> bb2exit;
  std::map<tree, Inst *> tree2instruction;
  std::map<tree, Inst *> tree2indef;
  std::map<tree, Inst *> tree2prov;
  std::map<tree, Inst *> decl2instruction;
  std::map<Inst *, Inst *> inst2memory_flagsx;
  std::map<Inst *, Inst *> inst2id;
  Inst *retval = nullptr;

  bool is_tgt_func;

  Inst *extract_id(Inst *inst);
  uint64_t bytesize_for_type(tree type);
  std::pair<Inst *, Inst *> to_mem_repr(Inst *inst, Inst *indef, tree type);
  Inst *to_mem_repr(Inst *inst, tree type);
  std::pair<Inst *, Inst *> from_mem_repr(Inst *inst, Inst *indef, tree type);
  Inst *from_mem_repr(Inst *inst, tree type);
  uint8_t bitfield_padding_at_offset(tree fld, int64_t offset);
  uint8_t padding_at_offset(tree type, uint64_t offset);
  Inst *build_memory_inst(uint64_t id, uint64_t size, uint32_t flags);
  void build_ub_if_not_zero(Inst *inst, Inst *cmp = nullptr);
  void constrain_range(Basic_block *bb, tree expr, Inst *inst, Inst *indef=nullptr);
  void store_ub_check(Inst *ptr, Inst *prov, uint64_t size, Inst *cond = nullptr);
  void load_ub_check(Inst *ptr, Inst *prov, uint64_t size, Inst *cond = nullptr);
  void load_vec_ub_check(Inst *ptr, Inst *prov, uint64_t size, tree expr);
  void overlap_ub_check(Inst *src_ptr, Inst *dst_ptr, uint64_t size);
  bool is_extracting_from_value(tree expr);
  std::tuple<Inst *, Inst *, Inst *> extract_component_ref(tree expr);
  std::tuple<Inst *, Inst *, Inst *> tree2inst_indef_prov(tree expr);
  std::pair<Inst *, Inst *>tree2inst_indef(tree expr);
  std::pair<Inst *, Inst *>tree2inst_prov(tree expr);
  Inst *tree2inst(tree expr);
  std::tuple<Inst *, Inst *, Inst *> tree2inst_constructor(tree expr);
  Inst *get_res_indef(Inst *arg1_indef, tree lhs_type);
  Inst *get_res_indef(Inst *arg1_indef, Inst *arg2_indef, tree lhs_type);
  Inst *get_res_indef(Inst *arg1_indef, Inst *arg2_indef, Inst *arg3_indef, tree lhs_type);
  void alignment_check(tree expr, Inst *ptr);
  void process_decl(tree decl);
  Addr process_array_ref(tree expr, bool is_mem_access);
  Addr process_component_ref(tree expr, bool is_mem_access);
  Addr process_bit_field_ref(tree expr, bool is_mem_access);
  Addr process_address(tree expr, bool is_mem_access);
  std::tuple<Inst *, Inst *, Inst *> vector_as_array(tree expr);
  std::tuple<Inst *, Inst *, Inst *> process_load(tree expr);
  bool is_load(tree expr);
  void load_store_overlap_ub_check(tree store_expr, tree load_expr);
  void process_store(tree addr_expr, tree value_expr);
  std::tuple<Inst *, Inst *, Inst *> load_value(Inst *ptr, uint64_t size);
  void store_value(Inst *ptr, Inst *value, Inst *indef = nullptr);
  std::tuple<Inst *, Inst *, Inst *> type_convert(Inst *inst, Inst *indef, Inst *prov, tree src_type, tree dest_type);
  Inst *type_convert(Inst *inst, tree src_type, tree dest_type);
  std::pair<Inst *, Inst *> process_unary_bool(enum tree_code code, Inst *arg1, Inst *arg1_indef, tree lhs_type, tree arg1_type);
  void check_wide_bool(Inst *inst, tree type);
  std::tuple<Inst *, Inst *, Inst *> process_unary_int(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg1_prov, tree lhs_type, tree arg1_type, bool ignore_overflow = false);
  Inst *process_unary_scalar(enum tree_code code, Inst *arg1, tree lhs_type, tree arg1_type, bool ignore_overflow = false);
  std::tuple<Inst *, Inst *, Inst *> process_unary_scalar(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg1_prov, tree lhs_type, tree arg1_type, bool ignore_overflow = false);
  std::pair<Inst *, Inst *> process_vec_duplicate(Inst *arg1, Inst *arg1_indef, tree lhs_elem_type, tree arg1_elem_type);
  std::pair<Inst *, Inst *> process_unary_vec(enum tree_code code, Inst *arg1, Inst *arg1_indef, tree lhs_elem_type, tree arg1_elem_type, bool ignore_overflow = false);
  std::pair<Inst *, Inst *> process_unary_float(enum tree_code code, Inst *arg1, Inst *arg1_indef, tree lhs_type, tree arg1_type);
  Inst *process_unary_complex(enum tree_code code, Inst *arg1, tree lhs_type);
  std::pair<Inst *, Inst *> process_binary_float(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type);
  Inst *process_binary_complex(enum tree_code code, Inst *arg1, Inst *arg2, tree lhs_type);
  Inst *process_binary_complex_cmp(enum tree_code code, Inst *arg1, Inst *arg2, tree lhs_type, tree arg1_type);
  std::pair<Inst *, Inst *> process_binary_bool(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type, tree arg1_type, tree arg2_type);
  std::tuple<Inst *, Inst *, Inst *> process_binary_int(enum tree_code code, bool is_unsigned, Inst *arg1, Inst *arg1_indef, Inst *arg1_prov, Inst *arg2, Inst *arg2_indef, Inst *arg2_prov, tree lhs_type, tree arg1_type, tree arg2_type, bool ignore_overflow = false);
  Inst *process_binary_scalar(enum tree_code code, Inst *arg1, Inst *arg2, tree lhs_type, tree arg1_type, tree arg2_type, bool ignore_overflow = false);
  std::tuple<Inst *, Inst *, Inst *> process_binary_scalar(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg1_prov, Inst *arg2, Inst *arg2_indef, Inst *arg2_prov, tree lhs_type, tree arg1_type, tree arg2_type, bool ignore_overflow = false);
  std::pair<Inst *, Inst *> process_binary_vec(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type, tree arg1_type, tree arg2_type, bool ignore_overflow = false);
  std::pair<Inst *, Inst *> process_widen_sum_vec(Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type, tree arg1_type, tree arg2_type);
  std::pair<Inst *, Inst *> process_widen_mult_evenodd(Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type, tree arg1_type, tree arg2_type, bool is_odd);
  std::pair<Inst *, Inst *> process_vec_series(Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type);
  Inst *process_ternary(enum tree_code code, Inst *arg1, Inst *arg2, Inst *arg3, tree arg1_type, tree arg2_type, tree arg3_type);
  std::tuple<Inst *, Inst *, Inst *> process_ternary(enum tree_code code, tree arg1_tree, tree arg2_tree, tree arg3_tree);
  Inst *process_ternary_vec(enum tree_code code, Inst *arg1, Inst *arg2, Inst *arg3, tree lhs_type, tree arg1_type, tree arg2_type, tree arg3_type);
  std::pair<Inst *, Inst *> gen_vec_cond(Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, Inst *arg3, Inst *arg3_indef, tree arg1_type, tree arg2_type, Inst *len = nullptr);
  std::pair<Inst *, Inst *> process_vec_perm_expr(gimple *stmt);
  std::tuple<Inst *, Inst *, Inst *> vector_constructor(tree expr);
  void process_constructor(tree lhs, tree rhs);
  void process_gimple_assign(gimple *stmt);
  void process_gimple_asm(gimple *stmt);
  void process_cfn_unary(gimple *stmt, const std::function<std::pair<Inst *, Inst *>(Inst *, Inst *, tree)>& gen_elem);
  void process_cfn_binary(gimple *stmt, const std::function<std::pair<Inst *, Inst *>(Inst *, Inst *, Inst *, Inst *, tree)>& gen_elem);
  void process_cfn_abd(gimple *stmt);
  void process_cfn_abort(gimple *stmt);
  void process_cfn_add_overflow(gimple *stmt);
  void process_cfn_bit_andn(gimple *stmt);
  void process_cfn_bit_iorn(gimple *stmt);
  void process_cfn_assume_aligned(gimple *stmt);
  void process_cfn_bswap(gimple *stmt);
  void process_cfn_check_war_ptrs(gimple *stmt);
  void process_cfn_check_raw_ptrs(gimple *stmt);
  void process_cfn_clrsb(gimple *stmt);
  void process_cfn_clz(gimple *stmt);
  std::pair<Inst*, Inst*> gen_cfn_cond_unary(tree_code code, Inst *cond, Inst *cond_indef, Inst *arg1, Inst *arg1_indef, Inst *orig, Inst *orig_indef, Inst *len, tree cond_type, tree arg1_type, tree orig_type);
  std::pair<Inst*, Inst*> gen_cfn_cond_binary(tree_code code, Inst *cond, Inst *cond_indef, Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, Inst *orig, Inst *orig_indef, Inst *len, tree cond_type, tree arg1_type, tree arg2_type, tree orig_type);
  void process_cfn_cond_unary(gimple *stmt, tree_code code);
  void process_cfn_cond_binary(gimple *stmt, tree_code code);
  void process_cfn_cond_len_binary(gimple *stmt, tree_code code);
  void process_cfn_cond_fminmax(gimple *stmt);
  void process_cfn_copysign(gimple *stmt);
  void process_cfn_ctz(gimple *stmt);
  void process_cfn_divmod(gimple *stmt);
  void process_cfn_exit(gimple *stmt);
  void process_cfn_expect(gimple *stmt);
  void process_cfn_fabs(gimple *stmt);
  void process_cfn_ffs(gimple *stmt);
  void process_cfn_fmax(gimple *stmt);
  void process_cfn_fmin(gimple *stmt);
  void process_cfn_isfinite(gimple *stmt);
  void process_cfn_isinf(gimple *stmt);
  void process_cfn_loop_vectorized(gimple *stmt);
  std::tuple<Inst*, Inst*> mask_len_load(Inst *ptr, Inst *ptr_indef, Inst *ptr_prov, uint64_t alignment, Inst *mask, Inst *mask_indef, tree mask_type, Inst *len, tree lhs_type, Inst *orig, Inst *orig_indef);
  void process_cfn_mask_len_load(gimple *stmt);
  void process_cfn_mask_load(gimple *stmt);
  void mask_len_store(Inst *ptr, Inst *ptr_indef, Inst *ptr_prov, uint64_t alignment, Inst *mask, Inst *mask_indef, tree mask_type, Inst *len, tree value_type, Inst *value, Inst *value_indef);
  void process_cfn_mask_len_store(gimple *stmt);
  void process_cfn_mask_store(gimple *stmt);
  void process_cfn_memcpy(gimple *stmt);
  void process_cfn_memmove(gimple *stmt);
  void process_cfn_mempcpy(gimple *stmt);
  void process_cfn_memset(gimple *stmt);
  void process_cfn_mul_overflow(gimple *stmt);
  void process_cfn_mulh(gimple *stmt);
  void process_cfn_nan(gimple *stmt);
  void process_cfn_parity(gimple *stmt);
  void process_cfn_popcount(gimple *stmt);
  void process_cfn_sat_add(gimple *stmt);
  void process_cfn_sat_sub(gimple *stmt);
  void process_cfn_sat_trunc(gimple *stmt);
  void process_cfn_select_vl(gimple *stmt);
  void process_cfn_signbit(gimple *stmt);
  void process_cfn_sub_overflow(gimple *stmt);
  void process_cfn_reduc(gimple *stmt, tree_code code);
  void process_cfn_reduc_fminmax(gimple *stmt);
  void process_cfn_trap(gimple *stmt);
  void process_cfn_uaddc(gimple *stmt);
  void process_cfn_unreachable(gimple *stmt);
  void process_cfn_usubc(gimple *stmt);
  void process_cfn_vcond_mask(gimple *stmt);
  void process_cfn_vcond_mask_len(gimple *stmt);
  void process_cfn_vec_addsub(gimple *stmt);
  void process_cfn_vec_convert(gimple *stmt);
  void process_cfn_vec_extract(gimple *stmt);
  void process_cfn_vec_set(gimple *stmt);
  void process_cfn_vec_widen(gimple *stmt, Op op, bool high);
  void process_cfn_vec_widen_abd(gimple *stmt, bool high);
  void process_cfn_while_ult(gimple *stmt);
  void process_cfn_xorsign(gimple *stmt);
  void process_gimple_call_combined_fn(gimple *stmt);
  void process_gimple_call(gimple *stmt);
  void process_gimple_return(gimple *stmt);
  Inst *build_label_cond(tree index_expr, tree label,
				Basic_block *bb);
  void process_gimple_switch(gimple *stmt, Basic_block *bb);
  Basic_block *get_phi_arg_bb(gphi *phi, int i);
  void generate_exit_inst();
  void generate_return_inst();
  void init_var_values(tree initial, Inst *mem_inst);
  void init_var(tree decl, Inst *mem_inst);
  void make_uninit(Inst *ptr, uint64_t size);
  void constrain_src_value(Inst *inst, tree type, Inst *mem_flags = nullptr);
  void process_variables();
  void process_func_args();
  bool need_prov_phi(gimple *phi);
  void process_instructions(int nof_blocks, int *postorder);
  Function *process_function();
};

// Build the minimal signed integer value for the bitsize.
// The bitsize may be larger than 128.
Inst *build_min_int(Basic_block *bb, uint64_t bitsize)
{
  Inst *top_bit = bb->value_inst(1, 1);
  if (bitsize == 1)
    return top_bit;
  Inst *zero = bb->value_inst(0, bitsize - 1);
  return bb->build_inst(Op::CONCAT, top_bit, zero);
}

unsigned __int128 get_widest_int_val(widest_int v)
{
  unsigned int len = v.get_len();
  const HOST_WIDE_INT *p = v.get_val();
  if (len != 1 && len != 2)
    throw Not_implemented("get_widest_int_val: precision > 128");
  assert(len == 1 || len == 2);
  unsigned __int128 value = 0;
  if (len == 2)
    value = ((unsigned __int128)p[1]) << 64;
  else
    {
      int64_t t = p[0] >> 63;
      value = ((unsigned __int128)t) << 64;
    }
  value |= (uint64_t)p[0];
  return value;
}

unsigned __int128 get_wide_int_val(wide_int v)
{
  unsigned int len = v.get_len();
  const HOST_WIDE_INT *p = v.get_val();
  if (len != 1 && len != 2)
    throw Not_implemented("get_wide_int_val: precision > 128");
  assert(len == 1 || len == 2);
  unsigned __int128 value = 0;
  if (len == 2)
    value = ((unsigned __int128)p[1]) << 64;
  else
    {
      int64_t t = p[0] >> 63;
      value = ((unsigned __int128)t) << 64;
    }
  value |= (uint64_t)p[0];
  return value;
}

void check_type(tree type)
{
  if (AGGREGATE_TYPE_P(type) && TYPE_REVERSE_STORAGE_ORDER(type))
    throw Not_implemented("reverse storage order");

  // Note: We do not check that all elements in structures/arrays have
  // valid type -- they will be checked when the fields are accessed.
  // This makes us able to analyze progams having invalid elements in
  // unused structures/arrays.
  if (DECIMAL_FLOAT_TYPE_P(type))
    throw Not_implemented("check_type: DECIMAL_FLOAT_TYPE");
  else if (type == bfloat16_type_node)
    throw Not_implemented("check_type: bfloat16");
  else if (VECTOR_TYPE_P(type) || TREE_CODE(type) == COMPLEX_TYPE)
    check_type(TREE_TYPE(type));
  else if (FLOAT_TYPE_P(type))
    {
      // We do not support 80-bit floating point because of two
      // reasons:
      // 1. It is a 128-bit type in memory and registers, so we must
      //    special case it in load/store or for all floating point
      //    operations to chose the correct 128 or 80 bit operation.
      // 2. It does not follow IEEE, so we must do some extra work
      //    to get the correct bits (or report bogus errors when it
      //    is constant folded).
      uint64_t precision = TYPE_PRECISION(type);
      if (precision != 16 && precision != 32 && precision != 64
	  && precision != 128)
	throw Not_implemented("check_type: fp" + std::to_string(precision));
    }
  else if (TREE_CODE(type) == BITINT_TYPE)
    {
      // The bitintlower pass creates loops that iterate over 64-bit
      // chunks of the integer, leading us to report miscompilations
      // if the loop is not fully unrolled. Therefore, we limit the
      // size of bitint integers to dimensions we can manage.
      if (TYPE_PRECISION(type) > 64 * unroll_limit)
	throw Not_implemented("check_type: too wide BITINT");
    }
}

Inst *Converter::extract_id(Inst *inst)
{
  if (auto it = inst2id.find(inst); it != inst2id.end())
    return it->second;

  Inst *id = bb->build_extract_id(inst);
  if (inst->op == Op::VALUE)
    id = constant_fold_inst(id);
  else
    id->move_after(inst);
  inst2id.insert({inst, id});
  return id;
}

// The size of the GCC type when stored in memory etc.
uint64_t Converter::bytesize_for_type(tree type)
{
  tree size_tree = TYPE_SIZE(type);
  if (size_tree == NULL_TREE)
    throw Not_implemented("bytesize_for_type: incomplete type");
  if (TREE_CODE(size_tree) != INTEGER_CST && !POLY_INT_CST_P(size_tree))
    {
      // Things like function parameters
      //   int foo(int n, struct T { char a[n]; } b);
      throw Not_implemented("bytesize_for_type: complicated type");
    }
  uint64_t bitsize = get_int_cst_val(size_tree);
  assert((bitsize & 7) == 0);
  uint64_t size = bitsize / 8;
  if (size >= (uint64_t(1) << module->ptr_offset_bits))
    throw Not_implemented("bytesize_for_type: too large type");
  return size;
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

std::tuple<Inst *, Inst *> extract_vec_elem(Basic_block *bb, Inst *inst, Inst *indef, uint32_t elem_bitsize, uint32_t idx)
{
  inst = extract_vec_elem(bb, inst, elem_bitsize, idx);
  if (indef)
    indef = extract_vec_elem(bb, indef, elem_bitsize, idx);
  return {inst, indef};
}

std::tuple<Inst *, Inst *, Inst *> extract_vec_elem(Basic_block *bb, Inst *inst, Inst *indef, Inst *prov, uint32_t elem_bitsize, uint32_t idx)
{
  inst = extract_vec_elem(bb, inst, elem_bitsize, idx);
  if (indef)
    indef = extract_vec_elem(bb, indef, elem_bitsize, idx);
  if (prov)
    prov = extract_vec_elem(bb, prov, elem_bitsize, idx);
  return {inst, indef, prov};
}

Inst *extract_elem(Basic_block *bb, Inst *vec, uint32_t elem_bitsize, Inst *idx)
{
  // The shift calculation below may overflow if idx is not wide enough,
  // so we extend it to a safe width.
  // Note: We could have extended it to the full vector bit size, but that
  // would limit optimizations such as constant folding for the shift
  // calculation for vectors wider than 128 bits.
  if (idx->bitsize < 32)
    idx = bb->build_inst(Op::ZEXT, idx, 32);

  Inst *elm_bsize = bb->value_inst(elem_bitsize, idx->bitsize);
  Inst *shift = bb->build_inst(Op::MUL, idx, elm_bsize);
  if (shift->bitsize > vec->bitsize)
    shift = bb->build_trunc(shift, vec->bitsize);
  else if (shift->bitsize < vec->bitsize)
    shift = bb->build_inst(Op::ZEXT, shift, vec->bitsize);
  Inst *inst = bb->build_inst(Op::LSHR, vec, shift);
  return bb->build_trunc(inst, elem_bitsize);
}

std::pair<Inst *, Inst *> extract_elem(Basic_block *bb, Inst *vec, Inst *vec_indef, uint32_t elem_bitsize, Inst *idx, Inst *idx_indef)
{
  Inst *res = extract_elem(bb, vec, elem_bitsize, idx);
  Inst *res_indef = nullptr;
  if (vec_indef)
    res_indef = extract_elem(bb, vec_indef, elem_bitsize, idx);
  if (idx_indef)
    {
      Inst *zero = bb->value_inst(0, idx_indef->bitsize);
      idx_indef = bb->build_inst(Op::NE, idx_indef, zero);
      if (elem_bitsize > 1)
	idx_indef = bb->build_inst(Op::SEXT, idx_indef, elem_bitsize);
      if (res_indef)
	res_indef = bb->build_inst(Op::OR, res_indef, idx_indef);
      else
	res_indef = idx_indef;
    }
  return {res, res_indef};
}

bool is_bit_field(tree expr)
{
  tree_code code = TREE_CODE(expr);
  if (code == COMPONENT_REF)
    {
      tree field = TREE_OPERAND(expr, 1);
      if (DECL_BIT_FIELD_TYPE(field))
	return true;
    }
  else if (code == BIT_FIELD_REF)
    {
      uint64_t bitsize = get_int_cst_val(TREE_OPERAND(expr, 1));
      uint64_t bit_offset = get_int_cst_val(TREE_OPERAND(expr, 2));
      return (bitsize % 8) != 0 || (bit_offset % 8) != 0;
    }

  return false;
}

// Add checks in the src function to ensure that the value is a valid value
// for the type. The main use is to make sure the initial state is valid
// (for example, global pointers can't point to local memory).
//
// * FLOAT_TYPE - make sure the value isn't a "non-canonical" NaN.
//   SMT solvers canonicalize NaN values, so if we get a non-canonical value,
//   then the SMT solver will change the result. This doesn't matter in most
//   cases (as it will change it consistently for the source and target) but
//   it fails in e.g., gcc.dg/tree-ssa/mult-abs-2.c where GCC optimizes
//     return x * (x > 0.f ? -1.f : 1.f);
//   to
//     if (x > 0.f)
//       return -x;
//     else
//       return x;
//   which only do the NaN canonicalization in one path.
// * POINTER_TYPE - ensures the pointer doesn't point to local memory
//   unless it has a memory_flag value that isn't 0.
// * BOOLEAN_TYPE - ensure the value is 0 or 1.
void Converter::constrain_src_value(Inst *inst, tree type, Inst *mem_flags)
{
  if (is_tgt_func)
    return;

  // TODO: We should invert the meaning of mem_flags.
  // TODO: mem_flags is not correct name -- it is only one flag.
  if (POINTER_TYPE_P(type))
    {
      Inst *id = extract_id(inst);
      Inst *zero = bb->value_inst(0, id->bitsize);
      Inst *not_written = nullptr;
      if (mem_flags)
	{
	  not_written = extract_id(mem_flags);
	  not_written = bb->build_inst(Op::EQ, not_written, zero);
	}
      Inst *cond = bb->build_inst(Op::SLT, id, zero);
      if (not_written)
	cond = bb->build_inst(Op::AND, cond, not_written);
      bb->build_inst(Op::UB, cond);

      for (auto restrict_id : state->restrict_ids)
	{
	  Inst *cond = bb->build_inst(Op::EQ, id, restrict_id);
	  if (not_written)
	    cond = bb->build_inst(Op::AND, cond, not_written);
	  bb->build_inst(Op::UB, cond);
	}
      return;
    }
  if (SCALAR_FLOAT_TYPE_P(type))
    {
      bb->build_inst(Op::UB, bb->build_inst(Op::IS_NONCANONICAL_NAN, inst));
      return;
    }
  if (INTEGRAL_TYPE_P(type) && inst->bitsize != bitsize_for_type(type))
    {
      Inst *tmp = bb->build_trunc(inst, bitsize_for_type(type));
      Op op = TYPE_UNSIGNED(type) ? Op::ZEXT : Op::SEXT;
      tmp = bb->build_inst(op, tmp, inst->bitsize);
      bb->build_inst(Op::UB, bb->build_inst(Op::NE, inst, tmp));
      return;
    }

  if (TREE_CODE(type) == RECORD_TYPE)
    {
      for (tree fld = TYPE_FIELDS(type); fld; fld = DECL_CHAIN(fld))
	{
	  if (TREE_CODE(fld) != FIELD_DECL)
	    continue;
	  if (DECL_BIT_FIELD_TYPE(fld))
	    continue;
	  tree elem_type = TREE_TYPE(fld);
	  uint64_t elem_size = bytesize_for_type(elem_type);
	  if (elem_size == 0)
	    continue;
	  uint64_t elem_offset = get_int_cst_val(DECL_FIELD_OFFSET(fld));
	  elem_offset += get_int_cst_val(DECL_FIELD_BIT_OFFSET(fld)) / 8;
	  Inst *high = bb->value_inst((elem_offset + elem_size) * 8 - 1, 32);
	  Inst *low = bb->value_inst(elem_offset * 8, 32);
	  Inst *extract1 = bb->build_inst(Op::EXTRACT, inst, high, low);
	  Inst *extract2 = nullptr;
	  if (mem_flags)
	    extract2 = bb->build_inst(Op::EXTRACT, mem_flags, high, low);
	  constrain_src_value(extract1, elem_type, extract2);
	}
      return;
    }
  if (VECTOR_TYPE_P(type)
      || TREE_CODE(type) == COMPLEX_TYPE
      || TREE_CODE(type) == ARRAY_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      uint32_t elem_bitsize = bytesize_for_type(elem_type) * 8;
      assert(inst->bitsize % elem_bitsize == 0);
      assert(inst->bitsize == bitsize_for_type(type));
      uint32_t nof_elt = inst->bitsize / elem_bitsize;
      for (uint64_t i = 0; i < nof_elt; i++)
	{
	  Inst *extract = extract_vec_elem(bb, inst, elem_bitsize, i);
	  Inst *extract2 = nullptr;
	  if (mem_flags)
	    extract2 = extract_vec_elem(bb, mem_flags, elem_bitsize, i);
	  constrain_src_value(extract, elem_type, extract2);
	}
      return;
    }
}

Inst *Converter::build_memory_inst(uint64_t id, uint64_t size, uint32_t flags)
{
  Basic_block *entry_bb = func->bbs[0];
  Inst *arg1 = entry_bb->value_inst(id, module->ptr_id_bits);
  Inst *arg2 = entry_bb->value_inst(size, module->ptr_offset_bits);
  Inst *arg3 = entry_bb->value_inst(flags, 32);
  return entry_bb->build_inst(Op::MEMORY, arg1, arg2, arg3);
}

void Converter::build_ub_if_not_zero(Inst *inst, Inst *cmp)
{
  Inst *zero = bb->value_inst(0, inst->bitsize);
  Inst *is_ub = bb->build_inst(Op::NE, inst, zero);
  if (cmp)
    is_ub = bb->build_inst(Op::AND, is_ub, cmp);
  bb->build_inst(Op::UB, is_ub);
}

void Converter::constrain_range(Basic_block *bb, tree expr, Inst *inst, Inst *indef)
{
  assert(TREE_CODE(expr) == SSA_NAME);

  // The constraints are added when we create a inst for the expr, so the work
  // is already done if tree2instruction contains this expr.
  if (tree2instruction.contains(expr))
    return;

  prange pr;
  if (pr.supports_type_p(TREE_TYPE(expr)))
    {
      if (!get_range_query(cfun)->range_of_expr(pr, expr))
	return;
      if (pr.undefined_p() || pr.varying_p())
	return;
      if (pr.nonzero_p())
	{
	  Inst *zero = bb->value_inst(0, inst->bitsize);
	  Inst *is_ub = bb->build_inst(Op::EQ, inst, zero);
	  if (indef)
	    {
	      Inst *cmp = bb->build_inst(Op::EQ, indef, zero);
	      is_ub = bb->build_inst(Op::AND, is_ub, cmp);
	    }
	  bb->build_inst(Op::UB, is_ub);
	}
      return;
    }

  int_range_max r;
  if (!r.supports_type_p(TREE_TYPE(expr)))
    return;
  if (!get_range_query(cfun)->range_of_expr(r, expr))
    return;
  if (r.undefined_p() || r.varying_p())
    return;

  // TODO: Implement wide types.
  if (inst->bitsize > 128)
    return;

  // TODO: get_nonzero_bits is deprecated if I understand correctly. This
  // should be updated to the new API.
  Inst *is_ub1 = nullptr;
  wide_int nz = r.get_nonzero_bits();
  if (nz != -1)
    {
      unsigned __int128 nonzero_bits = get_wide_int_val(nz);
      // The SMT solver get confused, and becomes much slower, when we have
      // both a mask and a range describing the same value. We therefore
      // skip adding a check for the mask if it does not constrain the value
      // more than what the range does.
      // TODO: Implement this for real. For now, we just assume that a mask
      // representing the top n bits as zero is fully represented by the
      // range.
      if (clz(nonzero_bits) + popcount(nonzero_bits) != 128)
	{
	  Inst *mask = bb->value_inst(~nonzero_bits, inst->bitsize);
	  Inst *bits = bb->build_inst(Op::AND, inst, mask);
	  Inst *zero = bb->value_inst(0, bits->bitsize);
	  is_ub1 = bb->build_inst(Op::NE, bits, zero);
	}
    }

  Inst *is_ub2 = nullptr;
  for (unsigned i = 0; i < r.num_pairs(); i++)
    {
      unsigned __int128 low_val = get_wide_int_val(r.lower_bound(i));
      unsigned __int128 high_val = get_wide_int_val(r.upper_bound(i));
      Inst *is_not_in_range;
      if (low_val == high_val)
	{
	  Inst *val = bb->value_inst(low_val, inst->bitsize);
	  is_not_in_range = bb->build_inst(Op::NE, inst, val);
	}
      else
	{
	  Inst *low = bb->value_inst(low_val, inst->bitsize);
	  Inst *high = bb->value_inst(high_val, inst->bitsize);
	  Op op = TYPE_UNSIGNED(TREE_TYPE(expr)) ? Op::ULT : Op::SLT;
	  Inst *cmp_low = bb->build_inst(op, inst, low);
	  Inst *cmp_high = bb->build_inst(op, high, inst);
	  is_not_in_range = bb->build_inst(Op::OR, cmp_low, cmp_high);
	}
      if (is_ub2)
	is_ub2 = bb->build_inst(Op::AND, is_not_in_range, is_ub2);
      else
	is_ub2 = is_not_in_range;
    }
  assert(is_ub2 != nullptr);

  // Ranges do not take uninitialized values into account, so, e.g., a phi
  // node may receive a range, even if one of its arguments is uninitialized.
  // We therefore need to filter out the indefinite cases from the check,
  // otherwise, we will report miscompilation for
  //
  //   int k;
  //   void foo (int x) {
  //     int y;
  //     if (x == 0)
  //       y = 1;
  //     else if (x == 1)
  //       y = 2;
  //     k = y;
  //   }
  //
  // This is safe because any use of the value that can use the range info
  // will be marked as UB if an uninitialized value is used in that operation.
  // TODO: This is not completely true since some recent changes. The
  // semantics need to be better specified.
  if (indef)
    {
      Inst *zero = bb->value_inst(0, indef->bitsize);
      Inst *cmp = bb->build_inst(Op::EQ, indef, zero);
      if (is_ub1)
	is_ub1 = bb->build_inst(Op::AND, is_ub1, cmp);
      is_ub2 = bb->build_inst(Op::AND, is_ub2, cmp);
    }

  if (is_ub1)
    bb->build_inst(Op::UB, is_ub1);
  bb->build_inst(Op::UB, is_ub2);
}

void Converter::store_ub_check(Inst *ptr, Inst *prov, uint64_t size, Inst *cond)
{
  // It is UB to write to constant memory.
  Inst *is_const = bb->build_inst(Op::IS_CONST_MEM, prov);
  if (cond)
    is_const = bb->build_inst(Op::AND, is_const, cond);
  bb->build_inst(Op::UB, is_const);

  load_ub_check(ptr, prov, size, cond);
}

void Converter::load_ub_check(Inst *ptr, Inst *prov, uint64_t size, Inst *cond)
{
  // It is UB if the pointer provenance does not correspond to the address.
  Inst *ptr_mem_id = extract_id(ptr);
  Inst *is_ub = bb->build_inst(Op::NE, prov, ptr_mem_id);
  if (cond)
    is_ub = bb->build_inst(Op::AND, is_ub, cond);
  bb->build_inst(Op::UB, is_ub);

  if (size != 0)
    {
      // It is UB if the size overflows the offset field.
      Inst *size_inst = bb->value_inst(size - 1, ptr->bitsize);
      Inst *end = bb->build_inst(Op::ADD, ptr, size_inst);
      Inst *end_mem_id = extract_id(end);
      Inst *overflow = bb->build_inst(Op::NE, prov, end_mem_id);
      if (cond)
	overflow = bb->build_inst(Op::AND, overflow, cond);
      bb->build_inst(Op::UB, overflow);

      // It is UB if the end is outside the memory object.
      // Note: ptr is within the memory object; otherwise, the provenance check
      // or the offset overflow check would have failed.
      Inst *mem_size = bb->build_inst(Op::GET_MEM_SIZE, prov);
      Inst *offset = bb->build_extract_offset(end);
      Inst *out_of_bound = bb->build_inst(Op::ULE, mem_size, offset);
      if (cond)
	out_of_bound = bb->build_inst(Op::AND, out_of_bound, cond);
      bb->build_inst(Op::UB, out_of_bound);
    }
  else
    {
      // The pointer must point to valid memory, or be one position past
      // valid memory.
      // TODO: Handle zero-sized memory blocks (such as malloc(0)).
      Inst *mem_size = bb->build_inst(Op::GET_MEM_SIZE, prov);
      Inst *offset = bb->build_extract_offset(ptr);
      Inst *out_of_bound = bb->build_inst(Op::ULT, mem_size, offset);
      if (cond)
	out_of_bound = bb->build_inst(Op::AND, out_of_bound, cond);
      bb->build_inst(Op::UB, out_of_bound);
    }
}

void Converter::load_vec_ub_check(Inst *ptr, Inst *prov, uint64_t size, tree expr)
{
  tree type = TREE_TYPE(expr);
  assert(VECTOR_TYPE_P(type));
  tree elem_type = TREE_TYPE(type);
  uint64_t alignment = get_object_alignment(expr) / 8;
  uint64_t elem_size = bytesize_for_type(elem_type);

  // It is UB if the pointer provenance does not correspond to the address.
  Inst *ptr_mem_id = extract_id(ptr);
  Inst *is_ub = bb->build_inst(Op::NE, prov, ptr_mem_id);
  bb->build_inst(Op::UB, is_ub);

  // It is UB if the size overflows the offset field.
  Inst *size_inst = bb->value_inst(size - 1, ptr->bitsize);
  Inst *end = bb->build_inst(Op::ADD, ptr, size_inst);
  Inst *end_mem_id = extract_id(end);
  Inst *overflow = bb->build_inst(Op::NE, prov, end_mem_id);
  bb->build_inst(Op::UB, overflow);

  // A vector load may read outside the object, as long as the first element
  // is within the object and the rest of the vector does not cross a page
  // boundary. The compiler does not, in general, know where the page
  // boundaries are, so in practice this means that out-of-bounds reads are
  // valid as long as the extra bytes are within the same alignment line as
  // the last valid byte.
  // TODO: Should the bytes read out of bounds be marked as indef?
  if (size <= alignment)
    {
      size_inst = bb->value_inst(elem_size - 1, ptr->bitsize);
      end = bb->build_inst(Op::ADD, ptr, size_inst);
    }
  Inst *mem_size = bb->build_inst(Op::GET_MEM_SIZE, prov);
  Inst *offset = bb->build_extract_offset(end);
  Inst *out_of_bound = bb->build_inst(Op::ULE, mem_size, offset);
  bb->build_inst(Op::UB, out_of_bound);
}

// Mark the execution as UB if the source and destination ranges overlap
// unless they are identical.
void Converter::overlap_ub_check(Inst *src_ptr, Inst *dst_ptr, uint64_t size)
{
  if (size <= 1)
    return;

  Inst *size_inst = bb->value_inst(size - 1, src_ptr->bitsize);
  Inst *src_end = bb->build_inst(Op::ADD, src_ptr, size_inst);
  Inst *dst_end = bb->build_inst(Op::ADD, dst_ptr, size_inst);

  Inst *cond1 = bb->build_inst(Op::ULT, src_ptr, dst_ptr);
  Inst *cond2 = bb->build_inst(Op::ULE, dst_ptr, src_end);
  bb->build_inst(Op::UB, bb->build_inst(Op::AND, cond1, cond2));

  Inst *cond3 = bb->build_inst(Op::ULT, dst_ptr, src_ptr);
  Inst *cond4 = bb->build_inst(Op::ULE, src_ptr, dst_end);
  bb->build_inst(Op::UB, bb->build_inst(Op::AND, cond3, cond4));
}

std::pair<Inst *, Inst *> Converter::to_mem_repr(Inst *inst, Inst *indef, tree type)
{
  uint64_t bitsize = bytesize_for_type(type) * 8;
  if (inst->bitsize == bitsize)
    return {inst, indef};

  assert(inst->bitsize < bitsize);
  if (INTEGRAL_TYPE_P(type))
    {
      Op op = TYPE_UNSIGNED(type) ? Op::ZEXT : Op::SEXT;
      inst = bb->build_inst(op, inst, bitsize);
      if (indef)
	indef = bb->build_inst(Op::SEXT, indef, bitsize);
    }
  return {inst, indef};
}

Inst *Converter::to_mem_repr(Inst *inst, tree type)
{
  auto [new_inst, indef] = to_mem_repr(inst, nullptr, type);
  return new_inst;
}

// TODO: Imput does not necessaily be mem_repr -- it can be BITFIELD_REF reads
// from vector elements. So should probably be "to_ir_repr" or similar.
std::pair<Inst *, Inst *> Converter::from_mem_repr(Inst *inst, Inst *indef, tree type)
{
  uint64_t bitsize = bitsize_for_type(type);
  assert(bitsize <= inst->bitsize);
  if (inst->bitsize == bitsize)
    return {inst, indef};

  inst = bb->build_trunc(inst, bitsize);
  if (indef)
    indef = bb->build_trunc(indef, bitsize);

  return {inst, indef};
}

Inst *Converter::from_mem_repr(Inst *inst, tree type)
{
  auto [new_inst, indef] = from_mem_repr(inst, nullptr, type);
  return new_inst;
}

// Helper function to padding_at_offset.
// TODO: Implement a sane version. And test.
uint8_t Converter::bitfield_padding_at_offset(tree fld, int64_t offset)
{
  uint8_t used_bits = 0;
  for (; fld; fld = DECL_CHAIN(fld))
    {
      if (TREE_CODE(fld) != FIELD_DECL)
	continue;

      if (!DECL_BIT_FIELD_TYPE(fld))
	break;

      tree elem_type = TREE_TYPE(fld);
      int64_t elem_bit_size = bitsize_for_type(elem_type);
      if (elem_bit_size == 0)
	continue;

      int64_t elem_size = bytesize_for_type(elem_type);
      int64_t elem_offset = get_int_cst_val(DECL_FIELD_OFFSET(fld));
      int64_t elem_bit_offset = get_int_cst_val(DECL_FIELD_BIT_OFFSET(fld));
      elem_offset += elem_bit_offset / 8;
      elem_bit_offset &= 7;
      elem_size = (elem_bit_offset + elem_bit_size + 7) / 8;
      if (elem_offset <= offset && offset < (elem_offset + elem_size))
	{
	  if (elem_offset < offset)
	    {
	      elem_bit_size -= 8 - elem_bit_offset;
	      elem_bit_offset = 0;
	      elem_offset += 1;
	      if (elem_bit_size < 0)
		continue;
	    }

	  if (elem_offset < offset)
	    {
	      assert(elem_bit_offset == 0);
	      elem_bit_size -= 8 * (offset - elem_offset);
	      if (elem_bit_size < 0)
		continue;
	    }

	  if (elem_bit_size > 8)
	    elem_bit_size = 8;

	  used_bits |= ((1 << elem_bit_size) - 1) << elem_bit_offset;
	}
    }
  return ~used_bits;
}

// Return a bitmask telling which bits are padding (i.e., where the value is
// unspecified) for an offset into the type.
uint8_t Converter::padding_at_offset(tree type, uint64_t offset)
{
  if (TREE_CODE(type) == ARRAY_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      uint64_t elem_size = bytesize_for_type(elem_type);
      if (elem_size == 0)
	throw Not_implemented("padding_at_offset: zero-sized array element");
      return padding_at_offset(elem_type, offset % elem_size);
    }
  if (TREE_CODE(type) == RECORD_TYPE)
    {
      for (tree fld = TYPE_FIELDS(type); fld; fld = DECL_CHAIN(fld))
	{
	  if (TREE_CODE(fld) != FIELD_DECL)
	    continue;
	  tree elem_type = TREE_TYPE(fld);
	  uint64_t elem_size = bytesize_for_type(elem_type);
	  uint64_t elem_offset = get_int_cst_val(DECL_FIELD_OFFSET(fld));
	  uint64_t elem_bit_offset =
	    get_int_cst_val(DECL_FIELD_BIT_OFFSET(fld));
	  elem_offset += elem_bit_offset / 8;
	  elem_bit_offset &= 7;
	  if (DECL_BIT_FIELD_TYPE(fld))
	    {
	      uint64_t elem_bit_size = bitsize_for_type(elem_type);
	      elem_size = (elem_bit_offset + elem_bit_size + 7) / 8;
	      if (elem_offset <= offset && offset < (elem_offset + elem_size))
		return bitfield_padding_at_offset(fld, offset);
	    }
	  else if (elem_offset <= offset && offset < (elem_offset + elem_size))
	    {
	      uint8_t padding =
		padding_at_offset(elem_type, offset - elem_offset);
	      // Record types in IR generated from C++ may have overlapping
	      // fields where one field is an empty record. We therefore
	      // continue iterating if we got a padding 0xff to try to find
	      // a following "real" field.
	      if (padding != 0xff)
		return padding;
	    }
	}
      return 0xff;
    }
  if (TREE_CODE(type) == UNION_TYPE)
    {
      // For unions, we mark it as padding if it is padding in all elements.
      uint8_t padding = 0xff;
      for (tree fld = TYPE_FIELDS(type); fld; fld = DECL_CHAIN(fld))
	{
	  if (TREE_CODE(fld) != FIELD_DECL)
	    continue;
	  tree elem_type = TREE_TYPE(fld);
	  padding &= padding_at_offset(elem_type, offset);
	}
      return padding;
    }

  // The other bytes does not have padding (well, Booleans sort of have
  // padding, but the padding must be 0 so it is defined).
  return 0;
}

bool Converter::is_extracting_from_value(tree expr)
{
  if (tree2instruction.contains(expr))
    return true;
  if (TREE_CODE(expr) == SSA_NAME)
    return true;
  if (TREE_CODE(expr) == COMPONENT_REF)
    return is_extracting_from_value(TREE_OPERAND(expr, 0));
  return false;
}

std::tuple<Inst *, Inst *, Inst *> Converter::extract_component_ref(tree expr)
{
  tree object = TREE_OPERAND(expr, 0);
  tree field = TREE_OPERAND(expr, 1);
  tree type = TREE_TYPE(expr);
  auto [inst, indef, prov] = tree2inst_indef_prov(object);

  // TODO: This will need implementation of index checking in variably sized
  //       array too, otherwise we will fail to catch when k is too big in
  //         struct A {int c[k]; int x[n];};
  //       from gcc.dg/pr51628-18.c.
  if (TREE_CODE(DECL_FIELD_OFFSET(field)) != INTEGER_CST)
    throw Not_implemented("process_component_ref: non-constant field offset");
  uint64_t offset = get_int_cst_val(DECL_FIELD_OFFSET(field));
  uint64_t bit_offset = get_int_cst_val(DECL_FIELD_BIT_OFFSET(field));
  uint64_t low_val = 8 * offset + bit_offset;
  if (is_bit_field(expr))
    {
      uint64_t high_val = low_val + bitsize_for_type(type) - 1;
      inst = bb->build_inst(Op::EXTRACT, inst, high_val, low_val);
      if (indef)
	indef = bb->build_inst(Op::EXTRACT, indef, high_val, low_val);
    }
  else
    {
      uint64_t high_val = low_val + bytesize_for_type(type) * 8 - 1;
      inst = bb->build_inst(Op::EXTRACT, inst, high_val, low_val);
      if (indef)
	indef = bb->build_inst(Op::EXTRACT, indef, high_val, low_val);
      constrain_src_value(inst, type);
      inst = bb->build_trunc(inst, bitsize_for_type(type));
      if (indef)
	indef = bb->build_trunc(indef, bitsize_for_type(type));
    }
  if (!prov && POINTER_TYPE_P(type))
    prov = extract_id(inst);
  return {inst, indef, prov};
}

std::tuple<Inst *, Inst *, Inst *> Converter::tree2inst_indef_prov(tree expr)
{
  check_type(TREE_TYPE(expr));

  auto it = tree2instruction.find(expr);
  if (it != tree2instruction.end())
    {
      Inst *inst = it->second;
      Inst *indef = nullptr;
      auto it2 = tree2indef.find(expr);
      if (it2 != tree2indef.end())
	{
	  indef = it2->second;
	  assert(indef);
	  assert(indef->bitsize == inst->bitsize);
	}
      Inst *prov = nullptr;
      auto it3 = tree2prov.find(expr);
      if (it3 != tree2prov.end())
	{
	  prov = it3->second;
	  assert(prov);
	}
      return {inst, indef, prov};
    }

  switch (TREE_CODE(expr))
    {
    case SSA_NAME:
      {
	tree var = SSA_NAME_VAR(expr);
	if (var && TREE_CODE(var) == PARM_DECL)
	  {
	    if (tree2instruction.contains(var))
	      {
		Inst *inst = tree2instruction.at(var);

		// Place the range check in the entry block as it is
		// invalid to call the function with invalid values.
		// This solves the problem that we "randomly" could
		// mark execution as UB depending on where the param
		// were used when passes were sinking/hoisting the params.
		// See gcc.dg/analyzer/pointer-merging.c for a test where
		// this check makes a difference.
		constrain_range(func->bbs[0], expr, inst);

		assert(!POINTER_TYPE_P(TREE_TYPE(expr))
		       || tree2prov.contains(var));
		Inst *prov = nullptr;
		if (tree2prov.contains(var))
		  prov = tree2prov.at(var);

		return {inst, nullptr, prov};
	      }
	  }
	if (var && TREE_CODE(var) == VAR_DECL)
	  {
	    uint64_t bitsize = bitsize_for_type(TREE_TYPE(expr));
	    Inst *inst = bb->value_inst(0, bitsize);
	    Inst *indef = bb->value_m1_inst(bitsize);
	    Inst *prov = nullptr;
	    if (POINTER_TYPE_P(TREE_TYPE(expr)))
	      prov = extract_id(inst);
	    return {inst, indef, prov};
	  }
	throw Not_implemented("tree2inst: unhandled ssa_name");
      }
    case CONSTRUCTOR:
      if (!VECTOR_TYPE_P(TREE_TYPE(expr)))
	{
	  // Constructors are not supposed to reach this point as they are
	  // only used in
	  //   * store instructions
	  //   * when initializing global variables
	  //   * constructing vectors from scalars
	  throw Not_implemented("tree2inst: constructor");
	}
      return vector_constructor(expr);
    case INTEGER_CST:
      {
	uint32_t precision = bitsize_for_type(TREE_TYPE(expr));
	Inst *inst = nullptr;
	Inst *prov = nullptr;
	int remaining = precision;
	for (int i = 0; i < TREE_INT_CST_NUNITS(expr); i++)
	  {
	    uint64_t elt_value = TREE_INT_CST_ELT(expr, i);
	    Inst *elt = bb->value_inst(elt_value, std::min(remaining, 64));
	    remaining -= elt->bitsize;
	    if (inst)
	      inst = bb->build_inst(Op::CONCAT, elt, inst);
	    else
	      inst = elt;
	  }
	if (inst->bitsize < precision)
	  inst = bb->build_inst(Op::SEXT, inst, precision);

	if (POINTER_TYPE_P(TREE_TYPE(expr)))
	  prov = extract_id(inst);

	return {inst, nullptr, prov};
      }
    case POLY_INT_CST:
      {
	uint32_t precision = bitsize_for_type(TREE_TYPE(expr));
	unsigned __int128 value = get_int_cst_val(expr);
	Inst *inst = bb->value_inst(value, precision);
	Inst *prov = nullptr;
	if (POINTER_TYPE_P(TREE_TYPE(expr)))
	  {
	    uint32_t ptr_id_bits = module->ptr_id_bits;
	    uint32_t ptr_id_low = module->ptr_id_low;
	    uint64_t id = (value >> ptr_id_low) & ((1 << ptr_id_bits) - 1);
	    prov = bb->value_inst(id, ptr_id_bits);
	  }
	return {inst, nullptr, prov};
      }
    case REAL_CST:
      {
	tree type = TREE_TYPE(expr);
	check_type(type);
	int nof_bytes = GET_MODE_SIZE(SCALAR_FLOAT_TYPE_MODE(type));
	Inst *res;
	if (REAL_VALUE_ISNAN(TREE_REAL_CST(expr)))
	  res = bb->build_inst(Op::NAN, TYPE_PRECISION(type));
	else
	  {
	    assert(nof_bytes <= 16);
	    long buf[4];
	    real_to_target(buf, TREE_REAL_CST_PTR(expr), TYPE_MODE(type));
	    union {
	      uint32_t buf[4];
	      unsigned __int128 i;
	    } u;
	    // real_to_target saves 32 bits in each long, so we copy the
	    // values to a uint32_t array to get rid of the extra bits.
	    // TODO: Big endian.
	    for (int i = 0; i < 4; i++)
	      u.buf[i] = buf[i];
	    res = bb->value_inst(u.i, TYPE_PRECISION(type));
	  }
	return {res, nullptr, nullptr};
      }
    case VECTOR_CST:
      {
	uint32_t bitsize = bitsize_for_type(TREE_TYPE(expr));
	uint32_t elem_bitsize = bitsize_for_type(TREE_TYPE(TREE_TYPE(expr)));
	uint32_t nof_elem = bitsize / elem_bitsize;
	Inst *ret = tree2inst(VECTOR_CST_ELT(expr, 0));
	for (uint32_t i = 1; i < nof_elem; i++)
	  {
	    Inst *elem = tree2inst(VECTOR_CST_ELT(expr, i));
	    ret = bb->build_inst(Op::CONCAT, elem, ret);
	  }
	return {ret, nullptr, nullptr};
      }
    case COMPLEX_CST:
      {
	tree elem_type = TREE_TYPE(TREE_TYPE(expr));
	Inst *real = tree2inst(TREE_REALPART(expr));
	real = to_mem_repr(real, elem_type);
	Inst *imag = tree2inst(TREE_IMAGPART(expr));
	imag = to_mem_repr(imag, elem_type);
	Inst *res = bb->build_inst(Op::CONCAT, imag, real);
	return {res, nullptr, nullptr};
      }
    case IMAGPART_EXPR:
      {
	tree elem_type = TREE_TYPE(expr);
	auto [arg, indef] = tree2inst_indef(TREE_OPERAND(expr, 0));
	Inst *high = bb->value_inst(arg->bitsize - 1, 32);
	Inst *low = bb->value_inst(arg->bitsize / 2, 32);
	Inst *res = bb->build_inst(Op::EXTRACT, arg, high, low);
	if (indef)
	  indef = bb->build_inst(Op::EXTRACT, indef, high, low);
	std::tie(res, indef) = from_mem_repr(res, indef, elem_type);
	return {res, indef, nullptr};
      }
    case REALPART_EXPR:
      {
	tree elem_type = TREE_TYPE(expr);
	auto [arg, indef] = tree2inst_indef(TREE_OPERAND(expr, 0));
	Inst *res = bb->build_trunc(arg, arg->bitsize / 2);
	if (indef)
	  indef = bb->build_trunc(indef, arg->bitsize / 2);
	std::tie(res, indef) = from_mem_repr(res, indef, elem_type);
	return {res, indef, nullptr};
      }
    case VIEW_CONVERT_EXPR:
      {
	auto [arg, indef, prov] = tree2inst_indef_prov(TREE_OPERAND(expr, 0));
	tree src_type = TREE_TYPE(TREE_OPERAND(expr, 0));
	tree dest_type = TREE_TYPE(expr);
	std::tie(arg, indef) = to_mem_repr(arg, indef, src_type);
	std::tie(arg, indef) = from_mem_repr(arg, indef, dest_type);
	constrain_src_value(arg, dest_type);
	if (POINTER_TYPE_P(dest_type))
	  {
	    assert(!POINTER_TYPE_P(src_type) || prov);
	    if (!prov)
	      prov = extract_id(arg);
	  }
	return {arg, indef, prov};
      }
    case ADDR_EXPR:
      {
	Addr addr = process_address(TREE_OPERAND(expr, 0), false);
	assert(!addr.bitoffset);
	return {addr.ptr, nullptr, addr.prov};
      }
    case BIT_FIELD_REF:
      {
	tree arg = TREE_OPERAND(expr, 0);
	if (TREE_CODE(arg) != SSA_NAME)
	  return process_load(expr);

	auto [value, indef, prov] = tree2inst_indef_prov(arg);
	uint64_t bitsize = get_int_cst_val(TREE_OPERAND(expr, 1));
	uint64_t bit_offset = get_int_cst_val(TREE_OPERAND(expr, 2));
	uint64_t hi = bitsize + bit_offset - 1;
	if (bit_offset >= value->bitsize || hi >= value->bitsize)
	  {
	    if (POINTER_TYPE_P(TREE_TYPE(expr)) && !prov)
	      prov = bb->value_inst(0, func->module->ptr_id_bits);
	    value = bb->value_inst(0, bitsize);
	    return {value, nullptr, prov};
	  }
	Inst *high = bb->value_inst(hi, 32);
	Inst *low = bb->value_inst(bit_offset, 32);
	value = to_mem_repr(value, TREE_TYPE(arg));
	value = bb->build_inst(Op::EXTRACT, value, high, low);
	if (indef)
	  indef = bb->build_inst(Op::EXTRACT, indef, high, low);
	std::tie(value, indef) = from_mem_repr(value, indef, TREE_TYPE(expr));
	if (POINTER_TYPE_P(TREE_TYPE(expr)) && !prov)
	  prov = extract_id(value);
	return {value, indef, prov};
      }
    case ARRAY_REF:
      {
	tree array = TREE_OPERAND(expr, 0);
	// Indexing element of a vector as vec[2] is done by an  ARRAY_REF of
	// a VIEW_CONVERT_EXPR of the vector.
	if (TREE_CODE(array) == VIEW_CONVERT_EXPR
	    && VECTOR_TYPE_P(TREE_TYPE(TREE_OPERAND(array, 0))))
	  return vector_as_array(expr);
	return process_load(expr);
      }
    case COMPONENT_REF:
      if (is_extracting_from_value(expr))
	return extract_component_ref(expr);
      else
	return process_load(expr);
    case MEM_REF:
    case TARGET_MEM_REF:
    case VAR_DECL:
    case RESULT_DECL:
      return process_load(expr);
    default:
      {
	const char *name = get_tree_code_name(TREE_CODE(expr));
	throw Not_implemented("tree2inst: "s + name);
      }
    }
}

std::pair<Inst *, Inst *> Converter::tree2inst_prov(tree expr)
{
  auto [inst, indef, prov] = tree2inst_indef_prov(expr);
  if (indef)
    build_ub_if_not_zero(indef);
  return {inst, prov};
}

std::pair<Inst *, Inst *> Converter::tree2inst_indef(tree expr)
{
  auto [inst, indef, _] = tree2inst_indef_prov(expr);
  return {inst, indef};
}

Inst *Converter::tree2inst(tree expr)
{
  auto [inst, indef, _] = tree2inst_indef_prov(expr);
  if (indef)
    build_ub_if_not_zero(indef);
  return inst;
}

// Processing constructors may give us more complex expr than what we get
// from normal operations. For example, initializing an array of pointers
// may have an initializer &a-&b that in the function body would be
// calculated by its own stmt.
std::tuple<Inst *, Inst *, Inst *> Converter::tree2inst_constructor(tree expr)
{
  check_type(TREE_TYPE(expr));

  tree_code code = TREE_CODE(expr);
  if (TREE_OPERAND_LENGTH(expr) == 2)
    {
      tree arg1_expr = TREE_OPERAND(expr, 0);
      tree arg2_expr = TREE_OPERAND(expr, 1);
      tree arg1_type = TREE_TYPE(arg1_expr);
      tree arg2_type = TREE_TYPE(arg2_expr);
      auto [arg1, arg1_indef, arg1_prov] = tree2inst_constructor(arg1_expr);
      auto [arg2, arg2_indef, arg2_prov] = tree2inst_constructor(arg2_expr);
      return process_binary_scalar(code, arg1, arg1_indef, arg1_prov, arg2,
				   arg2_indef, arg2_prov, TREE_TYPE(expr),
				   arg1_type, arg2_type);
    }
  switch (code)
    {
    case ABS_EXPR:
    case ABSU_EXPR:
    case BIT_NOT_EXPR:
    case NEGATE_EXPR:
    case NOP_EXPR:
    case CONVERT_EXPR:
      {
	tree arg_expr = TREE_OPERAND(expr, 0);
	auto [arg, arg_indef, arg_prov] = tree2inst_constructor(arg_expr);
	return process_unary_scalar(code, arg, arg_indef, arg_prov,
				    TREE_TYPE(expr), TREE_TYPE(arg_expr));
      }
    default:
      return tree2inst_indef_prov(expr);
    }
}

Inst *Converter::get_res_indef(Inst *arg1_indef, tree lhs_type)
{
  Inst *res_indef = nullptr;
  if (arg1_indef)
    {
      Inst *zero = bb->value_inst(0, arg1_indef->bitsize);
      res_indef = bb->build_inst(Op::NE, arg1_indef, zero);
      uint64_t bitsize = bitsize_for_type(lhs_type);
      if (bitsize > res_indef->bitsize)
	res_indef = bb->build_inst(Op::SEXT, res_indef, bitsize);
    }
  return res_indef;
}

Inst *Converter::get_res_indef(Inst *arg1_indef, Inst *arg2_indef, tree lhs_type)
{
  Inst *res_indef = nullptr;
  if (arg1_indef)
    {
      Inst *zero = bb->value_inst(0, arg1_indef->bitsize);
      res_indef = bb->build_inst(Op::NE, arg1_indef, zero);
    }
  if (arg2_indef)
    {
      Inst *zero = bb->value_inst(0, arg2_indef->bitsize);
      Inst *tmp = bb->build_inst(Op::NE, arg2_indef, zero);
      if (res_indef)
	res_indef = bb->build_inst(Op::OR, res_indef, tmp);
      else
	res_indef = tmp;
    }
  if (res_indef)
    {
      uint64_t bitsize = bitsize_for_type(lhs_type);
      if (bitsize > res_indef->bitsize)
	res_indef = bb->build_inst(Op::SEXT, res_indef, bitsize);
    }
  return res_indef;
}

Inst *Converter::get_res_indef(Inst *arg1_indef, Inst *arg2_indef, Inst *arg3_indef, tree lhs_type)
{
  Inst *res_indef = nullptr;
  if (arg1_indef)
    {
      Inst *zero = bb->value_inst(0, arg1_indef->bitsize);
      res_indef = bb->build_inst(Op::NE, arg1_indef, zero);
    }
  if (arg2_indef)
    {
      Inst *zero = bb->value_inst(0, arg2_indef->bitsize);
      Inst *tmp = bb->build_inst(Op::NE, arg2_indef, zero);
      if (res_indef)
	res_indef = bb->build_inst(Op::OR, res_indef, tmp);
      else
	res_indef = tmp;
    }
  if (arg3_indef)
    {
      Inst *zero = bb->value_inst(0, arg3_indef->bitsize);
      Inst *tmp = bb->build_inst(Op::NE, arg3_indef, zero);
      if (res_indef)
	res_indef = bb->build_inst(Op::OR, res_indef, tmp);
      else
	res_indef = tmp;
    }
  if (res_indef)
    {
      uint64_t bitsize = bitsize_for_type(lhs_type);
      if (bitsize > res_indef->bitsize)
	res_indef = bb->build_inst(Op::SEXT, res_indef, bitsize);
    }
  return res_indef;
}

Addr Converter::process_array_ref(tree expr, bool is_mem_access)
{
  tree array = TREE_OPERAND(expr, 0);
  tree index = TREE_OPERAND(expr, 1);
  tree array_type = TREE_TYPE(array);
  tree elem_type = TREE_TYPE(array_type);
  tree domain = TYPE_DOMAIN(array_type);

  Addr addr = process_address(array, is_mem_access);
  assert(!addr.bitoffset);
  Inst *idx = tree2inst(index);
  if (!TYPE_UNSIGNED(TREE_TYPE(index)))
    {
      Inst *zero = bb->value_inst(0, idx->bitsize);
      bb->build_inst(Op::UB, bb->build_inst(Op::SLT, idx, zero));
    }
  if (idx->bitsize < addr.ptr->bitsize)
    idx = bb->build_inst(Op::ZEXT, idx, addr.ptr->bitsize);
  else if (idx->bitsize > addr.ptr->bitsize)
    {
      Inst *high = bb->value_inst(idx->bitsize - 1, 32);
      Inst *low = bb->value_inst(addr.ptr->bitsize, 32);
      Inst *top = bb->build_inst(Op::EXTRACT, idx, high, low);
      Inst *zero = bb->value_inst(0, top->bitsize);
      bb->build_inst(Op::UB, bb->build_inst(Op::NE, top, zero));
      idx = bb->build_trunc(idx, addr.ptr->bitsize);
    }

  uint64_t elem_size = bytesize_for_type(elem_type);
  Inst *elm_size = bb->value_inst(elem_size, idx->bitsize);
  Inst *offset = bb->build_inst(Op::MUL, idx, elm_size);
  Inst *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);

  Inst *max_inst = nullptr;
  if (domain && TYPE_MAX_VALUE(domain))
    {
      if (!integer_zerop(TYPE_MIN_VALUE(domain)))
	throw Not_implemented("process_array_ref: index TYPE_MIN_VALUE != 0");
      tree max = TYPE_MAX_VALUE(domain);
      // TODO: Handle variable size arrays. This is (currently) not needed
      //       for correctness -- the array is its own object or last in a
      //       structure, so overflow is detected on the memory block level.
      //       But it will start failing when we support non-constant
      //       field offsets in structures.
      if (TREE_CODE(max) == INTEGER_CST)
	{
	  uint64_t max_val = get_int_cst_val(max);
	  max_inst = bb->value_inst(max_val, idx->bitsize);
	}
    }

  // Check that the index is within range. If the array has no max index,
  // we instead check that it fits within the maximal object size.
  if (max_inst && !array_ref_flexible_size_p(expr))
    {
      if (is_value_m1(max_inst))
	bb->build_inst(Op::UB, bb->value_inst(1, 1));
      else
	bb->build_inst(Op::UB, bb->build_inst(Op::ULT, max_inst, idx));
    }
  else
    {
      Inst *eidx = bb->build_inst(Op::ZEXT, idx, ptr->bitsize * 2);
      Inst *eelm_size = bb->value_inst(elem_size, ptr->bitsize * 2);
      Inst *eoffset = bb->build_inst(Op::MUL, eidx, eelm_size);
      Inst *emax_offset =
	bb->value_inst((uint64_t)1 << module->ptr_offset_bits,
		       eoffset->bitsize);
      bb->build_inst(Op::UB, bb->build_inst(Op::ULE, emax_offset, eoffset));
    }
  return {ptr, 0, addr.prov};
}

Addr Converter::process_component_ref(tree expr, bool is_mem_access)
{
  tree object = TREE_OPERAND(expr, 0);
  tree field = TREE_OPERAND(expr, 1);

  // TODO: This will need implementation of index checking in variably sized
  //       array too, otherwise we will fail to catch when k is too big in
  //         struct A {int c[k]; int x[n];};
  //       from gcc.dg/pr51628-18.c.
  if (TREE_CODE(DECL_FIELD_OFFSET(field)) != INTEGER_CST)
    throw Not_implemented("process_component_ref: non-constant field offset");
  uint64_t offset = get_int_cst_val(DECL_FIELD_OFFSET(field));
  uint64_t bit_offset = get_int_cst_val(DECL_FIELD_BIT_OFFSET(field));
  offset += bit_offset / 8;
  bit_offset &= 7;

  Addr addr = process_address(object, is_mem_access);
  assert(!addr.bitoffset);
  Inst *off = bb->value_inst(offset, addr.ptr->bitsize);
  Inst *ptr = bb->build_inst(Op::ADD, addr.ptr, off);

  return {ptr, bit_offset, addr.prov};
}

Addr Converter::process_bit_field_ref(tree expr, bool is_mem_access)
{
  tree object = TREE_OPERAND(expr, 0);
  tree position = TREE_OPERAND(expr, 2);
  uint64_t bit_offset = get_int_cst_val(position);
  Addr addr = process_address(object, is_mem_access);
  assert(!addr.bitoffset);
  Inst *ptr = addr.ptr;
  if (bit_offset > 7)
    {
      uint64_t offset = bit_offset / 8;
      Inst *off = bb->value_inst(offset, ptr->bitsize);
      ptr = bb->build_inst(Op::ADD, ptr, off);
      bit_offset &= 7;
    }
  return {ptr, bit_offset, addr.prov};
}

void Converter::alignment_check(tree expr, Inst *ptr)
{
  uint32_t alignment = get_object_alignment(expr) / 8;
  if (alignment > 1)
    {
      assert((alignment & (alignment - 1)) == 0);
      Inst *extract = bb->build_trunc(ptr, __builtin_ctz(alignment));
      Inst *zero = bb->value_inst(0, extract->bitsize);
      Inst *cond = bb->build_inst(Op::NE, extract, zero);
      bb->build_inst(Op::UB, cond);
    }
}

void Converter::process_decl(tree decl)
{
  if (DECL_REGISTER(decl))
    throw Not_implemented("process_decl: DECL_REGISTER variable");
  uint64_t size = bytesize_for_type(TREE_TYPE(decl));
  if (size >= ((uint64_t)1 << module->ptr_offset_bits))
    throw Not_implemented("process_decl: too large variable");
  if (size == 0)
    throw Not_implemented("process_decl: unknown size");
  uint64_t id;
  if (state->decl2id.contains(decl))
    id = state->decl2id.at(decl);
  else
    {
      if (DECL_ARTIFICIAL(decl) || !TREE_PUBLIC(decl))
	{
	  if (state->id_local <= state->ptr_id_min)
	    throw Not_implemented("process_decl: too many local variables");
	  id = --state->id_local;
	}
      else
	{
	  if (state->id_global >= state->ptr_id_max)
	    throw Not_implemented("process_decl: too many global variables");
	  id = ++state->id_global;
	}
      state->decl2id.insert({decl, id});
    }
  uint64_t flags = 0;
  if (TREE_READONLY(decl))
    flags |= MEM_CONST;
  if (auto_var_p(decl))
    {
      if (size > MAX_MEMORY_UNROLL_LIMIT)
	throw Not_implemented("process_decl: too large local variable");
      flags |= MEM_UNINIT;
    }
  Inst *memory_inst = build_memory_inst(id, size, flags);
  decl2instruction.insert({decl, memory_inst});
  if (TREE_READONLY(decl))
    init_var(decl, memory_inst);

  if (!auto_var_p(decl) && DECL_ASSEMBLER_NAME(decl))
    {
      const char *name = IDENTIFIER_POINTER(DECL_ASSEMBLER_NAME(decl));
      state->memory_objects.push_back({name, id, size, flags});
    }
}

Addr Converter::process_address(tree expr, bool is_mem_access)
{
  tree_code code = TREE_CODE(expr);
  if (code == MEM_REF)
    {
      auto [arg1, arg1_prov] = tree2inst_prov(TREE_OPERAND(expr, 0));
      assert(arg1_prov);
      Inst *arg2 = tree2inst(TREE_OPERAND(expr, 1));
      Inst *ptr = bb->build_inst(Op::ADD, arg1, arg2);
      if (is_mem_access)
	alignment_check(expr, ptr);
      return {ptr, 0, arg1_prov};
    }
  if (code == TARGET_MEM_REF)
    {
      // base + (step * index + index2 + offset)
      auto [base, base_prov] = tree2inst_prov(TREE_OPERAND(expr, 0));
      assert(base_prov);
      Inst *offset = tree2inst(TREE_OPERAND(expr, 1));
      Inst *off = offset;
      if (TREE_OPERAND(expr, 2))
	{
	  Inst *index = tree2inst(TREE_OPERAND(expr, 2));
	  if (TREE_OPERAND(expr, 3))
	    {
	      Inst *step = tree2inst(TREE_OPERAND(expr, 3));
	      index = bb->build_inst(Op::MUL, step, index);
	    }
	  off = bb->build_inst(Op::ADD, off, index);
	}
      if (TREE_OPERAND(expr, 4))
	{
	  Inst *index2 = tree2inst(TREE_OPERAND(expr, 4));
	  off = bb->build_inst(Op::ADD, off, index2);
	}
      Inst *ptr = bb->build_inst(Op::ADD, base, off);
      if (is_mem_access)
	alignment_check(expr, ptr);
      return {ptr, 0, base_prov};
    }
  if (code == VAR_DECL)
    {
      if (TREE_STATIC(expr) || DECL_EXTERNAL(expr))
	{
	  varpool_node *node = varpool_node::get(expr);
	  if (node)
	    expr = node->ultimate_alias_target()->decl;
	}

      if (!decl2instruction.contains(expr))
	process_decl(expr);
      Inst *ptr = decl2instruction.at(expr);
      assert(ptr->op == Op::MEMORY);
      Inst *id = extract_id(ptr);
      return {ptr, 0, id};
    }
  if (code == ARRAY_REF)
    return process_array_ref(expr, is_mem_access);
  if (code == COMPONENT_REF)
    return process_component_ref(expr, is_mem_access);
  if (code == BIT_FIELD_REF)
    return process_bit_field_ref(expr, is_mem_access);
  if (code == VIEW_CONVERT_EXPR)
    return process_address(TREE_OPERAND(expr, 0), is_mem_access);
  if (code == REALPART_EXPR)
    return process_address(TREE_OPERAND(expr, 0), is_mem_access);
  if (code == IMAGPART_EXPR)
    {
      Addr addr = process_address(TREE_OPERAND(expr, 0), is_mem_access);
      assert(!addr.bitoffset);
      uint64_t offset_val = bytesize_for_type(TREE_TYPE(expr));
      Inst *offset = bb->value_inst(offset_val, addr.ptr->bitsize);
      Inst *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);
      return {ptr, 0, addr.prov};
    }
  if (code == INTEGER_CST)
    {
      Inst *ptr = tree2inst(expr);
      Inst *prov = extract_id(ptr);
      return {ptr, 0, prov};
    }
  if (code == RESULT_DECL)
    {
      if (!decl2instruction.contains(expr))
	process_decl(expr);
      Inst *ptr = decl2instruction.at(expr);
      assert(ptr->op == Op::MEMORY);
      Inst *id = extract_id(ptr);
      return {ptr, 0, id};
    }

  const char *name = get_tree_code_name(TREE_CODE(expr));
  throw Not_implemented("process_address: "s + name);
}

std::tuple<Inst *, Inst *, Inst *> Converter::vector_as_array(tree expr)
{
  assert(TREE_CODE(expr) == ARRAY_REF);
  tree array = TREE_OPERAND(expr, 0);
  tree index = TREE_OPERAND(expr, 1);
  tree array_type = TREE_TYPE(array);
  tree elem_type = TREE_TYPE(array_type);
  assert(TREE_CODE(array) == VIEW_CONVERT_EXPR);
  tree vector_expr = TREE_OPERAND(array, 0);
  assert(VECTOR_TYPE_P(TREE_TYPE(vector_expr)));

  auto [inst, indef] = tree2inst_indef(vector_expr);

  uint64_t vector_size = bytesize_for_type(array_type);
  uint64_t elem_size = bytesize_for_type(elem_type);
  assert(vector_size % elem_size == 0);

  Inst *idx = tree2inst(index);
  Inst *nof_elems = bb->value_inst(vector_size / elem_size, idx->bitsize);
  Inst *cond = bb->build_inst(Op::ULE, nof_elems, idx);
  bb->build_inst(Op::UB, cond);

  Inst *elm_bitsize = bb->value_inst(elem_size * 8, idx->bitsize);
  Inst *shift = bb->build_inst(Op::MUL, idx, elm_bitsize);

  if (inst->bitsize > shift->bitsize)
    shift = bb->build_inst(Op::ZEXT, shift, inst->bitsize);
  else if (inst->bitsize < shift->bitsize)
    shift = bb->build_trunc(shift, inst->bitsize);
  inst = bb->build_inst(Op::LSHR, inst, shift);
  inst = bb->build_trunc(inst, elem_size * 8);
  if (indef)
    {
      indef = bb->build_inst(Op::LSHR, indef, shift);
      indef = bb->build_trunc(indef, elem_size * 8);
    }
  std::tie(inst, indef) = from_mem_repr(inst, indef, elem_type);

  return {inst, indef, nullptr};
}

std::tuple<Inst *, Inst *, Inst *> Converter::process_load(tree expr)
{
  if (reverse_storage_order_for_component_p(expr))
    throw Not_implemented("reverse storage order");

  tree type = TREE_TYPE(expr);
  uint64_t bitsize = bitsize_for_type(type);
  uint64_t size = bytesize_for_type(type);
  if (bitsize == 0)
    throw Not_implemented("process_load: load unhandled size 0");
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("process_load: load size too big");
  Addr addr = process_address(expr, true);
  bool is_bitfield = is_bit_field(expr);
  assert(is_bitfield || !addr.bitoffset);
  if (is_bitfield)
    size = (bitsize + addr.bitoffset + 7) / 8;
  if (VECTOR_TYPE_P(type))
    load_vec_ub_check(addr.ptr, addr.prov, size, expr);
  else
    load_ub_check(addr.ptr, addr.prov, size);
  Inst *value = nullptr;
  Inst *indef = nullptr;
  Inst *mem_flags2 = nullptr;
  for (uint64_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, addr.ptr->bitsize);
      Inst *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);

      Inst *data_byte;
      Inst *indef_byte;
      uint8_t padding = padding_at_offset(type, i);
      data_byte = bb->build_inst(Op::LOAD, ptr);
      indef_byte = bb->build_inst(Op::GET_MEM_INDEF, ptr);
      if (padding != 0)
	{
	  Inst *padding_inst = bb->value_inst(padding, 8);
	  indef_byte = bb->build_inst(Op::OR, indef_byte, padding_inst);
	}

      if (value)
	value = bb->build_inst(Op::CONCAT, data_byte, value);
      else
	value = data_byte;
      if (indef)
	indef = bb->build_inst(Op::CONCAT, indef_byte, indef);
      else
	indef = indef_byte;

      // TODO: Rename. This is not mem_flags -- we only splats one flag.
      Inst *flag = bb->build_inst(Op::GET_MEM_FLAG, ptr);
      flag = bb->build_inst(Op::SEXT, flag, 8);
      if (mem_flags2)
	mem_flags2 = bb->build_inst(Op::CONCAT, flag, mem_flags2);
      else
	mem_flags2 = flag;
    }
  if (is_bitfield)
    {
      Inst *high = bb->value_inst(bitsize + addr.bitoffset - 1, 32);
      Inst *low = bb->value_inst(addr.bitoffset, 32);
      value = bb->build_inst(Op::EXTRACT, value, high, low);
      indef = bb->build_inst(Op::EXTRACT, indef, high, low);
      mem_flags2 = bb->build_inst(Op::EXTRACT, mem_flags2, high, low);
    }
  else
    {
      if (expr != DECL_RESULT(fun->decl))
	constrain_src_value(value, type, mem_flags2);

      // TODO: What if the extracted bits are defined, but the extra bits
      // undefined?
      // E.g. a bool where the least significant bit is defined, but the rest
      // undefined. I guess it should be undefined?
      std::tie(value, indef) = from_mem_repr(value, indef, type);
      std::tie(value, mem_flags2) = from_mem_repr(value, mem_flags2, type);
      inst2memory_flagsx.insert({value, mem_flags2});
    }

  Inst *prov = nullptr;
  if (POINTER_TYPE_P(type))
    prov = extract_id(value);

  return {value, indef, prov};
}

// Read value/indef from memory. No UB checks etc. are done.
std::tuple<Inst *, Inst *, Inst *> Converter::load_value(Inst *orig_ptr, uint64_t size)
{
  Inst *value = nullptr;
  Inst *indef = nullptr;
  Inst *mem_flags = nullptr;
  for (uint64_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_ptr->bitsize);
      Inst *ptr = bb->build_inst(Op::ADD, orig_ptr, offset);
      Inst *data_byte = bb->build_inst(Op::LOAD, ptr);
      if (value)
	value = bb->build_inst(Op::CONCAT, data_byte, value);
      else
	value = data_byte;
      Inst *indef_byte = bb->build_inst(Op::GET_MEM_INDEF, ptr);
      if (indef)
	indef = bb->build_inst(Op::CONCAT, indef_byte, indef);
      else
	indef = indef_byte;
      Inst *flag = bb->build_inst(Op::GET_MEM_FLAG, ptr);
      flag = bb->build_inst(Op::SEXT, flag, 8);
      if (mem_flags)
	mem_flags = bb->build_inst(Op::CONCAT, flag, mem_flags);
      else
	mem_flags = flag;
    }
  return {value, indef, mem_flags};
}

// Write value to memory. No UB checks etc. are done, and memory flags
// are not updated.
void Converter::store_value(Inst *orig_ptr, Inst *value, Inst *indef)
{
  if ((value->bitsize & 7) != 0)
    throw Not_implemented("store_value: not byte aligned");
  uint64_t size = value->bitsize / 8;
  for (uint64_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_ptr->bitsize);
      Inst *ptr = bb->build_inst(Op::ADD, orig_ptr, offset);
      Inst *high = bb->value_inst(i * 8 + 7, 32);
      Inst *low = bb->value_inst(i * 8, 32);
      Inst *byte = bb->build_inst(Op::EXTRACT, value, high, low);
      bb->build_inst(Op::STORE, ptr, byte);
      if (indef)
	{
	  byte = bb->build_inst(Op::EXTRACT, indef, high, low);
	  bb->build_inst(Op::SET_MEM_INDEF, ptr, byte);
	}
    }
}

bool Converter::is_load(tree expr)
{
  switch (TREE_CODE(expr))
    {
    case ARRAY_REF:
      {
	tree array = TREE_OPERAND(expr, 0);
	// Indexing element of a vector as vec[2] is done by an  ARRAY_REF of
	// a VIEW_CONVERT_EXPR of the vector.
	if (TREE_CODE(array) == VIEW_CONVERT_EXPR
	    && VECTOR_TYPE_P(TREE_TYPE(TREE_OPERAND(array, 0))))
	  return false;
	return true;
      }
    case COMPONENT_REF:
      if (is_extracting_from_value(expr))
	return false;
      return true;
    case MEM_REF:
    case TARGET_MEM_REF:
    case VAR_DECL:
    case RESULT_DECL:
      return true;
    default:
      return false;
    }
}

// It is UB if the objects in a gimple_assign doing both a load and store
// overlap (unless the load/store addresses are identical).
void Converter::load_store_overlap_ub_check(tree store_expr, tree load_expr)
{
  assert(!is_bit_field(load_expr));
  assert(!is_bit_field(store_expr));

  Addr store_addr = process_address(store_expr, true);
  Addr load_addr = process_address(load_expr, true);
  uint64_t size = bytesize_for_type(TREE_TYPE(load_expr));
  if (size <= 1)
    return;

  uint32_t load_alignment = get_object_alignment(load_expr) / 8;
  uint32_t store_alignment = get_object_alignment(store_expr) / 8;
  if (size <= load_alignment && size <= store_alignment)
    return;

  overlap_ub_check(load_addr.ptr, store_addr.ptr, size);
}

void Converter::process_store(tree addr_expr, tree value_expr)
{
  if (reverse_storage_order_for_component_p(addr_expr))
    throw Not_implemented("reverse storage order");

  if (is_load(value_expr))
    load_store_overlap_ub_check(addr_expr, value_expr);

  if (TREE_CODE(value_expr) == STRING_CST)
    {
      uint64_t str_len = TREE_STRING_LENGTH(value_expr);
      uint64_t size = bytesize_for_type(TREE_TYPE(addr_expr));
      assert(str_len <= size);
      const char *p = TREE_STRING_POINTER(value_expr);
      Addr addr = process_address(addr_expr, true);
      assert(!addr.bitoffset);
      Inst *memory_flag = bb->value_inst(0, 1);
      Inst *indef = bb->value_inst(0, 8);
      if (size > MAX_MEMORY_UNROLL_LIMIT)
	throw Not_implemented("process_store: too large string");

      store_ub_check(addr.ptr, addr.prov, size);
      for (uint64_t i = 0; i < size; i++)
	{
	  Inst *offset = bb->value_inst(i, addr.ptr->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);
	  uint8_t byte = (i < str_len) ? p[i] : 0;
	  Inst *value = bb->value_inst(byte, 8);
	  bb->build_inst(Op::STORE, ptr, value);
	  bb->build_inst(Op::SET_MEM_FLAG, ptr, memory_flag);
	  bb->build_inst(Op::SET_MEM_INDEF, ptr, indef);
	}
      return;
    }

  tree value_type = TREE_TYPE(value_expr);
  bool is_bitfield = is_bit_field(addr_expr);
  Addr addr = process_address(addr_expr, true);
  assert(is_bitfield || !addr.bitoffset);
  assert(addr.bitoffset < 8);
  auto [value, indef, prov] = tree2inst_indef_prov(value_expr);
  if (!indef)
    indef = bb->value_inst(0, value->bitsize);

  // We are not tracking provenance through memory yet, which may lead to
  // false positives. For example,
  //
  //     #include <stdint.h>
  //     int a[10];
  //     int b[10];
  //     int *p;
  //     int foo(uintptr_t i) {
  //       int *q = a;
  //       q = q + i;
  //       p = q;
  //       b[0] = 1;
  //       *p = 0;
  //       return b[0];
  //     }
  //
  // SMTGCC reports a miscompilation when fre changes the return to "return 1;"
  // because it believes *p may modify 'b'. We mitigate this issue by marking
  // the store as UB if it stores a pointer with incorrect provenance.
  // This workaround is not correct for GIMPLE, but it doesn't matter much
  // as the original C or C++ program would be UB anyway if it stored a pointer
  // having incorrect provenance.
  // TODO: Implement provenance tracking through memory.
  if (POINTER_TYPE_P(value_type))
    {
      assert(prov);
      Inst *value_mem_id = extract_id(value);
      Inst *is_ub = bb->build_inst(Op::NE, prov, value_mem_id);
      bb->build_inst(Op::UB, is_ub);
    }

  // The addresses assigned to local variables differ between GIMPLE and the
  // RISC-V assembly, which makes smtgcc-tv-backend incorrectly report that
  // the code is miscompiled when a local variable is written to global
  // memory, such as
  //
  //   int *p;
  //   void foo(void) {
  //     int i;
  //     p = &i;
  //   }
  //
  // For now, we solve this by making storing a local pointer UB.
  //
  // TODO: Find a better way of handling this.
  if (state->arch != Arch::gimple
      && !is_tgt_func && POINTER_TYPE_P(value_type))
    {
      Inst *value_mem_id = extract_id(value);
      Inst *zero = bb->value_inst(0, module->ptr_id_bits);
      Inst *is_ub = bb->build_inst(Op::SLT, value_mem_id, zero);
      bb->build_inst(Op::UB, is_ub);
    }

  uint64_t size;
  if (is_bitfield)
    {
      uint64_t bitsize = bitsize_for_type(value_type);
      size = (bitsize + addr.bitoffset + 7) / 8;

      if (addr.bitoffset)
	{
	  Inst *first_byte = bb->build_inst(Op::LOAD, addr.ptr);
	  Inst *bits = bb->build_trunc(first_byte, addr.bitoffset);
	  value = bb->build_inst(Op::CONCAT, value, bits);

	  first_byte = bb->build_inst(Op::GET_MEM_INDEF, addr.ptr);
	  bits = bb->build_trunc(first_byte, addr.bitoffset);
	  indef = bb->build_inst(Op::CONCAT, indef, bits);
	}

      if (bitsize + addr.bitoffset != size * 8)
	{
	  Inst *offset = bb->value_inst(size - 1, addr.ptr->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);

	  uint64_t remaining = size * 8 - (bitsize + addr.bitoffset);
	  assert(remaining < 8);
	  Inst *high = bb->value_inst(7, 32);
	  Inst *low = bb->value_inst(8 - remaining, 32);

	  Inst *last_byte = bb->build_inst(Op::LOAD, ptr);
	  Inst *bits = bb->build_inst(Op::EXTRACT, last_byte, high, low);
	  value = bb->build_inst(Op::CONCAT, bits, value);

	  last_byte = bb->build_inst(Op::GET_MEM_INDEF, ptr);
	  bits = bb->build_inst(Op::EXTRACT, last_byte, high, low);
	  indef = bb->build_inst(Op::CONCAT, bits, indef);
	}
    }
  else
    {
      size = bytesize_for_type(value_type);
      std::tie(value, indef) = to_mem_repr(value, indef, value_type);
    }

  store_ub_check(addr.ptr, addr.prov, size);

  // TODO: Adjust for bitfield?
  Inst *memory_flagsx = nullptr;
  if (inst2memory_flagsx.contains(value))
    memory_flagsx = inst2memory_flagsx.at(value);

  for (uint64_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, addr.ptr->bitsize);
      Inst *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);

      Inst *high = bb->value_inst(i * 8 + 7, 32);
      Inst *low = bb->value_inst(i * 8, 32);

      uint8_t padding = padding_at_offset(value_type, i);
      Inst *byte = bb->build_inst(Op::EXTRACT, value, high, low);
      bb->build_inst(Op::STORE, ptr, byte);

      byte = bb->build_inst(Op::EXTRACT, indef, high, low);
      if (padding != 0)
	{
	  Inst *padding_inst = bb->value_inst(padding, 8);
	  byte = bb->build_inst(Op::OR, byte, padding_inst);
	}
      bb->build_inst(Op::SET_MEM_INDEF, ptr, byte);

      Inst *memory_flag;
      if (memory_flagsx)
	{
	  memory_flag = bb->build_inst(Op::EXTRACT, memory_flagsx, high, low);
	  Inst *zero = bb->value_inst(0, memory_flag->bitsize);
	  memory_flag = bb->build_inst(Op::NE, memory_flag, zero);
	}
      else
	memory_flag = bb->value_inst(1, 1);
      bb->build_inst(Op::SET_MEM_FLAG, ptr, memory_flag);
    }
}

// Convert a scalar value of src_type to dest_type.
std::tuple<Inst *, Inst *, Inst *> Converter::type_convert(Inst *inst, Inst *indef, Inst *prov, tree src_type, tree dest_type)
{
  Inst *res_indef = get_res_indef(indef, dest_type);

  // The addresses assigned to local variables differ between GIMPLE and the
  // RISC-V assembly, which makes smtgcc-tv-backend incorrectly report that
  // the code is miscompiled when a local variable is written to global
  // memory, such as
  //
  //   uintptr_t p;
  //   void foo(void) {
  //     int i;
  //     p = &i;
  //   }
  //
  // For now, we solve this by making converting a local pointer to a non-
  // pointer type UB.
  //
  // TODO: Find a better way of handling this.
  if (state->arch != Arch::gimple
      && !is_tgt_func
      && POINTER_TYPE_P(src_type)
      && !POINTER_TYPE_P(dest_type))
    {
      Inst *value_mem_id = extract_id(inst);
      Inst *zero = bb->value_inst(0, module->ptr_id_bits);
      Inst *is_ub = bb->build_inst(Op::SLT, value_mem_id, zero);
      bb->build_inst(Op::UB, is_ub);
    }

  if (INTEGRAL_TYPE_P(src_type)
      || POINTER_TYPE_P(src_type)
      || TREE_CODE(src_type) == OFFSET_TYPE)
    {
      if (INTEGRAL_TYPE_P(dest_type)
	  || POINTER_TYPE_P(dest_type)
	  || TREE_CODE(dest_type) == OFFSET_TYPE)
	{
	  res_indef = indef;
	  unsigned src_prec = inst->bitsize;
	  unsigned dest_prec = bitsize_for_type(dest_type);
	  if (src_prec > dest_prec)
	    {
	      inst = bb->build_trunc(inst, dest_prec);
	      if (indef)
		res_indef = bb->build_trunc(res_indef, dest_prec);
	    }
	  else if (src_prec < dest_prec)
	    {
	      Op op = TYPE_UNSIGNED(src_type) ? Op::ZEXT : Op::SEXT;
	      inst =  bb->build_inst(op, inst, dest_prec);
	      if (indef)
		res_indef =  bb->build_inst(op, res_indef, dest_prec);
	      prov = nullptr;
	    }
	  if (POINTER_TYPE_P(dest_type))
	    {
	      assert(!POINTER_TYPE_P(src_type) || prov);
	      if (!prov)
		prov = extract_id(inst);
	    }
	  if (TREE_CODE(dest_type) == BOOLEAN_TYPE && dest_prec > 1)
	    check_wide_bool(inst, dest_type);
	  return {inst, res_indef, prov};
	}
      if (FLOAT_TYPE_P(dest_type))
	{
	  Op op = TYPE_UNSIGNED(src_type) ? Op::U2F : Op::S2F;
	  Inst *res = bb->build_inst(op, inst, TYPE_PRECISION(dest_type));
	  return {res, res_indef, nullptr};
	}
    }

  if (FLOAT_TYPE_P(src_type))
    {
      if (TREE_CODE(dest_type) == INTEGER_TYPE
	  || TREE_CODE(dest_type) == BITINT_TYPE
	  || TREE_CODE(dest_type) == ENUMERAL_TYPE)
	{
	  unsigned src_bitsize = inst->bitsize;
	  unsigned dest_bitsize = bitsize_for_type(dest_type);

	  // The result is UB if the floating point value is out of range
	  // for the integer.
	  Inst *min = tree2inst(TYPE_MIN_VALUE(dest_type));
	  Inst *max = tree2inst(TYPE_MAX_VALUE(dest_type));
	  // TODO: Handle dest bitsize > 128
	  if (inst->bitsize == 16 && max->bitsize <= 128)
	    {
	      max = bb->value_inst(65504, max->bitsize);
	      if (!TYPE_UNSIGNED(dest_type))
		min = bb->value_inst(-65504, min->bitsize);
	    }
	  Op op = TYPE_UNSIGNED(dest_type) ? Op::U2F : Op::S2F;
	  Inst *fmin = bb->build_inst(op, min, src_bitsize);
	  Inst *fmax = bb->build_inst(op, max, src_bitsize);
	  Inst *clow = bb->build_inst(Op::FLE, fmin, inst);
	  Inst *chigh = bb->build_inst(Op::FLE, inst, fmax);
	  Inst *is_in_range = bb->build_inst(Op::AND, clow, chigh);
	  Inst *is_ub = bb->build_inst(Op::NOT, is_in_range);
	  bb->build_inst(Op::UB, is_ub);

	  op = TYPE_UNSIGNED(dest_type) ? Op::F2U : Op::F2S;
	  Inst *res = bb->build_inst(op, inst, dest_bitsize);

	  // The UB checks above are not completely correct when the
	  // floating-point mantissa has fewer bits than the source
	  // bitsize, as the limit then gets rounded up and the result
	  // may overflow.
	  // Check this case by converting to a larger size.
	  if (src_bitsize <= dest_bitsize)
	    {
	      Inst *val = bb->build_inst(op, inst, dest_bitsize + 1);
	      op = TYPE_UNSIGNED(dest_type) ? Op::ZEXT : Op::SEXT;
	      Inst *eres = bb->build_inst(op, res, val->bitsize);
	      Inst *is_ub = bb->build_inst(Op::NE, val, eres);
	      bb->build_inst(Op::UB, is_ub);
	    }
	  // TODO: Implement better UB checks that are both efficient
	  // and correct.

	  return {res, res_indef, nullptr};
	}
      if (FLOAT_TYPE_P(dest_type))
	{
	  unsigned src_prec = TYPE_PRECISION(src_type);
	  unsigned dest_prec = TYPE_PRECISION(dest_type);
	  if (src_prec == dest_prec)
	    return {inst, res_indef, nullptr};
	  Inst *res = bb->build_inst(Op::FCHPREC, inst, dest_prec);
	  return {res, res_indef, nullptr};
	}
    }

  throw Not_implemented("type_convert: unknown type");
}

Inst *Converter::type_convert(Inst *inst, tree src_type, tree dest_type)
{
  return std::get<0>(type_convert(inst, nullptr, nullptr, src_type, dest_type));
}

void Converter::check_wide_bool(Inst *inst, tree type)
{
  Inst *false_inst = bb->value_inst(0, inst->bitsize);
  Inst *true_inst = bb->value_inst(1, inst->bitsize);
  if (!TYPE_UNSIGNED(type))
    true_inst = bb->build_inst(Op::NEG, true_inst);
  Inst *cond0 = bb->build_inst(Op::NE, inst, true_inst);
  Inst *cond1 = bb->build_inst(Op::NE, inst, false_inst);
  Inst *cond = bb->build_inst(Op::AND, cond0, cond1);
  bb->build_inst(Op::UB, cond);
}

std::pair<Inst *, Inst *> Converter::process_unary_bool(enum tree_code code, Inst *arg1, Inst *arg1_indef, tree lhs_type, tree arg1_type)
{
  assert(TREE_CODE(lhs_type) == BOOLEAN_TYPE);

  auto [lhs, lhs_indef, _] =
    process_unary_int(code, arg1, arg1_indef, nullptr, lhs_type, arg1_type);

  if (lhs->bitsize > 1)
    check_wide_bool(lhs, lhs_type);

  assert(lhs->bitsize == TYPE_PRECISION(lhs_type));
  assert(!lhs_indef || lhs_indef->bitsize == TYPE_PRECISION(lhs_type));
  return {lhs, lhs_indef};
}

std::tuple<Inst *, Inst *, Inst *> Converter::process_unary_int(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg1_prov, tree lhs_type, tree arg1_type, bool ignore_overflow)
{
  // Handle instructions that have special requirements for the propagation
  // of indef bits.
  switch (code)
    {
    case BIT_NOT_EXPR:
      return {bb->build_inst(Op::NOT, arg1), arg1_indef, arg1_prov};
    case CONVERT_EXPR:
    case NOP_EXPR:
      return type_convert(arg1, arg1_indef, arg1_prov, arg1_type, lhs_type);
    case PAREN_EXPR:
      return {arg1, arg1_indef, arg1_prov};
    default:
      break;
    }

  // Handle instructions where the result is indef if any input bit is indef.
  Inst *res_indef = get_res_indef(arg1_indef, lhs_type);
  switch (code)
    {
    case ABS_EXPR:
      {
	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Inst *min_int_inst = build_min_int(bb, arg1->bitsize);
	    Inst *cond = bb->build_inst(Op::EQ, arg1, min_int_inst);
	    bb->build_inst(Op::UB, cond);
	  }
	assert(!TYPE_UNSIGNED(arg1_type));
	Inst *neg = bb->build_inst(Op::NEG, arg1);
	Inst *zero = bb->value_inst(0, arg1->bitsize);
	Inst *cond = bb->build_inst(Op::SLE, zero, arg1);
	return {bb->build_inst(Op::ITE, cond, arg1, neg), res_indef, nullptr};
      }
    case ABSU_EXPR:
      {
	assert(!TYPE_UNSIGNED(arg1_type));
	Inst *neg = bb->build_inst(Op::NEG, arg1);
	Inst *zero = bb->value_inst(0, arg1->bitsize);
	Inst *cond = bb->build_inst(Op::SLE, zero, arg1);
	return {bb->build_inst(Op::ITE, cond, arg1, neg), res_indef, nullptr};
      }
    case FIX_TRUNC_EXPR:
      return type_convert(arg1, arg1_indef, arg1_prov, arg1_type, lhs_type);
    case NEGATE_EXPR:
      if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	{
	  Inst *min_int_inst = build_min_int(bb, arg1->bitsize);
	  bb->build_inst(Op::UB, bb->build_inst(Op::EQ, arg1, min_int_inst));
	}
      return {bb->build_inst(Op::NEG, arg1), res_indef, nullptr};
    default:
      break;
    }

  throw Not_implemented("process_unary_int: "s + get_tree_code_name(code));
}

std::pair<Inst *, Inst *> Converter::process_unary_float(enum tree_code code, Inst *arg1, Inst *arg1_indef, tree lhs_type, tree arg1_type)
{
  // Handle instructions that have special requirements for the propagation
  // of indef bits.
  switch (code)
    {
    case FLOAT_EXPR:
    case CONVERT_EXPR:
    case NOP_EXPR:
      {
	auto [inst, indef, prov] =
	  type_convert(arg1, arg1_indef, nullptr, arg1_type, lhs_type);
	return {inst, indef};
      }
    default:
      break;
    }

  // Handle instructions where the result is indef if any input bit is indef.
  Inst *res_indef = get_res_indef(arg1_indef, lhs_type);
  switch (code)
    {
    case ABS_EXPR:
      if (state->arch != Arch::gimple)
	{
	  // Backends may choose to implement fabs as a bit operation. Skip
	  // checking if this would generate a non-canonical NaN.
	  Inst *shift = bb->value_inst(1, arg1->bitsize);
	  Inst *inst = bb->build_inst(Op::SHL, arg1, shift);
	  inst = bb->build_inst(Op::LSHR, inst, shift);
	  constrain_src_value(inst, arg1_type);
	}
      return {bb->build_inst(Op::FABS, arg1), res_indef};
    case NEGATE_EXPR:
      if (state->arch != Arch::gimple)
	{
	  // Backends may choose to implement fneg as a bit operation. Skip
	  // checking if this would generate a non-canonical NaN.
	  unsigned __int128 v = ((unsigned __int128)1) << (arg1->bitsize - 1);
	  Inst *inst = bb->value_inst(v, arg1->bitsize);
	  inst = bb->build_inst(Op::XOR, arg1, inst);
	  Inst *is_ub = bb->build_inst(Op::IS_NONCANONICAL_NAN, inst);
	  bb->build_inst(Op::UB, is_ub);
	}
      return {bb->build_inst(Op::FNEG, arg1), res_indef};
    case PAREN_EXPR:
      return {arg1, res_indef};
    default:
      break;
    }

  throw Not_implemented("process_unary_float: "s + get_tree_code_name(code));
}

Inst *Converter::process_unary_complex(enum tree_code code, Inst *arg1, tree lhs_type)
{
  tree elem_type = TREE_TYPE(lhs_type);
  uint64_t bitsize = arg1->bitsize;
  uint64_t elem_bitsize = bitsize / 2;
  Inst *real_high = bb->value_inst(elem_bitsize - 1, 32);
  Inst *real_low = bb->value_inst(0, 32);
  Inst *imag_high = bb->value_inst(bitsize - 1, 32);
  Inst *imag_low = bb->value_inst(elem_bitsize, 32);
  Inst *arg1_real = bb->build_inst(Op::EXTRACT, arg1, real_high, real_low);
  arg1_real = from_mem_repr(arg1_real, elem_type);
  Inst *arg1_imag = bb->build_inst(Op::EXTRACT, arg1, imag_high, imag_low);
  arg1_imag = from_mem_repr(arg1_imag, elem_type);

  switch (code)
    {
    case CONJ_EXPR:
      {
	Inst *inst_imag;
	inst_imag = process_unary_scalar(NEGATE_EXPR, arg1_imag,
					 elem_type, elem_type);
	arg1_real = to_mem_repr(arg1_real, elem_type);
	inst_imag = to_mem_repr(inst_imag, elem_type);
	return bb->build_inst(Op::CONCAT, inst_imag, arg1_real);
      }
    case NEGATE_EXPR:
      {
	Inst * inst_real =
	  process_unary_scalar(code, arg1_real, elem_type, elem_type);
	Inst *inst_imag =
	  process_unary_scalar(code, arg1_imag, elem_type, elem_type);
	inst_real = to_mem_repr(inst_real, elem_type);
	inst_imag = to_mem_repr(inst_imag, elem_type);
	return bb->build_inst(Op::CONCAT, inst_imag, inst_real);
      }
    case PAREN_EXPR:
      return arg1;
    default:
      break;
    }

  throw Not_implemented("process_unary_complex: "s + get_tree_code_name(code));
}

Inst *Converter::process_unary_scalar(enum tree_code code, Inst *arg1, tree lhs_type, tree arg1_type, bool ignore_overflow)
{
  auto [inst, indef, prov] =
    process_unary_scalar(code, arg1, nullptr, nullptr, lhs_type, arg1_type,
			 ignore_overflow);
  assert(!indef);
  assert(!prov);
  return inst;
}

std::tuple<Inst *, Inst *, Inst *> Converter::process_unary_scalar(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg1_prov, tree lhs_type, tree arg1_type, bool ignore_overflow)
{
  Inst *inst;
  Inst *indef = nullptr;
  Inst *prov = nullptr;
  if (TREE_CODE(lhs_type) == BOOLEAN_TYPE)
    {
      std::tie(inst, indef) =
	process_unary_bool(code, arg1, arg1_indef, lhs_type, arg1_type);
    }
  else if (FLOAT_TYPE_P(lhs_type))
    {
      std::tie(inst, indef) =
	process_unary_float(code, arg1, arg1_indef, lhs_type, arg1_type);
    }
  else
    {
      std::tie(inst, indef, prov) =
	process_unary_int(code, arg1, arg1_indef, arg1_prov, lhs_type,
			  arg1_type, ignore_overflow);
    }
  return {inst, indef, prov};
}

std::pair<Inst *, Inst *> Converter::process_vec_duplicate(Inst *arg1, Inst *arg1_indef, tree lhs_type, tree arg1_type)
{
  uint32_t elem_bitsize = bitsize_for_type(arg1_type);
  assert(bitsize_for_type(TREE_TYPE(lhs_type)) == elem_bitsize);
  uint32_t nof_elt = bitsize_for_type(lhs_type) / elem_bitsize;
  Inst *res = arg1;
  Inst *res_indef = arg1_indef;
  for (uint64_t i = 1; i < nof_elt; i++)
    {
      res = bb->build_inst(Op::CONCAT, arg1, res);
      if (res_indef)
	res_indef = bb->build_inst(Op::CONCAT, arg1_indef, res_indef);
    }
  return {res, res_indef};
}

std::pair<Inst *, Inst *> Converter::process_unary_vec(enum tree_code code, Inst *arg1, Inst *arg1_indef, tree lhs_elem_type, tree arg1_elem_type, bool ignore_overflow)
{
  uint32_t elem_bitsize = bitsize_for_type(arg1_elem_type);
  uint32_t nof_elt = arg1->bitsize / elem_bitsize;
  uint32_t start_idx = 0;

  if (code == VEC_UNPACK_LO_EXPR
      || code == VEC_UNPACK_HI_EXPR
      || code == VEC_UNPACK_FLOAT_LO_EXPR
      || code == VEC_UNPACK_FLOAT_HI_EXPR)
    {
      if (code == VEC_UNPACK_HI_EXPR || code == VEC_UNPACK_FLOAT_HI_EXPR)
	start_idx = nof_elt / 2;
      else
	nof_elt = nof_elt / 2;
      code = CONVERT_EXPR;
    }

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint64_t i = start_idx; i < nof_elt; i++)
    {
      Inst *a1_indef = nullptr;
      Inst *a1 = extract_vec_elem(bb, arg1, elem_bitsize, i);
      if (arg1_indef)
	a1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, i);
      auto [inst, inst_indef, _] =
	process_unary_scalar(code, a1, a1_indef, nullptr, lhs_elem_type,
			     arg1_elem_type, ignore_overflow);

      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg1_indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, inst_indef, res_indef);
	  else
	    res_indef = inst_indef;
	}
    }
  return {res, res_indef};
}

std::pair<Inst *, Inst *> Converter::process_binary_float(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type)
{
  Inst *res_indef = get_res_indef(arg1_indef, arg2_indef, lhs_type);
  switch (code)
    {
    case EQ_EXPR:
      return {bb->build_inst(Op::FEQ, arg1, arg2), res_indef};
    case NE_EXPR:
      return {bb->build_inst(Op::FNE, arg1, arg2), res_indef};
    case GE_EXPR:
      return {bb->build_inst(Op::FLE, arg2, arg1), res_indef};
    case GT_EXPR:
      return {bb->build_inst(Op::FLT, arg2, arg1), res_indef};
    case LE_EXPR:
      return {bb->build_inst(Op::FLE, arg1, arg2), res_indef};
    case LT_EXPR:
      return {bb->build_inst(Op::FLT, arg1, arg2), res_indef};
    case UNEQ_EXPR:
      {
	Inst *isnan1 = bb->build_inst(Op::IS_NAN, arg1);
	Inst *isnan2 = bb->build_inst(Op::IS_NAN, arg2);
	Inst *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Inst *cmp = bb->build_inst(Op::FEQ, arg1, arg2);
	return {bb->build_inst(Op::OR, isnan, cmp), res_indef};
      }
    case UNLT_EXPR:
      {
	Inst *isnan1 = bb->build_inst(Op::IS_NAN, arg1);
	Inst *isnan2 = bb->build_inst(Op::IS_NAN, arg2);
	Inst *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Inst *cmp = bb->build_inst(Op::FLT, arg1, arg2);
	return {bb->build_inst(Op::OR, isnan, cmp), res_indef};
      }
    case UNLE_EXPR:
      {
	Inst *isnan1 = bb->build_inst(Op::IS_NAN, arg1);
	Inst *isnan2 = bb->build_inst(Op::IS_NAN, arg2);
	Inst *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Inst *cmp = bb->build_inst(Op::FLE, arg1, arg2);
	return {bb->build_inst(Op::OR, isnan, cmp), res_indef};
      }
    case UNGT_EXPR:
      {
	Inst *isnan1 = bb->build_inst(Op::IS_NAN, arg1);
	Inst *isnan2 = bb->build_inst(Op::IS_NAN, arg2);
	Inst *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Inst *cmp = bb->build_inst(Op::FLT, arg2, arg1);
	return {bb->build_inst(Op::OR, isnan, cmp), res_indef};
      }
    case UNGE_EXPR:
      {
	Inst *isnan1 = bb->build_inst(Op::IS_NAN, arg1);
	Inst *isnan2 = bb->build_inst(Op::IS_NAN, arg2);
	Inst *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Inst *cmp = bb->build_inst(Op::FLE, arg2, arg1);
	return {bb->build_inst(Op::OR, isnan, cmp), res_indef};
      }
    case UNORDERED_EXPR:
      {
	Inst *isnan1 = bb->build_inst(Op::IS_NAN, arg1);
	Inst *isnan2 = bb->build_inst(Op::IS_NAN, arg2);
	return {bb->build_inst(Op::OR, isnan1, isnan2), res_indef};
      }
    case ORDERED_EXPR:
      {
	Inst *isnan1 = bb->build_inst(Op::IS_NAN, arg1);
	Inst *isnan2 = bb->build_inst(Op::IS_NAN, arg2);
	Inst *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	return {bb->build_inst(Op::NOT, isnan), res_indef};
      }
    case LTGT_EXPR:
      {
	Inst *lt = bb->build_inst(Op::FLT, arg1, arg2);
	Inst *gt = bb->build_inst(Op::FLT, arg2, arg1);
	return {bb->build_inst(Op::OR, lt, gt), res_indef};
      }
    case RDIV_EXPR:
      return {bb->build_inst(Op::FDIV, arg1, arg2), res_indef};
    case MAX_EXPR:
      return {gen_fmax(bb, arg1, arg2), res_indef};
    case MIN_EXPR:
      return {gen_fmin(bb, arg1, arg2), res_indef};
    case MINUS_EXPR:
      return {bb->build_inst(Op::FSUB, arg1, arg2), res_indef};
    case MULT_EXPR:
      return {bb->build_inst(Op::FMUL, arg1, arg2), res_indef};
    case PLUS_EXPR:
      return {bb->build_inst(Op::FADD, arg1, arg2), res_indef};
    default:
      break;
    }

  throw Not_implemented("process_binary_float: "s + get_tree_code_name(code));
}

Inst *Converter::process_binary_complex(enum tree_code code, Inst *arg1, Inst *arg2, tree lhs_type)
{
  tree elem_type = TREE_TYPE(lhs_type);
  uint64_t bitsize = arg1->bitsize;
  uint64_t elem_bitsize = bitsize / 2;
  Inst *real_high = bb->value_inst(elem_bitsize - 1, 32);
  Inst *real_low = bb->value_inst(0, 32);
  Inst *imag_high = bb->value_inst(bitsize - 1, 32);
  Inst *imag_low = bb->value_inst(elem_bitsize, 32);
  Inst *arg1_real = bb->build_inst(Op::EXTRACT, arg1, real_high, real_low);
  arg1_real = from_mem_repr(arg1_real, elem_type);
  Inst *arg1_imag = bb->build_inst(Op::EXTRACT, arg1, imag_high, imag_low);
  arg1_imag = from_mem_repr(arg1_imag, elem_type);
  Inst *arg2_real = bb->build_inst(Op::EXTRACT, arg2, real_high, real_low);
  arg2_real = from_mem_repr(arg2_real, elem_type);
  Inst *arg2_imag = bb->build_inst(Op::EXTRACT, arg2, imag_high, imag_low);
  arg2_imag = from_mem_repr(arg2_imag, elem_type);

  switch (code)
    {
    case MINUS_EXPR:
    case PLUS_EXPR:
      {
	Inst *inst_real =
	  process_binary_scalar(code, arg1_real, arg2_real,
				elem_type, elem_type, elem_type);
	Inst *inst_imag =
	  process_binary_scalar(code, arg1_imag, arg2_imag,
				elem_type, elem_type, elem_type);
	inst_real = to_mem_repr(inst_real, elem_type);
	inst_imag = to_mem_repr(inst_imag, elem_type);
	return bb->build_inst(Op::CONCAT, inst_imag, inst_real);
      }
    default:
      break;
    }

  throw Not_implemented("process_binary_complex: "s + get_tree_code_name(code));
}

Inst *Converter::process_binary_complex_cmp(enum tree_code code, Inst *arg1, Inst *arg2, tree lhs_type, tree arg1_type)
{
  tree elem_type = TREE_TYPE(arg1_type);
  uint64_t bitsize = arg1->bitsize;
  uint64_t elem_bitsize = bitsize / 2;
  Inst *real_high = bb->value_inst(elem_bitsize - 1, 32);
  Inst *real_low = bb->value_inst(0, 32);
  Inst *imag_high = bb->value_inst(bitsize - 1, 32);
  Inst *imag_low = bb->value_inst(elem_bitsize, 32);
  Inst *arg1_real = bb->build_inst(Op::EXTRACT, arg1, real_high, real_low);
  arg1_real = from_mem_repr(arg1_real, elem_type);
  Inst *arg1_imag = bb->build_inst(Op::EXTRACT, arg1, imag_high, imag_low);
  arg1_imag = from_mem_repr(arg1_imag, elem_type);
  Inst *arg2_real = bb->build_inst(Op::EXTRACT, arg2, real_high, real_low);
  arg2_real = from_mem_repr(arg2_real, elem_type);
  Inst *arg2_imag = bb->build_inst(Op::EXTRACT, arg2, imag_high, imag_low);
  arg2_imag = from_mem_repr(arg2_imag, elem_type);

  switch (code)
    {
    case EQ_EXPR:
    case NE_EXPR:
      {
	Inst *cmp_real =
	  process_binary_scalar(code, arg1_real, arg2_real,
				lhs_type, elem_type, elem_type);
	Inst *cmp_imag =
	  process_binary_scalar(code, arg1_imag, arg2_imag,
				lhs_type, elem_type, elem_type);
	Inst *cmp;
	if (code == EQ_EXPR)
	  cmp = bb->build_inst(Op::AND, cmp_real, cmp_imag);
	else
	  cmp = bb->build_inst(Op::OR, cmp_real, cmp_imag);
	return cmp;
      }
    default:
      break;
    }

  throw Not_implemented("process_binary_complex_cmp: "s + get_tree_code_name(code));
}

std::pair<Inst *, Inst *> Converter::process_binary_bool(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type, tree arg1_type, tree arg2_type)
{
  assert(TREE_CODE(lhs_type) == BOOLEAN_TYPE);

  Inst *lhs;
  Inst *lhs_indef = nullptr;
  if (VECTOR_TYPE_P(arg1_type))
    {
      assert(code == EQ_EXPR || code == NE_EXPR);
      // We can compare vectors in the same way as scalars, but it is
      // more efficient to do it elementwise since we then can perform
      // more optimizations on the generated code.
      tree elem_type = TREE_TYPE(arg1_type);
      uint32_t elem_bitsize = bitsize_for_type(elem_type);
      uint32_t nof_elt = bitsize_for_type(arg1_type) / elem_bitsize;
      lhs = bb->value_inst(code == EQ_EXPR, 1);
      Op op = code == EQ_EXPR ? Op::AND : Op::OR;
      if (arg1_indef || arg2_indef)
	lhs_indef = bb->value_inst(0, 1);
      for (uint64_t i = 0; i < nof_elt; i++)
	{
	  Inst *a1 = extract_vec_elem(bb, arg1, elem_bitsize, i);
	  Inst *a1_indef = nullptr;
	  if (arg1_indef)
	    a1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, i);
	  Inst *a2 = extract_vec_elem(bb, arg2, elem_bitsize, i);
	  Inst *a2_indef = nullptr;
	  if (arg2_indef)
	    a2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, i);
	  if (a1_indef || a2_indef)
	    {
	      if (!a1_indef)
		a1_indef = bb->value_inst(0, elem_bitsize);
	      if (!a2_indef)
		a2_indef = bb->value_inst(0, elem_bitsize);
	    }
	  auto [inst, inst_indef] =
	    process_binary_bool(code, a1, a1_indef, a2, a2_indef,
				boolean_type_node, elem_type, elem_type);
	  lhs = bb->build_inst(op, lhs, inst);
	  if (lhs_indef)
	    lhs_indef = bb->build_inst(Op::OR, lhs_indef, inst_indef);
	}
    }
  else if (FLOAT_TYPE_P(arg1_type))
    {
      std::tie(lhs, lhs_indef) =
	process_binary_float(code, arg1, arg1_indef, arg2, arg2_indef,
			     lhs_type);
    }
  else
    {
      Inst *lhs_prov;
      std::tie(lhs, lhs_indef, lhs_prov) =
	process_binary_int(code, TYPE_UNSIGNED(arg1_type), arg1, arg1_indef,
			   nullptr, arg2, arg2_indef, nullptr, lhs_type,
			   arg1_type, arg2_type);
    }

  // GCC may use non-standard Boolean types (such as signed-boolean:8), so
  // we may need to extend the value if we have generated a standard 1-bit
  // Boolean for a comparison.
  uint64_t precision = TYPE_PRECISION(lhs_type);
  if (lhs->bitsize == 1 && precision > 1)
    {
      Op op = TYPE_UNSIGNED(lhs_type) ? Op::ZEXT : Op::SEXT;
      lhs = bb->build_inst(op, lhs, precision);
      // process_binary_* creates an indef result of the correct bitsize
      // for the return type, but the result for comparisons is returned
      // as a 1-bit value. So we must check if the indef bitsize is already
      // correct before extending.
      if (lhs_indef && lhs_indef->bitsize == 1)
	lhs_indef = bb->build_inst(Op::SEXT, lhs_indef, precision);
    }
  else if (lhs->bitsize > 1)
    check_wide_bool(lhs, lhs_type);

  assert(lhs->bitsize == precision);
  assert(!lhs_indef || lhs_indef->bitsize == precision);
  return {lhs, lhs_indef};
}

std::tuple<Inst *, Inst *, Inst *> Converter::process_binary_int(enum tree_code code, bool is_unsigned, Inst *arg1, Inst *arg1_indef, Inst *arg1_prov, Inst *arg2, Inst *arg2_indef, Inst *arg2_prov, tree lhs_type, tree arg1_type, tree arg2_type, bool ignore_overflow)
{
  // Handle instructions that have special requirements for the propagation
  // of indef bits.
  switch (code)
    {
    case BIT_AND_EXPR:
      {
	Inst *res = bb->build_inst(Op::AND, arg1, arg2);
	Inst *res_indef = nullptr;
	if (arg1_indef || arg2_indef)
	  {
	    if (!arg1_indef)
	      arg1_indef = bb->value_inst(0, arg1->bitsize);
	    if (!arg2_indef)
	      arg2_indef = bb->value_inst(0, arg2->bitsize);

	    // (0 & uninitialized) is 0.
	    // (1 & uninitialized) is uninitialized.
	    Inst *mask =
	      bb->build_inst(Op::AND,
			     bb->build_inst(Op::OR, arg1, arg1_indef),
			     bb->build_inst(Op::OR, arg2, arg2_indef));
	    res_indef =
	      bb->build_inst(Op::AND,
			     bb->build_inst(Op::OR, arg1_indef, arg2_indef),
			     mask);
	  }

	Inst *prov = nullptr;
	if (arg1_prov && arg2_prov && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in BIT_AND_EXPR");
	if (arg1_prov)
	  prov = arg1_prov;
	if (arg2_prov)
	  prov = arg2_prov;

	return {res, res_indef, prov};
      }
    case BIT_IOR_EXPR:
      {
	Inst *res = bb->build_inst(Op::OR, arg1, arg2);
	Inst *res_indef = nullptr;
	if (arg1_indef || arg2_indef)
	  {
	    if (!arg1_indef)
	      arg1_indef = bb->value_inst(0, arg1->bitsize);
	    if (!arg2_indef)
	      arg2_indef = bb->value_inst(0, arg2->bitsize);

	    // (0 | uninitialized) is uninitialized.
	    // (1 | uninitialized) is 1.
	    Inst *mask =
	      bb->build_inst(Op::AND,
			     bb->build_inst(Op::OR,
					    bb->build_inst(Op::NOT, arg1),
					    arg1_indef),
			     bb->build_inst(Op::OR,
					    bb->build_inst(Op::NOT, arg2),
					    arg2_indef));
	    res_indef =
	      bb->build_inst(Op::AND,
			     bb->build_inst(Op::OR, arg1_indef, arg2_indef),
			     mask);
	  }

	Inst *prov = nullptr;
	if (arg1_prov && arg2_prov && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in BIT_IOR_EXPR");
	if (arg1_prov)
	  prov = arg1_prov;
	if (arg2_prov)
	  prov = arg2_prov;

	return {res, res_indef, prov};
      }
    case BIT_XOR_EXPR:
      {
	Inst *res_indef = nullptr;
	if (arg1_indef || arg2_indef)
	  {
	    if (!arg1_indef)
	      arg1_indef = bb->value_inst(0, arg1->bitsize);
	    if (!arg2_indef)
	      arg2_indef = bb->value_inst(0, arg2->bitsize);
	    res_indef = bb->build_inst(Op::OR, arg1_indef, arg2_indef);
	  }
	return {bb->build_inst(Op::XOR, arg1, arg2), res_indef, nullptr};
      }
    case MULT_EXPR:
      {
	Inst *res_indef = nullptr;
	if (arg1_indef || arg2_indef)
	  {
	    // The result is defined if no input is uninitialized, or if one of
	    // the arguments is an initialized zero.
	    Inst *zero = bb->value_inst(0, arg1->bitsize);
	    if (!arg1_indef)
	      arg1_indef = zero;
	    if (!arg2_indef)
	      arg2_indef = zero;
	    Inst *arg1_unini = bb->build_inst(Op::NE, arg1_indef, zero);
	    Inst *arg1_nonzero = bb->build_inst(Op::NE, arg1, zero);
	    Inst *arg2_unini = bb->build_inst(Op::NE, arg2_indef, zero);
	    Inst *arg2_nonzero = bb->build_inst(Op::NE, arg2, zero);
	    Inst *is_indef =
	      bb->build_inst(Op::OR,
			     bb->build_inst(Op::AND,
					    arg1_unini,
					    bb->build_inst(Op::OR, arg2_unini,
							   arg2_nonzero)),
			     bb->build_inst(Op::AND,
					    arg2_unini,
					    bb->build_inst(Op::OR, arg1_unini,
							   arg1_nonzero)));
	    res_indef = bb->build_inst(Op::SEXT, is_indef, arg1->bitsize);
	  }

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  bb->build_inst(Op::UB, bb->build_inst(Op::SMUL_WRAPS, arg1, arg2));

	return {bb->build_inst(Op::MUL, arg1, arg2), res_indef, nullptr};
      }
    case LSHIFT_EXPR:
      {
	if (arg2_indef)
	  build_ub_if_not_zero(arg2_indef);
	Inst *bitsize = bb->value_inst(arg1->bitsize, arg2->bitsize);
	bb->build_inst(Op::UB, bb->build_inst(Op::ULE, bitsize, arg2));
	arg2 = type_convert(arg2, arg2_type, arg1_type);
	Inst *res_indef = arg1_indef;
	if (res_indef)
	  res_indef = bb->build_inst(Op::SHL, res_indef, arg2);
	return {bb->build_inst(Op::SHL, arg1, arg2), res_indef, nullptr};
      }
    case RROTATE_EXPR:
      {
	if (arg2_indef)
	  build_ub_if_not_zero(arg2_indef);
	Inst *bitsize = bb->value_inst(arg1->bitsize, arg2->bitsize);
	bb->build_inst(Op::UB, bb->build_inst(Op::ULE, bitsize, arg2));
	arg2 = type_convert(arg2, arg2_type, arg1_type);
	Inst *concat = bb->build_inst(Op::CONCAT, arg1, arg1);
	Inst *shift = bb->build_inst(Op::ZEXT, arg2, concat->bitsize);
	Inst *shifted = bb->build_inst(Op::LSHR, concat, shift);
	Inst *res_indef = arg1_indef;
	if (res_indef)
	  {
	    Inst *concat = bb->build_inst(Op::CONCAT, res_indef, res_indef);
	    Inst *shifted = bb->build_inst(Op::LSHR, concat, shift);
	    res_indef = bb->build_trunc(shifted, arg1->bitsize);
	  }
	return {bb->build_trunc(shifted, arg1->bitsize), res_indef, nullptr};
      }
    case LROTATE_EXPR:
      {
	if (arg2_indef)
	  build_ub_if_not_zero(arg2_indef);
	Inst *bitsize = bb->value_inst(arg1->bitsize, arg2->bitsize);
	bb->build_inst(Op::UB, bb->build_inst(Op::ULE, bitsize, arg2));
	arg2 = type_convert(arg2, arg2_type, arg1_type);
	Inst *concat = bb->build_inst(Op::CONCAT, arg1, arg1);
	Inst *shift = bb->build_inst(Op::ZEXT, arg2, concat->bitsize);
	Inst *shifted = bb->build_inst(Op::SHL, concat, shift);
	Inst *high = bb->value_inst(2 * arg1->bitsize - 1, 32);
	Inst *low = bb->value_inst(arg1->bitsize, 32);
	Inst *ret = bb->build_inst(Op::EXTRACT, shifted, high, low);
	Inst *res_indef = arg1_indef;
	if (res_indef)
	  {
	    Inst *concat = bb->build_inst(Op::CONCAT, res_indef, res_indef);
	    Inst *shifted = bb->build_inst(Op::SHL, concat, shift);
	    res_indef = bb->build_inst(Op::EXTRACT, shifted, high, low);
	  }
	return {ret, res_indef, nullptr};
      }
    case RSHIFT_EXPR:
      {
	if (arg2_indef)
	  build_ub_if_not_zero(arg2_indef);
	Inst *bitsize = bb->value_inst(arg1->bitsize, arg2->bitsize);
	bb->build_inst(Op::UB, bb->build_inst(Op::ULE, bitsize, arg2));
	Op op = is_unsigned ? Op::LSHR : Op::ASHR;
	arg2 = type_convert(arg2, arg2_type, arg1_type);
	Inst *res_indef = arg1_indef;
	if (res_indef)
	  res_indef = bb->build_inst(op, res_indef, arg2);
	return {bb->build_inst(op, arg1, arg2), res_indef, nullptr};
      }
    case NE_EXPR:
    case EQ_EXPR:
      {
	Inst *res_indef = nullptr;
	if (arg1_indef || arg2_indef)
	  {
	    // The result is defined if the value of the indefinite bits
	    // does not matter. That is, if there are bits that are defined
	    // in both arg1 and arg2, and at least one of those bits differs
	    // between arg1 and arg2, then the result is defined.
	    if (!arg1_indef)
	      arg1_indef = bb->value_inst(0, arg1->bitsize);
	    if (!arg2_indef)
	      arg2_indef = bb->value_inst(0, arg1->bitsize);
	    Inst *arg1_mask = bb->build_inst(Op::NOT, arg1_indef);
	    Inst *arg2_mask = bb->build_inst(Op::NOT, arg2_indef);
	    Inst *mask = bb->build_inst(Op::AND, arg1_mask, arg2_mask);
	    Inst *a1 = bb->build_inst(Op::AND, arg1, mask);
	    Inst *a2 = bb->build_inst(Op::AND, arg2, mask);
	    Inst *c1 = bb->build_inst(Op::EQ, a1, a2);
	    Inst *m1 = bb->value_m1_inst(arg1->bitsize);
	    Inst *c2 = bb->build_inst(Op::NE, mask, m1);
	    res_indef = bb->build_inst(Op::AND, c1, c2);
	  }
	Op op = code == NE_EXPR ? Op::NE : Op::EQ;
	return {bb->build_inst(op, arg1, arg2), res_indef, nullptr};
      }
    default:
      break;
    }

  // Handle instructions where the result is indef if any input bit is indef.
  Inst *res_indef = get_res_indef(arg1_indef, arg2_indef, lhs_type);
  switch (code)
    {
    case EXACT_DIV_EXPR:
      {
	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Inst *min_int_inst = build_min_int(bb, arg1->bitsize);
	    Inst *minus1_inst = bb->value_inst(-1, arg1->bitsize);
	    Inst *cond1 = bb->build_inst(Op::EQ, arg1, min_int_inst);
	    Inst *cond2 = bb->build_inst(Op::EQ, arg2, minus1_inst);
	    bb->build_inst(Op::UB, bb->build_inst(Op::AND, cond1, cond2));
	  }
	Inst *zero = bb->value_inst(0, arg1->bitsize);
	Op rem_op = is_unsigned ? Op::UREM : Op::SREM;
	Inst *rem = bb->build_inst(rem_op, arg1, arg2);
	Inst *ub_cond = bb->build_inst(Op::NE, rem, zero);
	bb->build_inst(Op::UB, ub_cond);
	Inst *ub_cond2 = bb->build_inst(Op::EQ, arg2, zero);
	bb->build_inst(Op::UB, ub_cond2);
	Op div_op = is_unsigned ? Op::UDIV : Op::SDIV;
	return {bb->build_inst(div_op, arg1, arg2), res_indef, nullptr};
      }
    case TRUNC_DIV_EXPR:
      {
	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Inst *min_int_inst = build_min_int(bb, arg1->bitsize);
	    Inst *minus1_inst = bb->value_inst(-1, arg1->bitsize);
	    Inst *cond1 = bb->build_inst(Op::EQ, arg1, min_int_inst);
	    Inst *cond2 = bb->build_inst(Op::EQ, arg2, minus1_inst);
	    bb->build_inst(Op::UB, bb->build_inst(Op::AND, cond1, cond2));
	  }
	Inst *zero_inst = bb->value_inst(0, arg1->bitsize);
	Inst *cond = bb->build_inst(Op::EQ, arg2, zero_inst);
	bb->build_inst(Op::UB, cond);
	Op op = is_unsigned ? Op::UDIV : Op::SDIV;
	return {bb->build_inst(op, arg1, arg2), res_indef, nullptr};
      }
    case TRUNC_MOD_EXPR:
      {
	if (!TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Inst *min_int_inst = build_min_int(bb, arg1->bitsize);
	    Inst *minus1_inst = bb->value_inst(-1, arg1->bitsize);
	    Inst *cond1 = bb->build_inst(Op::EQ, arg1, min_int_inst);
	    Inst *cond2 = bb->build_inst(Op::EQ, arg2, minus1_inst);
	    bb->build_inst(Op::UB, bb->build_inst(Op::AND, cond1, cond2));
	  }
	Inst *zero_inst = bb->value_inst(0, arg1->bitsize);
	Inst *cond = bb->build_inst(Op::EQ, arg2, zero_inst);
	bb->build_inst(Op::UB, cond);
	Op op = is_unsigned ? Op::UREM : Op::SREM;
	return {bb->build_inst(op, arg1, arg2), res_indef, nullptr};
      }
    case MAX_EXPR:
      {
	if ((arg1_prov || arg2_prov) && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in MAX_EXPR");
	Op op = is_unsigned ? Op::ULE : Op::SLE;
	Inst *cond = bb->build_inst(op, arg2, arg1);
	Inst *res = bb->build_inst(Op::ITE, cond, arg1, arg2);
	return {res, res_indef, arg1_prov};
      }
    case MIN_EXPR:
      {
	if ((arg1_prov || arg2_prov) && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in MIN_EXPR");
	Op op = is_unsigned ? Op::ULT : Op::SLT;
	Inst *cond = bb->build_inst(op, arg1, arg2);
	Inst *res = bb->build_inst(Op::ITE, cond, arg1, arg2);
	return {res, res_indef, arg1_prov};
      }
    case POINTER_PLUS_EXPR:
      {
	assert(arg1_prov);
	Inst *ptr = bb->build_inst(Op::ADD, arg1, arg2);

	// The resulting pointer cannot be NULL if arg1 or arg2 is non-zero.
	// (However, GIMPLE allows NULL + 0).
	//
	// It is enough to check one of the arguments (as, e.g., arg1 must be
	// zero if arg2 is zero and arg1 + arg2 == 0), and checking arg2 is
	// in general more efficient compared to arg1 as arg2 often is constant
	// and the comparison therefore can be constant folded.
	if (flag_delete_null_pointer_checks)
	  {
	    Inst *zero = bb->value_inst(0, ptr->bitsize);
	    Inst *cond1 = bb->build_inst(Op::EQ, ptr, zero);
	    Inst *cond2 = bb->build_inst(Op::NE, arg2, zero);
	    bb->build_inst(Op::UB, bb->build_inst(Op::AND, cond1, cond2));
	  }

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Inst *sub_overflow = bb->build_inst(Op::ULT, arg1, ptr);
	    Inst *add_overflow = bb->build_inst(Op::ULT, ptr, arg1);
	    Inst *zero = bb->value_inst(0, arg2->bitsize);
	    Inst *is_sub = bb->build_inst(Op::SLT, arg2, zero);
	    Inst *is_ub =
	      bb->build_inst(Op::ITE, is_sub, sub_overflow, add_overflow);
	    bb->build_inst(Op::UB, is_ub);
	  }

	return {ptr, res_indef, arg1_prov};
      }
    case MINUS_EXPR:
      {
	Inst *prov = nullptr;
	if (arg1_prov && !arg2_prov)
	  prov = arg1_prov;
	if (arg2_prov && !arg1_prov)
	  prov = arg2_prov;

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  bb->build_inst(Op::UB, bb->build_inst(Op::SSUB_WRAPS, arg1, arg2));
	return {bb->build_inst(Op::SUB, arg1, arg2), res_indef, prov};
      }
    case PLUS_EXPR:
      {
	Inst *prov = nullptr;
	if (arg1_prov && arg2_prov && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in PLUS_EXPR");
	if (arg1_prov)
	  prov = arg1_prov;
	if (arg2_prov)
	  prov = arg2_prov;

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  bb->build_inst(Op::UB, bb->build_inst(Op::SADD_WRAPS, arg1, arg2));
	return {bb->build_inst(Op::ADD, arg1, arg2), res_indef, prov};
      }
    case GE_EXPR:
      {
	Op op = is_unsigned ? Op::ULE : Op::SLE;
	return {bb->build_inst(op, arg2, arg1), res_indef, nullptr};
      }
    case GT_EXPR:
      {
	Op op = is_unsigned ? Op::ULT : Op::SLT;
	return {bb->build_inst(op, arg2, arg1), res_indef, nullptr};
      }
    case LE_EXPR:
      {
	Op op = is_unsigned ? Op::ULE : Op::SLE;
	return {bb->build_inst(op, arg1, arg2), res_indef, nullptr};
      }
    case LT_EXPR:
      {
	Op op = is_unsigned ? Op::ULT : Op::SLT;
	return {bb->build_inst(op, arg1, arg2), res_indef, nullptr};
      }
    case POINTER_DIFF_EXPR:
      {
	Inst *res = bb->build_inst(Op::SUB, arg1, arg2);

	// Pointers are treated as unsigned, and the result must fit in
	// a signed integer of the same width.
	assert(arg1->bitsize == arg2->bitsize);
	Inst *earg1 = bb->build_inst(Op::ZEXT, arg1, arg1->bitsize + 1);
	Inst *earg2 = bb->build_inst(Op::ZEXT, arg2, arg2->bitsize + 1);
	Inst *esub = bb->build_inst(Op::SUB, earg1, earg2);
	Inst *eres = bb->build_inst(Op::SEXT, res, arg2->bitsize + 1);
	Inst *is_ub = bb->build_inst(Op::NE, esub, eres);
	bb->build_inst(Op::UB, is_ub);

	return {res, res_indef, nullptr};
      }
    case WIDEN_MULT_EXPR:
      {
	assert(arg1->bitsize == arg2->bitsize);
	uint32_t new_bitsize = bitsize_for_type(lhs_type);
	Op op1 = TYPE_UNSIGNED(arg1_type) ? Op::ZEXT : Op::SEXT;
	arg1 = bb->build_inst(op1, arg1, new_bitsize);
	Op op2 = TYPE_UNSIGNED(arg2_type) ? Op::ZEXT : Op::SEXT;
	arg2 = bb->build_inst(op2, arg2, new_bitsize);
	return {bb->build_inst(Op::MUL, arg1, arg2), res_indef, nullptr};
      }
    case WIDEN_SUM_EXPR:
      {
	uint32_t new_bitsize = bitsize_for_type(lhs_type);
	Op op1 = TYPE_UNSIGNED(arg1_type) ? Op::ZEXT : Op::SEXT;
	arg1 = bb->build_inst(op1, arg1, new_bitsize);
	return {bb->build_inst(Op::ADD, arg1, arg2), res_indef, nullptr};
      }
    case MULT_HIGHPART_EXPR:
      {
	assert(arg1->bitsize == arg2->bitsize);
	assert(TYPE_UNSIGNED(arg1_type) == TYPE_UNSIGNED(arg2_type));
	Op op = is_unsigned ? Op::ZEXT : Op::SEXT;
	arg1 = bb->build_inst(op, arg1, 2 * arg1->bitsize);
	arg2 = bb->build_inst(op, arg2, 2 * arg2->bitsize);
	Inst *mul = bb->build_inst(Op::MUL, arg1, arg2);
	Inst *high = bb->value_inst(mul->bitsize - 1, 32);
	Inst *low = bb->value_inst(mul->bitsize / 2, 32);
	Inst *res = bb->build_inst(Op::EXTRACT, mul, high, low);
	return {res, res_indef, nullptr};
      }
    default:
      break;
    }

  throw Not_implemented("process_binary_int: "s + get_tree_code_name(code));
}

Inst *Converter::process_binary_scalar(enum tree_code code, Inst *arg1, Inst *arg2, tree lhs_type, tree arg1_type, tree arg2_type, bool ignore_overflow)
{
  auto [inst, indef, prov] =
    process_binary_scalar(code, arg1, nullptr, nullptr, arg2, nullptr, nullptr, lhs_type, arg1_type, arg2_type, ignore_overflow);
  assert(!indef);
  assert(!prov);
  return inst;
}

std::tuple<Inst *, Inst *, Inst *> Converter::process_binary_scalar(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg1_prov, Inst *arg2, Inst *arg2_indef, Inst *arg2_prov, tree lhs_type, tree arg1_type, tree arg2_type, bool ignore_overflow)
{
  if (TREE_CODE(lhs_type) == BOOLEAN_TYPE)
    {
      auto [inst, indef] =
	process_binary_bool(code, arg1, arg1_indef, arg2, arg2_indef,
			    lhs_type, arg1_type, arg2_type);
      return {inst, indef, nullptr};
    }
  else if (FLOAT_TYPE_P(lhs_type))
    {
      auto [inst, indef] =
	process_binary_float(code, arg1, arg1_indef, arg2, arg2_indef,
			     lhs_type);
      return {inst, indef, nullptr};
    }
  else
    return process_binary_int(code, TYPE_UNSIGNED(arg1_type),
			      arg1, arg1_indef, arg1_prov, arg2, arg2_indef,
			      arg2_prov, lhs_type, arg1_type, arg2_type,
			      ignore_overflow);
}

std::pair<Inst *, Inst *> Converter::process_binary_vec(enum tree_code code, Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type, tree arg1_type, tree arg2_type, bool ignore_overflow)
{
  if (code == WIDEN_SUM_EXPR)
    return process_widen_sum_vec(arg1, arg1_indef, arg2, arg2_indef,
				 lhs_type, arg1_type, arg2_type);
  if (code == VEC_WIDEN_MULT_EVEN_EXPR)
    return process_widen_mult_evenodd(arg1, arg1_indef, arg2, arg2_indef,
				      lhs_type, arg1_type, arg2_type, false);
  if (code == VEC_WIDEN_MULT_ODD_EXPR)
    return process_widen_mult_evenodd(arg1, arg1_indef, arg2, arg2_indef,
				      lhs_type, arg1_type, arg2_type, true);
  if (code == VEC_SERIES_EXPR)
    return process_vec_series(arg1, arg1_indef, arg2, arg2_indef, lhs_type);

  assert(VECTOR_TYPE_P(lhs_type));
  assert(VECTOR_TYPE_P(arg1_type));
  tree lhs_elem_type = TREE_TYPE(lhs_type);
  tree arg1_elem_type = TREE_TYPE(arg1_type);
  tree arg2_elem_type;
  if (VECTOR_TYPE_P(arg2_type))
    arg2_elem_type = TREE_TYPE(arg2_type);
  else
    arg2_elem_type = arg2_type;

  if (code == VEC_PACK_TRUNC_EXPR || code == VEC_PACK_FIX_TRUNC_EXPR)
    {
      Inst *arg = bb->build_inst(Op::CONCAT, arg2, arg1);
      Inst *arg_indef = nullptr;
      if (arg1_indef || arg2_indef)
	{
	  if (!arg1_indef)
	    arg1_indef = bb->value_inst(0, arg1->bitsize);
	  if (!arg2_indef)
	    arg2_indef = bb->value_inst(0, arg2->bitsize);
	  arg_indef = bb->build_inst(Op::CONCAT, arg2_indef, arg1_indef);
	}
      return process_unary_vec(CONVERT_EXPR, arg, arg_indef, lhs_elem_type,
			       arg1_elem_type);
    }

  uint32_t elem_bitsize = bitsize_for_type(arg1_elem_type);
  uint32_t nof_elt = bitsize_for_type(arg1_type) / elem_bitsize;
  uint32_t start_idx = 0;

  if (code == VEC_WIDEN_MULT_LO_EXPR || code == VEC_WIDEN_MULT_HI_EXPR)
    {
      if (code == VEC_WIDEN_MULT_HI_EXPR)
	start_idx = nof_elt / 2;
      else
	nof_elt = nof_elt / 2;
      code = WIDEN_MULT_EXPR;
    }

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint64_t i = start_idx; i < nof_elt; i++)
    {
      Inst *a1_indef = nullptr;
      Inst *a2_indef = nullptr;
      Inst *a1 = extract_vec_elem(bb, arg1, elem_bitsize, i);
      if (arg1_indef)
	a1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, i);
      Inst *a2;
      if (VECTOR_TYPE_P(arg2_type))
	{
	  a2 = extract_vec_elem(bb, arg2, elem_bitsize, i);
	  if (arg2_indef)
	    a2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, i);
	}
      else
	{
	  a2 = arg2;
	  if (arg2_indef)
	    a2_indef = arg2_indef;
	}
      auto [inst, inst_indef, _] =
	process_binary_scalar(code, a1, a1_indef, nullptr, a2, a2_indef,
			      nullptr, lhs_elem_type, arg1_elem_type,
			      arg2_elem_type, ignore_overflow);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg1_indef || arg2_indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, inst_indef, res_indef);
	  else
	    res_indef = inst_indef;
	}
    }
  return {res, res_indef};
}

std::pair<Inst *, Inst *> Converter::process_widen_sum_vec(Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type, tree arg1_type, tree arg2_type)
{
  assert(VECTOR_TYPE_P(lhs_type));
  assert(VECTOR_TYPE_P(arg1_type));
  tree lhs_elem_type = TREE_TYPE(lhs_type);
  tree arg1_elem_type = TREE_TYPE(arg1_type);
  tree arg2_elem_type = TREE_TYPE(arg2_type);

  uint32_t elem_bitsize = bitsize_for_type(arg2_elem_type);
  uint32_t nof_elt = bitsize_for_type(arg2_type) / elem_bitsize;

  uint32_t arg1_elem_bitsize = bitsize_for_type(arg1_elem_type);
  uint32_t arg1_nof_elt = bitsize_for_type(arg1_type) / arg1_elem_bitsize;
  assert(arg1_nof_elt >= nof_elt);
  assert(arg1_nof_elt % nof_elt == 0);

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint64_t i = 0; i < arg1_nof_elt; i++)
    {
      Inst *a1_indef = nullptr;
      Inst *a2_indef = nullptr;
      Inst *a1 = extract_vec_elem(bb, arg1, arg1_elem_bitsize, i);
      if (arg1_indef)
	a1_indef = extract_vec_elem(bb, arg1_indef, arg1_elem_bitsize, i);
      Inst *a2 = extract_vec_elem(bb, arg2, elem_bitsize, i % nof_elt);
      if (arg2_indef)
	a2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, i % nof_elt);
      auto [inst, inst_indef, _] =
	process_binary_scalar(WIDEN_SUM_EXPR,
			      a1, a1_indef, nullptr,
			      a2, a2_indef, nullptr,
			      lhs_elem_type, arg1_elem_type, arg2_elem_type,
			      true);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg1_indef || arg2_indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, inst_indef, res_indef);
	  else
	    res_indef = inst_indef;
	}

      if ((i + 1) % nof_elt == 0)
	{
	  arg2 = res;
	  arg2_indef = res_indef;
	  res = nullptr;
	  res_indef = nullptr;
	}
    }
  return {arg2, arg2_indef};
}

std::pair<Inst *, Inst *> Converter::process_widen_mult_evenodd(Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type, tree arg1_type, tree arg2_type, bool is_odd)
{
  assert(VECTOR_TYPE_P(lhs_type));
  assert(VECTOR_TYPE_P(arg1_type));
  tree lhs_elem_type = TREE_TYPE(lhs_type);
  tree arg1_elem_type = TREE_TYPE(arg1_type);
  tree arg2_elem_type = TREE_TYPE(arg2_type);

  uint32_t elem_bitsize = bitsize_for_type(arg1_elem_type);
  uint32_t nof_elt = bitsize_for_type(arg1_type) / elem_bitsize;

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint64_t i = is_odd; i < nof_elt; i += 2)
    {
      Inst *a1_indef = nullptr;
      Inst *a2_indef = nullptr;
      Inst *a1 = extract_vec_elem(bb, arg1, elem_bitsize, i);
      if (arg1_indef)
	a1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, i);
      Inst *a2 = extract_vec_elem(bb, arg2, elem_bitsize, i);
      if (arg2_indef)
	a2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, i);
      auto [inst, inst_indef, _] =
	process_binary_scalar(WIDEN_MULT_EXPR, a1, a1_indef, nullptr,
			      a2, a2_indef, nullptr,
			      lhs_elem_type, arg1_elem_type, arg2_elem_type);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg1_indef || arg2_indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, inst_indef, res_indef);
	  else
	    res_indef = inst_indef;
	}
    }
  return {res, res_indef};
}

std::pair<Inst *, Inst *> Converter::process_vec_series(Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, tree lhs_type)
{
  tree elem_type = TREE_TYPE(lhs_type);
  uint32_t elem_bitsize = bitsize_for_type(elem_type);
  uint32_t bitsize = bitsize_for_type(lhs_type);
  uint32_t nof_elem = bitsize / elem_bitsize;
  assert(arg1->bitsize == elem_bitsize);
  assert(arg2->bitsize == elem_bitsize);

  Inst *elem_indef = nullptr;
  if (arg1_indef || arg2_indef)
    {
      if (!arg1_indef)
	arg1_indef = bb->value_inst(0, arg1->bitsize);
      elem_indef = get_res_indef(arg1_indef, arg2_indef, elem_type);
    }
  Inst *res = arg1;
  Inst *res_indef = arg1_indef;
  Inst *elem = arg1;
  for (uint32_t i = 1; i < nof_elem; i++)
    {
      elem = bb->build_inst(Op::ADD, elem, arg2);
      res = bb->build_inst(Op::CONCAT, elem, res);
      if (res_indef)
	res_indef = bb->build_inst(Op::CONCAT, elem_indef, res_indef);
    }
  return {res, res_indef};
}

Inst *Converter::process_ternary(enum tree_code code, Inst *arg1, Inst *arg2, Inst *arg3, tree arg1_type, tree arg2_type, tree arg3_type)
{
  switch (code)
    {
    case SAD_EXPR:
      {
	arg1 = type_convert(arg1, arg1_type, arg3_type);
	arg2 = type_convert(arg2, arg2_type, arg3_type);
	Inst *inst = bb->build_inst(Op::SUB, arg1, arg2);
	Inst *zero = bb->value_inst(0, inst->bitsize);
	Inst *cmp = bb->build_inst(Op::SLE, zero, inst);
	Inst *neg = bb->build_inst(Op::NEG, inst);
	inst = bb->build_inst(Op::ITE, cmp, inst, neg);
	return bb->build_inst(Op::ADD, inst, arg3);
      }
    case DOT_PROD_EXPR:
      {
	arg1 = type_convert(arg1, arg1_type, arg3_type);
	arg2 = type_convert(arg2, arg2_type, arg3_type);
	Inst *inst = bb->build_inst(Op::MUL, arg1, arg2);
	return bb->build_inst(Op::ADD, inst, arg3);
      }
    case WIDEN_MULT_MINUS_EXPR:
      {
	assert(arg1->bitsize == arg2->bitsize);
	uint32_t new_bitsize = bitsize_for_type(arg3_type);
	Op op1 = TYPE_UNSIGNED(arg1_type) ? Op::ZEXT : Op::SEXT;
	arg1 = bb->build_inst(op1, arg1, new_bitsize);
	Op op2 = TYPE_UNSIGNED(arg2_type) ? Op::ZEXT : Op::SEXT;
	arg2 = bb->build_inst(op2, arg2, new_bitsize);
	Inst *mul = bb->build_inst(Op::MUL, arg1, arg2);
	return bb->build_inst(Op::SUB, arg3, mul);
      }
    default:
      throw Not_implemented("process_ternary: "s + get_tree_code_name(code));
    }
}

std::tuple<Inst *, Inst *, Inst *> Converter::process_ternary(enum tree_code code, tree arg1_tree, tree arg2_tree, tree arg3_tree)
{
  switch (code)
    {
    case WIDEN_MULT_PLUS_EXPR:
      {
	tree src_type = TREE_TYPE(arg1_tree);
	tree dest_type = TREE_TYPE(arg3_tree);
	auto [arg1, arg1_indef, arg1_prov] = tree2inst_indef_prov(arg1_tree);
	auto [arg2, arg2_indef, arg2_prov] = tree2inst_indef_prov(arg2_tree);
	auto [arg3, arg3_indef, arg3_prov] = tree2inst_indef_prov(arg3_tree);
	std::tie(arg1, arg1_indef, arg1_prov) =
	  type_convert(arg1, arg1_indef, arg1_prov, src_type, dest_type);
	std::tie(arg2, arg2_indef, arg2_prov) =
	  type_convert(arg2, arg2_indef, arg2_prov, src_type, dest_type);
	Inst *mul = bb->build_inst(Op::MUL, arg1, arg2);
	Inst *res = bb->build_inst(Op::ADD, mul, arg3);
	if (arg1_prov || arg2_prov)
	  throw Not_implemented("arg1/arg2 provenance in WIDEN_MULT_PLUS_EXPR");
	Inst *res_indef =
	  get_res_indef(arg1_indef, arg2_indef, arg3_indef, dest_type);
	return {res, res_indef, arg3_prov};
      }
    default:
      throw Not_implemented("process_ternary: "s + get_tree_code_name(code));
    }
}

Inst *Converter::process_ternary_vec(enum tree_code code, Inst *arg1, Inst *arg2, Inst *arg3, tree lhs_type, tree arg1_type, tree arg2_type, tree arg3_type)
{
  assert(VECTOR_TYPE_P(lhs_type));
  assert(VECTOR_TYPE_P(arg1_type));
  assert(VECTOR_TYPE_P(arg2_type));
  assert(VECTOR_TYPE_P(arg3_type));

  tree arg1_elem_type = TREE_TYPE(arg1_type);
  uint32_t arg1_elem_bitsize = bitsize_for_type(arg1_elem_type);
  tree arg2_elem_type = TREE_TYPE(arg2_type);
  uint32_t arg2_elem_bitsize = bitsize_for_type(arg2_elem_type);
  tree arg3_elem_type = TREE_TYPE(arg3_type);
  uint32_t arg3_elem_bitsize = bitsize_for_type(arg3_elem_type);

  uint32_t nof_elt3 = bitsize_for_type(arg3_type) / arg3_elem_bitsize;
  uint32_t nof_elt = bitsize_for_type(arg1_type) / arg1_elem_bitsize;
  Inst *res = nullptr;
  for (uint64_t i = 0; i < nof_elt; i++)
    {
      Inst *a1 = extract_vec_elem(bb, arg1, arg1_elem_bitsize, i);
      Inst *a2 = extract_vec_elem(bb, arg2, arg2_elem_bitsize, i);
      // Instructions such as SAD_EXPR has fewer elements in the arg3,
      // and it iterates multiple times and updates that.
      uint32_t i3 = i % nof_elt3;
      if (!i3 && res)
	{
	  arg3 = res;
	  res = nullptr;
	}
      Inst *a3 = extract_vec_elem(bb, arg3, arg3_elem_bitsize, i3);
      Inst *inst = process_ternary(code, a1, a2, a3, arg1_elem_type,
					  arg2_elem_type, arg3_elem_type);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  return res;
}

std::pair<Inst *, Inst *> Converter::gen_vec_cond(Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, Inst *arg3, Inst *arg3_indef, tree arg1_type, tree arg2_type, Inst *len)
{
  assert(VECTOR_TYPE_P(arg1_type));
  assert(VECTOR_TYPE_P(arg2_type));
  assert(arg2->bitsize == arg3->bitsize);

  if (arg2_indef || arg3_indef)
    {
      if (!arg2_indef)
	arg2_indef = bb->value_inst(0, arg2->bitsize);
      if (!arg3_indef)
	arg3_indef = bb->value_inst(0, arg3->bitsize);
    }

  tree arg1_elem_type = TREE_TYPE(arg1_type);
  assert(TREE_CODE(arg1_elem_type) == BOOLEAN_TYPE);
  tree arg2_elem_type = TREE_TYPE(arg2_type);

  uint32_t elem_bitsize1 = bitsize_for_type(arg1_elem_type);
  uint32_t elem_bitsize2 = bitsize_for_type(arg2_elem_type);

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  uint32_t nof_elt = bitsize_for_type(arg1_type) / elem_bitsize1;
  for (uint64_t i = 0; i < nof_elt; i++)
    {
      Inst *a1 = extract_vec_elem(bb, arg1, elem_bitsize1, i);
      if (a1->bitsize > 1)
	a1 = bb->build_extract_bit(a1, 0);
      Inst *a2 = extract_vec_elem(bb, arg2, elem_bitsize2, i);
      Inst *a3 = extract_vec_elem(bb, arg3, elem_bitsize2, i);
      if (len)
	{
	  Inst *i_inst = bb->value_inst(i, len->bitsize);
	  Inst *cmp = bb->build_inst(Op::ULT, i_inst, len);
	  a1 = bb->build_inst(Op::AND, cmp, a1);
	}
      Inst *inst = bb->build_inst(Op::ITE, a1, a2, a3);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      Inst *indef = nullptr;
      if (arg1_indef)
	{
	  Inst *a1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize1, i);
	  if (a1_indef->bitsize > 1)
	    a1_indef = bb->build_extract_bit(a1_indef, 0);
	  if (len)
	    {
	      Inst *i_inst = bb->value_inst(i, len->bitsize);
	      Inst *cmp = bb->build_inst(Op::ULT, i_inst, len);
	      a1_indef = bb->build_inst(Op::AND, cmp, a1_indef);
	    }
	  indef = get_res_indef(a1_indef, arg2_elem_type);
	}
      if (arg2_indef)
	{
	  Inst *a2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize2, i);
	  Inst *a3_indef = extract_vec_elem(bb, arg3_indef, elem_bitsize2, i);
	  Inst *indef2 = bb->build_inst(Op::ITE, a1, a2_indef, a3_indef);
	  if (indef)
	    indef = bb->build_inst(Op::OR, indef, indef2);
	  else
	    indef = indef2;
	}
      if (indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, indef, res_indef);
	  else
	    res_indef = indef;
	}
    }

  return {res, res_indef};
}

std::pair<Inst *, Inst *> Converter::process_vec_perm_expr(gimple *stmt)
{
  auto [arg1, arg1_indef] = tree2inst_indef(gimple_assign_rhs1(stmt));
  auto [arg2, arg2_indef] = tree2inst_indef(gimple_assign_rhs2(stmt));
  Inst *arg3 = tree2inst(gimple_assign_rhs3(stmt));
  assert(arg1->bitsize == arg2->bitsize);
  tree arg1_type = TREE_TYPE(gimple_assign_rhs1(stmt));
  tree arg1_elem_type = TREE_TYPE(arg1_type);
  tree arg3_type = TREE_TYPE(gimple_assign_rhs3(stmt));
  tree arg3_elem_type = TREE_TYPE(arg3_type);
  uint32_t elem_bitsize1 = bitsize_for_type(arg1_elem_type);
  uint32_t elem_bitsize3 = bitsize_for_type(arg3_elem_type);
  uint32_t nof_elt1 = bitsize_for_type(arg1_type) / elem_bitsize1;
  uint32_t nof_elt3 = bitsize_for_type(arg3_type) / elem_bitsize3;

  if (arg1_indef || arg2_indef)
    {
      if (!arg1_indef)
	arg1_indef = bb->value_inst(0, arg1->bitsize);
      if (!arg2_indef)
	arg2_indef = bb->value_inst(0, arg2->bitsize);
    }

  Inst *mask1 = bb->value_inst(nof_elt1 * 2 - 1, elem_bitsize3);
  Inst *mask2 = bb->value_inst(nof_elt1 - 1, elem_bitsize3);
  Inst *nof_elt_inst = bb->value_inst(nof_elt1, elem_bitsize3);
  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint64_t i = 0; i < nof_elt3; i++)
    {
      Inst *idx1 = extract_vec_elem(bb, arg3, elem_bitsize3, i);
      idx1 = bb->build_inst(Op::AND, idx1, mask1);
      Inst *idx2 = bb->build_inst(Op::AND, idx1, mask2);
      Inst *cmp = bb->build_inst(Op::ULT, idx1,  nof_elt_inst);
      Inst *elt1 = extract_elem(bb, arg1, elem_bitsize1, idx2);
      Inst *elt2 = extract_elem(bb, arg2, elem_bitsize1, idx2);
      Inst *inst = bb->build_inst(Op::ITE, cmp, elt1, elt2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg1_indef)
	{
	  Inst *indef1 = extract_elem(bb, arg1_indef, elem_bitsize1, idx2);
	  Inst *indef2 = extract_elem(bb, arg2_indef, elem_bitsize1, idx2);
	  Inst *indef = bb->build_inst(Op::ITE, cmp, indef1, indef2);
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, indef, res_indef);
	  else
	    res_indef = indef;
	}
    }
  return {res, res_indef};
}

std::tuple<Inst *, Inst *, Inst *> Converter::vector_constructor(tree expr)
{
  assert(TREE_CODE(expr) == CONSTRUCTOR);
  assert(VECTOR_TYPE_P(TREE_TYPE(expr)));
  unsigned HOST_WIDE_INT idx;
  tree value;
  uint32_t vector_bitsize = bitsize_for_type(TREE_TYPE(expr));
  Inst *res = nullptr;
  Inst *indef = nullptr;
  bool any_elem_has_indef = false;
  // Note: The contstuctor elements may have different sizes. For example,
  // we may create a vector by concatenating a scalar with a vector.
  FOR_EACH_CONSTRUCTOR_VALUE(CONSTRUCTOR_ELTS(expr), idx, value)
    {
      auto [elem, elem_indef, elem_prov] = tree2inst_constructor(value);
      if (elem_indef)
	{
	  any_elem_has_indef = true;
	}
      else
	elem_indef = bb->value_inst(0, elem->bitsize);
      if (res)
	{
	  res = bb->build_inst(Op::CONCAT, elem, res);
	  indef = bb->build_inst(Op::CONCAT, elem_indef, indef);
	}
      else
	{
	  assert(idx == 0);
	  res = elem;
	  indef = elem_indef;
	}
    }
  if (CONSTRUCTOR_NO_CLEARING(expr))
    throw Not_implemented("vector_constructor: CONSTRUCTOR_NO_CLEARING");
  if (!res)
    res = bb->value_inst(0, vector_bitsize);
  else if (res->bitsize != vector_bitsize)
    {
      assert(res->bitsize < vector_bitsize);
      Inst *zero = bb->value_inst(0, vector_bitsize - res->bitsize);
      res = bb->build_inst(Op::CONCAT, zero, res);
      indef = bb->build_inst(Op::CONCAT, zero, indef);
    }

  if (!any_elem_has_indef)
    {
      // No element had indef information, so `indef` only consists of the
      // zero values we creates. Change it to the `nullptr` so that later
      // code does not need to add UB comparisions each place the result
      // is used.
      indef = nullptr;
    }
  return {res, indef, nullptr};
}

void Converter::process_constructor(tree lhs, tree rhs)
{
  Addr addr = process_address(lhs, true);
  assert(!addr.bitoffset);

  if (TREE_CLOBBER_P(rhs) && CLOBBER_KIND(rhs) == CLOBBER_STORAGE_END)
    {
      bb->build_inst(Op::FREE, extract_id(addr.ptr));
      return;
    }

  assert(!CONSTRUCTOR_NO_CLEARING(rhs));
  uint64_t size = bytesize_for_type(TREE_TYPE(rhs));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("process_constructor: too large constructor");
  store_ub_check(addr.ptr, addr.prov, size);

  if (TREE_CLOBBER_P(rhs))
    make_uninit(addr.ptr, size);
  else
    {
      Inst *zero = bb->value_inst(0, 8);
      Inst *memory_flag = bb->value_inst(0, 1);
      for (uint64_t i = 0; i < size; i++)
	{
	  Inst *offset = bb->value_inst(i, addr.ptr->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);
	  uint8_t padding = padding_at_offset(TREE_TYPE(rhs), i);
	  Inst *indef = bb->value_inst(padding, 8);
	  bb->build_inst(Op::STORE, ptr, zero);
	  bb->build_inst(Op::SET_MEM_INDEF, ptr, indef);
	  bb->build_inst(Op::SET_MEM_FLAG, ptr, memory_flag);
	}
    }

  assert(!CONSTRUCTOR_NELTS(rhs));
}

void Converter::process_gimple_assign(gimple *stmt)
{
  tree lhs = gimple_assign_lhs(stmt);
  check_type(TREE_TYPE(lhs));
  enum tree_code code = gimple_assign_rhs_code(stmt);

  if (TREE_CODE(lhs) != SSA_NAME)
    {
      assert(get_gimple_rhs_class(code) == GIMPLE_SINGLE_RHS);
      tree rhs = gimple_assign_rhs1(stmt);
      if (TREE_CODE(rhs) == CONSTRUCTOR)
	process_constructor(lhs, rhs);
      else
	process_store(lhs, rhs);
      return;
    }

  tree rhs1 = gimple_assign_rhs1(stmt);
  check_type(TREE_TYPE(rhs1));
  Inst *inst;
  Inst *indef = nullptr;
  Inst *prov = nullptr;
  switch (get_gimple_rhs_class(code))
    {
    case GIMPLE_TERNARY_RHS:
      {
	if (code == SAD_EXPR
	    || code == DOT_PROD_EXPR
	    || code == WIDEN_MULT_MINUS_EXPR)
	  {
	    Inst *arg1 = tree2inst(gimple_assign_rhs1(stmt));
	    Inst *arg2 = tree2inst(gimple_assign_rhs2(stmt));
	    Inst *arg3 = tree2inst(gimple_assign_rhs3(stmt));
	    tree lhs_type = TREE_TYPE(gimple_assign_lhs(stmt));
	    tree arg1_type = TREE_TYPE(gimple_assign_rhs1(stmt));
	    tree arg2_type = TREE_TYPE(gimple_assign_rhs2(stmt));
	    tree arg3_type = TREE_TYPE(gimple_assign_rhs3(stmt));
	    if (VECTOR_TYPE_P(lhs_type))
	      inst = process_ternary_vec(code, arg1, arg2, arg3, lhs_type,
					 arg1_type, arg2_type, arg3_type);
	    else
	      inst = process_ternary(code, arg1, arg2, arg3, arg1_type,
				     arg2_type, arg3_type);
	  }
	else if (code == WIDEN_MULT_PLUS_EXPR)
	  {
	    tree lhs_type = TREE_TYPE(gimple_assign_lhs(stmt));
	    if (VECTOR_TYPE_P(lhs_type))
	      throw Not_implemented("process_gimple_assign: vector "s
				    + get_tree_code_name(code));
	    tree arg1_tree = gimple_assign_rhs1(stmt);
	    tree arg2_tree = gimple_assign_rhs2(stmt);
	    tree arg3_tree = gimple_assign_rhs3(stmt);
	    std::tie(inst, indef, prov) =
	      process_ternary(code, arg1_tree, arg2_tree, arg3_tree);
	  }
	else if (code == VEC_PERM_EXPR)
	  std::tie(inst, indef) = process_vec_perm_expr(stmt);
	else if (code == VEC_COND_EXPR)
	  {
	    auto [arg1, arg1_indef] =
	      tree2inst_indef(gimple_assign_rhs1(stmt));
	    auto [arg2, arg2_indef] =
	      tree2inst_indef(gimple_assign_rhs2(stmt));
	    auto [arg3, arg3_indef] =
	      tree2inst_indef(gimple_assign_rhs3(stmt));
	    tree arg1_type = TREE_TYPE(gimple_assign_rhs1(stmt));
	    tree arg2_type = TREE_TYPE(gimple_assign_rhs2(stmt));
	    std::tie(inst, indef) =
	      gen_vec_cond(arg1, arg1_indef, arg2, arg2_indef,
			   arg3, arg3_indef, arg1_type, arg2_type);
	  }
	else if (code == COND_EXPR)
	  {
	    tree rhs1_type = TREE_TYPE(gimple_assign_rhs1(stmt));
	    tree rhs2_type = TREE_TYPE(gimple_assign_rhs2(stmt));
	    assert(TREE_CODE(rhs1_type) == BOOLEAN_TYPE);
	    auto [arg1, arg1_indef, arg1_prov] =
	      tree2inst_indef_prov(gimple_assign_rhs1(stmt));
	    if (TYPE_PRECISION(rhs1_type) != 1)
	      {
		arg1 = bb->build_extract_bit(arg1, 0);
		if (arg1_indef)
		  arg1_indef = bb->build_extract_bit(arg1_indef, 0);
	      }
	    auto [arg2, arg2_indef, arg2_prov] =
	      tree2inst_indef_prov(gimple_assign_rhs2(stmt));
	    auto [arg3, arg3_indef, arg3_prov] =
	      tree2inst_indef_prov(gimple_assign_rhs3(stmt));
	    if (arg1_indef)
	      indef = get_res_indef(arg1_indef, rhs2_type);
	    if (arg2_indef || arg3_indef)
	      {
		if (!arg2_indef)
		  arg2_indef = bb->value_inst(0, arg2->bitsize);
		if (!arg3_indef)
		  arg3_indef = bb->value_inst(0, arg3->bitsize);
		Inst *arg_indef =
		  bb->build_inst(Op::ITE, arg1, arg2_indef, arg3_indef);
		if (indef)
		  indef = bb->build_inst(Op::OR, indef, arg_indef);
		else
		  indef = arg_indef;
	      }
	    if (arg2_prov && arg3_prov)
	      prov = bb->build_inst(Op::ITE, arg1, arg2_prov, arg3_prov);
	    inst = bb->build_inst(Op::ITE, arg1, arg2, arg3);
	  }
	else if (code == BIT_INSERT_EXPR)
	  {
	    auto [arg1, arg1_indef] =
	      tree2inst_indef(gimple_assign_rhs1(stmt));
	    tree arg2_expr = gimple_assign_rhs2(stmt);
	    auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
	    bool has_indef = arg1_indef || arg2_indef;
	    if (has_indef)
	      {
		if (!arg1_indef)
		  arg1_indef = bb->value_inst(0, arg1->bitsize);
		if (!arg2_indef)
		  arg2_indef = bb->value_inst(0, arg2->bitsize);
	      }
	    uint64_t bit_pos = get_int_cst_val(gimple_assign_rhs3(stmt));
	    if (bit_pos > 0)
	      {
		Inst *extract = bb->build_trunc(arg1, bit_pos);
		inst = bb->build_inst(Op::CONCAT, arg2, extract);
		if (has_indef)
		  {
		    Inst *extract_indef = bb->build_trunc(arg1_indef, bit_pos);
		    indef =
		      bb->build_inst(Op::CONCAT, arg2_indef, extract_indef);
		  }
	      }
	    else
	      {
		inst = arg2;
		if (has_indef)
		  indef = arg2_indef;
	      }
	    if (bit_pos + arg2->bitsize != arg1->bitsize)
	      {
		Inst *high = bb->value_inst(arg1->bitsize - 1, 32);
		Inst *low = bb->value_inst(bit_pos + arg2->bitsize, 32);
		Inst *extract = bb->build_inst(Op::EXTRACT, arg1, high, low);
		inst = bb->build_inst(Op::CONCAT, extract, inst);
		if (has_indef)
		  {
		    Inst *extract_indef =
		      bb->build_inst(Op::EXTRACT, arg1_indef, high, low);
		    indef = bb->build_inst(Op::CONCAT, extract_indef, indef);
		  }
	      }
	  }
	else
	  throw Not_implemented("GIMPLE_TERNARY_RHS: "s + get_tree_code_name(code));
      }
      break;
    case GIMPLE_BINARY_RHS:
      {
	tree lhs_type = TREE_TYPE(gimple_assign_lhs(stmt));
	tree rhs1 = gimple_assign_rhs1(stmt);
	tree rhs2 = gimple_assign_rhs2(stmt);
	tree arg1_type = TREE_TYPE(rhs1);
	tree arg2_type = TREE_TYPE(rhs2);
	if (TREE_CODE(lhs_type) == COMPLEX_TYPE && code == COMPLEX_EXPR)
	  {
	    auto [arg1, arg1_indef] = tree2inst_indef(rhs1);
	    auto [arg2, arg2_indef] = tree2inst_indef(rhs2);
	    arg1 = to_mem_repr(arg1, TREE_TYPE(rhs1));
	    arg2 = to_mem_repr(arg2, TREE_TYPE(rhs2));
	    inst = bb->build_inst(Op::CONCAT, arg2, arg1);
	    if (arg1_indef || arg2_indef)
	      {
		if (arg1_indef)
		  arg1_indef = to_mem_repr(arg1_indef, TREE_TYPE(rhs1));
		if (arg2_indef)
		  arg2_indef = to_mem_repr(arg2_indef, TREE_TYPE(rhs2));
		if (!arg1_indef)
		  arg1_indef = bb->value_inst(0, arg1->bitsize);
		if (!arg2_indef)
		  arg2_indef = bb->value_inst(0, arg2->bitsize);
		indef =
		  bb->build_inst(Op::CONCAT, arg2_indef, arg1_indef);
	      }
	  }
	else
	  {
	    if (TREE_CODE(lhs_type) == COMPLEX_TYPE)
	      {
		Inst *arg1 = tree2inst(rhs1);
		Inst *arg2 = tree2inst(rhs2);
		inst = process_binary_complex(code, arg1, arg2, lhs_type);
	      }
	    else if (TREE_CODE(arg1_type) == COMPLEX_TYPE)
	      {
		Inst *arg1 = tree2inst(rhs1);
		Inst *arg2 = tree2inst(rhs2);
		inst = process_binary_complex_cmp(code, arg1, arg2, lhs_type,
						  arg1_type);
	      }
	    else if (VECTOR_TYPE_P(lhs_type))
	      {
		auto [arg1, arg1_indef] = tree2inst_indef(rhs1);
		auto [arg2, arg2_indef] = tree2inst_indef(rhs2);
		std::tie(inst, indef) =
		  process_binary_vec(code, arg1, arg1_indef, arg2,
				     arg2_indef, lhs_type, arg1_type,
				     arg2_type);
	      }
	    else
	      {
		auto [arg1, arg1_indef, arg1_prov] =
		  tree2inst_indef_prov(rhs1);
		auto [arg2, arg2_indef, arg2_prov] =
		  tree2inst_indef_prov(rhs2);
		std::tie(inst, indef, prov) =
		  process_binary_scalar(code, arg1, arg1_indef, arg1_prov,
					arg2, arg2_indef, arg2_prov, lhs_type,
					arg1_type, arg2_type);
	      }
	  }
      }
      break;
    case GIMPLE_UNARY_RHS:
      {
	tree rhs1 = gimple_assign_rhs1(stmt);
	tree lhs_type = TREE_TYPE(gimple_assign_lhs(stmt));
	tree arg1_type = TREE_TYPE(rhs1);
	if (TREE_CODE(lhs_type) == COMPLEX_TYPE
	    || TREE_CODE(arg1_type) == COMPLEX_TYPE)
	  {
	    Inst *arg1 = tree2inst(rhs1);
	    inst = process_unary_complex(code, arg1, lhs_type);
	  }
	else if (VECTOR_TYPE_P(lhs_type))
	  {
	    auto [arg1, arg1_indef] = tree2inst_indef(rhs1);
	    if (code == VEC_DUPLICATE_EXPR)
	      std::tie(inst, indef) =
		process_vec_duplicate(arg1, arg1_indef, lhs_type, arg1_type);
	    else
	      {
		tree lhs_elem_type = TREE_TYPE(lhs_type);
		tree arg1_elem_type = TREE_TYPE(arg1_type);
		std::tie(inst, indef) =
		  process_unary_vec(code, arg1, arg1_indef, lhs_elem_type,
				    arg1_elem_type);
	      }
	  }
	else
	  {
	    auto [arg1, arg1_indef, arg1_prov] = tree2inst_indef_prov(rhs1);
	    std::tie(inst, indef, prov) =
	      process_unary_scalar(code, arg1, arg1_indef, arg1_prov, lhs_type,
				   arg1_type);
	  }
      }
      break;
    case GIMPLE_SINGLE_RHS:
      std::tie(inst, indef, prov) =
	tree2inst_indef_prov(gimple_assign_rhs1(stmt));
      break;
    default:
      throw Not_implemented("unknown get_gimple_rhs_class");
    }

  constrain_range(bb, lhs, inst, indef);

  assert(TREE_CODE(lhs) == SSA_NAME);
  tree2instruction.insert({lhs, inst});
  if (indef)
    tree2indef.insert({lhs, indef});
  if (prov)
    tree2prov.insert({lhs, prov});
}

void Converter::process_gimple_asm(gimple *stmt)
{
  gasm *asm_stmt = as_a<gasm *>(stmt);
  const char *p = gimple_asm_string(asm_stmt);

  // We can ignore asm having an empty string (as they only constrain
  // optimizations in ways that does not affect us).
  while (*p)
    {
      if (!ISSPACE(*p++))
	throw Not_implemented("process_function: gimple_asm");
    }

  // Empty asm goto gives us problems with GIMPLE BBs with the wrong
  // number of EDGE_COUNT preds/succs. This is easy to fix, but does
  // not give us any benefit until we have real asm handling.
  if (gimple_asm_nlabels(asm_stmt))
    throw Not_implemented("process_function: gimple_asm");

  // Asm with output, such as
  //   asm volatile ("" : "+rm" (p));
  // needs some more work to constrain the values correctly. Report
  // "not implemented" for now.
  if (gimple_asm_noutputs(asm_stmt))
    throw Not_implemented("process_function: gimple_asm");
}

void Converter::process_cfn_binary(gimple *stmt, const std::function<std::pair<Inst *, Inst *>(Inst *, Inst *, Inst *, Inst *, tree)>& gen_elem)
{
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  assert(VECTOR_TYPE_P(TREE_TYPE(arg2_expr)) == VECTOR_TYPE_P(arg1_type));
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);

  uint32_t nof_elem;
  uint32_t elem_bitsize;
  tree elem_type;
  if (VECTOR_TYPE_P(arg1_type))
    {
      elem_type = TREE_TYPE(arg1_type);
      elem_bitsize = bitsize_for_type(elem_type);
      nof_elem = bitsize_for_type(arg1_type) / elem_bitsize;
    }
  else
    {
      elem_type = arg1_type;
      elem_bitsize = arg1->bitsize;
      nof_elem = 1;
    }

  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  assert(VECTOR_TYPE_P(TREE_TYPE(lhs)) == VECTOR_TYPE_P(arg1_type));

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint32_t j = 0; j < nof_elem; j++)
    {
      Inst *elem1 = extract_vec_elem(bb, arg1, elem_bitsize, j);
      Inst *elem2 = extract_vec_elem(bb, arg2, elem_bitsize, j);
      Inst *elem1_indef = nullptr;
      if (arg1_indef)
	elem1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, j);
      Inst *elem2_indef = nullptr;
      if (arg2_indef)
	elem2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, j);

      auto [inst, indef] =
	gen_elem(elem1, elem1_indef, elem2, elem2_indef, elem_type);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
      if (indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, indef, res_indef);
	  else
	    res_indef = indef;
	}
    }
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_unary(gimple *stmt, const std::function<std::pair<Inst *, Inst *>(Inst *, Inst *, tree)>& gen_elem)
{
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);

  uint32_t nof_elem;
  uint32_t elem_bitsize;
  tree elem_type;
  if (VECTOR_TYPE_P(arg1_type))
    {
      elem_type = TREE_TYPE(arg1_type);
      elem_bitsize = bitsize_for_type(elem_type);
      nof_elem = bitsize_for_type(arg1_type) / elem_bitsize;
    }
  else
    {
      elem_type = arg1_type;
      elem_bitsize = arg1->bitsize;
      nof_elem = 1;
    }

  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  assert(VECTOR_TYPE_P(TREE_TYPE(lhs)) == VECTOR_TYPE_P(arg1_type));
  tree lhs_elem_type = TREE_TYPE(lhs);
  if (VECTOR_TYPE_P(arg1_type))
    {
      lhs_elem_type = TREE_TYPE(lhs_elem_type);
    }

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint32_t j = 0; j < nof_elem; j++)
    {
      Inst *elem1 = extract_vec_elem(bb, arg1, elem_bitsize, j);
      Inst *elem1_indef = nullptr;
      if (arg1_indef)
	elem1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, j);

      auto [inst, indef] = gen_elem(elem1, elem1_indef, lhs_elem_type);

      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
      if (indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, indef, res_indef);
	  else
	    res_indef = indef;
	}
    }
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_abd(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Op op = TYPE_UNSIGNED(elem_type) ? Op::ZEXT : Op::SEXT;
      Inst *sub = bb->build_inst(Op::SUB, elem1, elem2);
      Inst *neg_sub = bb->build_inst(Op::NEG, sub);
      Inst *eelem1 = bb->build_inst(op, elem1, elem1->bitsize + 1);
      Inst *eelem2 = bb->build_inst(op, elem2, elem2->bitsize + 1);
      Inst *esub = bb->build_inst(Op::SUB, eelem1, eelem2);
      Inst *zero = bb->value_inst(0, esub->bitsize);
      Inst *cmp = bb->build_inst(Op::SLT, esub, zero);
      Inst *res = bb->build_inst(Op::ITE, cmp, neg_sub, sub);
      res = bb->build_trunc(res, elem1->bitsize);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_abort(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 0);
  // We may fail to check the abort call if there is UB after the call
  // in this basic block. This should not happen, as abort is marked as
  // noreturn, but it is unclear to me if this is guaranteed in GIMPLE.
  // We therefore ensure that we branch to the exit block and create a
  // new dead block for the remaining instructions, if any.
  basic_block gcc_exit_block = EXIT_BLOCK_PTR_FOR_FN(fun);
  Basic_block *exit_bb = gccbb_top2bb.at(gcc_exit_block);
  Basic_block *dead_bb = func->build_bb();
  bb->build_br_inst(bb->value_inst(1, 1), exit_bb, dead_bb);
  bb_abort.insert(bb);
  bb = dead_bb;
}

void Converter::process_cfn_add_overflow(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_add_overflow: vector type");
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  Inst *res_indef = get_res_indef(arg1_indef, arg2_indef, lhs_elem_type);
  if (res_indef)
    {
      Inst *overflow_indef = bb->build_trunc(res_indef, 1);
      overflow_indef =
	bb->build_inst(Op::ZEXT, overflow_indef, res_indef->bitsize);
      res_indef = to_mem_repr(res_indef, lhs_elem_type);
      overflow_indef = to_mem_repr(overflow_indef, lhs_elem_type);
      res_indef = bb->build_inst(Op::CONCAT, overflow_indef, res_indef);
    }

  unsigned lhs_elem_bitsize = bitsize_for_type(lhs_elem_type);
  unsigned bitsize = 1 + std::max(arg1->bitsize, arg2->bitsize);
  bitsize = 1 + std::max(bitsize, lhs_elem_bitsize);
  if (TYPE_UNSIGNED(arg1_type))
    arg1 = bb->build_inst(Op::ZEXT, arg1, bitsize);
  else
    arg1 = bb->build_inst(Op::SEXT, arg1, bitsize);
  if (TYPE_UNSIGNED(arg2_type))
    arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize);
  else
    arg2 = bb->build_inst(Op::SEXT, arg2, bitsize);
  Inst *inst = bb->build_inst(Op::ADD, arg1, arg2);
  Inst *res = bb->build_trunc(inst, lhs_elem_bitsize);
  Inst *eres;
  if (TYPE_UNSIGNED(lhs_elem_type))
    eres = bb->build_inst(Op::ZEXT, res, bitsize);
  else
    eres = bb->build_inst(Op::SEXT, res, bitsize);
  Inst *overflow = bb->build_inst(Op::NE, inst, eres);

  res = to_mem_repr(res, lhs_elem_type);
  overflow = bb->build_inst(Op::ZEXT, overflow, res->bitsize);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_bit_andn(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      elem2 = bb->build_inst(Op::NOT, elem2);
      auto [res, res_indef, _] =
      process_binary_int(BIT_AND_EXPR, false,
			 elem1, elem1_indef, nullptr,
			 elem2, elem2_indef, nullptr,
			 elem_type, elem_type, elem_type);
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_bit_iorn(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      elem2 = bb->build_inst(Op::NOT, elem2);
      auto [res, res_indef, _] =
      process_binary_int(BIT_IOR_EXPR, false,
			 elem1, elem1_indef, nullptr,
			 elem2, elem2_indef, nullptr,
			 elem_type, elem_type, elem_type);
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_assume_aligned(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2
	 || gimple_call_num_args(stmt) == 3);
  auto [arg1, arg1_prov] = tree2inst_prov(gimple_call_arg(stmt, 0));
  Inst *arg2 = tree2inst(gimple_call_arg(stmt, 1));
  // TODO: handle arg3
  assert(arg1->bitsize == arg2->bitsize);
  Inst *one = bb->value_inst(1, arg2->bitsize);
  Inst *mask = bb->build_inst(Op::SUB, arg2, one);
  Inst *val = bb->build_inst(Op::AND, arg1, mask);
  Inst *zero = bb->value_inst(0, val->bitsize);
  Inst *cond = bb->build_inst(Op::NE, val, zero);
  bb->build_inst(Op::UB, cond);
  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      constrain_range(bb, lhs, arg1);
      tree2instruction.insert({lhs, arg1});
      tree2prov.insert({lhs, arg1_prov});
    }
}

void Converter::process_cfn_bswap(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  auto [arg, arg_indef, arg_prov] =
    tree2inst_indef_prov(gimple_call_arg(stmt, 0));
  // Determine the width from lhs as bswap16 has 32-bit arg.
  int bitwidth = TYPE_PRECISION(TREE_TYPE(lhs));
  Inst *inst = bb->build_trunc(arg, 8);
  Inst *inst_indef = nullptr;
  if (arg_indef)
    inst_indef = bb->build_trunc(arg_indef, 8);
  for (int i = 8; i < bitwidth; i += 8)
    {
      Inst *high = bb->value_inst(i + 7, 32);
      Inst *low = bb->value_inst(i, 32);
      Inst *byte = bb->build_inst(Op::EXTRACT, arg, high, low);
      inst = bb->build_inst(Op::CONCAT, inst, byte);
      if (arg_indef)
	{
	  Inst *byte_indef =
	    bb->build_inst(Op::EXTRACT, arg_indef, high, low);
	  inst_indef = bb->build_inst(Op::CONCAT, inst_indef, byte_indef);
	}
    }
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  if (inst_indef)
    tree2indef.insert({lhs, inst_indef});
  if (arg_prov)
    tree2prov.insert({lhs, arg_prov});
}

void Converter::process_cfn_check_war_ptrs(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 4);
  Inst *arg1 = tree2inst(gimple_call_arg(stmt, 0));
  Inst *arg2 = tree2inst(gimple_call_arg(stmt, 1));
  Inst *arg3 = tree2inst(gimple_call_arg(stmt, 2));

  Inst *arg1_end = bb->build_inst(Op::ADD, arg1, arg3);
  Inst *res = bb->build_inst(Op::ULE, arg2, arg1);
  Inst *cmp1 = bb->build_inst(Op::ULE, arg1_end, arg2);
  res = bb->build_inst(Op::OR, res, cmp1);
  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      uint32_t lhs_bitsize = bitsize_for_type(TREE_TYPE(lhs));
      if (res->bitsize < lhs_bitsize)
	res = bb->build_inst(Op::ZEXT, res, lhs_bitsize);
      tree2instruction.insert({lhs, res});
    }
}

void Converter::process_cfn_check_raw_ptrs(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 4);
  Inst *arg1 = tree2inst(gimple_call_arg(stmt, 0));
  Inst *arg2 = tree2inst(gimple_call_arg(stmt, 1));
  Inst *arg3 = tree2inst(gimple_call_arg(stmt, 2));

  Inst *arg1_end = bb->build_inst(Op::ADD, arg1, arg3);
  Inst *arg2_end = bb->build_inst(Op::ADD, arg2, arg3);
  Inst *res = bb->build_inst(Op::EQ, arg1, arg2);
  Inst *cmp1 = bb->build_inst(Op::ULE, arg1_end, arg2);
  res = bb->build_inst(Op::OR, res, cmp1);
  Inst *cmp2 = bb->build_inst(Op::ULE, arg2_end, arg1);
  res = bb->build_inst(Op::OR, res, cmp2);
  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      uint32_t lhs_bitsize = bitsize_for_type(TREE_TYPE(lhs));
      if (res->bitsize < lhs_bitsize)
	res = bb->build_inst(Op::ZEXT, res, lhs_bitsize);
      tree2instruction.insert({lhs, res});
    }
}

void Converter::process_cfn_clrsb(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, tree lhs_elem_type)
    -> std::pair<Inst *, Inst *>
    {
      int bitsize = bitsize_for_type(lhs_elem_type);
      Inst *signbit = bb->build_extract_bit(elem1, elem1->bitsize - 1);
      Inst *inst = bb->value_inst(elem1->bitsize - 1, bitsize);
      for (unsigned i = 0; i < elem1->bitsize - 1; i++)
	{
	  Inst *bit = bb->build_extract_bit(elem1, i);
	  Inst *cmp = bb->build_inst(Op::NE, bit, signbit);
	  Inst *val = bb->value_inst(elem1->bitsize - i - 2, bitsize);
	  inst = bb->build_inst(Op::ITE, cmp, val, inst);
	}
      Inst *indef = get_res_indef(elem1_indef, lhs_elem_type);
      return {inst, indef};
    };
  process_cfn_unary(stmt, gen_elem);
}

void Converter::process_cfn_clz(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1
	 || gimple_call_num_args(stmt) == 2);
  tree arg_expr = gimple_call_arg(stmt, 0);
  tree arg_type = TREE_TYPE(arg_expr);
  Inst *arg = tree2inst(arg_expr);
  uint32_t nof_elem;
  uint32_t elem_bitsize;
  tree arg_elem_type;
  if (VECTOR_TYPE_P(arg_type))
    {
      arg_elem_type = TREE_TYPE(arg_type);
      elem_bitsize = bitsize_for_type(arg_elem_type);
      nof_elem = bitsize_for_type(arg_type) / elem_bitsize;
    }
  else
    {
      arg_elem_type = arg_type;
      elem_bitsize = arg->bitsize;
      nof_elem = 1;
    }

  int nargs = gimple_call_num_args(stmt);
  if (nargs == 1)
    {
      Inst *zero = bb->value_inst(0, elem_bitsize);
      for (uint32_t i = 0; i < nof_elem; i++)
	{
	  Inst *elem = extract_vec_elem(bb, arg, elem_bitsize, i);
	  Inst *ub = bb->build_inst(Op::EQ, elem, zero);
	  bb->build_inst(Op::UB, ub);
	}
    }

  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  tree lhs_type = TREE_TYPE(lhs);
  assert(VECTOR_TYPE_P(arg_type) == VECTOR_TYPE_P(lhs_type));
  uint32_t bitsize;
  tree lhs_elem_type;
  if (VECTOR_TYPE_P(arg_type))
    {
      lhs_elem_type = TREE_TYPE(lhs_type);
      bitsize = bitsize_for_type(TREE_TYPE(lhs_type));
    }
  else
    {
      lhs_elem_type = lhs_type;
      bitsize = bitsize_for_type(lhs_type);
    }
  Inst *inst0;
  if (nargs == 1)
    inst0 = bb->value_inst(elem_bitsize, bitsize);
  else
    {
      inst0 = tree2inst(gimple_call_arg(stmt, 1));
      if (inst0->bitsize != bitsize)
	inst0 = type_convert(inst0, arg_elem_type, lhs_elem_type);
    }

  Inst *res = nullptr;
  for (uint32_t j = 0; j < nof_elem; j++)
    {
      Inst *elem = extract_vec_elem(bb, arg, elem_bitsize, j);
      Inst *inst = inst0;
      for (unsigned i = 0; i < elem_bitsize; i++)
	{
	  Inst *bit = bb->build_extract_bit(elem, i);
	  Inst *val = bb->value_inst(elem_bitsize - i - 1, bitsize);
	  inst = bb->build_inst(Op::ITE, bit, val, inst);
	}
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
}

std::pair<Inst*, Inst*> Converter::gen_cfn_cond_unary(tree_code code, Inst *cond, Inst *cond_indef, Inst *arg1, Inst *arg1_indef, Inst *orig, Inst *orig_indef, Inst *len, tree cond_type, tree arg1_type, tree orig_type)
{
  Inst *op_inst;
  Inst *op_indef = nullptr;
  // TODO: We ignore overflow for now, but we may need to modify this to
  // check for oveflow when the condition is true.
  // TODO: Indef cond is UB.
  if (VECTOR_TYPE_P(arg1_type))
    {
      tree elem_type = TREE_TYPE(arg1_type);
      std::tie(op_inst, op_indef) =
	process_unary_vec(code, arg1, arg1_indef, elem_type, elem_type, true);
    }
  else
    {
      Inst *op_prov;
      std::tie(op_inst, op_indef, op_prov) =
	process_unary_scalar(code, arg1, arg1_indef, nullptr,
			     arg1_type, arg1_type, true);
    }

  Inst *res;
  Inst *res_indef = nullptr;
  if (VECTOR_TYPE_P(cond_type))
    {
      std::tie(res, res_indef) =
	gen_vec_cond(cond, cond_indef, op_inst, op_indef, orig, orig_indef,
		     cond_type, orig_type, len);
    }
  else
    {
      assert(TREE_CODE(cond_type) == BOOLEAN_TYPE);
      if (TYPE_PRECISION(cond_type) != 1)
	cond = bb->build_extract_bit(cond, 0);
      res = bb->build_inst(Op::ITE, cond, op_inst, orig);
      if (op_indef || orig_indef)
	{
	  if (!op_indef)
	    op_indef = bb->value_inst(0, op_inst->bitsize);
	  if (!orig_indef)
	    orig_indef = bb->value_inst(0, orig->bitsize);
	  res_indef = bb->build_inst(Op::ITE, cond, op_indef, orig_indef);
	}
    }

  return {res, res_indef};
}

std::pair<Inst*, Inst*> Converter::gen_cfn_cond_binary(tree_code code, Inst *cond, Inst *cond_indef, Inst *arg1, Inst *arg1_indef, Inst *arg2, Inst *arg2_indef, Inst *orig, Inst *orig_indef, Inst *len, tree cond_type, tree arg1_type, tree arg2_type, tree orig_type)
{
  Inst *op_inst;
  Inst *op_indef = nullptr;
  // TODO: We ignore overflow for now, but we may need to modify this to
  // check for oveflow when the condition is true.
  // TODO: Indef cond is UB.
  if (VECTOR_TYPE_P(arg1_type))
    {
      std::tie(op_inst, op_indef) =
	process_binary_vec(code, arg1, arg1_indef, arg2, arg2_indef,
			   arg1_type, arg1_type, arg2_type, true);
    }
  else
    {
      Inst *op_prov;
      std::tie(op_inst, op_indef, op_prov) =
	process_binary_scalar(code, arg1, arg1_indef, nullptr,
			      arg2, arg2_indef, nullptr,
			      arg1_type, arg1_type, arg2_type, true);
    }

  Inst *res;
  Inst *res_indef = nullptr;
  if (VECTOR_TYPE_P(cond_type))
    {
      std::tie(res, res_indef) =
	gen_vec_cond(cond, cond_indef, op_inst, op_indef, orig, orig_indef,
		     cond_type, orig_type, len);
    }
  else
    {
      assert(TREE_CODE(cond_type) == BOOLEAN_TYPE);
      if (TYPE_PRECISION(cond_type) != 1)
	cond = bb->build_extract_bit(cond, 0);
      res = bb->build_inst(Op::ITE, cond, op_inst, orig);
      if (op_indef || orig_indef)
	{
	  if (!op_indef)
	    op_indef = bb->value_inst(0, op_inst->bitsize);
	  if (!orig_indef)
	    orig_indef = bb->value_inst(0, orig->bitsize);
	  res_indef = bb->build_inst(Op::ITE, cond, op_indef, orig_indef);
	}
    }

  return {res, res_indef};
}

void Converter::process_cfn_cond_unary(gimple *stmt, tree_code code)
{
  assert(gimple_call_num_args(stmt) == 3);
  tree cond_expr = gimple_call_arg(stmt, 0);
  tree cond_type = TREE_TYPE(cond_expr);
  auto[cond, cond_indef] = tree2inst_indef(cond_expr);
  tree arg1_expr = gimple_call_arg(stmt, 1);
  tree arg1_type = TREE_TYPE(arg1_expr);
  auto[arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  tree orig_expr = gimple_call_arg(stmt, 2);
  tree orig_type = TREE_TYPE(orig_expr);
  auto[orig, orig_indef] = tree2inst_indef(orig_expr);
  tree lhs = gimple_call_lhs(stmt);

  auto [res, res_indef] =
    gen_cfn_cond_unary(code, cond, cond_indef, arg1, arg1_indef,
		       orig, orig_indef, nullptr, cond_type, arg1_type,
		       orig_type);

  if (lhs)
    {
      tree2instruction.insert({lhs, res});
      if (res_indef)
	tree2indef.insert({lhs, res_indef});
    }
}

void Converter::process_cfn_cond_binary(gimple *stmt, tree_code code)
{
  assert(gimple_call_num_args(stmt) == 4);
  tree cond_expr = gimple_call_arg(stmt, 0);
  tree cond_type = TREE_TYPE(cond_expr);
  auto[cond, cond_indef] = tree2inst_indef(cond_expr);
  tree arg1_expr = gimple_call_arg(stmt, 1);
  tree arg1_type = TREE_TYPE(arg1_expr);
  auto[arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 2);
  tree arg2_type = TREE_TYPE(arg2_expr);
  auto[arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  tree orig_expr = gimple_call_arg(stmt, 3);
  tree orig_type = TREE_TYPE(orig_expr);
  auto[orig, orig_indef] = tree2inst_indef(orig_expr);
  tree lhs = gimple_call_lhs(stmt);

  auto [res, res_indef] =
    gen_cfn_cond_binary(code, cond, cond_indef, arg1, arg1_indef,
			arg2, arg2_indef, orig, orig_indef, nullptr,
			cond_type, arg1_type, arg2_type, orig_type);

  if (lhs)
    {
      tree2instruction.insert({lhs, res});
      if (res_indef)
	tree2indef.insert({lhs, res_indef});
    }
}

void Converter::process_cfn_cond_len_binary(gimple *stmt, tree_code code)
{
  assert(gimple_call_num_args(stmt) == 6);
  tree cond_expr = gimple_call_arg(stmt, 0);
  tree cond_type = TREE_TYPE(cond_expr);
  auto[cond, cond_indef] = tree2inst_indef(cond_expr);
  tree arg1_expr = gimple_call_arg(stmt, 1);
  tree arg1_type = TREE_TYPE(arg1_expr);
  auto[arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 2);
  tree arg2_type = TREE_TYPE(arg2_expr);
  auto[arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  tree orig_expr = gimple_call_arg(stmt, 3);
  tree orig_type = TREE_TYPE(orig_expr);
  auto[orig, orig_indef] = tree2inst_indef(orig_expr);
  tree len_expr = gimple_call_arg(stmt, 4);
  assert(TYPE_UNSIGNED(TREE_TYPE(len_expr)));
  Inst *len = tree2inst(len_expr);
  tree len_type = TREE_TYPE(len_expr);
  tree bias_expr = gimple_call_arg(stmt, 5);
  Inst *bias = tree2inst(bias_expr);
  tree bias_type = TREE_TYPE(bias_expr);

  tree lhs = gimple_call_lhs(stmt);

  bias = type_convert(bias, bias_type, len_type);
  len = bb->build_inst(Op::ADD, len, bias);
  auto [res, res_indef] =
    gen_cfn_cond_binary(code, cond, cond_indef, arg1, arg1_indef,
			arg2, arg2_indef, orig, orig_indef, len,
			cond_type, arg1_type, arg2_type, orig_type);

  if (lhs)
    {
      tree2instruction.insert({lhs, res});
      if (res_indef)
	tree2indef.insert({lhs, res_indef});
    }
}

void Converter::process_cfn_cond_fminmax(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 4);
  auto gen_elem_fmin =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *res = gen_fmin(bb, elem1, elem2);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };
  auto gen_elem_fmax =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *res = gen_fmax(bb, elem1, elem2);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };

  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  auto[arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree arg2_elem_type = TREE_TYPE(arg2_type);
  auto[arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  auto[arg3, arg3_indef] = tree2inst_indef(arg3_expr);
  tree arg4_expr = gimple_call_arg(stmt, 3);
  tree arg4_type = TREE_TYPE(arg4_expr);
  auto[arg4, arg4_indef] = tree2inst_indef(arg4_expr);
  tree lhs = gimple_call_lhs(stmt);

  combined_fn code = gimple_call_combined_fn(stmt);
  assert(code == CFN_COND_FMIN || code == CFN_COND_FMAX);

  uint32_t elem_bitsize = bitsize_for_type(arg2_elem_type);
  uint32_t nof_elt = bitsize_for_type(arg2_type) / elem_bitsize;
  Inst *op_inst = nullptr;
  Inst *op_indef = nullptr;
  for (uint64_t i = 0; i < nof_elt; i++)
    {
      Inst *a2 = extract_vec_elem(bb, arg2, elem_bitsize, i);
      Inst *a2_indef = nullptr;
      if (arg2_indef)
	a2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, i);
      Inst *a3 = extract_vec_elem(bb, arg3, elem_bitsize, i);
      Inst *a3_indef = nullptr;
      if (arg3_indef)
	a3_indef = extract_vec_elem(bb, arg3_indef, elem_bitsize, i);
      Inst *inst;
      Inst *indef;
      if (code == CFN_COND_FMIN)
	std::tie(inst, indef) =
	  gen_elem_fmin(a2, a2_indef, a3, a3_indef, arg2_type);
      else
	std::tie(inst, indef) =
	  gen_elem_fmax(a2, a2_indef, a3, a3_indef, arg2_type);

      if (op_inst)
	op_inst = bb->build_inst(Op::CONCAT, inst, op_inst);
      else
	op_inst = inst;

      if (indef)
	{
	  if (op_indef)
	    op_indef = bb->build_inst(Op::CONCAT, indef, op_indef);
	  else
	    op_indef = indef;
	}
    }

  auto [ret_inst, ret_indef] =
    gen_vec_cond(arg1, arg1_indef, op_inst, op_indef, arg4, arg4_indef,
		 arg1_type, arg4_type);

  if (lhs)
    {
      tree2instruction.insert({lhs, ret_inst});
      if (ret_indef)
	tree2indef.insert({lhs, ret_indef});
    }
}

void Converter::process_cfn_copysign(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *signbit = bb->build_extract_bit(elem2, elem2->bitsize - 1);
      Inst *res = bb->build_trunc(elem1, elem1->bitsize - 1);
      res = bb->build_inst(Op::CONCAT, signbit, res);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      if (state->arch == Arch::gimple)
	{
	  // SMT solvers has only one NaN value, so NEGATE_EXPR of NaN
	  // does not change the value. This leads to incorrect reports
	  // of miscompilations for transformations like
	  //   -ABS_EXPR(x) -> .COPYSIGN(x, -1.0)
	  // because copysign has introduceed a non-canonical NaN.
	  // For now, treat copying the sign to NaN as always produce the
	  // original canonical NaN.
	  // TODO: Remove this when Op::IS_NONCANONICAL_NAN is removed.
	  Inst *is_nan = bb->build_inst(Op::IS_NAN, elem1);
	  res = bb->build_inst(Op::ITE, is_nan, elem1, res);
	}
      constrain_src_value(res, elem_type);
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_ctz(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1
	 || gimple_call_num_args(stmt) == 2);
  tree arg_expr = gimple_call_arg(stmt, 0);
  tree arg_type = TREE_TYPE(arg_expr);
  Inst *arg = tree2inst(arg_expr);
  uint32_t nof_elem;
  uint32_t elem_bitsize;
  tree arg_elem_type;
  if (VECTOR_TYPE_P(arg_type))
    {
      arg_elem_type = TREE_TYPE(arg_type);
      elem_bitsize = bitsize_for_type(arg_elem_type);
      nof_elem = bitsize_for_type(arg_type) / elem_bitsize;
    }
  else
    {
      arg_elem_type = arg_type;
      elem_bitsize = arg->bitsize;
      nof_elem = 1;
    }

  int nargs = gimple_call_num_args(stmt);
  if (nargs == 1)
    {
      Inst *zero = bb->value_inst(0, elem_bitsize);
      for (uint32_t i = 0; i < nof_elem; i++)
	{
	  Inst *elem = extract_vec_elem(bb, arg, elem_bitsize, i);
	  Inst *ub = bb->build_inst(Op::EQ, elem, zero);
	  bb->build_inst(Op::UB, ub);
	}
    }

  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  tree lhs_type = TREE_TYPE(lhs);
  assert(VECTOR_TYPE_P(arg_type) == VECTOR_TYPE_P(lhs_type));
  uint32_t bitsize;
  tree lhs_elem_type;
  if (VECTOR_TYPE_P(arg_type))
    {
      lhs_elem_type = TREE_TYPE(lhs_type);
      bitsize = bitsize_for_type(TREE_TYPE(lhs_type));
    }
  else
    {
      lhs_elem_type = lhs_type;
      bitsize = bitsize_for_type(lhs_type);
    }
  Inst *inst0;
  if (nargs == 1)
    inst0 = bb->value_inst(elem_bitsize, bitsize);
  else
    {
      inst0 = tree2inst(gimple_call_arg(stmt, 1));
      if (inst0->bitsize != bitsize)
	inst0 = type_convert(inst0, arg_elem_type, lhs_elem_type);
    }

  Inst *res = nullptr;
  for (uint32_t j = 0; j < nof_elem; j++)
    {
      Inst *elem = extract_vec_elem(bb, arg, elem_bitsize, j);
      Inst *inst = inst0;
      for (int i = elem_bitsize - 1; i >= 0; i--)
	{
	  Inst *bit = bb->build_extract_bit(elem, i);
	  Inst *val = bb->value_inst(i, bitsize);
	  inst = bb->build_inst(Op::ITE, bit, val, inst);
	}
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
}

void Converter::process_cfn_exit(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  Inst *arg1 = tree2inst(gimple_call_arg(stmt, 0));
  // We may fail to check the exit call if there is UB after the call
  // in this basic block. This should not happen, as exit is marked as
  // noreturn, but it is unclear to me if this is guaranteed in GIMPLE.
  // We therefore ensure that we branch to the exit block and create a
  // new dead block for the remaining instructions, if any.
  basic_block gcc_exit_block = EXIT_BLOCK_PTR_FOR_FN(fun);
  Basic_block *exit_bb = gccbb_top2bb.at(gcc_exit_block);
  Basic_block *dead_bb = func->build_bb();
  bb->build_br_inst(bb->value_inst(1, 1), exit_bb, dead_bb);
  bb2exit.insert({bb, arg1});
  bb = dead_bb;
}

void Converter::process_cfn_divmod(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_divmod_: vector type");
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  Inst *arg1 = tree2inst(arg1_expr);
  Inst *arg2 = tree2inst(arg2_expr);
  Inst *mod = process_binary_scalar(TRUNC_MOD_EXPR, arg1, arg2,
					   lhs_elem_type, arg1_type,
					   arg2_type);
  mod = to_mem_repr(mod, lhs_elem_type);
  Inst *div = process_binary_scalar(TRUNC_DIV_EXPR, arg1, arg2,
					   lhs_elem_type, arg1_type,
					   arg2_type);
  div = to_mem_repr(div, lhs_elem_type);
  Inst *inst = bb->build_inst(Op::CONCAT, mod, div);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_expect(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2
	 || gimple_call_num_args(stmt) == 3);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  Inst *arg1 = tree2inst(gimple_call_arg(stmt, 0));
  // Ignore arg2 and arg3. They only contain the most common value and the
  // probability, which we can't use for anything.
  constrain_range(bb, lhs, arg1);
  tree2instruction.insert({lhs, arg1});
}

void Converter::process_cfn_fabs(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, tree lhs_elem_type)
    -> std::pair<Inst *, Inst *>
    {
      if (state->arch != Arch::gimple)
	{
	  // Backends may choose to implement fabs as a bit operation. Skip
	  // checking if this would generate a non-canonical NaN.
	  Inst *shift = bb->value_inst(1, elem1->bitsize);
	  Inst *inst = bb->build_inst(Op::SHL, elem1, shift);
	  inst = bb->build_inst(Op::LSHR, inst, shift);
	  constrain_src_value(inst, lhs_elem_type);
	}
      Inst *res = bb->build_inst(Op::FABS, elem1);
      Inst *res_indef = get_res_indef(elem1_indef, lhs_elem_type);
      return {res, res_indef};
    };
  process_cfn_unary(stmt, gen_elem);
}

void Converter::process_cfn_ffs(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  Inst *arg = tree2inst(gimple_call_arg(stmt, 0));
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_ffs: vector type");
  int bitsize = bitsize_for_type(TREE_TYPE(lhs));
  Inst *inst;
  inst = bb->value_inst(0, bitsize);
  for (int i = arg->bitsize - 1; i >= 0; i--)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *val = bb->value_inst(i + 1, bitsize);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_fmax(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *res = gen_fmax(bb, elem1, elem2);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_fmin(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *res = gen_fmin(bb, elem1, elem2);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_isfinite(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, tree elem_type)
    -> std::pair<Inst *, Inst *>
    {
      Inst *is_inf = bb->build_inst(Op::IS_INF, elem1);
      Inst *is_nan = bb->build_inst(Op::IS_NAN, elem1);
      Inst *res = bb->build_inst(Op::OR, is_inf, is_nan);
      res = bb->build_inst(Op::NOT, res);
      uint32_t bitsize = bitsize_for_type(elem_type);
      if (bitsize > res->bitsize)
	res = bb->build_inst(Op::ZEXT, res, bitsize);
      Inst *res_indef = get_res_indef(elem1_indef, elem_type);
      return {res, res_indef};
    };
  process_cfn_unary(stmt, gen_elem);
}

void Converter::process_cfn_isinf(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, tree elem_type)
    -> std::pair<Inst *, Inst *>
    {
      Inst *res = bb->build_inst(Op::IS_INF, elem1);
      uint32_t bitsize = bitsize_for_type(elem_type);
      if (bitsize > res->bitsize)
	res = bb->build_inst(Op::ZEXT, res, bitsize);
      Inst *neg = bb->build_inst(Op::NEG, res);
      Inst *zero = bb->value_inst(0, elem1->bitsize);
      Inst *is_neg = bb->build_inst(Op::FLT, elem1, zero);
      res = bb->build_inst(Op::ITE, is_neg, neg, res);
      Inst *res_indef = get_res_indef(elem1_indef, elem_type);
      return {res, res_indef};
    };
  process_cfn_unary(stmt, gen_elem);
}

void Converter::process_cfn_loop_vectorized(gimple *stmt)
{
  tree lhs = gimple_call_lhs(stmt);
  tree2instruction.insert({lhs, bb->value_inst(0, 1)});
}

std::tuple<Inst*, Inst*> Converter::mask_len_load(Inst *ptr, Inst *ptr_indef, Inst *ptr_prov, uint64_t alignment, Inst *mask, Inst *mask_indef, tree mask_type, Inst *len, tree lhs_type, Inst *orig, Inst *orig_indef)
{
  tree elem_type = VECTOR_TYPE_P(lhs_type) ? TREE_TYPE(lhs_type) : lhs_type;
  tree mask_elem_type =
    VECTOR_TYPE_P(mask_type) ? TREE_TYPE(mask_type) : mask_type;
  uint64_t elem_size = bytesize_for_type(elem_type);
  uint64_t elem_bitsize = bitsize_for_type(elem_type);
  assert(TREE_CODE(mask_elem_type) == BOOLEAN_TYPE);
  uint64_t mask_elem_bitsize = bitsize_for_type(mask_elem_type);

  uint64_t size = bytesize_for_type(lhs_type);
  uint64_t nof_elem = size / elem_size;
  assert((size % elem_size) == 0);
  Inst *inst = nullptr;
  Inst *indef = nullptr;
  Inst *mem_flags = nullptr;
  Inst *mem_accessed = bb->value_inst(0, 1);
  for (uint64_t i = 0; i < nof_elem; i++)
    {
      Inst *cond = extract_vec_elem(bb, mask, mask_elem_bitsize, i);
      if (cond->bitsize != 1)
	cond = bb->build_trunc(cond, 1);
      Inst *mask_elem_indef = nullptr;
      if (mask_indef)
	mask_elem_indef =
	  extract_vec_elem(bb, mask_indef, mask_elem_bitsize, i);
      if (len)
	{
	  Inst *i_inst = bb->value_inst(i, len->bitsize);
	  Inst *cmp = bb->build_inst(Op::ULT, i_inst, len);
	  if (mask_elem_indef)
	    build_ub_if_not_zero(mask_elem_indef, cmp);
	  cond = bb->build_inst(Op::AND, cmp, cond);
	}
      else
	{
	  if (mask_elem_indef)
	    build_ub_if_not_zero(mask_elem_indef);
	}
      if (ptr_indef)
	build_ub_if_not_zero(ptr_indef, cond);

      Inst *offset = bb->value_inst(i * elem_size, ptr->bitsize);
      Inst *src_ptr = bb->build_inst(Op::ADD, ptr, offset);
      load_ub_check(src_ptr, ptr_prov, elem_size, cond);
      auto [elem, elem_indef, elem_flags] = load_value(src_ptr, elem_size);
      if (!VECTOR_TYPE_P(lhs_type))
	std::tie(elem, elem_indef) = from_mem_repr(elem, elem_indef, elem_type);
      auto [orig_elem, orig_elem_indef] =
	extract_vec_elem(bb, orig, orig_indef, elem_bitsize, i);
      // TODO: We should call constrain_src_value in order to constrain
      // non-canonical NaN etc. But this should only be done when not masked.
      // constrain_src_value(elem, elem_type, elem_flags);
      elem = bb->build_inst(Op::ITE, cond, elem, orig_elem);
      if (!orig_elem_indef)
	orig_elem_indef = bb->value_inst(0, elem_indef->bitsize);
      elem_indef = bb->build_inst(Op::ITE, cond, elem_indef, orig_elem_indef);
      mem_accessed = bb->build_inst(Op::OR, mem_accessed, cond);

      if (inst)
	inst = bb->build_inst(Op::CONCAT, elem, inst);
      else
	inst = elem;
      if (indef)
	indef = bb->build_inst(Op::CONCAT, elem_indef, indef);
      else
	indef = elem_indef;
      if (mem_flags)
	mem_flags = bb->build_inst(Op::CONCAT, elem_flags, mem_flags);
      else
	mem_flags = elem_flags;
    }

  if (alignment > 1)
    {
      assert((alignment & (alignment - 1)) == 0);
      Inst *extract = bb->build_trunc(ptr, __builtin_ctz(alignment));
      Inst *zero = bb->value_inst(0, extract->bitsize);
      Inst *is_misaligned = bb->build_inst(Op::NE, extract, zero);
      Inst *is_ub = bb->build_inst(Op::AND, mem_accessed, is_misaligned);
      bb->build_inst(Op::UB, is_ub);
    }

  return {inst, indef};
}

void Converter::process_cfn_mask_len_load(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 6);
  tree ptr_expr = gimple_call_arg(stmt, 0);
  tree alignment_expr = gimple_call_arg(stmt, 1);
  tree mask_expr = gimple_call_arg(stmt, 2);
  tree mask_type = TREE_TYPE(mask_expr);
  tree orig_expr = gimple_call_arg(stmt, 3);
  tree len_expr = gimple_call_arg(stmt, 4);
  assert(TYPE_UNSIGNED(TREE_TYPE(len_expr)));
  tree bias = gimple_call_arg(stmt, 5);
  if (tree2inst(bias)->value() != 0)
    throw Not_implemented("process_cfn_mask_len_load: bias != 0");
  tree lhs = gimple_call_lhs(stmt);
  assert(lhs);
  tree lhs_type = TREE_TYPE(lhs);

  auto [ptr, ptr_indef, ptr_prov] = tree2inst_indef_prov(ptr_expr);
  auto [orig, orig_indef] = tree2inst_indef(orig_expr);
  uint64_t alignment = get_int_cst_val(alignment_expr) / 8;
  auto [mask, mask_indef] = tree2inst_indef(mask_expr);
  Inst *len = tree2inst(len_expr);

  auto [inst, indef] =
    mask_len_load(ptr, ptr_indef, ptr_prov, alignment, mask, mask_indef,
		  mask_type, len, lhs_type, orig, orig_indef);
  std::tie(inst, indef) = from_mem_repr(inst, indef, lhs_type);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  tree2indef.insert({lhs, indef});
  if (POINTER_TYPE_P(lhs_type))
    tree2prov.insert({lhs, extract_id(inst)});
}

void Converter::process_cfn_mask_load(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 4);
  tree ptr_expr = gimple_call_arg(stmt, 0);
  tree alignment_expr = gimple_call_arg(stmt, 1);
  tree mask_expr = gimple_call_arg(stmt, 2);
  tree mask_type = TREE_TYPE(mask_expr);
  tree orig_expr = gimple_call_arg(stmt, 3);
  tree lhs = gimple_call_lhs(stmt);
  assert(lhs);
  tree lhs_type = TREE_TYPE(lhs);

  auto [ptr, ptr_indef, ptr_prov] = tree2inst_indef_prov(ptr_expr);
  auto [orig, orig_indef] = tree2inst_indef(orig_expr);
  uint64_t alignment = get_int_cst_val(alignment_expr) / 8;
  auto [mask, mask_indef] = tree2inst_indef(mask_expr);

  auto [inst, indef] =
    mask_len_load(ptr, ptr_indef, ptr_prov, alignment, mask, mask_indef,
		  mask_type, nullptr, lhs_type, orig, orig_indef);
  std::tie(inst, indef) = from_mem_repr(inst, indef, lhs_type);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  tree2indef.insert({lhs, indef});
  if (POINTER_TYPE_P(lhs_type))
    tree2prov.insert({lhs, extract_id(inst)});
}

void Converter::mask_len_store(Inst *ptr, Inst *ptr_indef, Inst *ptr_prov, uint64_t alignment, Inst *mask, Inst *mask_indef, tree mask_type, Inst *len, tree value_type, Inst *value, Inst *value_indef)
{
 tree elem_type =
   VECTOR_TYPE_P(value_type) ? TREE_TYPE(value_type) : value_type;
  tree mask_elem_type =
    VECTOR_TYPE_P(mask_type) ? TREE_TYPE(mask_type) : mask_type;
  uint64_t elem_size = bytesize_for_type(elem_type);
  uint64_t elem_bitsize = bitsize_for_type(elem_type);
  assert(TREE_CODE(mask_elem_type) == BOOLEAN_TYPE);
  uint64_t mask_elem_bitsize = bitsize_for_type(mask_elem_type);

  Inst *is_misaligned = nullptr;
  if (alignment > 1)
    {
      assert((alignment & (alignment - 1)) == 0);
      Inst *extract = bb->build_trunc(ptr, __builtin_ctz(alignment));
      Inst *zero = bb->value_inst(0, extract->bitsize);
      is_misaligned = bb->build_inst(Op::NE, extract, zero);
    }

  if (!value_indef)
    value_indef = bb->value_inst(0, value->bitsize);

  uint64_t size = bytesize_for_type(value_type);
  uint64_t nof_elem = size / elem_size;
  assert((size % elem_size) == 0);
  for (uint64_t i = 0; i < nof_elem; i++)
    {
      Inst *cond = extract_vec_elem(bb, mask, mask_elem_bitsize, i);
      if (cond->bitsize != 1)
	cond = bb->build_trunc(cond, 1);
      Inst *mask_elem_indef = nullptr;
      if (mask_indef)
	mask_elem_indef =
	  extract_vec_elem(bb, mask_indef, mask_elem_bitsize, i);
      if (len)
	{
	  Inst *i_inst = bb->value_inst(i, len->bitsize);
	  Inst *cmp = bb->build_inst(Op::ULT, i_inst, len);
	  if (mask_elem_indef)
	    build_ub_if_not_zero(mask_elem_indef, cmp);
	  cond = bb->build_inst(Op::AND, cmp, cond);
	}
      else
	{
	  if (mask_elem_indef)
	    build_ub_if_not_zero(mask_elem_indef);
	}
      if (ptr_indef)
	build_ub_if_not_zero(ptr_indef, cond);

      Basic_block *true_bb = func->build_bb();
      Basic_block *false_bb = func->build_bb();
      bb->build_br_inst(cond, true_bb, false_bb);
      bb = true_bb;
      auto [elem, elem_indef] =
	extract_vec_elem(bb, value, value_indef, elem_bitsize, i);
      Inst *offset = bb->value_inst(i * elem_size, ptr->bitsize);
      Inst *dst_ptr = bb->build_inst(Op::ADD, ptr, offset);
      if (!VECTOR_TYPE_P(value_type))
	std::tie(elem, elem_indef) = to_mem_repr(elem, elem_indef, elem_type);
      if (is_misaligned)
	bb->build_inst(Op::UB, is_misaligned);
      store_ub_check(dst_ptr, ptr_prov, elem_size, cond);
      store_value(dst_ptr, elem, elem_indef);
      bb->build_br_inst(false_bb);

      bb = false_bb;
    }
}

void Converter::process_cfn_mask_len_store(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 6);
  tree ptr_expr = gimple_call_arg(stmt, 0);
  tree alignment_expr = gimple_call_arg(stmt, 1);
  tree mask_expr = gimple_call_arg(stmt, 2);
  tree mask_type = TREE_TYPE(mask_expr);
  tree len_expr = gimple_call_arg(stmt, 3);
  assert(TYPE_UNSIGNED(TREE_TYPE(len_expr)));
  tree bias = gimple_call_arg(stmt, 4);
  if (tree2inst(bias)->value() != 0)
    throw Not_implemented("process_cfn_mask_len_store: bias != 0");
  tree value_expr = gimple_call_arg(stmt, 5);
  tree value_type = TREE_TYPE(value_expr);

  auto [ptr, ptr_indef, ptr_prov] = tree2inst_indef_prov(ptr_expr);
  uint64_t alignment = get_int_cst_val(alignment_expr) / 8;
  auto [mask, mask_indef] = tree2inst_indef(mask_expr);
  Inst *len = tree2inst(len_expr);
  auto [value, value_indef] = tree2inst_indef(value_expr);
  mask_len_store(ptr, ptr_indef, ptr_prov, alignment, mask, mask_indef,
		 mask_type, len, value_type, value, value_indef);
}

void Converter::process_cfn_mask_store(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 4);
  tree ptr_expr = gimple_call_arg(stmt, 0);
  tree alignment_expr = gimple_call_arg(stmt, 1);
  tree mask_expr = gimple_call_arg(stmt, 2);
  tree mask_type = TREE_TYPE(mask_expr);
  tree value_expr = gimple_call_arg(stmt, 3);
  tree value_type = TREE_TYPE(value_expr);

  auto [ptr, ptr_indef, ptr_prov] = tree2inst_indef_prov(ptr_expr);
  uint64_t alignment = get_int_cst_val(alignment_expr) / 8;
  auto [mask, mask_indef] = tree2inst_indef(mask_expr);
  auto [value, value_indef] = tree2inst_indef(value_expr);
  mask_len_store(ptr, ptr_indef, ptr_prov, alignment, mask, mask_indef,
		 mask_type, nullptr, value_type, value, value_indef);
}

void Converter::process_cfn_memcpy(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  if (TREE_CODE(gimple_call_arg(stmt, 2)) != INTEGER_CST)
    throw Not_implemented("non-constant memcpy size");
  auto [orig_dest_ptr, dest_prov] = tree2inst_prov(gimple_call_arg(stmt, 0));
  auto [orig_src_ptr, src_prov] = tree2inst_prov(gimple_call_arg(stmt, 1));
  unsigned __int128 size = get_int_cst_val(gimple_call_arg(stmt, 2));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("too large memcpy");

  store_ub_check(orig_dest_ptr, dest_prov, size);
  load_ub_check(orig_src_ptr, src_prov, size);
  overlap_ub_check(orig_src_ptr, orig_dest_ptr, size);

  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      constrain_range(bb, lhs, orig_dest_ptr);
      tree2instruction.insert({lhs, orig_dest_ptr});
      tree2prov.insert({lhs, dest_prov});
    }

  std::vector<Inst*> bytes;
  bytes.reserve(size);
  std::vector<Inst*> mem_flags;
  mem_flags.reserve(size);
  std::vector<Inst*> indefs;
  indefs.reserve(size);
  for (size_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_src_ptr->bitsize);
      Inst *src_ptr = bb->build_inst(Op::ADD, orig_src_ptr, offset);
      bytes.push_back(bb->build_inst(Op::LOAD, src_ptr));
      mem_flags.push_back(bb->build_inst(Op::GET_MEM_FLAG, src_ptr));
      indefs.push_back(bb->build_inst(Op::GET_MEM_INDEF, src_ptr));
    }
  for (size_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_src_ptr->bitsize);
      Inst *dest_ptr = bb->build_inst(Op::ADD, orig_dest_ptr, offset);
      bb->build_inst(Op::STORE, dest_ptr, bytes[i]);
      bb->build_inst(Op::SET_MEM_FLAG, dest_ptr, mem_flags[i]);
      bb->build_inst(Op::SET_MEM_INDEF, dest_ptr, indefs[i]);
    }
}

void Converter::process_cfn_memmove(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  if (TREE_CODE(gimple_call_arg(stmt, 2)) != INTEGER_CST)
    throw Not_implemented("non-constant memmove size");
  auto [orig_dest_ptr, dest_prov] = tree2inst_prov(gimple_call_arg(stmt, 0));
  auto [orig_src_ptr, src_prov] = tree2inst_prov(gimple_call_arg(stmt, 1));
  unsigned __int128 size = get_int_cst_val(gimple_call_arg(stmt, 2));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("too large memmove");

  store_ub_check(orig_dest_ptr, dest_prov, size);
  load_ub_check(orig_src_ptr, src_prov, size);

  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      constrain_range(bb, lhs, orig_dest_ptr);
      tree2instruction.insert({lhs, orig_dest_ptr});
      tree2prov.insert({lhs, dest_prov});
    }

  std::vector<Inst*> bytes;
  bytes.reserve(size);
  std::vector<Inst*> mem_flags;
  mem_flags.reserve(size);
  std::vector<Inst*> indefs;
  indefs.reserve(size);
  for (size_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_src_ptr->bitsize);
      Inst *src_ptr = bb->build_inst(Op::ADD, orig_src_ptr, offset);
      bytes.push_back(bb->build_inst(Op::LOAD, src_ptr));
      mem_flags.push_back(bb->build_inst(Op::GET_MEM_FLAG, src_ptr));
      indefs.push_back(bb->build_inst(Op::GET_MEM_INDEF, src_ptr));
    }
  for (size_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_src_ptr->bitsize);
      Inst *dest_ptr = bb->build_inst(Op::ADD, orig_dest_ptr, offset);
      bb->build_inst(Op::STORE, dest_ptr, bytes[i]);
      bb->build_inst(Op::SET_MEM_FLAG, dest_ptr, mem_flags[i]);
      bb->build_inst(Op::SET_MEM_INDEF, dest_ptr, indefs[i]);
    }
}

void Converter::process_cfn_mempcpy(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  if (TREE_CODE(gimple_call_arg(stmt, 2)) != INTEGER_CST)
    throw Not_implemented("non-constant mempcpy size");
  auto [orig_dest_ptr, dest_prov] = tree2inst_prov(gimple_call_arg(stmt, 0));
  auto [orig_src_ptr, src_prov] = tree2inst_prov(gimple_call_arg(stmt, 1));
  unsigned __int128 size = get_int_cst_val(gimple_call_arg(stmt, 2));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("too large mempcpy");

  store_ub_check(orig_dest_ptr, dest_prov, size);
  load_ub_check(orig_src_ptr, src_prov, size);
  overlap_ub_check(orig_src_ptr, orig_dest_ptr, size);

  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      Inst *offset = bb->value_inst(size, orig_src_ptr->bitsize);
      Inst *dest_ptr = bb->build_inst(Op::ADD, orig_dest_ptr, offset);
      constrain_range(bb, lhs, dest_ptr);
      tree2instruction.insert({lhs, dest_ptr});
      tree2prov.insert({lhs, dest_prov});
    }

  std::vector<Inst*> bytes;
  bytes.reserve(size);
  std::vector<Inst*> mem_flags;
  mem_flags.reserve(size);
  std::vector<Inst*> indefs;
  indefs.reserve(size);
  for (size_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_src_ptr->bitsize);
      Inst *src_ptr = bb->build_inst(Op::ADD, orig_src_ptr, offset);
      bytes.push_back(bb->build_inst(Op::LOAD, src_ptr));
      mem_flags.push_back(bb->build_inst(Op::GET_MEM_FLAG, src_ptr));
      indefs.push_back(bb->build_inst(Op::GET_MEM_INDEF, src_ptr));
    }
  for (size_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_src_ptr->bitsize);
      Inst *dest_ptr = bb->build_inst(Op::ADD, orig_dest_ptr, offset);
      bb->build_inst(Op::STORE, dest_ptr, bytes[i]);
      bb->build_inst(Op::SET_MEM_FLAG, dest_ptr, mem_flags[i]);
      bb->build_inst(Op::SET_MEM_INDEF, dest_ptr, indefs[i]);
    }
}

void Converter::process_cfn_memset(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  if (TREE_CODE(gimple_call_arg(stmt, 2)) != INTEGER_CST)
    throw Not_implemented("non-constant memset size");
  auto [orig_ptr, ptr_prov] = tree2inst_prov(gimple_call_arg(stmt, 0));
  Inst *value = tree2inst(gimple_call_arg(stmt, 1));
  unsigned __int128 size = get_int_cst_val(gimple_call_arg(stmt, 2));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("too large memset");

  store_ub_check(orig_ptr, ptr_prov, size);

  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      constrain_range(bb, lhs, orig_ptr);
      tree2instruction.insert({lhs, orig_ptr});
      tree2prov.insert({lhs, ptr_prov});
    }

  assert(value->bitsize >= 8);
  if (value->bitsize > 8)
    value = bb->build_trunc(value, 8);
  Inst *mem_flag = bb->value_inst(0, 1);
  Inst *indef = bb->value_inst(0, 8);
  for (size_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_ptr->bitsize);
      Inst *ptr = bb->build_inst(Op::ADD, orig_ptr, offset);
      bb->build_inst(Op::STORE, ptr, value);
      bb->build_inst(Op::SET_MEM_FLAG, ptr, mem_flag);
      bb->build_inst(Op::SET_MEM_INDEF, ptr, indef);
    }
}

void Converter::process_cfn_mul_overflow(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_mul_overflow: vector type");
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  Inst *res_indef = get_res_indef(arg1_indef, arg2_indef, lhs_elem_type);
  if (res_indef)
    {
      Inst *overflow_indef = bb->build_trunc(res_indef, 1);
      overflow_indef =
	bb->build_inst(Op::ZEXT, overflow_indef, res_indef->bitsize);
      res_indef = to_mem_repr(res_indef, lhs_elem_type);
      overflow_indef = to_mem_repr(overflow_indef, lhs_elem_type);
      res_indef = bb->build_inst(Op::CONCAT, overflow_indef, res_indef);
    }
  unsigned lhs_elem_bitsize = bitsize_for_type(lhs_elem_type);
  unsigned bitsize =
    1 + std::max(arg1->bitsize + arg2->bitsize, lhs_elem_bitsize);
  if (TYPE_UNSIGNED(arg1_type))
    arg1 = bb->build_inst(Op::ZEXT, arg1, bitsize);
  else
    arg1 = bb->build_inst(Op::SEXT, arg1, bitsize);
  if (TYPE_UNSIGNED(arg2_type))
    arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize);
  else
    arg2 = bb->build_inst(Op::SEXT, arg2, bitsize);
  Inst *inst = bb->build_inst(Op::MUL, arg1, arg2);
  Inst *res = bb->build_trunc(inst, lhs_elem_bitsize);
  Inst *eres;
  if (TYPE_UNSIGNED(lhs_elem_type))
    eres = bb->build_inst(Op::ZEXT, res, bitsize);
  else
    eres = bb->build_inst(Op::SEXT, res, bitsize);
  Inst *overflow = bb->build_inst(Op::NE, inst, eres);

  res = to_mem_repr(res, lhs_elem_type);
  overflow = bb->build_inst(Op::ZEXT, overflow, res->bitsize);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_mulh(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Op op = TYPE_UNSIGNED(elem_type) ? Op::ZEXT : Op::SEXT;
      Inst *eelem1 = bb->build_inst(op, elem1, 2 * elem1->bitsize);
      Inst *eelem2 = bb->build_inst(op, elem2, 2 * elem2->bitsize);
      Inst *res = bb->build_inst(Op::MUL, eelem1, eelem2);
      uint32_t hi = res->bitsize - 1;
      uint32_t lo = res->bitsize / 2;
      res = bb->build_inst(Op::EXTRACT, res, hi, lo);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_nan(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  // TODO: Implement the argument setting NaN payload when support for
  // noncanonical NaNs is implemented in the SMT solvers.
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_nan: vector type");

  Inst *bs = bb->value_inst(bitsize_for_type(TREE_TYPE(lhs)), 32);
  tree2instruction.insert({lhs, bb->build_inst(Op::NAN, bs)});
}

void Converter::process_cfn_parity(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_parity: vector type");
  Inst *arg = tree2inst(gimple_call_arg(stmt, 0));
  int bitwidth = arg->bitsize;
  Inst *inst = bb->build_extract_bit(arg, 0);
  for (int i = 1; i < bitwidth; i++)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      inst = bb->build_inst(Op::XOR, inst, bit);
    }
  bitwidth = TYPE_PRECISION(TREE_TYPE(lhs));
  inst = bb->build_inst(Op::ZEXT, inst, bitwidth);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_popcount(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1
	 || gimple_call_num_args(stmt) == 2);
  // TODO: Handle popcount arg2.
  // arg2 indicates that the result is used in a way that may allow for
  // a simpler version of the operation. For example, arg2 == 0 tells us
  // that arg1 cannot be 0. This does not help us, but we may want to
  // make it UB if the condition specified by arg2 is not satisfied.
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, tree lhs_elem_type)
    -> std::pair<Inst *, Inst *>
    {
      unsigned bitsize = bitsize_for_type(lhs_elem_type);
      Inst *res = gen_popcount(bb, elem1);
      if (res->bitsize > bitsize)
	res = bb->build_trunc(res, bitsize);
      else if (res->bitsize < bitsize)
	res = bb->build_inst(Op::ZEXT, res, bitsize);
      Inst *res_indef = get_res_indef(elem1_indef, lhs_elem_type);
      return {res, res_indef};
    };
  process_cfn_unary(stmt, gen_elem);
}

void Converter::process_cfn_sat_add(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *res = bb->build_inst(Op::ADD, elem1, elem2);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      if (TYPE_UNSIGNED(elem_type))
	{
	  Inst *cmp = bb->build_inst(Op::ULT, res, elem1);
	  Inst *m1 = bb->value_inst(-1, res->bitsize);
	  res = bb->build_inst(Op::ITE, cmp, m1, res);
	}
      else
	{
	  Inst *zero = bb->value_inst(0, res->bitsize);
	  Inst *is_neg_elem2 = bb->build_inst(Op::SLT, elem2, zero);
	  Inst *is_pos_elem2 = bb->build_inst(Op::NOT, is_neg_elem2);
	  Inst *is_neg_oflw = bb->build_inst(Op::SLT, elem1, res);
	  is_neg_oflw = bb->build_inst(Op::AND, is_neg_oflw, is_neg_elem2);
	  Inst *is_pos_oflw = bb->build_inst(Op::SLT, res, elem1);
	  is_pos_oflw = bb->build_inst(Op::AND, is_pos_oflw, is_pos_elem2);
	  unsigned __int128 maxint =
	    (((unsigned __int128)1) << (res->bitsize - 1)) - 1;
	  unsigned __int128 minint =
	    ((unsigned __int128)1) << (res->bitsize - 1);
	  Inst *maxint_inst = bb->value_inst(maxint, res->bitsize);
	  Inst *minint_inst = bb->value_inst(minint, res->bitsize);
	  res = bb->build_inst(Op::ITE, is_pos_oflw, maxint_inst, res);
	  res = bb->build_inst(Op::ITE, is_neg_oflw, minint_inst, res);
	}
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_sat_sub(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *res = bb->build_inst(Op::SUB, elem1, elem2);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      if (TYPE_UNSIGNED(elem_type))
	{
	  Inst *cmp = bb->build_inst(Op::ULT, elem1, res);
	  Inst *zero = bb->value_inst(0, res->bitsize);
	  res = bb->build_inst(Op::ITE, cmp, zero, res);
	}
      else
	{
	  Inst *zero = bb->value_inst(0, res->bitsize);
	  Inst *is_neg_elem2 = bb->build_inst(Op::SLT, elem2, zero);
	  Inst *is_pos_elem2 = bb->build_inst(Op::NOT, is_neg_elem2);
	  Inst *is_neg_oflw = bb->build_inst(Op::SLT, elem1, res);
	  is_neg_oflw = bb->build_inst(Op::AND, is_neg_oflw, is_pos_elem2);
	  Inst *is_pos_oflw = bb->build_inst(Op::SLT, res, elem1);
	  is_pos_oflw = bb->build_inst(Op::AND, is_pos_oflw, is_neg_elem2);
	  unsigned __int128 maxint =
	    (((unsigned __int128)1) << (res->bitsize - 1)) - 1;
	  unsigned __int128 minint =
	    ((unsigned __int128)1) << (res->bitsize - 1);
	  Inst *maxint_inst = bb->value_inst(maxint, res->bitsize);
	  Inst *minint_inst = bb->value_inst(minint, res->bitsize);
	  res = bb->build_inst(Op::ITE, is_pos_oflw, maxint_inst, res);
	  res = bb->build_inst(Op::ITE, is_neg_oflw, minint_inst, res);
	}
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_cfn_sat_trunc(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, tree elem_type)
    -> std::pair<Inst *, Inst *>
    {
      Inst *res = bb->build_trunc(elem1, bitsize_for_type(elem_type));
      Inst *res_indef = get_res_indef(elem1_indef, elem_type);
      if (TYPE_UNSIGNED(elem_type))
	{
	  Inst *m1_res = bb->value_inst(-1, res->bitsize);
	  Inst *m1_ext = bb->value_inst(m1_res->value(), elem1->bitsize);
	  Inst *cmp = bb->build_inst(Op::ULT, elem1, m1_ext);
	  res = bb->build_inst(Op::ITE, cmp, res, m1_res);
	}
      else
	{
	  unsigned __int128 maxint =
	    (((unsigned __int128)1) << (res->bitsize - 1)) - 1;
	  unsigned __int128 minint =
	    ((unsigned __int128)1) << (res->bitsize - 1);
	  Inst *max_res = bb->value_inst(maxint, res->bitsize);
	  Inst *max_ext =
	    bb->value_inst(max_res->signed_value(), elem1->bitsize);
	  Inst *min_res = bb->value_inst(minint, res->bitsize);
	  Inst *min_ext =
	    bb->value_inst(min_res->signed_value(), elem1->bitsize);
	  Inst *cmp_max = bb->build_inst(Op::SLT, elem1, max_ext);
	  Inst *cmp_min = bb->build_inst(Op::SLT, elem1, min_ext);
	  res = bb->build_inst(Op::ITE, cmp_max, res, max_res);
	  res = bb->build_inst(Op::ITE, cmp_min, min_res, res);
	}
      return {res, res_indef};
    };
  process_cfn_unary(stmt, gen_elem);
}

void Converter::process_cfn_select_vl(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  Inst *arg1 = tree2inst(arg1_expr);
  assert(TYPE_UNSIGNED(arg1_type));
  tree arg2_expr = gimple_call_arg(stmt, 1);
  uint32_t nof_elem = get_int_cst_val(arg2_expr);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;

  Inst *nof = bb->value_inst(nof_elem, arg1->bitsize);
  Inst *cmp = bb->build_inst(Op::ULT, arg1, nof);
  Inst *res = bb->build_inst(Op::ITE, cmp, arg1, nof);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
}

void Converter::process_cfn_signbit(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, tree lhs_elem_type)
    -> std::pair<Inst *, Inst *>
    {
      Inst *res_indef = get_res_indef(elem1_indef, lhs_elem_type);
      Inst *res;
      uint32_t bitsize = bitsize_for_type(lhs_elem_type);
      if (bitsize >= elem1->bitsize)
	{
	  assert(bitsize <= 128);
	  unsigned __int128 mask =
	    ((unsigned __int128)1) << (elem1->bitsize - 1);
	  Inst *mask_inst = bb->value_inst(mask, elem1->bitsize);
	  res = bb->build_inst(Op::AND, elem1, mask_inst);
	}
      else
	res = bb->build_extract_bit(elem1, elem1->bitsize - 1);
      if (res->bitsize < bitsize)
	res = bb->build_inst(Op::ZEXT, res, bitsize);
      return {res, res_indef};
    };
  process_cfn_unary(stmt, gen_elem);
}

void Converter::process_cfn_sub_overflow(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_sub_overflow: vector type");
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  Inst *res_indef = get_res_indef(arg1_indef, arg2_indef, lhs_elem_type);
  if (res_indef)
    {
      Inst *overflow_indef = bb->build_trunc(res_indef, 1);
      overflow_indef
	= bb->build_inst(Op::ZEXT, overflow_indef, res_indef->bitsize);
      res_indef = to_mem_repr(res_indef, lhs_elem_type);
      overflow_indef = to_mem_repr(overflow_indef, lhs_elem_type);
      res_indef = bb->build_inst(Op::CONCAT, overflow_indef, res_indef);
    }
  unsigned lhs_elem_bitsize = bitsize_for_type(lhs_elem_type);
  unsigned bitsize = 1 + std::max(arg1->bitsize, arg2->bitsize);
  bitsize = 1 + std::max(bitsize, lhs_elem_bitsize);
  if (TYPE_UNSIGNED(arg1_type))
    arg1 = bb->build_inst(Op::ZEXT, arg1, bitsize);
  else
    arg1 = bb->build_inst(Op::SEXT, arg1, bitsize);
  if (TYPE_UNSIGNED(arg2_type))
    arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize);
  else
    arg2 = bb->build_inst(Op::SEXT, arg2, bitsize);
  Inst *inst = bb->build_inst(Op::SUB, arg1, arg2);
  Inst *res = bb->build_trunc(inst, lhs_elem_bitsize);
  Inst *eres;
  if (TYPE_UNSIGNED(lhs_elem_type))
    eres = bb->build_inst(Op::ZEXT, res, bitsize);
  else
    eres = bb->build_inst(Op::SEXT, res, bitsize);
  Inst *overflow = bb->build_inst(Op::NE, inst, eres);

  res = to_mem_repr(res, lhs_elem_type);
  overflow = bb->build_inst(Op::ZEXT, overflow, res->bitsize);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_reduc(gimple *stmt, tree_code code)
{
  assert(gimple_call_num_args(stmt) == 1);
  tree arg_expr = gimple_call_arg(stmt, 0);
  tree arg_type = TREE_TYPE(arg_expr);
  assert(VECTOR_TYPE_P(arg_type));
  tree elem_type = TREE_TYPE(arg_type);
  auto[arg, arg_indef, arg_prov] = tree2inst_indef_prov(arg_expr);
  tree lhs = gimple_call_lhs(stmt);

  uint32_t elem_bitsize = bitsize_for_type(elem_type);
  uint32_t nof_elt = bitsize_for_type(arg_type) / elem_bitsize;
  auto [inst, indef, prov] =
    extract_vec_elem(bb, arg, arg_indef, arg_prov, elem_bitsize, 0);
  for (uint64_t i = 1; i < nof_elt; i++)
    {
      auto [elem, elem_indef, elem_prov] =
	extract_vec_elem(bb, arg, arg_indef, arg_prov, elem_bitsize, i);
      std::tie(inst, indef, prov) =
	process_binary_scalar(code, inst, indef, prov, elem, elem_indef,
			      elem_prov, elem_type, elem_type, elem_type, true);
    }

  if (lhs)
    {
      tree2instruction.insert({lhs, inst});
      if (indef)
	tree2indef.insert({lhs, indef});
      if (prov)
	tree2prov.insert({lhs, prov});
    }
}

void Converter::process_cfn_reduc_fminmax(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  auto gen_elem_fmin =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *res = gen_fmin(bb, elem1, elem2);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };
  auto gen_elem_fmax =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *res = gen_fmax(bb, elem1, elem2);
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };

  tree arg_expr = gimple_call_arg(stmt, 0);
  tree arg_type = TREE_TYPE(arg_expr);
  assert(VECTOR_TYPE_P(arg_type));
  tree elem_type = TREE_TYPE(arg_type);
  auto[arg, arg_indef] = tree2inst_indef(arg_expr);
  tree lhs = gimple_call_lhs(stmt);

  combined_fn code = gimple_call_combined_fn(stmt);
  assert(code == CFN_REDUC_FMIN || code == CFN_REDUC_FMAX);

  uint32_t elem_bitsize = bitsize_for_type(elem_type);
  uint32_t nof_elt = bitsize_for_type(arg_type) / elem_bitsize;
  auto [inst, indef] = extract_vec_elem(bb, arg, arg_indef, elem_bitsize, 0);
  for (uint64_t i = 1; i < nof_elt; i++)
    {
      auto [elem, elem_indef] =
	extract_vec_elem(bb, arg, arg_indef, elem_bitsize, i);
      if (code == CFN_REDUC_FMIN)
	std::tie(inst, indef) =
	  gen_elem_fmin(inst, indef, elem, elem_indef, elem_type);
      else
	std::tie(inst, indef) =
	  gen_elem_fmax(inst, indef, elem, elem_indef, elem_type);
    }

  if (lhs)
    {
      tree2instruction.insert({lhs, inst});
      if (indef)
	tree2indef.insert({lhs, indef});
    }
}

void Converter::process_cfn_trap(gimple *stmt ATTRIBUTE_UNUSED)
{
  assert(gimple_call_num_args(stmt) == 0);
  // TODO: Some passes add __builtin_trap for cases that are UB (so that
  // the program terminates instead of continuing in a random state).
  // We threat these as UB for now, but they should arguably be handled
  // in a special way to verify that we actually are termininating.
  bb->build_inst(Op::UB, bb->value_inst(1, 1));
}

void Converter::process_cfn_vcond_mask(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;

  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  auto [arg3, arg3_indef] = tree2inst_indef(arg3_expr);
  auto [inst, indef] = gen_vec_cond(arg1, arg1_indef, arg2, arg2_indef,
				    arg3, arg3_indef, arg1_type, arg2_type);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  if (indef)
    tree2indef.insert({lhs, indef});
}

void Converter::process_cfn_vcond_mask_len(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 5);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  tree len_expr = gimple_call_arg(stmt, 3);
  assert(TYPE_UNSIGNED(TREE_TYPE(len_expr)));
  Inst *len = tree2inst(len_expr);
  tree len_type = TREE_TYPE(len_expr);
  tree bias_expr = gimple_call_arg(stmt, 4);
  Inst *bias = tree2inst(bias_expr);
  tree bias_type = TREE_TYPE(bias_expr);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;

  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  auto [arg3, arg3_indef] = tree2inst_indef(arg3_expr);
  bias = type_convert(bias, bias_type, len_type);
  len = bb->build_inst(Op::ADD, len, bias);
  auto [inst, indef] = gen_vec_cond(arg1, arg1_indef, arg2, arg2_indef,
				    arg3, arg3_indef, arg1_type, arg2_type,
				    len);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  if (indef)
    tree2indef.insert({lhs, indef});
}

void Converter::process_cfn_vec_addsub(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  assert(VECTOR_TYPE_P(TREE_TYPE(arg2_expr)) == VECTOR_TYPE_P(arg1_type));
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);

  tree elem_type = TREE_TYPE(arg1_type);
  uint32_t elem_bitsize = bitsize_for_type(elem_type);
  uint32_t nof_elem = bitsize_for_type(arg1_type) / elem_bitsize;

  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  assert(VECTOR_TYPE_P(TREE_TYPE(lhs)) == VECTOR_TYPE_P(arg1_type));

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint32_t j = 0; j < nof_elem; j++)
    {
      Inst *elem1 = extract_vec_elem(bb, arg1, elem_bitsize, j);
      Inst *elem2 = extract_vec_elem(bb, arg2, elem_bitsize, j);
      Inst *elem1_indef = nullptr;
      if (arg1_indef)
	elem1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, j);
      Inst *elem2_indef = nullptr;
      if (arg2_indef)
	elem2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, j);
      Op op = (j & 1) ? Op::FADD : Op::FSUB;
      Inst *inst = bb->build_inst(op, elem1, elem2);
      Inst *indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
      if (indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, indef, res_indef);
	  else
	    res_indef = indef;
	}
    }
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_vec_convert(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 1);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  Inst *arg1 = tree2inst(arg1_expr);
  tree arg1_elem_type = TREE_TYPE(TREE_TYPE(arg1_expr));
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  auto [inst, indef] =
    process_unary_vec(CONVERT_EXPR, arg1, nullptr, lhs_elem_type,
		      arg1_elem_type);
  assert(!indef);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_vec_extract(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;

  uint32_t elem_bitsize = bitsize_for_type(TREE_TYPE(arg1_type));
  uint32_t nof_elem = bitsize_for_type(arg1_type) / elem_bitsize;
  Inst *nof = bb->value_inst(nof_elem, arg2->bitsize);
  Inst *in_range = bb->build_inst(Op::ULT, arg2, nof);
  bb->build_inst(Op::UB, bb->build_inst(Op::NOT, in_range));

  Inst *res;
  Inst *res_indef = nullptr;
  if (TREE_CODE(arg2_expr) == INTEGER_CST || POLY_INT_CST_P(arg2_expr))
    {
      unsigned __int128 idx = get_int_cst_val(arg2_expr);
      if (idx >= nof_elem)
	{
	  // This is UB, so does not matter what we return.
	  res = bb->value_inst(0, elem_bitsize);
	}
      else
	std::tie(res, res_indef) =
	  extract_vec_elem(bb, arg1, arg1_indef, elem_bitsize, idx);
    }
  else
    std::tie(res, res_indef) =
      extract_elem(bb, arg1, arg1_indef, elem_bitsize, arg2, arg2_indef);

  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_vec_set(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  auto [arg3, arg3_indef] = tree2inst_indef(arg3_expr);

  uint32_t elem_bitsize = bitsize_for_type(TREE_TYPE(arg1_type));
  uint32_t nof_elem = bitsize_for_type(arg1_type) / elem_bitsize;
  assert(elem_bitsize == bitsize_for_type(TREE_TYPE(arg2_expr)));

  Inst *nof = bb->value_inst(nof_elem, arg3->bitsize);
  Inst *in_range = bb->build_inst(Op::ULT, arg3, nof);
  bb->build_inst(Op::UB, bb->build_inst(Op::NOT, in_range));

  if (arg1_indef || arg2_indef)
    {
      if (!arg1_indef)
	arg1_indef = bb->value_inst(0, arg1->bitsize);
      if (!arg2_indef)
	arg2_indef = bb->value_inst(0, arg2->bitsize);
    }
  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(bb, arg1, elem_bitsize, i);
      Inst *idx = bb->value_inst(i, arg3->bitsize);
      Inst *cmp = bb->build_inst(Op::EQ, idx, arg3);
      Inst *inst = bb->build_inst(Op::ITE, cmp, arg2, elem1);
      Inst *indef = nullptr;
      if (arg1_indef)
	{
	  Inst *elem1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, i);
	  indef = bb->build_inst(Op::ITE, cmp, arg2_indef, elem1_indef);
	}

      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
      if (indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, indef, res_indef);
	  else
	    res_indef = indef;
	}
    }

  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_vec_widen(gimple *stmt, Op op, bool high)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);

  tree elem_type = TREE_TYPE(arg1_type);
  uint32_t elem_bitsize = bitsize_for_type(elem_type);
  uint32_t nof_elem = (bitsize_for_type(arg1_type) / elem_bitsize) / 2;

  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  assert(bitsize_for_type(lhs_elem_type) == 2 * elem_bitsize);

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      uint32_t idx = high ? i + nof_elem : i;
      Inst *elem1 = extract_vec_elem(bb, arg1, elem_bitsize, idx);
      elem1 = type_convert(elem1, elem_type, lhs_elem_type);
      Inst *elem2 = extract_vec_elem(bb, arg2, elem_bitsize, idx);
      elem2 = type_convert(elem2, elem_type, lhs_elem_type);
      Inst *inst = bb->build_inst(op, elem1, elem2);

      Inst *elem1_indef = nullptr;
      if (arg1_indef)
	elem1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, idx);
      Inst *elem2_indef = nullptr;
      if (arg2_indef)
	elem2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, idx);
      Inst *indef = get_res_indef(elem1_indef, elem2_indef, lhs_elem_type);

      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
      if (indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, indef, res_indef);
	  else
	    res_indef = indef;
	}
    }
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_vec_widen_abd(gimple *stmt, bool high)
{
  assert(gimple_call_num_args(stmt) == 2);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);

  tree elem_type = TREE_TYPE(arg1_type);
  uint32_t elem_bitsize = bitsize_for_type(elem_type);
  uint32_t nof_elem = (bitsize_for_type(arg1_type) / elem_bitsize) / 2;

  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  assert(bitsize_for_type(lhs_elem_type) == 2 * elem_bitsize);

  Inst *res = nullptr;
  Inst *res_indef = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      uint32_t idx = high ? i + nof_elem : i;
      Inst *elem1 = extract_vec_elem(bb, arg1, elem_bitsize, idx);
      elem1 = type_convert(elem1, elem_type, lhs_elem_type);
      Inst *elem2 = extract_vec_elem(bb, arg2, elem_bitsize, idx);
      elem2 = type_convert(elem2, elem_type, lhs_elem_type);
      Inst *inst = bb->build_inst(Op::SUB, elem1, elem2);
      Inst *neg_inst = bb->build_inst(Op::NEG, inst);
      Inst *zero = bb->value_inst(0, inst->bitsize);
      Inst *cmp = bb->build_inst(Op::SLT, inst, zero);
      inst = bb->build_inst(Op::ITE, cmp, neg_inst, inst);

      Inst *elem1_indef = nullptr;
      if (arg1_indef)
	elem1_indef = extract_vec_elem(bb, arg1_indef, elem_bitsize, idx);
      Inst *elem2_indef = nullptr;
      if (arg2_indef)
	elem2_indef = extract_vec_elem(bb, arg2_indef, elem_bitsize, idx);
      Inst *indef = get_res_indef(elem1_indef, elem2_indef, lhs_elem_type);

      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
      if (indef)
	{
	  if (res_indef)
	    res_indef = bb->build_inst(Op::CONCAT, indef, res_indef);
	  else
	    res_indef = indef;
	}
    }
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_uaddc(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_uaddc: vector type");
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  unsigned lhs_elem_bitsize = bitsize_for_type(lhs_elem_type);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  auto [arg3, arg3_indef] = tree2inst_indef(arg3_expr);
  Inst *res_indef =
    get_res_indef(arg1_indef, arg2_indef, arg3_indef, TREE_TYPE(lhs));
  assert(arg1->bitsize == arg2->bitsize);
  assert(arg1->bitsize == arg3->bitsize);
  assert(lhs_elem_bitsize == arg1->bitsize);

  arg1 = bb->build_inst(Op::ZEXT, arg1, arg1->bitsize + 2);
  arg2 = bb->build_inst(Op::ZEXT, arg2, arg2->bitsize + 2);
  arg3 = bb->build_inst(Op::ZEXT, arg3, arg3->bitsize + 2);
  Inst *sum = bb->build_inst(Op::ADD, arg1, arg2);
  sum = bb->build_inst(Op::ADD, sum, arg3);
  Inst *res = bb->build_trunc(sum, lhs_elem_bitsize);

  Inst *high = bb->value_inst(sum->bitsize - 1, 32);
  Inst *low = bb->value_inst(sum->bitsize - 2, 32);
  Inst *overflow = bb->build_inst(Op::EXTRACT, sum, high, low);
  Inst *zero = bb->value_inst(0, overflow->bitsize);
  overflow = bb->build_inst(Op::NE, overflow, zero);
  overflow = bb->build_inst(Op::ZEXT, overflow, lhs_elem_bitsize);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_unreachable(gimple *stmt ATTRIBUTE_UNUSED)
{
  assert(gimple_call_num_args(stmt) == 0);
  bb->build_inst(Op::UB, bb->value_inst(1, 1));
}

void Converter::process_cfn_usubc(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_usubc: vector type");
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  unsigned lhs_elem_bitsize = bitsize_for_type(lhs_elem_type);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
  auto [arg3, arg3_indef] = tree2inst_indef(arg3_expr);
  assert(arg1->bitsize == arg2->bitsize);
  assert(arg1->bitsize == arg3->bitsize);
  assert(lhs_elem_bitsize == arg1->bitsize);
  Inst *res_indef =
    get_res_indef(arg1_indef, arg2_indef, arg3_indef, TREE_TYPE(lhs));

  arg1 = bb->build_inst(Op::ZEXT, arg1, arg1->bitsize + 2);
  arg2 = bb->build_inst(Op::ZEXT, arg2, arg2->bitsize + 2);
  arg3 = bb->build_inst(Op::ZEXT, arg3, arg3->bitsize + 2);
  Inst *sum = bb->build_inst(Op::SUB, arg1, arg2);
  sum = bb->build_inst(Op::SUB, sum, arg3);
  Inst *res = bb->build_trunc(sum, lhs_elem_bitsize);

  Inst *high = bb->value_inst(sum->bitsize - 1, 32);
  Inst *low = bb->value_inst(sum->bitsize - 2, 32);
  Inst *overflow = bb->build_inst(Op::EXTRACT, sum, high, low);
  Inst *zero = bb->value_inst(0, overflow->bitsize);
  overflow = bb->build_inst(Op::NE, overflow, zero);
  overflow = bb->build_inst(Op::ZEXT, overflow, lhs_elem_bitsize);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_indef)
    tree2indef.insert({lhs, res_indef});
}

void Converter::process_cfn_while_ult(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 3);
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  // TODO: Handle arg3_expr.
  Inst *arg1 = tree2inst(arg1_expr);
  Inst *arg2 = tree2inst(arg2_expr);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  tree lhs_type = TREE_TYPE(lhs);

  uint32_t elem_bitsize = bitsize_for_type(TREE_TYPE(lhs_type));
  uint32_t nof_elem = bitsize_for_type(lhs_type) / elem_bitsize;
  Inst *res = nullptr;
  Inst *one = bb->value_inst(1, arg1->bitsize);
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *inst = bb->build_inst(Op::ULT, arg1, arg2);
      if (elem_bitsize > 1)
	inst = bb->build_inst(Op::SEXT, inst, elem_bitsize);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
      arg1 = bb->build_inst(Op::ADD, arg1, one);
    }

  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
}

void Converter::process_cfn_xorsign(gimple *stmt)
{
  assert(gimple_call_num_args(stmt) == 2);
  auto gen_elem =
    [this](Inst *elem1, Inst *elem1_indef, Inst *elem2, Inst *elem2_indef,
	   tree elem_type) -> std::pair<Inst *, Inst *>
    {
      Inst *signbit1 = bb->build_extract_bit(elem1, elem1->bitsize - 1);
      Inst *signbit2 = bb->build_extract_bit(elem2, elem2->bitsize - 1);
      Inst *signbit = bb->build_inst(Op::XOR, signbit1, signbit2);
      Inst *res = bb->build_trunc(elem1, elem1->bitsize - 1);
      res = bb->build_inst(Op::CONCAT, signbit, res);
      if (state->arch == Arch::gimple)
	{
	  // For now, treat copying the sign to NaN as always produce the
	  // original canonical NaN.
	  // TODO: Remove this when Op::IS_NONCANONICAL_NAN is removed.
	  Inst *is_nan = bb->build_inst(Op::IS_NAN, elem1);
	  res = bb->build_inst(Op::ITE, is_nan, elem1, res);
	}
      Inst *res_indef = get_res_indef(elem1_indef, elem2_indef, elem_type);
      return {res, res_indef};
    };
  process_cfn_binary(stmt, gen_elem);
}

void Converter::process_gimple_call_combined_fn(gimple *stmt)
{
  switch (gimple_call_combined_fn(stmt))
    {
    case CFN_ABD:
      process_cfn_abd(stmt);
      break;
    case CFN_ADD_OVERFLOW:
      process_cfn_add_overflow(stmt);
      break;
    case CFN_BIT_ANDN:
      process_cfn_bit_andn(stmt);
      break;
    case CFN_BIT_IORN:
      process_cfn_bit_iorn(stmt);
      break;
    case CFN_BUILT_IN_ABORT:
      process_cfn_abort(stmt);
      break;
    case CFN_BUILT_IN_ASSUME_ALIGNED:
      process_cfn_assume_aligned(stmt);
      break;
    case CFN_BUILT_IN_BSWAP16:
    case CFN_BUILT_IN_BSWAP32:
    case CFN_BUILT_IN_BSWAP64:
    case CFN_BUILT_IN_BSWAP128:
      process_cfn_bswap(stmt);
      break;
    case CFN_BUILT_IN_CLRSB:
    case CFN_BUILT_IN_CLRSBL:
    case CFN_BUILT_IN_CLRSBLL:
    case CFN_CLRSB:
      process_cfn_clrsb(stmt);
      break;
    case CFN_BUILT_IN_CLZ:
    case CFN_BUILT_IN_CLZL:
    case CFN_BUILT_IN_CLZLL:
    case CFN_CLZ:
      process_cfn_clz(stmt);
      break;
    case CFN_BUILT_IN_COPYSIGN:
    case CFN_BUILT_IN_COPYSIGNF:
    case CFN_BUILT_IN_COPYSIGNL:
    case CFN_BUILT_IN_COPYSIGNF16:
    case CFN_BUILT_IN_COPYSIGNF32:
    case CFN_BUILT_IN_COPYSIGNF32X:
    case CFN_BUILT_IN_COPYSIGNF64:
    case CFN_BUILT_IN_COPYSIGNF128:
    case CFN_COPYSIGN:
      process_cfn_copysign(stmt);
      break;
    case CFN_BUILT_IN_CTZ:
    case CFN_BUILT_IN_CTZL:
    case CFN_BUILT_IN_CTZLL:
    case CFN_CTZ:
      process_cfn_ctz(stmt);
      break;
    case CFN_BUILT_IN_EXIT:
      process_cfn_exit(stmt);
      break;
    case CFN_BUILT_IN_EXPECT:
    case CFN_BUILT_IN_EXPECT_WITH_PROBABILITY:
      process_cfn_expect(stmt);
      break;
    case CFN_BUILT_IN_FABSF:
    case CFN_BUILT_IN_FABS:
    case CFN_BUILT_IN_FABSL:
      process_cfn_fabs(stmt);
      break;
    case CFN_BUILT_IN_FFS:
    case CFN_BUILT_IN_FFSL:
    case CFN_BUILT_IN_FFSLL:
    case CFN_FFS:
      process_cfn_ffs(stmt);
      break;
    case CFN_BUILT_IN_FMAX:
    case CFN_BUILT_IN_FMAXF:
    case CFN_BUILT_IN_FMAXL:
    case CFN_BUILT_IN_FMAXF16:
    case CFN_BUILT_IN_FMAXF32:
    case CFN_BUILT_IN_FMAXF64:
    case CFN_FMAX:
      process_cfn_fmax(stmt);
      break;
    case CFN_BUILT_IN_FMIN:
    case CFN_BUILT_IN_FMINF:
    case CFN_BUILT_IN_FMINL:
    case CFN_BUILT_IN_FMINF16:
    case CFN_BUILT_IN_FMINF32:
    case CFN_BUILT_IN_FMINF64:
    case CFN_FMIN:
      process_cfn_fmin(stmt);
      break;
    case CFN_BUILT_IN_ISFINITE:
      process_cfn_isfinite(stmt);
      break;
    case CFN_BUILT_IN_ISINF:
    case CFN_BUILT_IN_ISINFF:
    case CFN_BUILT_IN_ISINFL:
      process_cfn_isinf(stmt);
      break;
    case CFN_BUILT_IN_MEMCPY:
      process_cfn_memcpy(stmt);
      break;
    case CFN_BUILT_IN_MEMMOVE:
      process_cfn_memmove(stmt);
      break;
    case CFN_BUILT_IN_MEMPCPY:
      process_cfn_mempcpy(stmt);
      break;
    case CFN_BUILT_IN_MEMSET:
      process_cfn_memset(stmt);
      break;
    case CFN_BUILT_IN_NAN:
    case CFN_BUILT_IN_NANF:
    case CFN_BUILT_IN_NANL:
      process_cfn_nan(stmt);
      break;
    case CFN_BUILT_IN_PARITY:
    case CFN_BUILT_IN_PARITYL:
    case CFN_BUILT_IN_PARITYLL:
    case CFN_PARITY:
      process_cfn_parity(stmt);
      break;
    case CFN_BUILT_IN_POPCOUNT:
    case CFN_BUILT_IN_POPCOUNTL:
    case CFN_BUILT_IN_POPCOUNTLL:
    case CFN_POPCOUNT:
      process_cfn_popcount(stmt);
      break;
    case CFN_SAT_ADD:
      process_cfn_sat_add(stmt);
      break;
    case CFN_SAT_SUB:
      process_cfn_sat_sub(stmt);
      break;
    case CFN_SAT_TRUNC:
      process_cfn_sat_trunc(stmt);
      break;
    case CFN_SELECT_VL:
      process_cfn_select_vl(stmt);
      break;
    case CFN_BUILT_IN_SIGNBIT:
    case CFN_BUILT_IN_SIGNBITF:
    case CFN_BUILT_IN_SIGNBITL:
    case CFN_SIGNBIT:
      process_cfn_signbit(stmt);
      break;
    case CFN_BUILT_IN_TRAP:
      process_cfn_trap(stmt);
      break;
    case CFN_BUILT_IN_UNREACHABLE:
    case CFN_BUILT_IN_UNREACHABLE_TRAP:
      process_cfn_unreachable(stmt);
      break;
    case CFN_CHECK_RAW_PTRS:
      process_cfn_check_raw_ptrs(stmt);
      break;
    case CFN_CHECK_WAR_PTRS:
      process_cfn_check_war_ptrs(stmt);
      break;
    case CFN_COND_ADD:
      process_cfn_cond_binary(stmt, PLUS_EXPR);
      break;
    case CFN_COND_AND:
      process_cfn_cond_binary(stmt, BIT_AND_EXPR);
      break;
    case CFN_COND_IOR:
      process_cfn_cond_binary(stmt, BIT_IOR_EXPR);
      break;
    case CFN_COND_MAX:
      process_cfn_cond_binary(stmt, MAX_EXPR);
      break;
    case CFN_COND_MIN:
      process_cfn_cond_binary(stmt, MIN_EXPR);
      break;
    case CFN_COND_MUL:
      process_cfn_cond_binary(stmt, MULT_EXPR);
      break;
    case CFN_COND_NEG:
      process_cfn_cond_unary(stmt, NEGATE_EXPR);
      break;
    case CFN_COND_NOT:
      process_cfn_cond_unary(stmt, BIT_NOT_EXPR);
      break;
    case CFN_COND_RDIV:
      process_cfn_cond_binary(stmt, RDIV_EXPR);
      break;
    case CFN_COND_SHL:
      process_cfn_cond_binary(stmt, LSHIFT_EXPR);
      break;
    case CFN_COND_SHR:
      process_cfn_cond_binary(stmt, RSHIFT_EXPR);
      break;
    case CFN_COND_SUB:
      process_cfn_cond_binary(stmt, MINUS_EXPR);
      break;
    case CFN_COND_XOR:
      process_cfn_cond_binary(stmt, BIT_XOR_EXPR);
      break;
    case CFN_COND_LEN_ADD:
      process_cfn_cond_len_binary(stmt, PLUS_EXPR);
      break;
    case CFN_COND_LEN_AND:
      process_cfn_cond_len_binary(stmt, BIT_AND_EXPR);
      break;
    case CFN_COND_LEN_IOR:
      process_cfn_cond_len_binary(stmt, BIT_IOR_EXPR);
      break;
    case CFN_COND_LEN_MAX:
      process_cfn_cond_len_binary(stmt, MAX_EXPR);
      break;
    case CFN_COND_LEN_MIN:
      process_cfn_cond_len_binary(stmt, MIN_EXPR);
      break;
    case CFN_COND_LEN_MUL:
      process_cfn_cond_len_binary(stmt, MULT_EXPR);
      break;
    case CFN_COND_LEN_RDIV:
      process_cfn_cond_len_binary(stmt, RDIV_EXPR);
      break;
    case CFN_COND_LEN_SHL:
      process_cfn_cond_len_binary(stmt, LSHIFT_EXPR);
      break;
    case CFN_COND_LEN_SHR:
      process_cfn_cond_len_binary(stmt, RSHIFT_EXPR);
      break;
    case CFN_COND_LEN_SUB:
      process_cfn_cond_len_binary(stmt, MINUS_EXPR);
      break;
    case CFN_COND_LEN_XOR:
      process_cfn_cond_len_binary(stmt, BIT_XOR_EXPR);
      break;
    case CFN_COND_FMIN:
    case CFN_COND_FMAX:
      process_cfn_cond_fminmax(stmt);
      break;
    case CFN_DEFERRED_INIT:
      // DEFERRED_INIT initializes the memory. But the value is still
      // considered uninitialized, so everything is already handled by
      // our indef handling. So there is nothing to do here.
      break;
    case CFN_DIVMOD:
      process_cfn_divmod(stmt);
      break;
    case CFN_FALLTHROUGH:
      break;
    case CFN_LOOP_VECTORIZED:
      process_cfn_loop_vectorized(stmt);
      break;
    case CFN_MASK_LEN_LOAD:
      process_cfn_mask_len_load(stmt);
      break;
    case CFN_MASK_LEN_STORE:
      process_cfn_mask_len_store(stmt);
      break;
    case CFN_MASK_LOAD:
      process_cfn_mask_load(stmt);
      break;
    case CFN_MASK_STORE:
      process_cfn_mask_store(stmt);
      break;
    case CFN_MUL_OVERFLOW:
      process_cfn_mul_overflow(stmt);
      break;
    case CFN_MULH:
      process_cfn_mulh(stmt);
      break;
    case CFN_REDUC_AND:
      process_cfn_reduc(stmt, BIT_AND_EXPR);
      break;
    case CFN_REDUC_IOR:
      process_cfn_reduc(stmt, BIT_IOR_EXPR);
      break;
    case CFN_REDUC_MAX:
      process_cfn_reduc(stmt, MAX_EXPR);
      break;
    case CFN_REDUC_MIN:
      process_cfn_reduc(stmt, MIN_EXPR);
      break;
    case CFN_REDUC_PLUS:
      process_cfn_reduc(stmt, PLUS_EXPR);
      break;
    case CFN_REDUC_XOR:
      process_cfn_reduc(stmt, BIT_XOR_EXPR);
      break;
    case CFN_REDUC_FMIN:
    case CFN_REDUC_FMAX:
      process_cfn_reduc_fminmax(stmt);
      break;
    case CFN_SUB_OVERFLOW:
      process_cfn_sub_overflow(stmt);
      break;
    case CFN_UADDC:
      process_cfn_uaddc(stmt);
      break;
    case CFN_USUBC:
      process_cfn_usubc(stmt);
      break;
    case CFN_VCOND_MASK:
      process_cfn_vcond_mask(stmt);
      break;
    case CFN_VCOND_MASK_LEN:
      process_cfn_vcond_mask_len(stmt);
      break;
    case CFN_VEC_ADDSUB:
      process_cfn_vec_addsub(stmt);
      break;
    case CFN_VEC_CONVERT:
      process_cfn_vec_convert(stmt);
      break;
    case CFN_VEC_EXTRACT:
      process_cfn_vec_extract(stmt);
      break;
    case CFN_VEC_SET:
      process_cfn_vec_set(stmt);
      break;
    case CFN_VEC_WIDEN_ABD_HI:
      process_cfn_vec_widen_abd(stmt, true);
      break;
    case CFN_VEC_WIDEN_ABD_LO:
      process_cfn_vec_widen_abd(stmt, false);
      break;
    case CFN_VEC_WIDEN_MINUS_HI:
      process_cfn_vec_widen(stmt, Op::SUB, true);
      break;
    case CFN_VEC_WIDEN_MINUS_LO:
      process_cfn_vec_widen(stmt, Op::SUB, false);
      break;
    case CFN_VEC_WIDEN_PLUS_HI:
      process_cfn_vec_widen(stmt, Op::ADD, true);
      break;
    case CFN_VEC_WIDEN_PLUS_LO:
      process_cfn_vec_widen(stmt, Op::ADD, false);
      break;
    case CFN_WHILE_ULT:
      process_cfn_while_ult(stmt);
      break;
    case CFN_XORSIGN:
      process_cfn_xorsign(stmt);
      break;
    default:
      {
	const char *name;
	if (gimple_call_builtin_p(stmt))
	  name = fndecl_name(gimple_call_fndecl(stmt));
	else
	  name = internal_fn_name(gimple_call_internal_fn(stmt));
	throw Not_implemented("process_gimple_call_combined_fn: "s + name);
      }
    }
}

void Converter::process_gimple_call(gimple *stmt)
{
  if (gimple_call_builtin_p(stmt) || gimple_call_internal_p(stmt))
    process_gimple_call_combined_fn(stmt);
  else
    {
      std::string name = fndecl_name(gimple_call_fndecl(stmt));
      if (name.starts_with("__gcov"))
	{
	  if (name == "__gcov_average_profiler"
	      || name == "__gcov_dump"
	      || name == "__gcov_exit"
	      || name == "__gcov_indirect_call_profiler_v4"
	      || name == "__gcov_init"
	      || name == "__gcov_interval_profiler"
	      || name == "__gcov_ior_profiler"
	      || name == "__gcov_pow2_profiler"
	      || name == "__gcov_reset"
	      || name == "__gcov_topn_values_profiler"
	      || name == "__gcov_write_unsigned")
	    {
	      // The __gcov functions do not affect the semantics, so they
	      // can be ignored.
	    }
	  else
	    throw Not_implemented("gimple_call: " + name);
	}
      else
	throw Not_implemented("gimple_call");
    }
}

Inst *Converter::build_label_cond(tree index_expr, tree label, Basic_block *bb)
{
  tree index_type = TREE_TYPE(index_expr);
  Inst *index = tree2inst(index_expr);
  tree low_expr = CASE_LOW(label);
  Inst *low = tree2inst(low_expr);
  low = type_convert(low, TREE_TYPE(low_expr), index_type);
  tree high_expr = CASE_HIGH(label);
  Inst *cond;
  if (high_expr)
    {
      Inst *high = tree2inst(high_expr);
      high = type_convert(high, TREE_TYPE(high_expr), index_type);
      Op op = TYPE_UNSIGNED(index_type) ?  Op::ULE: Op::SLE;
      Inst *cond_low = bb->build_inst(op, low, index);
      Inst *cond_high = bb->build_inst(op, index, high);
      cond = bb->build_inst(Op::AND, cond_low, cond_high);
    }
  else
    cond = bb->build_inst(Op::EQ, index, low);
  return cond;
}

// Expand switch statements to a series of compare and branch.
void Converter::process_gimple_switch(gimple *stmt, Basic_block *switch_bb)
{
  gswitch *switch_stmt = as_a<gswitch *>(stmt);
  tree index_expr = gimple_switch_index(switch_stmt);

  // We expand the switch case to a series of compare and branch. This
  // complicates the phi node handling -- phi arguments from the BB
  // containing the switch statement should use the correct BB in the compare
  // and branch chain, so we must keep track of which new BBs corresponds
  // to the switch statement.
  std::set<Basic_block *>& bbset = switch_bbs[switch_bb];

  // We start the chain by an unconditional branch to a new BB instead of
  // doing the first compare-and-branch at the end of the BB containing the
  // switch statement. This is not necessary, but it avoids confusion as
  // the phi argument from switch always comes from a BB we have introduced.
  Basic_block *bb = func->build_bb();
  bbset.insert(bb);
  switch_bb->build_br_inst(bb);

  // Multiple switch cases may branch to the same basic block. Collect these
  // so that we only do one branch (in order to prevent complications when
  // the target contains phi nodes that would otherwise need to be adjusted
  // for the additional edges).
  basic_block default_block = gimple_switch_label_bb(fun, switch_stmt, 0);
  std::map<basic_block, std::vector<tree>> block2labels;
  size_t n = gimple_switch_num_labels(switch_stmt);
  std::vector<basic_block> cases;
  for (size_t i = 1; i < n; i++)
    {
      tree label = gimple_switch_label(switch_stmt, i);
      basic_block block = label_to_block(fun, CASE_LABEL(label));
      if (block == default_block)
	continue;
      if (!block2labels.contains(block))
	cases.push_back(block);
      block2labels[block].push_back(label);
    }

  if (cases.empty())
    {
      // All cases branch to the default case.
      bb->build_br_inst(gccbb_top2bb.at(default_block));
      return;
    }

  n = cases.size();
  for (size_t i = 0; i < n; i++)
    {
      Inst *cond = nullptr;
      basic_block block = cases[i];
      const std::vector<tree>& labels = block2labels.at(block);
      for (auto label : labels)
	{
	  Inst *label_cond = build_label_cond(index_expr, label, bb);
	  if (cond)
	    cond = bb->build_inst(Op::OR, cond, label_cond);
	  else
	    cond = label_cond;
	}

      Basic_block *true_bb = gccbb_top2bb.at(block);
      Basic_block *false_bb;
      if (i != n - 1)
	{
	  false_bb = func->build_bb();
	  bbset.insert(false_bb);
	}
      else
	false_bb = gccbb_top2bb.at(default_block);
      bb->build_br_inst(cond, true_bb, false_bb);
      bb = false_bb;
    }
}

// Get the BB corresponding to the source of the phi argument i.
Basic_block *Converter::get_phi_arg_bb(gphi *phi, int i)
{
  edge e = gimple_phi_arg_edge(phi, i);
  Basic_block *arg_bb = gccbb_bottom2bb.at(e->src);
  Basic_block *phi_bb = gccbb_top2bb.at(e->dest);
  if (switch_bbs.contains(arg_bb))
    {
      std::set<Basic_block *>& bbset = switch_bbs[arg_bb];
      assert(bbset.size() > 0);
      for (auto bb : bbset)
	{
	  auto it = std::find(phi_bb->preds.begin(), phi_bb->preds.end(), bb);
	  if (it != phi_bb->preds.end())
	    return bb;
	}
      assert(false);
    }
  return arg_bb;
}

void Converter::process_gimple_return(gimple *stmt)
{
  greturn *return_stmt = dyn_cast<greturn *>(stmt);
  tree expr = gimple_return_retval(return_stmt);
  if (expr)
    bb2retval.insert({bb, tree2inst_indef(expr)});
  // TODO: Add assert that the successor goes to the exit block. We will
  // miscompile otherwise...
}

Inst *split_phi(Inst *phi, uint64_t elem_bitsize, std::map<std::pair<Inst *, uint64_t>, std::vector<Inst *>>& cache)
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

void Converter::generate_exit_inst()
{
  if (bb_abort.empty() && bb2exit.empty())
    return;

  Inst *abort_called = bb->build_phi_inst(1);
  Inst *exit_called = bb->build_phi_inst(1);
  Inst *exit_val = bb->build_phi_inst(bitsize_for_type(integer_type_node));
  for (Basic_block *pred_bb : bb->preds)
    {
      Inst *a = bb->value_inst(bb_abort.contains(pred_bb), 1);
      abort_called->add_phi_arg(a, pred_bb);

      Inst *e = bb->value_inst(bb2exit.contains(pred_bb), 1);
      exit_called->add_phi_arg(e, pred_bb);

      Inst *v;
      if (bb2exit.contains(pred_bb))
	v = bb2exit[pred_bb];
      else
	v = bb->value_inst(0, exit_val->bitsize);
      exit_val->add_phi_arg(v, pred_bb);
    }
  bb->build_inst(Op::EXIT, abort_called, exit_called, exit_val);
}

void Converter::generate_return_inst()
{
  tree retval_type = TREE_TYPE(DECL_RESULT(fun->decl));
  if (VOID_TYPE_P(retval_type))
    {
      bb->build_ret_inst();
      return;
    }
  int retval_bitsize = bitsize_for_type(retval_type);
  if (!retval_bitsize)
    {
      bb->build_ret_inst();
      return;
    }

  // Some predecessors to the exit block may not have a return value;
  // They may have a return without value, or the predecessor may be
  // a builtin_unreachable, etc. We therefore creates a dummy value,
  // marked as indefinite, for these predecessors to make the IR valid.
  {
    Inst *retval = nullptr;
    Inst *indef = nullptr;
    Basic_block *entry_bb = func->bbs[0];
    for (Basic_block *pred_bb : bb->preds)
      {
	if (!bb2retval.contains(pred_bb))
	  {
	    if (!retval)
	      {
		retval = entry_bb->value_inst(0, retval_bitsize);
		unsigned bitsize = retval_bitsize;
		while (bitsize)
		  {
		    uint32_t bs = std::min(bitsize, 128u);
		    bitsize -= bs;
		    Inst *inst = entry_bb->value_inst(-1, bs);
		    if (indef)
		      indef = entry_bb->build_inst(Op::CONCAT, inst, indef);
		    else
		      indef = inst;
		  }
	      }
	    bb2retval.insert({pred_bb, {retval, indef}});
	  }
      }
  }

  Inst *retval;
  Inst *retval_indef;
  if (bb->preds.size() == 1)
    std::tie(retval, retval_indef) = bb2retval.at(bb->preds[0]);
  else
    {
      Inst *phi = bb->build_phi_inst(retval_bitsize);
      Inst *phi_indef = bb->build_phi_inst(retval_bitsize);
      bool need_indef_phi = false;
      for (Basic_block *pred_bb : bb->preds)
	{
	  auto [ret, ret_indef] = bb2retval.at(pred_bb);
	  phi->add_phi_arg(ret, pred_bb);
	  need_indef_phi = need_indef_phi || ret_indef;
	  if (!ret_indef)
	    ret_indef = pred_bb->value_inst(0, retval_bitsize);
	  phi_indef->add_phi_arg(ret_indef, pred_bb);
	}
      retval = phi;
      retval_indef = need_indef_phi ? phi_indef : nullptr;

      std::map<std::pair<Inst *, uint64_t>, std::vector<Inst *>> cache;
      if (VECTOR_TYPE_P(retval_type) || TREE_CODE(retval_type) == COMPLEX_TYPE)
	{
	  uint32_t elem_bitsize;
	  if (VECTOR_TYPE_P(retval_type))
	    elem_bitsize = bitsize_for_type(TREE_TYPE(retval_type));
	  else
	    elem_bitsize = bytesize_for_type(TREE_TYPE(retval_type)) * 8;
	  retval = split_phi(retval, elem_bitsize, cache);
	  if (retval_indef)
	    retval_indef = split_phi(retval_indef, elem_bitsize, cache);
	}
    }

  // GCC treats it as UB to return the address of a local variable.
  if (POINTER_TYPE_P(retval_type))
    {
      Inst *mem_id = extract_id(retval);
      Inst *zero = bb->value_inst(0, module->ptr_id_bits);
      Inst *cond = bb->build_inst(Op::SLT, mem_id, zero);
      if (retval_indef)
	{
	  Inst *zero2 = bb->value_inst(0, retval_indef->bitsize);
	  Inst *cond2 = bb->build_inst(Op::EQ, retval_indef, zero2);
	  cond = bb->build_inst(Op::AND, cond, cond2);
	}
      bb->build_inst(Op::UB, cond);
    }

  if (retval_indef)
    bb->build_ret_inst(retval, retval_indef);
  else
    bb->build_ret_inst(retval);
}

// Write the values to initialized variables.
void Converter::init_var_values(tree initial, Inst *mem_inst)
{
  assert(bb == func->bbs[0]);
  if (TREE_CODE(initial) == ERROR_MARK)
    throw Not_implemented("init_var_values: ERROR_MARK");

  tree type = TREE_TYPE(initial);
  uint64_t size = bytesize_for_type(TREE_TYPE(initial));

  if (TREE_CODE(initial) == STRING_CST || TREE_CODE(initial) == RAW_DATA_CST)
    {
      uint64_t len;
      const char *p;
      if (TREE_CODE(initial) == RAW_DATA_CST)
	{
	  len = RAW_DATA_LENGTH(initial);
	  p = RAW_DATA_POINTER(initial);
	}
      else
	{
	  len = TREE_STRING_LENGTH(initial);
	  p = TREE_STRING_POINTER(initial);
	}
      for (uint64_t i = 0; i < len; i++)
	{
	  Inst *offset = bb->value_inst(i, mem_inst->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, mem_inst, offset);
	  Inst *byte = bb->value_inst(p[i], 8);
	  bb->build_inst(Op::STORE, ptr, byte);
	}
      for (uint64_t i = len; i < size; i++)
	{
	  Inst *offset = bb->value_inst(i, mem_inst->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, mem_inst, offset);
	  Inst *byte = bb->value_inst(0, 8);
	  bb->build_inst(Op::STORE, ptr, byte);
	}
      return;
    }

  if (INTEGRAL_TYPE_P(type)
      || TREE_CODE(type) == OFFSET_TYPE
      || FLOAT_TYPE_P(type)
      || POINTER_TYPE_P(type)
      || VECTOR_TYPE_P(type))
    {
      auto [value, indef, prov] = tree2inst_constructor(initial);
      value = to_mem_repr(value, type);
      store_value(mem_inst, value);
      return;
    }

  if (TREE_CODE(type) == ARRAY_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      uint64_t elem_size = bytesize_for_type(elem_type);
      Inst *elm_size = bb->value_inst(elem_size, mem_inst->bitsize);
      unsigned HOST_WIDE_INT idx;
      tree index;
      tree value;
      FOR_EACH_CONSTRUCTOR_ELT(CONSTRUCTOR_ELTS(initial), idx, index, value)
	{
	  Inst *off;
	  if (index)
	    {
	      if (TREE_CODE(index) == RANGE_EXPR)
		throw Not_implemented("init_var: RANGE_EXPR");
	      Inst *indx = tree2inst(index);
	      if (indx->bitsize < mem_inst->bitsize)
		{
		  if (TYPE_UNSIGNED(TREE_TYPE(index)))
		    indx = bb->build_inst(Op::ZEXT, indx, mem_inst->bitsize);
		  else
		    indx = bb->build_inst(Op::SEXT, indx, mem_inst->bitsize);
		}
	      else if (indx->bitsize > mem_inst->bitsize)
		indx = bb->build_trunc(indx, mem_inst->bitsize);
	      off = bb->build_inst(Op::MUL, indx, elm_size);
	    }
	  else
	    off = bb->value_inst(idx * elem_size, mem_inst->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, mem_inst, off);
	  init_var_values(value, ptr);
	}
      return;
    }

  if (TREE_CODE(type) == RECORD_TYPE || TREE_CODE(type) == UNION_TYPE)
    {
      unsigned HOST_WIDE_INT idx;
      tree index;
      tree value;
      if (TREE_CODE(initial) != CONSTRUCTOR)
	throw Not_implemented("init_var_values: initial record/union value "
			      "is not a CONSTRUCTOR");
      FOR_EACH_CONSTRUCTOR_ELT(CONSTRUCTOR_ELTS(initial), idx, index, value)
	{
	  uint64_t offset = get_int_cst_val(DECL_FIELD_OFFSET(index));
	  uint64_t bit_offset = get_int_cst_val(DECL_FIELD_BIT_OFFSET(index));
	  offset += bit_offset / 8;
	  bit_offset &= 7;
	  Inst *off = bb->value_inst(offset, mem_inst->bitsize);
	  Inst *ptr = bb->build_inst(Op::ADD, mem_inst, off);
	  tree elem_type = TREE_TYPE(value);
	  if (TREE_CODE(elem_type) == ARRAY_TYPE
	      || TREE_CODE(elem_type) == RECORD_TYPE
	      || TREE_CODE(elem_type) == UNION_TYPE)
	    init_var_values(value, ptr);
	  else
	    {
	      uint64_t bitsize = bitsize_for_type(elem_type);
	      auto [value_inst, indef, prov] = tree2inst_constructor(value);
	      size = (bitsize + bit_offset + 7) / 8;
	      if (DECL_BIT_FIELD_TYPE(index))
		{
		  if (bit_offset)
		    {
		      Inst *first_byte = bb->build_inst(Op::LOAD, ptr);
		      Inst *bits =
			bb->build_trunc(first_byte, bit_offset);
		      value_inst =
			bb->build_inst(Op::CONCAT, value_inst, bits);
		    }
		  if (bitsize + bit_offset != size * 8)
		    {
		      Inst *offset =
			bb->value_inst(size - 1, ptr->bitsize);
		      Inst *ptr3 = bb->build_inst(Op::ADD, ptr, offset);

		      uint64_t remaining = size * 8 - (bitsize + bit_offset);
		      assert(remaining < 8);
		      Inst *high = bb->value_inst(7, 32);
		      Inst *low = bb->value_inst(8 - remaining, 32);

		      Inst *last_byte = bb->build_inst(Op::LOAD, ptr3);
		      Inst *bits =
			bb->build_inst(Op::EXTRACT, last_byte, high, low);
		      value_inst =
			bb->build_inst(Op::CONCAT, bits, value_inst);
		    }
		}
	      else
		{
		  value_inst = to_mem_repr(value_inst, elem_type);
		}
	      store_value(ptr, value_inst);
	    }
	}
      return;
    }

  throw Not_implemented("init_var: unknown constructor");
}

void Converter::init_var(tree decl, Inst *mem_inst)
{
  uint64_t size = bytesize_for_type(TREE_TYPE(decl));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("init_var: too large constructor");
  check_type(TREE_TYPE(decl));

  Basic_block *entry_bb = func->bbs[0];
  assert(mem_inst->bb == entry_bb);

  tree initial = DECL_INITIAL(decl);
  if (!initial)
    {
      if (!TREE_STATIC(decl))
	return;

      // Uninitializied static variables are guaranted to be initialized to 0.
      Inst *zero = entry_bb->value_inst(0, 8);
      uint64_t size = bytesize_for_type(TREE_TYPE(decl));
      for (uint64_t i = 0; i < size; i++)
	{
	  Inst *offset = entry_bb->value_inst(i, mem_inst->bitsize);
	  Inst *ptr = entry_bb->build_inst(Op::ADD, mem_inst, offset);
	  entry_bb->build_inst(Op::STORE, ptr, zero);
	}
      return;
    }

  if (TREE_CODE(initial) == CONSTRUCTOR)
    {
      assert(TREE_CODE(initial) == CONSTRUCTOR);
      tree type = TREE_TYPE(initial);
      uint64_t size = bytesize_for_type(TREE_TYPE(initial));

      if (CONSTRUCTOR_NO_CLEARING(initial))
	throw Not_implemented("init_var: CONSTRUCTOR_NO_CLEARING");

      Inst *zero = entry_bb->value_inst(0, 8);
      if (size > MAX_MEMORY_UNROLL_LIMIT)
	throw Not_implemented("init_var: too large constructor");
      for (uint64_t i = 0; i < size; i++)
	{
	  Inst *offset = entry_bb->value_inst(i, mem_inst->bitsize);
	  Inst *ptr = entry_bb->build_inst(Op::ADD, mem_inst, offset);
	  uint8_t padding = padding_at_offset(type, i);
	  if (padding)
	    entry_bb->build_inst(Op::SET_MEM_INDEF, ptr,
				 entry_bb->value_inst(padding, 8));
	  entry_bb->build_inst(Op::STORE, ptr, zero);
	}
    }

  Basic_block *orig_bb = bb;
  bb = func->bbs[0];
  init_var_values(initial, mem_inst);
  bb = orig_bb;
}

void Converter::make_uninit(Inst *orig_ptr, uint64_t size)
{
  Inst *byte_m1 = bb->value_inst(255, 8);
  for (uint64_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, orig_ptr->bitsize);
      Inst *ptr = bb->build_inst(Op::ADD, orig_ptr, offset);
      bb->build_inst(Op::SET_MEM_INDEF, ptr, byte_m1);
    }
}

void Converter::process_variables()
{
  // Add anonymous memory.
  // TODO: Should only be done if we have unconstrained pointers?
  build_memory_inst(2, ANON_MEM_SIZE, MEM_KEEP);

  // Add global variables.
  //
  // Tgt must have the same global constant memory as src, even if it
  // is not used (for example, when arr[5] has been substituted with
  // the value from the initialization). We ensure this by always
  // adding all previously created global constant variables.
  for (auto [decl, _] : state->decl2id)
    {
      if (!decl2instruction.contains(decl)
	  && TREE_PUBLIC(decl)
	  && TREE_READONLY(decl))
	process_decl(decl);
    }
}

void Converter::process_func_args()
{
  std::vector<std::pair<Inst*, tree>> pointers;
  tree fntype = TREE_TYPE(fun->decl);
  bitmap nonnullargs = get_nonnull_args(fntype);
  tree decl;
  int param_number = 0;
  Basic_block *entry_bb = func->bbs[0];
  const char *decl_name = IDENTIFIER_POINTER(DECL_NAME(fun->decl));
  for (decl = DECL_ARGUMENTS(fun->decl); decl; decl = DECL_CHAIN(decl))
    {
      check_type(TREE_TYPE(decl));
      uint32_t bitsize = bitsize_for_type(TREE_TYPE(decl));
      if (bitsize <= 0)
	throw Not_implemented("Parameter size == 0");

      // TODO: There must be better ways to determine if this is the "this"
      // pointer of a C++ constructor.
      if (param_number == 0 && !strcmp(decl_name, "__ct_base "))
	{
	  assert(POINTER_TYPE_P(TREE_TYPE(decl)));

	  // We use constant ID as it must be the same between src and tgt.
	  int64_t id = 1;
	  uint64_t flags = MEM_UNINIT | MEM_KEEP;
	  uint64_t size = bytesize_for_type(TREE_TYPE(TREE_TYPE(decl)));

	  Inst *param_inst = build_memory_inst(id, size, flags);
	  tree2prov.insert({decl, param_inst->args[0]});
	  tree2instruction.insert({decl, param_inst});
	}
      else
	{
	  Inst *param_nbr = entry_bb->value_inst(param_number, 32);
	  Inst *param_bitsize = entry_bb->value_inst(bitsize, 32);
	  Inst *param_inst =
	    entry_bb->build_inst(Op::PARAM, param_nbr, param_bitsize);
	  tree2instruction.insert({decl, param_inst});

	  // Pointers cannot point to local variables or to the this pointer
	  // in constructors.
	  // TODO: Update all pointer UB checks for this.
	  if (POINTER_TYPE_P(TREE_TYPE(decl)))
	    {
	      if (TYPE_RESTRICT(TREE_TYPE(decl)))
		{
		  if (!is_tgt_func)
		    {
		      // Create a new memory object that is only used for this
		      // restrict pointer.
		      if (state->id_global >= state->ptr_id_max)
			throw Not_implemented("process_func_args: "
					      "too many global variables");
		      uint64_t restrict_id = ++state->id_global;
		      build_memory_inst(restrict_id, ANON_MEM_SIZE, MEM_KEEP);

		      // We force the Op::PARAM to use this memory by making all
		      // other memory ID UB instead of just setting it to the
		      // memory directly -- this is needed to ensure we test
		      // with different alignment/sizes.
		      Inst *id1 = extract_id(param_inst);
		      Inst *id2 =
			entry_bb->value_inst(restrict_id, module->ptr_id_bits);
		      entry_bb->build_inst(Op::UB,
					   entry_bb->build_inst(Op::NE, id1, id2));
		      state->restrict_ids.push_back(id2);
		    }
		}
	      else
		pointers.push_back({param_inst, decl});
	    }

	  // Constrain the value for non-pointers. Pointers must wait until
	  // all parameters have been processed so we have found all restrict
	  // pointers.
	  if (!POINTER_TYPE_P(TREE_TYPE(decl)))
	    constrain_src_value(param_inst, TREE_TYPE(decl));

	  // Params marked "nonnull" is UB if NULL.
	  if (POINTER_TYPE_P(TREE_TYPE(decl))
	      && nonnullargs
	      && (bitmap_empty_p(nonnullargs)
		  || bitmap_bit_p(nonnullargs, param_number)))
	    {
	      Inst *zero = entry_bb->value_inst(0, param_inst->bitsize);
	      Inst *cond = entry_bb->build_inst(Op::EQ, param_inst, zero);
	      entry_bb->build_inst(Op::UB, cond);
	    }

	  // VRP
	  // If there are recorded data, we get a constant value, and a mask
	  // indicating which bits varies. For example, for a funcion
	  //   static void foo(uint16_t x);
	  // is called as
	  //   foo(0xfffe);
	  //   foo(0xf0ff);
	  // we get value == 0xf0fe and mask == 0xf01
	  tree value;
	  widest_int mask;
	  if (param_inst->bitsize <= 128  // TODO: Implement wide types.
	      && ipcp_get_parm_bits(decl, &value, &mask))
	    {
	      unsigned __int128 m = get_widest_int_val(mask);
	      unsigned __int128 v = get_int_cst_val(value);
	      assert((m & v) == 0);

	      Inst *m_inst = entry_bb->value_inst(~m, param_inst->bitsize);
	      Inst *v_inst = entry_bb->value_inst(v, param_inst->bitsize);
	      Inst *and_inst = entry_bb->build_inst(Op::AND, param_inst, m_inst);
	      Inst *cond = entry_bb->build_inst(Op::NE, v_inst, and_inst);
	      entry_bb->build_inst(Op::UB, cond);
	    }
	}

      if (POINTER_TYPE_P(TREE_TYPE(decl)))
	{
	  Inst *param_inst = tree2instruction.at(decl);
	  Inst *id = extract_id(param_inst);
	  tree2prov.insert({decl, id});
	}

      param_number++;
    }

  for (auto [param_inst, decl] : pointers)
    {
      constrain_src_value(param_inst, TREE_TYPE(decl));

      // TODO: This should move into constrain_src_value.
      Inst *id = extract_id(param_inst);
      Inst *one = entry_bb->value_inst(1, module->ptr_id_bits);
      entry_bb->build_inst(Op::UB, entry_bb->build_inst(Op::EQ, id, one));
    }

  BITMAP_FREE(nonnullargs);
}

bool Converter::need_prov_phi(gimple *phi)
{
  tree phi_result = gimple_phi_result(phi);
  if (POINTER_TYPE_P(TREE_TYPE(phi_result)))
    return true;

  for (unsigned i = 0; i < gimple_phi_num_args(phi); i++)
    {
      tree arg = gimple_phi_arg_def(phi, i);
      if (tree2prov.contains(arg))
	return true;
    }

  return false;
}

void Converter::process_instructions(int nof_blocks, int *postorder)
{
  for (int i = 0; i < nof_blocks; i++)
    {
      basic_block gcc_bb =
	(*fun->cfg->x_basic_block_info)[postorder[nof_blocks - 1 - i]];
      bb = gccbb_top2bb.at(gcc_bb);
      gimple *switch_stmt = nullptr;
      gimple *cond_stmt = nullptr;
      gimple_stmt_iterator gsi;
      for (gsi = gsi_start_phis(gcc_bb); !gsi_end_p(gsi); gsi_next(&gsi))
	{
	  gimple *phi = gsi_stmt(gsi);
	  tree phi_result = gimple_phi_result(phi);
	  if (VOID_TYPE_P(TREE_TYPE(phi_result)))
	    {
	      // Skip phi nodes for the memory SSA virtual SSA names.
	      continue;
	    }
	  int bitwidth = bitsize_for_type(TREE_TYPE(phi_result));
	  Inst *phi_inst = bb->build_phi_inst(bitwidth);
	  Inst *phi_indef = bb->build_phi_inst(bitwidth);
	  constrain_range(bb, phi_result, phi_inst, phi_indef);
	  if (need_prov_phi(phi))
	    {
	      Inst *phi_prov = bb->build_phi_inst(module->ptr_id_bits);
	      tree2prov.insert({phi_result, phi_prov});
	    }
	  tree2instruction.insert({phi_result, phi_inst});
	  tree2indef.insert({phi_result, phi_indef});
	}
      for (gsi = gsi_start_bb(gcc_bb); !gsi_end_p(gsi); gsi_next(&gsi))
	{
	  gimple *stmt = gsi_stmt(gsi);
	  switch (gimple_code(stmt))
	    {
	    case GIMPLE_ASSIGN:
	      process_gimple_assign(stmt);
	      break;
	    case GIMPLE_ASM:
	      process_gimple_asm(stmt);
	      break;
	    case GIMPLE_CALL:
	      process_gimple_call(stmt);
	      break;
	    case GIMPLE_COND:
	      assert(!cond_stmt);
	      assert(!switch_stmt);
	      cond_stmt = stmt;
	      break;
	    case GIMPLE_RETURN:
	      process_gimple_return(stmt);
	      break;
	    case GIMPLE_SWITCH:
	      assert(!cond_stmt);
	      assert(!switch_stmt);
	      switch_stmt = stmt;
	      break;
	    case GIMPLE_LABEL:
	    case GIMPLE_PREDICT:
	    case GIMPLE_NOP:
	      // Nothing to do.
	      break;
	    default:
	      {
		const char *name = gimple_code_name[gimple_code(stmt)];
		throw Not_implemented("process_instructions: "s + name);
	      }
	    }
	}
      gccbb_bottom2bb.insert({gcc_bb, bb});

      // Check that we do not have any extra, unsupported, edges as that will
      // make the code below fail assertions when adding the branches.
      for (unsigned j = 0; j < EDGE_COUNT(gcc_bb->succs); j++)
	{
	  edge succ_edge = EDGE_SUCC(gcc_bb, j);
	  if (succ_edge->flags & EDGE_ABNORMAL)
	    throw Not_implemented("abnormal edge(exceptions)");
	  if (succ_edge->flags & EDGE_IRREDUCIBLE_LOOP)
	    throw Not_implemented("irreducible loop");
	}

      // Add the branch instruction(s) at the end of the basic block.
      if (switch_stmt)
	process_gimple_switch(switch_stmt, bb);
      else if (EDGE_COUNT(gcc_bb->succs) == 0)
	{
	  basic_block gcc_exit_block = EXIT_BLOCK_PTR_FOR_FN(fun);
	  if (gcc_bb != gcc_exit_block)
	    {
	      // This is not the exit block, but there are not any successors
	      // (I.e., this is a block from an __builting_unreachable() etc.)
	      // so we must add a branch to the real exit block as the smtgcc
	      // IR only can have one ret instruction.
	      bb->build_br_inst(gccbb_top2bb.at(gcc_exit_block));
	    }
	  else
	    {
	      generate_exit_inst();
	      generate_return_inst();
	    }
	}
      else if (cond_stmt)
	{
	  tree_code code = gimple_cond_code(cond_stmt);
	  tree arg1_expr = gimple_cond_lhs(cond_stmt);
	  tree arg2_expr = gimple_cond_rhs(cond_stmt);
	  tree arg1_type = TREE_TYPE(arg1_expr);
	  tree arg2_type = TREE_TYPE(arg2_expr);
	  auto [arg1, arg1_indef] = tree2inst_indef(arg1_expr);
	  auto [arg2, arg2_indef] = tree2inst_indef(arg2_expr);
	  Inst *cond;
	  Inst *cond_indef;
	  if (TREE_CODE(arg1_type) == COMPLEX_TYPE)
	    {
	      // TODO: Implement indef aguments.
	      cond = process_binary_complex_cmp(code, arg1, arg2,
						boolean_type_node,
						arg1_type);
	      cond_indef = nullptr;
	    }
	  else
	    {
	      Inst *cond_prov;
	      std::tie(cond, cond_indef, cond_prov) =
		process_binary_scalar(code, arg1, arg1_indef, nullptr,
				      arg2, arg2_indef, nullptr,
				      boolean_type_node,
				      arg1_type, arg2_type);
	    }
	  if (cond_indef)
	    build_ub_if_not_zero(cond_indef);
	  edge true_edge, false_edge;
	  extract_true_false_edges_from_block(gcc_bb, &true_edge, &false_edge);
	  Basic_block *true_bb = gccbb_top2bb.at(true_edge->dest);
	  Basic_block *false_bb = gccbb_top2bb.at(false_edge->dest);
	  bb->build_br_inst(cond, true_bb, false_bb);
	}
      else
	{
	  assert(EDGE_COUNT(gcc_bb->succs) == 1);
	  Basic_block *succ_bb =
	    gccbb_top2bb.at(single_succ_edge(gcc_bb)->dest);
	  bb->build_br_inst(succ_bb);
	}
    }
  bb = nullptr;

  // We have created all instructions, so it is now safe to add the phi
  // arguments (as they must have been created now).
  std::map<std::pair<Inst *, uint64_t>, std::vector<Inst *>> cache;
  for (int i = 0; i < nof_blocks; i++)
    {
      basic_block gcc_bb =
	(*fun->cfg->x_basic_block_info)[postorder[nof_blocks - 1 - i]];
      for (gphi_iterator gsi = gsi_start_phis(gcc_bb);
	   !gsi_end_p(gsi);
	   gsi_next(&gsi))
	{
	  gphi *phi = gsi.phi();
	  tree phi_result = gimple_phi_result(phi);
	  tree phi_type = TREE_TYPE(phi_result);
	  if (VOID_TYPE_P(phi_type))
	    {
	      // Skip phi nodes for the memory SSA virtual SSA names.
	      continue;
	    }
	  Inst *phi_inst = tree2instruction.at(phi_result);
	  Inst *phi_indef = tree2indef.at(phi_result);
	  Inst *phi_prov = nullptr;
	  if (tree2prov.contains(phi_result))
	    phi_prov = tree2prov.at(phi_result);
	  for (unsigned i = 0; i < gimple_phi_num_args(phi); i++)
	    {
	      Basic_block *arg_bb = get_phi_arg_bb(phi, i);
	      bb = arg_bb;
	      tree arg = gimple_phi_arg_def(phi, i);
	      auto [arg_inst, arg_indef, arg_prov] =
		tree2inst_indef_prov(arg);
	      phi_inst->add_phi_arg(arg_inst, arg_bb);
	      if (!arg_indef)
		arg_indef = arg_bb->value_inst(0, arg_inst->bitsize);
	      phi_indef->add_phi_arg(arg_indef, arg_bb);
	      if (phi_prov)
		{
		  assert(!POINTER_TYPE_P(phi_type) || arg_prov);
		  if (!arg_prov)
		    arg_prov = arg_bb->value_inst(0, module->ptr_id_bits);
		  phi_prov->add_phi_arg(arg_prov, arg_bb);
		}
	      bb = nullptr;
	    }

	  if (VECTOR_TYPE_P(phi_type) || TREE_CODE(phi_type) == COMPLEX_TYPE)
	    {
	      uint32_t bs;
	      if (VECTOR_TYPE_P(phi_type))
		bs = bitsize_for_type(TREE_TYPE(phi_type));
	      else
		bs = bytesize_for_type(TREE_TYPE(phi_type)) * 8;
	      tree2instruction[phi_result] = split_phi(phi_inst, bs, cache);
	      tree2indef[phi_result] = split_phi(phi_indef, bs, cache);
	    }
	}
    }
}

Function *Converter::process_function()
{
  if (!fun->cfg)
    throw Not_implemented("missing fun->cfg");

  if (fun->static_chain_decl)
    {
      // TODO: Should be possible to handle this by treating it as a normal
      // pointer argument?
      throw Not_implemented("nested functions");
    }

  const char *name = function_name(fun);
  func = module->build_function(name);

  int *postorder = nullptr;
  try {
    postorder = XNEWVEC(int, last_basic_block_for_fn(fun));
    int nof_blocks = post_order_compute(postorder, true, true);

    // Build the new basic blocks.
    for (int i = nof_blocks - 1; i >= 0; --i)
      {
	basic_block gcc_bb = (*fun->cfg->x_basic_block_info)[postorder[i]];
	gccbb_top2bb.insert({gcc_bb, func->build_bb()});
      }
    bb = func->bbs[0];

    process_func_args();
    process_variables();

    // GCC assumes that __builtin functions have the specified semantics
    // (see GCC PR 112949), so in practice, it is undefined behavior to
    // create a __builtin function. Mark such functions as always invoking
    // undefined behavior.
    // TODO: We encounter similar issues with normal functions, such as
    // memcpy. We should also detect such cases.
    if (!strncmp(function_name(cfun), "__builtin_", 10))
      func->bbs[0]->build_inst(Op::UB, func->bbs[0]->value_inst(1, 1));

    process_instructions(nof_blocks, postorder);

    free(postorder);
    postorder = nullptr;
  }
  catch (...) {
    free(postorder);
    throw;
  }

  reverse_post_order(func);

  validate(func);

  Function *f = func;
  func = nullptr;
  return f;
}

} // end anonymous namespace

Function *process_function(Module *module, CommonState *state, function *fun, bool is_tgt_func)
{
  Converter func(module, state, fun, is_tgt_func);
  return func.process_function();
}

void unroll_and_optimize(Function *func)
{
  simplify_insts(func);
  dead_code_elimination(func);
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
  vrp(func);
  simplify_insts(func);
  dead_code_elimination(func);
  validate(func);
}

void unroll_and_optimize(Module *module)
{
  for (auto func : module->functions)
    unroll_and_optimize(func);
}

CommonState::CommonState(Arch arch)
  : arch{arch}
{
  assert(POINTER_SIZE == 32 || POINTER_SIZE == 64);
  if (POINTER_SIZE == 32)
    {
      ptr_id_max = std::numeric_limits<int8_t>::max();
      ptr_id_min = std::numeric_limits<int8_t>::min();
    }
  else
    {
      if (arch == Arch::riscv)
	{
	  ptr_id_max = std::numeric_limits<int8_t>::max();
	  ptr_id_min = std::numeric_limits<int8_t>::min();
	}
      else
	{
	  ptr_id_max = std::numeric_limits<int16_t>::max();
	  ptr_id_min = std::numeric_limits<int16_t>::min();
	}
    }
}

Module *create_module(Arch arch)
{
  assert(POINTER_SIZE == 32 || POINTER_SIZE == 64);
  uint32_t ptr_bits;
  uint32_t ptr_id_bits;
  uint32_t ptr_offset_bits;
  if (POINTER_SIZE == 32)
    {
      ptr_bits = 32;
      ptr_id_bits = 8;
      ptr_offset_bits = 24;
    }
  else
    {
      if (arch == Arch::riscv)
	{
	  ptr_bits = 64;
	  ptr_id_bits = 40;
	  ptr_offset_bits = 24;
	}
      else
	{
	  ptr_bits = 64;
	  ptr_id_bits = 16;
	  ptr_offset_bits = 48;
	}
    }
  return create_module(ptr_bits, ptr_id_bits, ptr_offset_bits);
}

// The logical bitsize used in the IR for the GCC type/
uint64_t bitsize_for_type(tree type)
{
  check_type(type);

  if (INTEGRAL_TYPE_P(type))
    return TYPE_PRECISION(type);
  if (VECTOR_TYPE_P(type) && bitsize_for_type(TREE_TYPE(type)) == 1)
    return 1 << VECTOR_TYPE_CHECK(type)->type_common.precision;

  tree size_tree = TYPE_SIZE(type);
  assert(size_tree);
  if (size_tree == NULL_TREE)
    throw Not_implemented("bitsize_for_type: incomplete type");
  if (TREE_CODE(size_tree) != INTEGER_CST && !POLY_INT_CST_P(size_tree))
    {
      // Things like function parameters
      //   int foo(int n, struct T { char a[n]; } b);
      throw Not_implemented("bitsize_for_type: dynamically sized type");
    }
  return get_int_cst_val(size_tree);
}

unsigned __int128 get_int_cst_val(tree expr)
{
  assert(TREE_CODE(expr) == INTEGER_CST || POLY_INT_CST_P(expr));
  uint32_t precision = TYPE_PRECISION(TREE_TYPE(expr));
  if (precision > 128)
    throw Not_implemented("get_int_cst_val: INTEGER_CST precision > 128");
  assert(precision > 0 && precision <= 128);
  if (POLY_INT_CST_P(expr))
    expr = POLY_INT_CST_COEFF(expr, 0);
  unsigned __int128 value = 0;
  if (TREE_INT_CST_NUNITS(expr) == 2)
    {
      value = TREE_INT_CST_ELT(expr, 1);
      value <<= 64;
      value |= (uint64_t)TREE_INT_CST_ELT(expr, 0);
    }
  else
    value = (int64_t)TREE_INT_CST_ELT(expr, 0);
  return value;
}
