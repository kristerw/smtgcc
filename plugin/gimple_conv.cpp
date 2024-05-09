#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gcc-plugin.h"
#include "tree-pass.h"
#include "tree.h"
#include "tree-cfg.h"
#include "gimple.h"
#include "gimple-iterator.h"
#include "gimple-pretty-print.h"
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
#define ANON_MEM_SIZE 128

using namespace std::string_literals;
using namespace smtgcc;

namespace {

// Setting this to true makes us check that tgt is a refinement of src
// for both possible values of CFN_LOOP_VECTORIZED. This slows down the
// checking and reports issues for cases that are arguably correct (such as
// in gcc.c-torture/execute/pr94734.c where the true case has been changed
// from
//   if (x == (i & 0x25))
//     arr[y] = i;
// to
//   _25 = &arr[y_12(D)];
//   .MASK_STORE (_25, 32B, _2, i_17);
// which may result in invalid indexing for cases where `x == (i & 0x25)` is
// false, but would not be a problem after vectorization.
bool check_loop_vectorized = false;

struct Addr {
  Instruction *ptr;
  uint64_t bitoffset;
  Instruction *provenance;
};

struct Converter {
  Converter(Module *module, CommonState *state, function *fun, bool is_tgt_func = false)
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
  Instruction *loop_vect_sym = nullptr;
  std::string pass_name;
  std::map<Basic_block *, std::set<Basic_block *>> switch_bbs;
  std::map<basic_block, Basic_block *> gccbb2bb;
  std::map<Basic_block *, std::pair<Instruction *, Instruction *> > bb2retval;
  std::map<tree, Instruction *> tree2instruction;
  std::map<tree, Instruction *> tree2undef;
  std::map<tree, Instruction *> tree2provenance;
  std::map<tree, Instruction *> decl2instruction;
  std::map<Instruction *, Instruction *> inst2memory_flagsx;
  Instruction *retval = nullptr;
  int retval_bitsize;
  tree retval_type;

  bool is_tgt_func;

  Instruction *build_memory_inst(uint64_t id, uint64_t size, uint32_t flags);
  void constrain_range(Basic_block *bb, tree expr, Instruction *inst, Instruction *undef=nullptr);
  void mem_access_ub_check(Basic_block *bb, Instruction *ptr, Instruction *provenance, uint64_t size);
  std::tuple<Instruction *, Instruction *, Instruction *> tree2inst_undef_prov(Basic_block *bb, tree expr);
  std::pair<Instruction *, Instruction *>tree2inst_undef(Basic_block *bb, tree expr);
  std::pair<Instruction *, Instruction *>tree2inst_prov(Basic_block *bb, tree expr);
  Instruction *tree2inst(Basic_block *bb, tree expr);
  std::tuple<Instruction *, Instruction *, Instruction *> tree2inst_init_var(Basic_block *bb, tree expr);
  std::pair<Instruction *, Instruction *> get_res_undef(Instruction *arg1_undef, tree lhs_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> get_res_undef(Instruction *arg1_undef, Instruction *arg2_undef, tree lhs_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> get_res_undef(Instruction *arg1_undef, Instruction *arg2_undef, Instruction *arg3_undef, tree lhs_type, Basic_block *bb);
  Addr process_array_ref(Basic_block *bb, tree expr, bool is_mem_access);
  Addr process_component_ref(Basic_block *bb, tree expr, bool is_mem_access);
  Addr process_bit_field_ref(Basic_block *bb, tree expr, bool is_mem_access);
  Addr process_address(Basic_block *bb, tree expr, bool is_mem_access);
  std::tuple<Instruction *, Instruction *, Instruction *> vector_as_array(Basic_block *bb, tree expr);
  std::tuple<Instruction *, Instruction *, Instruction *> process_load(Basic_block *bb, tree expr);
  void process_store(tree addr_expr, tree value_expr, Basic_block *bb);
  std::tuple<Instruction *, Instruction *, Instruction *> load_value(Basic_block *bb, Instruction *ptr, uint64_t size);
  void store_value(Basic_block *bb, Instruction *ptr, Instruction *value, Instruction *undef = nullptr);
  std::tuple<Instruction *, Instruction *, Instruction *> type_convert(Instruction *inst, Instruction *undef, Instruction *provenance, tree src_type, tree dest_type, Basic_block *bb);
  Instruction *type_convert(Instruction *inst, tree src_type, tree dest_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> process_unary_bool(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, tree lhs_type, tree arg1_type, Basic_block *bb);
  std::tuple<Instruction *, Instruction *, Instruction *> process_unary_int(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg1_prov, tree lhs_type, tree arg1_type, Basic_block *bb);
  Instruction *process_unary_scalar(enum tree_code code, Instruction *arg1, tree lhs_type, tree arg1_type, Basic_block *bb);
  std::tuple<Instruction *, Instruction *, Instruction *> process_unary_scalar(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg1_prov, tree lhs_type, tree arg1_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> process_unary_vec(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, tree lhs_elem_type, tree arg1_elem_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> process_unary_float(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, tree lhs_type, tree arg1_type, Basic_block *bb);
  Instruction *process_unary_complex(enum tree_code code, Instruction *arg1, tree lhs_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> process_binary_float(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg2, Instruction *arg2_undef, tree lhs_type, Basic_block *bb);
  Instruction *process_binary_complex(enum tree_code code, Instruction *arg1, Instruction *arg2, tree lhs_type, Basic_block *bb);
  Instruction *process_binary_complex_cmp(enum tree_code code, Instruction *arg1, Instruction *arg2, tree lhs_type, tree arg1_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> process_binary_bool(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg2, Instruction *arg2_undef, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb);
  Instruction *process_binary_int(enum tree_code code, bool is_unsigned, Instruction *arg1, Instruction *arg2, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow = false);
  std::tuple<Instruction *, Instruction *, Instruction *> process_binary_int(enum tree_code code, bool is_unsigned, Instruction *arg1, Instruction *arg1_undef, Instruction *arg1_prov, Instruction *arg2, Instruction *arg2_undef, Instruction *arg2_prov, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow = false);
  Instruction *process_binary_scalar(enum tree_code code, Instruction *arg1, Instruction *arg2, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow = false);
  std::tuple<Instruction *, Instruction *, Instruction *> process_binary_scalar(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg1_prov, Instruction *arg2, Instruction *arg2_undef, Instruction *arg2_prov, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow = false);
  std::pair<Instruction *, Instruction *> process_binary_vec(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg2, Instruction *arg2_undef, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow = false);
  Instruction *process_ternary(enum tree_code code, Instruction *arg1, Instruction *arg2, Instruction *arg3, tree arg1_type, tree arg2_type, tree arg3_type, Basic_block *bb);
  Instruction *process_ternary_vec(enum tree_code code, Instruction *arg1, Instruction *arg2, Instruction *arg3, tree lhs_type, tree arg1_type, tree arg2_type, tree arg3_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> process_vec_cond(Instruction *arg1, Instruction *arg2, Instruction *arg2_undef, Instruction *arg3, Instruction *arg3_undef, tree arg1_type, tree arg2_type, Basic_block *bb);
  std::pair<Instruction *, Instruction *> process_vec_perm_expr(gimple *stmt, Basic_block *bb);
  std::tuple<Instruction *, Instruction *, Instruction *> vector_constructor(Basic_block *bb, tree expr);
  void process_constructor(tree lhs, tree rhs, Basic_block *bb);
  void process_gimple_assign(gimple *stmt, Basic_block *bb);
  void process_gimple_asm(gimple *stmt);
  void process_cfn_add_overflow(gimple *stmt, Basic_block *bb);
  void process_cfn_assume_aligned(gimple *stmt, Basic_block *bb);
  void process_cfn_bswap(gimple *stmt, Basic_block *bb);
  void process_cfn_clrsb(gimple *stmt, Basic_block *bb);
  void process_cfn_clz(gimple *stmt, Basic_block *bb);
  void process_cfn_cond(gimple *stmt, Basic_block *bb);
  void process_cfn_copysign(gimple *stmt, Basic_block *bb);
  void process_cfn_ctz(gimple *stmt, Basic_block *bb);
  void process_cfn_divmod(gimple *stmt, Basic_block *bb);
  void process_cfn_expect(gimple *stmt, Basic_block *bb);
  void process_cfn_ffs(gimple *stmt, Basic_block *bb);
  void process_cfn_fmax(gimple *stmt, Basic_block *bb);
  void process_cfn_fmin(gimple *stmt, Basic_block *bb);
  void process_cfn_loop_vectorized(gimple *stmt);
  void process_cfn_mask_load(gimple *stmt, Basic_block *bb);
  void process_cfn_mask_store(gimple *stmt, Basic_block *bb);
  void process_cfn_memcpy(gimple *stmt, Basic_block *bb);
  void process_cfn_memset(gimple *stmt, Basic_block *bb);
  void process_cfn_mul_overflow(gimple *stmt, Basic_block *bb);
  void process_cfn_nan(gimple *stmt, Basic_block *bb);
  void process_cfn_parity(gimple *stmt, Basic_block *bb);
  void process_cfn_popcount(gimple *stmt, Basic_block *bb);
  void process_cfn_signbit(gimple *stmt, Basic_block *bb);
  void process_cfn_sub_overflow(gimple *stmt, Basic_block *bb);
  void process_cfn_reduc(gimple *stmt, Basic_block *bb);
  void process_cfn_trap(gimple *stmt, Basic_block *bb);
  void process_cfn_uaddc(gimple *stmt, Basic_block *bb);
  void process_cfn_unreachable(gimple *stmt, Basic_block *bb);
  void process_cfn_usubc(gimple *stmt, Basic_block *bb);
  void process_cfn_vcond(gimple *stmt, Basic_block *bb);
  void process_cfn_vcond_mask(gimple *stmt, Basic_block *bb);
  void process_cfn_vec_convert(gimple *stmt, Basic_block *bb);
  void process_cfn_xorsign(gimple *stmt, Basic_block *bb);
  void process_gimple_call_combined_fn(gimple *stmt, Basic_block *bb);
  void process_gimple_call(gimple *stmt, Basic_block *bb);
  void process_gimple_return(gimple *stmt, Basic_block *bb);
  Instruction *build_label_cond(tree index_expr, tree label,
				Basic_block *bb);
  void process_gimple_switch(gimple *stmt, Basic_block *bb);
  Basic_block *get_phi_arg_bb(gphi *phi, int i);
  void generate_return_inst(Basic_block *bb);
  void init_var_values(tree initial, Instruction *mem_inst);
  void init_var(tree decl, Instruction *mem_inst);
  void make_uninit(Basic_block *bb, Instruction *ptr, uint64_t size);
  void constrain_src_value(Basic_block *bb, Instruction *inst, tree type, Instruction *mem_flags = nullptr);
  void process_variables();
  void process_func_args();
  bool need_prov_phi(gimple *phi);
  void process_instructions(int nof_blocks, int *postorder);
  Function *process_function();
};

Instruction *get_lv_inst(Function *func)
{
  for (Instruction *inst = func->bbs[0]->first_inst; inst; inst = inst->next)
    {
      if (inst->opcode == Op::SYMBOLIC
	  && inst->arguments[0]->value() == LOOP_VECT_SYM_IDX)
	return inst;
    }
  return nullptr;
}

// Build the minimal signed integer value for the bitsize.
// The bitsize may be larger than 128.
Instruction *build_min_int(Basic_block *bb, uint64_t bitsize)
{
  Instruction *top_bit = bb->value_inst(1, 1);
  if (bitsize == 1)
    return top_bit;
  Instruction *zero = bb->value_inst(0, bitsize - 1);
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

unsigned __int128 get_int_cst_val(tree expr)
{
  assert(TREE_CODE(expr) == INTEGER_CST);
  uint32_t precision = TYPE_PRECISION(TREE_TYPE(expr));
  if (precision > 128)
    throw Not_implemented("get_int_cst_val: INTEGER_CST precision > 128");
  assert(precision > 0 && precision <= 128);
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

void check_type(tree type)
{
  // Note: We do not check that all elements in structures/arrays have
  // valid type -- they will be checked when the fields are accessed.
  // This makes us able to analyze progams having invalid elements in
  // unused structures/arrays.
  if (DECIMAL_FLOAT_TYPE_P(type))
    throw Not_implemented("check_type: DECIMAL_FLOAT_TYPE");
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

// The logical bitsize used in the IR for the GCC type/
uint64_t bitsize_for_type(tree type)
{
  check_type(type);

  if (INTEGRAL_TYPE_P(type))
    return TYPE_PRECISION(type);
  if (VECTOR_TYPE_P(type) && bitsize_for_type(TREE_TYPE(type)) == 1)
    return 1 << VECTOR_TYPE_CHECK(type)->type_common.precision;

  tree size_tree = TYPE_SIZE(type);
  if (size_tree == NULL_TREE)
    throw Not_implemented("bitsize_for_type: incomplete type");
  if (TREE_CODE(size_tree) != INTEGER_CST)
    {
      // Things like function parameters
      //   int foo(int n, struct T { char a[n]; } b);
      throw Not_implemented("bitsize_for_type: dynamically sized type");
    }
  return TREE_INT_CST_LOW(size_tree);
}

// The size of the GCC type when stored in memory etc.
uint64_t bytesize_for_type(tree type)
{
  tree size_tree = TYPE_SIZE(type);
  if (size_tree == NULL_TREE)
    throw Not_implemented("bytesize_for_type: incomplete type");
  if (TREE_CODE(size_tree) != INTEGER_CST)
    {
      // Things like function parameters
      //   int foo(int n, struct T { char a[n]; } b);
      throw Not_implemented("bytesize_for_type: complicated type");
    }
  uint64_t bitsize = TREE_INT_CST_LOW(size_tree);
  assert((bitsize & 7) == 0);
  return bitsize / 8;
}

Instruction *extract_vec_elem(Basic_block *bb, Instruction *inst, uint32_t elem_bitsize, uint32_t idx)
{
  assert(inst->bitsize % elem_bitsize == 0);
  Instruction *high = bb->value_inst(idx * elem_bitsize + elem_bitsize - 1, 32);
  Instruction *low = bb->value_inst(idx * elem_bitsize, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
}

std::tuple<Instruction *, Instruction *, Instruction *> extract_vec_elem(Basic_block *bb, Instruction *inst, Instruction *undef, Instruction *prov, uint32_t elem_bitsize, uint32_t idx)
{
  inst = extract_vec_elem(bb, inst, elem_bitsize, idx);
  if (undef)
    undef = extract_vec_elem(bb, undef, elem_bitsize, idx);
  if (prov)
    prov = extract_vec_elem(bb, prov, elem_bitsize, idx);
  return {inst, undef, prov};
}

Instruction *extract_elem(Basic_block *bb, Instruction *vec, uint32_t elem_bitsize, Instruction *idx)
{
  // The shift calculation below may overflow if idx is not wide enough,
  // so we extend it to a safe width.
  // Note: We could have extended it to the full vector bit size, but that
  // would limit optimizations such as constant folding for the shift
  // calculation for vectors wider than 128 bits.
  if (idx->bitsize < 32)
    idx = bb->build_inst(Op::ZEXT, idx, bb->value_inst(32, 32));

  Instruction *elm_bsize = bb->value_inst(elem_bitsize, idx->bitsize);
  Instruction *shift = bb->build_inst(Op::MUL, idx, elm_bsize);
  if (shift->bitsize > vec->bitsize)
    {
      Instruction *high = bb->value_inst(vec->bitsize - 1, 32);
      Instruction *low = bb->value_inst(0, 32);
      shift = bb->build_inst(Op::EXTRACT, shift, high, low);
    }
  else if (shift->bitsize < vec->bitsize)
    {
      Instruction *bitsize_inst = bb->value_inst(vec->bitsize, 32);
      shift = bb->build_inst(Op::ZEXT, shift, bitsize_inst);
    }
  Instruction *inst = bb->build_inst(Op::LSHR, vec, shift);
  Instruction *high = bb->value_inst(elem_bitsize - 1, 32);
  Instruction *low = bb->value_inst(0, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
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
void Converter::constrain_src_value(Basic_block *bb, Instruction *inst, tree type, Instruction *mem_flags)
{
  if (is_tgt_func)
    return;

  // TODO: We should invert the meaning of mem_flags.
  // TODO: mem_flags is not correct name -- it is only one flag.
  if (POINTER_TYPE_P(type))
    {
      Instruction *id = bb->build_extract_id(inst);
      Instruction *zero = bb->value_inst(0, id->bitsize);
      Instruction *cond = bb->build_inst(Op::SLT, id, zero);
      if (mem_flags)
	{
	  Instruction *not_written = bb->build_extract_id(mem_flags);
	  not_written = bb->build_inst(Op::EQ, not_written, zero);
	  cond = bb->build_inst(Op::AND, cond, not_written);
	}
      bb->build_inst(Op::UB, cond);
      return;
    }
  if (SCALAR_FLOAT_TYPE_P(type))
    {
      bb->build_inst(Op::UB, bb->build_inst(Op::IS_NONCANONICAL_NAN, inst));
      return;
    }
  if (INTEGRAL_TYPE_P(type) && inst->bitsize != bitsize_for_type(type))
    {
      Instruction *tmp = bb->build_trunc(inst, bitsize_for_type(type));
      Op op = TYPE_UNSIGNED(type) ? Op::ZEXT : Op::SEXT;
      tmp = bb->build_inst(op, tmp, bb->value_inst(inst->bitsize, 32));
      bb->build_inst(Op::UB, bb->build_inst(Op::NE, inst, tmp));
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
	  Instruction *high =
	    bb->value_inst((elem_offset + elem_size) * 8 - 1, 32);
	  Instruction *low = bb->value_inst(elem_offset * 8, 32);
	  Instruction *extract1 = bb->build_inst(Op::EXTRACT, inst, high, low);
	  Instruction *extract2 = nullptr;
	  if (mem_flags)
	    extract2 = bb->build_inst(Op::EXTRACT, mem_flags, high, low);
	  constrain_src_value(bb, extract1, elem_type, extract2);
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
	  Instruction *extract = extract_vec_elem(bb, inst, elem_bitsize, i);
	  Instruction *extract2 = nullptr;
	  if (mem_flags)
	    extract2 = extract_vec_elem(bb, mem_flags, elem_bitsize, i);
	  constrain_src_value(bb, extract, elem_type, extract2);
	}
      return;
    }
}

Instruction *Converter::build_memory_inst(uint64_t id, uint64_t size, uint32_t flags)
{
  Basic_block *bb = func->bbs[0];
  uint32_t ptr_id_bits = func->module->ptr_id_bits;
  uint32_t ptr_offset_bits = func->module->ptr_offset_bits;
  Instruction *arg1 = bb->value_inst(id, ptr_id_bits);
  Instruction *arg2 = bb->value_inst(size, ptr_offset_bits);
  Instruction *arg3 = bb->value_inst(flags, 32);
  return bb->build_inst(Op::MEMORY, arg1, arg2, arg3);
}

void build_ub_if_not_zero(Basic_block *bb, Instruction *inst)
{
  Instruction *zero = bb->value_inst(0, inst->bitsize);
  Instruction *cmp = bb->build_inst(Op::NE, inst, zero);
  bb->build_inst(Op::UB, cmp);
}

int popcount(unsigned __int128 x)
{
  int result = 0;
  for (int i = 0; i < 4; i++)
    {
      uint32_t t = x >> (i* 32);
      result += __builtin_popcount(t);
    }
  return result;
}

int clz(unsigned __int128 x)
{
  int result = 0;
  for (int i = 0; i < 4; i++)
    {
      uint32_t t = x >> ((3 - i) * 32);
      if (t)
	return result + __builtin_clz(t);
      result += 32;
    }
  return result;
}

void Converter::constrain_range(Basic_block *bb, tree expr, Instruction *inst, Instruction *undef)
{
  assert(TREE_CODE(expr) == SSA_NAME);

  // The constraints are added when we create a inst for the expr, so the work
  // is already done if tree2instruction contains this expr.
  if (tree2instruction.contains(expr))
    return;

  tree type = TREE_TYPE(expr);
  if (!INTEGRAL_TYPE_P(type) && !POINTER_TYPE_P(type))
    return;

  int_range_max r;
  get_range_query(cfun)->range_of_expr(r, expr);
  if (r.undefined_p() || r.varying_p())
    return;

  // TODO: Implement wide types.
  if (inst->bitsize > 128)
    return;

  // TODO: get_nonzero_bits is deprecated if I understand correctly. This
  // should be updated to the new API.
  Instruction *is_ub1 = nullptr;
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
	  Instruction *mask = bb->value_inst(~nonzero_bits, inst->bitsize);
	  Instruction *bits = bb->build_inst(Op::AND, inst, mask);
	  Instruction *zero = bb->value_inst(0, bits->bitsize);
	  is_ub1 = bb->build_inst(Op::NE, bits, zero);
	}
    }

  Instruction *is_ub2 = nullptr;
  for (unsigned i = 0; i < r.num_pairs(); i++)
    {
      unsigned __int128 low_val = get_wide_int_val(r.lower_bound(i));
      unsigned __int128 high_val = get_wide_int_val(r.upper_bound(i));
      Instruction *is_not_in_range;
      if (low_val == high_val)
	{
	  Instruction *val = bb->value_inst(low_val, inst->bitsize);
	  is_not_in_range = bb->build_inst(Op::NE, inst, val);
	}
      else
	{
	  Instruction *low = bb->value_inst(low_val, inst->bitsize);
	  Instruction *high = bb->value_inst(high_val, inst->bitsize);
	  Op op = TYPE_UNSIGNED(type) ? Op::UGT : Op::SGT;
	  Instruction *cmp_low = bb->build_inst(op, low, inst);
	  Instruction *cmp_high = bb->build_inst(op, inst, high);
	  is_not_in_range = bb->build_inst(Op::OR, cmp_low, cmp_high);
	}
      if (is_ub2)
	is_ub2 = bb->build_inst(Op::AND, is_not_in_range, is_ub2);
      else
	is_ub2 = is_not_in_range;
    }
  assert(is_ub2 != nullptr);

  // Ranges do not take undefined values into account, so, e.g., a phi node
  // may get a range, even if one of the arguments is undefined. We therefore
  // need to filter out the undef cases from the check, otherwise we will
  // report miscompilation for
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
  // This is safe, because all use of the value that can use the range info
  // will be marked as UB if an undef value is used in that operation.
  if (undef)
    {
      Instruction *zero = bb->value_inst(0, undef->bitsize);
      Instruction *cmp = bb->build_inst(Op::EQ, undef, zero);
      if (is_ub1)
	is_ub1 = bb->build_inst(Op::AND, is_ub1, cmp);
      is_ub2 = bb->build_inst(Op::AND, is_ub2, cmp);
    }

  if (is_ub1)
    bb->build_inst(Op::UB, is_ub1);
  bb->build_inst(Op::UB, is_ub2);
}

void Converter::mem_access_ub_check(Basic_block *bb, Instruction *ptr, Instruction *provenance, uint64_t size)
{
  assert(size < (uint64_t(1) << func->module->ptr_offset_bits));

  // It is UB if the pointer provenance does not correspond to the address.
  Instruction *ptr_mem_id = bb->build_extract_id(ptr);
  Instruction *is_ub = bb->build_inst(Op::NE, provenance, ptr_mem_id);
  bb->build_inst(Op::UB, is_ub);

  // It is UB if the size overflows the offset field.
  Instruction *size_inst = bb->value_inst(size - 1, ptr->bitsize);
  Instruction *end = bb->build_inst(Op::ADD, ptr, size_inst);
  Instruction *end_mem_id = bb->build_extract_id(end);
  Instruction *overflow = bb->build_inst(Op::NE, provenance, end_mem_id);
  bb->build_inst(Op::UB, overflow);

  // It is UB if the end is outside the memory object.
  // Note: ptr is within the memory object; otherwise, the provenance check
  // or the offset overflow check would have failed.
  Instruction *mem_size = bb->build_inst(Op::GET_MEM_SIZE, provenance);
  Instruction *offset = bb->build_extract_offset(end);
  Instruction *out_of_bound = bb->build_inst(Op::UGE, offset, mem_size);
  bb->build_inst(Op::UB, out_of_bound);
}

void store_ub_check(Basic_block *bb, Instruction *ptr, Instruction *provenance, uint64_t size, Instruction *cond = nullptr)
{
  // It is UB to write to constant memory.
  Instruction *is_const = bb->build_inst(Op::IS_CONST_MEM, provenance);
  if (cond)
    is_const = bb->build_inst(Op::AND, is_const, cond);
  bb->build_inst(Op::UB, is_const);

  // It is UB if the pointer provenance does not correspond to the address.
  Instruction *ptr_mem_id = bb->build_extract_id(ptr);
  Instruction *is_ub = bb->build_inst(Op::NE, provenance, ptr_mem_id);
  if (cond)
    is_ub = bb->build_inst(Op::AND, is_ub, cond);
  bb->build_inst(Op::UB, is_ub);

  if (size != 0)
    {
      // It is UB if the size overflows the offset field.
      assert(size != 0);
      Instruction *size_inst = bb->value_inst(size - 1, ptr->bitsize);
      Instruction *end = bb->build_inst(Op::ADD, ptr, size_inst);
      Instruction *end_mem_id = bb->build_extract_id(end);
      Instruction *overflow = bb->build_inst(Op::NE, provenance, end_mem_id);
      if (cond)
	overflow = bb->build_inst(Op::AND, overflow, cond);
      bb->build_inst(Op::UB, overflow);

      // It is UB if the end is outside the memory object.
      // Note: ptr is within the memory object; otherwise, the provenance check
      // or the offset overflow check would have failed.
      Instruction *mem_size = bb->build_inst(Op::GET_MEM_SIZE, provenance);
      Instruction *offset = bb->build_extract_offset(end);
      Instruction *out_of_bound = bb->build_inst(Op::UGE, offset, mem_size);
      if (cond)
	out_of_bound = bb->build_inst(Op::AND, out_of_bound, cond);
      bb->build_inst(Op::UB, out_of_bound);
    }
  else
    {
      // The pointer must point to valid memory, or be one position past
      // valid memory.
      // TODO: Handle zero-sized memory blocks (such as malloc(0)).
      Instruction *mem_size = bb->build_inst(Op::GET_MEM_SIZE, provenance);
      Instruction *offset = bb->build_extract_offset(ptr);
      Instruction *out_of_bound = bb->build_inst(Op::UGT, offset, mem_size);
      if (cond)
	out_of_bound = bb->build_inst(Op::AND, out_of_bound, cond);
      bb->build_inst(Op::UB, out_of_bound);
    }
}

void load_ub_check(Basic_block *bb, Instruction *ptr, Instruction *provenance, uint64_t size, Instruction *cond = nullptr)
{
  // It is UB if the pointer provenance does not correspond to the address.
  Instruction *ptr_mem_id = bb->build_extract_id(ptr);
  Instruction *is_ub = bb->build_inst(Op::NE, provenance, ptr_mem_id);
  if (cond)
    is_ub = bb->build_inst(Op::AND, is_ub, cond);
  bb->build_inst(Op::UB, is_ub);

  if (size != 0)
    {
      // It is UB if the size overflows the offset field.
      Instruction *size_inst = bb->value_inst(size - 1, ptr->bitsize);
      Instruction *end = bb->build_inst(Op::ADD, ptr, size_inst);
      Instruction *end_mem_id = bb->build_extract_id(end);
      Instruction *overflow = bb->build_inst(Op::NE, provenance, end_mem_id);
      if (cond)
	overflow = bb->build_inst(Op::AND, overflow, cond);
      bb->build_inst(Op::UB, overflow);

      // It is UB if the end is outside the memory object.
      // Note: ptr is within the memory object; otherwise, the provenance check
      // or the offset overflow check would have failed.
      Instruction *mem_size = bb->build_inst(Op::GET_MEM_SIZE, provenance);
      Instruction *offset = bb->build_extract_offset(end);
      Instruction *out_of_bound = bb->build_inst(Op::UGE, offset, mem_size);
      if (cond)
	out_of_bound = bb->build_inst(Op::AND, out_of_bound, cond);
      bb->build_inst(Op::UB, out_of_bound);
    }
  else
    {
      // The pointer must point to valid memory, or be one position past
      // valid memory.
      // TODO: Handle zero-sized memory blocks (such as malloc(0)).
      Instruction *mem_size = bb->build_inst(Op::GET_MEM_SIZE, provenance);
      Instruction *offset = bb->build_extract_offset(ptr);
      Instruction *out_of_bound = bb->build_inst(Op::UGT, offset, mem_size);
      if (cond)
	out_of_bound = bb->build_inst(Op::AND, out_of_bound, cond);
      bb->build_inst(Op::UB, out_of_bound);
    }
}

std::pair<Instruction *, Instruction *> to_mem_repr(Basic_block *bb, Instruction *inst, Instruction *undef, tree type)
{
  uint64_t bitsize = bytesize_for_type(type) * 8;
  if (inst->bitsize == bitsize)
    return {inst, undef};

  assert(inst->bitsize < bitsize);
  if (INTEGRAL_TYPE_P(type))
    {
      Instruction *bitsize_inst = bb->value_inst(bitsize, 32);
      Op op = TYPE_UNSIGNED(type) ? Op::ZEXT : Op::SEXT;
      inst = bb->build_inst(op, inst, bitsize_inst);
      if (undef)
	undef = bb->build_inst(Op::SEXT, undef, bitsize_inst);
    }
  return {inst, undef};
}

Instruction *to_mem_repr(Basic_block *bb, Instruction *inst, tree type)
{
  auto [new_inst, undef] = to_mem_repr(bb, inst, nullptr, type);
  return new_inst;
}

// TODO: Imput does not necessaily be mem_repr -- it can be BITFIELD_REF reads
// from vector elements. So should probably be "to_ir_repr" or similar.
std::pair<Instruction *, Instruction *> from_mem_repr(Basic_block *bb, Instruction *inst, Instruction *undef, tree type)
{
  uint64_t bitsize = bitsize_for_type(type);
  assert(bitsize <= inst->bitsize);
  if (inst->bitsize == bitsize)
    return {inst, undef};

  inst = bb->build_trunc(inst, bitsize);
  if (undef)
    undef = bb->build_trunc(undef, bitsize);

  return {inst, undef};
}

Instruction *from_mem_repr(Basic_block *bb, Instruction *inst, tree type)
{
  auto [new_inst, undef] = from_mem_repr(bb, inst, nullptr, type);
  return new_inst;
}

// Helper function to padding_at_offset.
// TODO: Implement a sane version. And test.
uint8_t bitfield_padding_at_offset(tree fld, int64_t offset)
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
// undefined) for an offset into the type.
uint8_t padding_at_offset(tree type, uint64_t offset)
{
  if (TREE_CODE(type) == ARRAY_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      uint64_t elem_size = bytesize_for_type(elem_type);
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
	  tree elem_type = TREE_TYPE(fld);
	  padding &= padding_at_offset(elem_type, offset);
	}
      return padding;
    }

  // The other bytes does not have padding (well, Booleans sort of have
  // padding, but the padding must be 0 so it is not undefined).
  return 0;
}

std::tuple<Instruction *, Instruction *, Instruction *> Converter::tree2inst_undef_prov(Basic_block *bb, tree expr)
{
  check_type(TREE_TYPE(expr));

  auto it = tree2instruction.find(expr);
  if (it != tree2instruction.end())
    {
      Instruction *inst = it->second;
      Instruction *undef = nullptr;
      auto it2 = tree2undef.find(expr);
      if (it2 != tree2undef.end())
	{
	  undef = it2->second;
	  assert(undef);
	}
      Instruction *provenance = nullptr;
      auto it3 = tree2provenance.find(expr);
      if (it3 != tree2provenance.end())
	{
	  provenance = it3->second;
	  assert(provenance);
	}
      return {inst, undef, provenance};
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
		Instruction *inst = tree2instruction.at(var);

		// Place the range check in the entry block as it is
		// invalid to call the function with invalid values.
		// This solves the problem that we "randomly" could
		// mark execution as UB depending on where the param
		// were used when passes were sinking/hoisting the params.
		// See gcc.dg/analyzer/pointer-merging.c for a test where
		// this check makes a difference.
		constrain_range(func->bbs[0], expr, inst);

		assert(!POINTER_TYPE_P(TREE_TYPE(expr))
		       || tree2provenance.contains(var));
		Instruction *provenance = nullptr;
		if (tree2provenance.contains(var))
		  provenance = tree2provenance.at(var);

		return {inst, nullptr, provenance};
	      }
	  }
	if (var && TREE_CODE(var) == VAR_DECL)
	  {
	    uint64_t bitsize = bitsize_for_type(TREE_TYPE(expr));
	    Instruction *inst = bb->value_inst(0, bitsize);
	    Instruction *undef = bb->value_m1_inst(bitsize);
	    Instruction *provenance = nullptr;
	    if (POINTER_TYPE_P(TREE_TYPE(expr)))
	      provenance = bb->build_extract_id(inst);
	    return {inst, undef, provenance};
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
      return vector_constructor(bb, expr);
    case INTEGER_CST:
      {
	uint32_t precision = bitsize_for_type(TREE_TYPE(expr));
	unsigned __int128 value = get_int_cst_val(expr);
	Instruction *inst = bb->value_inst(value, precision);
	Instruction *provenance = nullptr;
	if (POINTER_TYPE_P(TREE_TYPE(expr)))
	  {
	    uint32_t ptr_id_bits = bb->func->module->ptr_id_bits;
	    uint32_t ptr_id_low = bb->func->module->ptr_id_low;
	    uint64_t id = (value >> ptr_id_low) & ((1 << ptr_id_bits) - 1);
	    provenance = bb->value_inst(id, ptr_id_bits);
	  }
	return {inst, nullptr, provenance};
      }
    case REAL_CST:
      {
	tree type = TREE_TYPE(expr);
	check_type(type);
	int nof_bytes = GET_MODE_SIZE(SCALAR_FLOAT_TYPE_MODE(type));
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
	return {bb->value_inst(u.i, TYPE_PRECISION(type)), nullptr, nullptr};
      }
    case VECTOR_CST:
      {
	unsigned HOST_WIDE_INT nunits;
	if (!VECTOR_CST_NELTS(expr).is_constant(&nunits))
	  throw Not_implemented("tree2inst: !VECTOR_CST_NELTS");
	Instruction *ret = tree2inst(bb, VECTOR_CST_ELT(expr, 0));
	for (unsigned i = 1; i < nunits; i++)
	  {
	    Instruction *elem = tree2inst(bb, VECTOR_CST_ELT(expr, i));
	    ret = bb->build_inst(Op::CONCAT, elem, ret);
	  }
	return {ret, nullptr, nullptr};
      }
    case COMPLEX_CST:
      {
	tree elem_type = TREE_TYPE(TREE_TYPE(expr));
	Instruction *real = tree2inst(bb, TREE_REALPART(expr));
	real = to_mem_repr(bb, real, elem_type);
	Instruction *imag = tree2inst(bb, TREE_IMAGPART(expr));
	imag = to_mem_repr(bb, imag, elem_type);
	Instruction *res = bb->build_inst(Op::CONCAT, imag, real);
	return {res, nullptr, nullptr};
      }
    case IMAGPART_EXPR:
      {
	tree elem_type = TREE_TYPE(expr);
	auto [arg, undef] = tree2inst_undef(bb, TREE_OPERAND(expr, 0));
	Instruction *high = bb->value_inst(arg->bitsize - 1, 32);
	Instruction *low = bb->value_inst(arg->bitsize / 2, 32);
	Instruction *res = bb->build_inst(Op::EXTRACT, arg, high, low);
	if (undef)
	  undef = bb->build_inst(Op::EXTRACT, undef, high, low);
	std::tie(res, undef) = from_mem_repr(bb, res, undef, elem_type);
	return {res, undef, nullptr};
      }
    case REALPART_EXPR:
      {
	tree elem_type = TREE_TYPE(expr);
	auto [arg, undef] = tree2inst_undef(bb, TREE_OPERAND(expr, 0));
	Instruction *res = bb->build_trunc(arg, arg->bitsize / 2);
	if (undef)
	  undef = bb->build_trunc(undef, arg->bitsize / 2);
	std::tie(res, undef) = from_mem_repr(bb, res, undef, elem_type);
	return {res, undef, nullptr};
      }
    case VIEW_CONVERT_EXPR:
      {
	auto [arg, undef, provenance] =
	  tree2inst_undef_prov(bb, TREE_OPERAND(expr, 0));
	tree src_type = TREE_TYPE(TREE_OPERAND(expr, 0));
	tree dest_type = TREE_TYPE(expr);
	std::tie(arg, undef) = to_mem_repr(bb, arg, undef, src_type);
	std::tie(arg, undef) = from_mem_repr(bb, arg, undef, dest_type);
	constrain_src_value(bb, arg, dest_type);
	if (POINTER_TYPE_P(dest_type))
	  {
	    assert(!POINTER_TYPE_P(src_type) || provenance);
	    if (!provenance)
	      provenance = bb->build_extract_id(arg);
	  }
	return {arg, undef, provenance};
      }
    case ADDR_EXPR:
      {
	Addr addr = process_address(bb, TREE_OPERAND(expr, 0), false);
	assert(addr.bitoffset == 0);
	return {addr.ptr, nullptr, addr.provenance};
      }
    case BIT_FIELD_REF:
      {
	tree arg = TREE_OPERAND(expr, 0);
	auto [value, undef] = tree2inst_undef(bb, arg);
	uint64_t bitsize = get_int_cst_val(TREE_OPERAND(expr, 1));
	uint64_t bit_offset = get_int_cst_val(TREE_OPERAND(expr, 2));
	Instruction *high =
	  bb->value_inst(bitsize + bit_offset - 1, 32);
	Instruction *low = bb->value_inst(bit_offset, 32);
	value = to_mem_repr(bb, value, TREE_TYPE(arg));
	value = bb->build_inst(Op::EXTRACT, value, high, low);
	if (undef)
	  undef = bb->build_inst(Op::EXTRACT, undef, high, low);
	std::tie(value, undef) =
	  from_mem_repr(bb, value, undef, TREE_TYPE(expr));
	return {value, undef, nullptr};
      }
    case ARRAY_REF:
      {
	tree array = TREE_OPERAND(expr, 0);
	// Indexing element of a vector as vec[2] is done by an  ARRAY_REF of
	// a VIEW_CONVERT_EXPR of the vector.
	if (TREE_CODE(array) == VIEW_CONVERT_EXPR
	    && VECTOR_TYPE_P(TREE_TYPE(TREE_OPERAND(array, 0))))
	  return vector_as_array(bb, expr);
	return process_load(bb, expr);
      }
    case MEM_REF:
    case COMPONENT_REF:
    case TARGET_MEM_REF:
    case VAR_DECL:
    case RESULT_DECL:
      return process_load(bb, expr);
    default:
      {
	const char *name = get_tree_code_name(TREE_CODE(expr));
	throw Not_implemented("tree2inst: "s + name);
      }
    }
}

std::pair<Instruction *, Instruction *> Converter::tree2inst_prov(Basic_block *bb, tree expr)
{
  auto [inst, undef, provenance] = tree2inst_undef_prov(bb, expr);
  if (undef)
    build_ub_if_not_zero(bb, undef);
  return {inst, provenance};
}

std::pair<Instruction *, Instruction *> Converter::tree2inst_undef(Basic_block *bb, tree expr)
{
  auto [inst, undef, _] = tree2inst_undef_prov(bb, expr);
  return {inst, undef};
}

Instruction *Converter::tree2inst(Basic_block *bb, tree expr)
{
  auto [inst, undef, _] = tree2inst_undef_prov(bb, expr);
  if (undef)
    build_ub_if_not_zero(bb, undef);
  return inst;
}

// Processing constructors for global variables may give us more complex expr
// than what we get from normal operations. For example, initializing an array
// of pointers may have an initializer &a-&b that in the function body would
// be calculated by its own stmt.
std::tuple<Instruction *, Instruction *, Instruction *> Converter::tree2inst_init_var(Basic_block *bb, tree expr)
{
  check_type(TREE_TYPE(expr));

  tree_code code = TREE_CODE(expr);
  if (TREE_OPERAND_LENGTH(expr) == 2)
    {
      tree arg1_expr = TREE_OPERAND(expr, 0);
      tree arg2_expr = TREE_OPERAND(expr, 1);
      tree arg1_type = TREE_TYPE(arg1_expr);
      tree arg2_type = TREE_TYPE(arg2_expr);
      auto [arg1, arg1_undef, arg1_prov] = tree2inst_init_var(bb, arg1_expr);
      auto [arg2, arg2_undef, arg2_prov] = tree2inst_init_var(bb, arg2_expr);
      return process_binary_scalar(code, arg1, arg1_undef, arg1_prov, arg2,
				   arg2_undef, arg2_prov, TREE_TYPE(expr),
				   arg1_type, arg2_type, bb);
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
	auto [arg, arg_undef, arg_prov] = tree2inst_init_var(bb, arg_expr);
	return process_unary_scalar(code, arg, arg_undef, arg_prov,
				    TREE_TYPE(expr), TREE_TYPE(arg_expr), bb);
      }
    default:
      return tree2inst_undef_prov(bb, expr);
    }
}

std::pair<Instruction *, Instruction *> Converter::get_res_undef(Instruction *arg1_undef, tree lhs_type, Basic_block *bb)
{
  Instruction *res_undef = nullptr;
  Instruction *res_def = nullptr;
  if (arg1_undef)
    {
      Instruction *zero = bb->value_inst(0, arg1_undef->bitsize);
      res_undef = bb->build_inst(Op::NE, arg1_undef, zero);
      res_def = bb->build_inst(Op::NOT, res_undef);
      uint64_t bitsize = bitsize_for_type(lhs_type);
      if (TREE_CODE(lhs_type) != BOOLEAN_TYPE && bitsize > res_undef->bitsize)
	{
	  Instruction *bs_inst = bb->value_inst(bitsize, 32);
	  res_undef = bb->build_inst(Op::SEXT, res_undef, bs_inst);
	}
    }
  return {res_undef, res_def};
}

std::pair<Instruction *, Instruction *> Converter::get_res_undef(Instruction *arg1_undef, Instruction *arg2_undef, tree lhs_type, Basic_block *bb)
{
  Instruction *res_undef = nullptr;
  Instruction *res_def = nullptr;
  if (arg1_undef)
    {
      Instruction *zero = bb->value_inst(0, arg1_undef->bitsize);
      res_undef = bb->build_inst(Op::NE, arg1_undef, zero);
    }
  if (arg2_undef)
    {
      Instruction *zero = bb->value_inst(0, arg2_undef->bitsize);
      Instruction *tmp = bb->build_inst(Op::NE, arg2_undef, zero);
      if (res_undef)
	res_undef = bb->build_inst(Op::OR, res_undef, tmp);
      else
	res_undef = tmp;
    }
  if (res_undef)
    {
      res_def = bb->build_inst(Op::NOT, res_undef);
      uint64_t bitsize = bitsize_for_type(lhs_type);
      if (TREE_CODE(lhs_type) != BOOLEAN_TYPE && bitsize > res_undef->bitsize)
	{
	  Instruction *bs_inst = bb->value_inst(bitsize, 32);
	  res_undef = bb->build_inst(Op::SEXT, res_undef, bs_inst);
	}
    }
  return {res_undef, res_def};
}

std::pair<Instruction *, Instruction *> Converter::get_res_undef(Instruction *arg1_undef, Instruction *arg2_undef, Instruction *arg3_undef, tree lhs_type, Basic_block *bb)
{
  Instruction *res_undef = nullptr;
  Instruction *res_def = nullptr;
  if (arg1_undef)
    {
      Instruction *zero = bb->value_inst(0, arg1_undef->bitsize);
      res_undef = bb->build_inst(Op::NE, arg1_undef, zero);
    }
  if (arg2_undef)
    {
      Instruction *zero = bb->value_inst(0, arg2_undef->bitsize);
      Instruction *tmp = bb->build_inst(Op::NE, arg2_undef, zero);
      if (res_undef)
	res_undef = bb->build_inst(Op::OR, res_undef, tmp);
      else
	res_undef = tmp;
    }
  if (arg3_undef)
    {
      Instruction *zero = bb->value_inst(0, arg3_undef->bitsize);
      Instruction *tmp = bb->build_inst(Op::NE, arg3_undef, zero);
      if (res_undef)
	res_undef = bb->build_inst(Op::OR, res_undef, tmp);
      else
	res_undef = tmp;
    }
  if (res_undef)
    {
      res_def = bb->build_inst(Op::NOT, res_undef);
      uint64_t bitsize = bitsize_for_type(lhs_type);
      if (TREE_CODE(lhs_type) != BOOLEAN_TYPE && bitsize > res_undef->bitsize)
	{
	  Instruction *bs_inst = bb->value_inst(bitsize, 32);
	  res_undef = bb->build_inst(Op::SEXT, res_undef, bs_inst);
	}
    }
  return {res_undef, res_def};
}

Addr Converter::process_array_ref(Basic_block *bb, tree expr, bool is_mem_access)
{
  tree array = TREE_OPERAND(expr, 0);
  tree index = TREE_OPERAND(expr, 1);
  tree array_type = TREE_TYPE(array);
  tree elem_type = TREE_TYPE(array_type);
  tree domain = TYPE_DOMAIN(array_type);

  Addr addr = process_address(bb, array, is_mem_access);
  Instruction *idx = tree2inst(bb, index);
  if (idx->bitsize < addr.ptr->bitsize)
    {
      Instruction *bitsize_inst = bb->value_inst(addr.ptr->bitsize, 32);
      if (TYPE_UNSIGNED(TREE_TYPE(index)))
	idx = bb->build_inst(Op::ZEXT, idx, bitsize_inst);
      else
	idx = bb->build_inst(Op::SEXT, idx, bitsize_inst);
    }
  else if (idx->bitsize > addr.ptr->bitsize)
    {
      Instruction *high = bb->value_inst(idx->bitsize - 1, 32);
      Instruction *low = bb->value_inst(addr.ptr->bitsize, 32);
      Instruction *top = bb->build_inst(Op::EXTRACT, idx, high, low);
      Instruction *zero = bb->value_inst(0, top->bitsize);
      Instruction *cond = bb->build_inst(Op::NE, top, zero);
      bb->build_inst(Op::UB, cond);
      idx = bb->build_trunc(idx, addr.ptr->bitsize);
    }

  uint64_t elem_size = bytesize_for_type(elem_type);
  Instruction *elm_size = bb->value_inst(elem_size, idx->bitsize);
  Instruction *offset = bb->build_inst(Op::MUL, idx, elm_size);
  Instruction *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);

  Instruction *max_inst = nullptr;
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
  if (max_inst)
    {
      Instruction *cond = bb->build_inst(Op::UGT, idx, max_inst);
      bb->build_inst(Op::UB, cond);
    }
  else
    {
      Op op = TYPE_UNSIGNED(TREE_TYPE(index)) ? Op::ZEXT : Op::SEXT;
      Instruction *ext_bitsize_inst = bb->value_inst(ptr->bitsize * 2, 32);
      Instruction *eidx = bb->build_inst(op, idx, ext_bitsize_inst);
      Instruction *eelm_size = bb->value_inst(elem_size, ptr->bitsize * 2);
      Instruction *eoffset = bb->build_inst(Op::MUL, eidx, eelm_size);
      uint32_t ptr_offset_bits = func->module->ptr_offset_bits;
      Instruction *emax_offset =
	bb->value_inst((uint64_t)1 << ptr_offset_bits, ptr->bitsize * 2);
      Instruction *cond = bb->build_inst(Op::UGE, eoffset, emax_offset);
      bb->build_inst(Op::UB, cond);
      if (is_mem_access)
	mem_access_ub_check(bb, ptr, addr.provenance, elem_size);
    }
  return {ptr, 0, addr.provenance};
}

Addr Converter::process_component_ref(Basic_block *bb, tree expr, bool is_mem_access)
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

  Addr addr = process_address(bb, object, is_mem_access);
  Instruction *off = bb->value_inst(offset, addr.ptr->bitsize);
  Instruction *ptr = bb->build_inst(Op::ADD, addr.ptr, off);

  return {ptr, bit_offset, addr.provenance};
}

Addr Converter::process_bit_field_ref(Basic_block *bb, tree expr, bool is_mem_access)
{
  tree object = TREE_OPERAND(expr, 0);
  tree position = TREE_OPERAND(expr, 2);
  uint64_t bit_offset = get_int_cst_val(position);
  Addr addr = process_address(bb, object, is_mem_access);
  Instruction *ptr = addr.ptr;
  if (bit_offset > 7)
    {
      uint64_t offset = bit_offset / 8;
      Instruction *off = bb->value_inst(offset, ptr->bitsize);
      ptr = bb->build_inst(Op::ADD, ptr, off);
      bit_offset &= 7;
    }
  return {ptr, bit_offset, addr.provenance};
}

void alignment_check(Basic_block *bb, tree expr, Instruction *ptr)
{
  // TODO: There are cases where bit_alignment1 and bit_alignment2
  // are inconsistent -- sometimes bit_alignment1 is larges, and
  // sometimes bit_alignment2. And they varies in strange ways.
  // E.g. bit_alignment1 contain info about __builtin_assume_aligned
  // and is often correct in size of type alignment. But sometimes it
  // has the element alignment for vectors. Or 128 when bit_alignment2
  // is 256.
  uint32_t bit_alignment1 = get_object_alignment(expr);
  uint32_t bit_alignment2 = TYPE_ALIGN(TREE_TYPE(expr));
  uint32_t bit_alignment = std::max(bit_alignment1, bit_alignment2);
  assert((bit_alignment1 & 7) == 0);
  assert((bit_alignment2 & 7) == 0);
  uint32_t alignment = bit_alignment / 8;
  if (alignment > 1)
    {
      uint32_t high_val = 0;
      for (;;)
	{
	  high_val++;
	  if (alignment == (1u << high_val))
	    break;
	}

      Instruction *extract = bb->build_trunc(ptr, high_val);
      Instruction *zero = bb->value_inst(0, high_val);
      Instruction *cond = bb->build_inst(Op::NE, extract, zero);
      bb->build_inst(Op::UB, cond);
    }
}

Addr Converter::process_address(Basic_block *bb, tree expr, bool is_mem_access)
{
  tree_code code = TREE_CODE(expr);
  if (code == MEM_REF)
    {
      auto [arg1, arg1_prov] = tree2inst_prov(bb, TREE_OPERAND(expr, 0));
      assert(arg1_prov);
      Instruction *arg2 = tree2inst(bb, TREE_OPERAND(expr, 1));
      Instruction *ptr = bb->build_inst(Op::ADD, arg1, arg2);
      if (is_mem_access)
	{
	  uint64_t size = bytesize_for_type(TREE_TYPE(expr));
	  mem_access_ub_check(bb, ptr, arg1_prov, size);
	  alignment_check(bb, expr, ptr);
	}
      return {ptr, 0, arg1_prov};
    }
  if (code == TARGET_MEM_REF)
    {
      // base + (step * index + index2 + offset)
      auto [base, base_prov] = tree2inst_prov(bb, TREE_OPERAND(expr, 0));
      assert(base_prov);
      Instruction *offset = tree2inst(bb, TREE_OPERAND(expr, 1));
      Instruction *off = offset;
      if (TREE_OPERAND(expr, 2))
	{
	  Instruction *index = tree2inst(bb, TREE_OPERAND(expr, 2));
	  if (TREE_OPERAND(expr, 3))
	    {
	      Instruction *step = tree2inst(bb, TREE_OPERAND(expr, 3));
	      index = bb->build_inst(Op::MUL, step, index);
	    }
	  off = bb->build_inst(Op::ADD, off, index);
	}
      if (TREE_OPERAND(expr, 4))
	{
	  Instruction *index2 = tree2inst(bb, TREE_OPERAND(expr, 4));
	  off = bb->build_inst(Op::ADD, off, index2);
	}
      Instruction *ptr = bb->build_inst(Op::ADD, base, off);
      if (is_mem_access)
	{
	  uint64_t size = bytesize_for_type(TREE_TYPE(expr));
	  mem_access_ub_check(bb, ptr, base_prov, size);
	  alignment_check(bb, expr, ptr);
	}
      return {ptr, 0, base_prov};
    }
  if (code == VAR_DECL)
    {
      // We are currently not adding RTTI structures, which makes
      // decl2instruction.at(expr) crash for e.g., g++.dg/analyzer/pr108003.C
      if (!decl2instruction.contains(expr))
	throw Not_implemented("process_address: unknown VAR_DECL");

      Instruction *ptr = decl2instruction.at(expr);
      assert(ptr->opcode == Op::MEMORY);
      Instruction *id = bb->build_extract_id(ptr);
      if (is_mem_access)
	{
	  // This reads/writes a variable, so we know the access is in range.
	  // However, we must verify the variable hasn't gone out of scope.
	  Instruction *mem_size = bb->build_inst(Op::GET_MEM_SIZE, id);
	  Instruction *zero = bb->value_inst(0, mem_size->bitsize);
	  bb->build_inst(Op::UB, bb->build_inst(Op::EQ, mem_size, zero));
	}
      return {ptr, 0, id};
    }
  if (code == ARRAY_REF)
    return process_array_ref(bb, expr, is_mem_access);
  if (code == COMPONENT_REF)
    return process_component_ref(bb, expr, is_mem_access);
  if (code == BIT_FIELD_REF)
    return process_bit_field_ref(bb, expr, is_mem_access);
  if (code == VIEW_CONVERT_EXPR)
    return process_address(bb, TREE_OPERAND(expr, 0), is_mem_access);
  if (code == REALPART_EXPR)
    return process_address(bb, TREE_OPERAND(expr, 0), is_mem_access);
  if (code == IMAGPART_EXPR)
    {
      Addr addr = process_address(bb, TREE_OPERAND(expr, 0), is_mem_access);
      uint64_t offset_val = bytesize_for_type(TREE_TYPE(expr));
      Instruction *offset = bb->value_inst(offset_val, addr.ptr->bitsize);
      Instruction *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);
      return {ptr, 0, addr.provenance};
    }
  if (code == INTEGER_CST)
    {
      Instruction *ptr = tree2inst(bb, expr);
      Instruction *provenance = bb->build_extract_id(ptr);
      return {ptr, 0, provenance};
    }
  if (code == RESULT_DECL)
    {
      Instruction *ptr = decl2instruction.at(expr);
      assert(ptr->opcode == Op::MEMORY);
      return {ptr, 0, ptr->arguments[0]};
    }

  const char *name = get_tree_code_name(TREE_CODE(expr));
  throw Not_implemented("process_address: "s + name);
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

std::tuple<Instruction *, Instruction *, Instruction *> Converter::vector_as_array(Basic_block *bb, tree expr)
{
  assert(TREE_CODE(expr) == ARRAY_REF);
  tree array = TREE_OPERAND(expr, 0);
  tree index = TREE_OPERAND(expr, 1);
  tree array_type = TREE_TYPE(array);
  tree elem_type = TREE_TYPE(array_type);
  assert(TREE_CODE(array) == VIEW_CONVERT_EXPR);
  tree vector_expr = TREE_OPERAND(array, 0);
  assert(VECTOR_TYPE_P(TREE_TYPE(vector_expr)));

  auto [inst, undef] = tree2inst_undef(bb, vector_expr);

  uint64_t vector_size = bytesize_for_type(array_type);
  uint64_t elem_size = bytesize_for_type(elem_type);
  assert(vector_size % elem_size == 0);

  Instruction *idx = tree2inst(bb, index);
  Instruction *nof_elems =
    bb->value_inst(vector_size / elem_size, idx->bitsize);
  Instruction *cond = bb->build_inst(Op::UGE, idx, nof_elems);
  bb->build_inst(Op::UB, cond);

  Instruction *elm_bitsize = bb->value_inst(elem_size * 8, idx->bitsize);
  Instruction *shift = bb->build_inst(Op::MUL, idx, elm_bitsize);

  if (inst->bitsize > shift->bitsize)
    {
      Instruction *bitsize_inst = bb->value_inst(inst->bitsize, 32);
      shift = bb->build_inst(Op::ZEXT, shift, bitsize_inst);
    }
  else if (inst->bitsize < shift->bitsize)
    shift = bb->build_trunc(shift, inst->bitsize);
  inst = bb->build_inst(Op::LSHR, inst, shift);
  inst = bb->build_trunc(inst, elem_size * 8);
  if (undef)
    {
      undef = bb->build_inst(Op::LSHR, undef, shift);
      undef = bb->build_trunc(undef, elem_size * 8);
    }
  std::tie(inst, undef) = from_mem_repr(bb, inst, undef, elem_type);

  return {inst, undef, nullptr};
}

std::tuple<Instruction *, Instruction *, Instruction *> Converter::process_load(Basic_block *bb, tree expr)
{
  tree type = TREE_TYPE(expr);
  uint64_t bitsize = bitsize_for_type(type);
  uint64_t size = bytesize_for_type(type);
  if (bitsize == 0)
    throw Not_implemented("process_load: load unhandled size 0");
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("process_load: load size too big");
  Addr addr = process_address(bb, expr, true);
  bool is_bitfield = is_bit_field(expr);
  assert(is_bitfield || addr.bitoffset == 0);
  if (is_bitfield)
    size = (bitsize + addr.bitoffset + 7) / 8;
  Instruction *value = nullptr;
  Instruction *undef = nullptr;
  Instruction *mem_flags2 = nullptr;
  for (uint64_t i = 0; i < size; i++)
    {
      Instruction *offset = bb->value_inst(i, addr.ptr->bitsize);
      Instruction *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);

      Instruction *data_byte;
      Instruction *undef_byte;
      uint8_t padding = padding_at_offset(type, i);
      if (padding == 255)
	{
	  // No need to load a value as its value is indeterminate.
	  data_byte = bb->value_inst(0, 8);
	  undef_byte = bb->value_inst(255, 8);
	}
      else
	{
	  data_byte = bb->build_inst(Op::LOAD, ptr);
	  undef_byte = bb->build_inst(Op::GET_MEM_UNDEF, ptr);
	  if (padding != 0)
	    {
	      Instruction *padding_inst = bb->value_inst(padding, 8);
	      undef_byte = bb->build_inst(Op::OR, undef_byte, padding_inst);
	    }
	}

      if (value)
	value = bb->build_inst(Op::CONCAT, data_byte, value);
      else
	value = data_byte;
      if (undef)
	undef = bb->build_inst(Op::CONCAT, undef_byte, undef);
      else
	undef = undef_byte;

      // TODO: Rename. This is not mem_flags -- we only splats one flag.
      Instruction *flag = bb->build_inst(Op::GET_MEM_FLAG, ptr);
      flag = bb->build_inst(Op::SEXT, flag, bb->value_inst(8, 32));
      if (mem_flags2)
	mem_flags2 = bb->build_inst(Op::CONCAT, flag, mem_flags2);
      else
	mem_flags2 = flag;
    }
  if (is_bitfield)
    {
      Instruction *high = bb->value_inst(bitsize + addr.bitoffset - 1, 32);
      Instruction *low = bb->value_inst(addr.bitoffset, 32);
      value = bb->build_inst(Op::EXTRACT, value, high, low);
      undef = bb->build_inst(Op::EXTRACT, undef, high, low);
      mem_flags2 = bb->build_inst(Op::EXTRACT, mem_flags2, high, low);
    }
  else
    {
      if (expr != DECL_RESULT(fun->decl))
	constrain_src_value(bb, value, type, mem_flags2);

      // TODO: What if the extracted bits are defined, but the extra bits
      // undefined?
      // E.g. a bool where the least significant bit is defined, but the rest
      // undefined. I guess it should be undefined?
      std::tie(value, undef) = from_mem_repr(bb, value, undef, type);
      std::tie(value, mem_flags2) = from_mem_repr(bb, value, mem_flags2, type);
      inst2memory_flagsx.insert({value, mem_flags2});
    }

  Instruction *provenance = nullptr;
  if (POINTER_TYPE_P(type))
    provenance = bb->build_extract_id(value);

  return {value, undef, provenance};
}

// Read value/undef from memory. No UB checks etc. are done.
std::tuple<Instruction *, Instruction *, Instruction *> Converter::load_value(Basic_block *bb, Instruction *orig_ptr, uint64_t size)
{
  Instruction *value = nullptr;
  Instruction *undef = nullptr;
  Instruction *mem_flags = nullptr;
  for (uint64_t i = 0; i < size; i++)
    {
      Instruction *offset = bb->value_inst(i, orig_ptr->bitsize);
      Instruction *ptr = bb->build_inst(Op::ADD, orig_ptr, offset);
      Instruction *data_byte = bb->build_inst(Op::LOAD, ptr);
      if (value)
	value = bb->build_inst(Op::CONCAT, data_byte, value);
      else
	value = data_byte;
      Instruction *undef_byte = bb->build_inst(Op::GET_MEM_UNDEF, ptr);
      if (undef)
	undef = bb->build_inst(Op::CONCAT, undef_byte, undef);
      else
	undef = undef_byte;
      Instruction *flag = bb->build_inst(Op::GET_MEM_FLAG, ptr);
      flag = bb->build_inst(Op::SEXT, flag, bb->value_inst(8, 32));
      if (mem_flags)
	mem_flags = bb->build_inst(Op::CONCAT, flag, mem_flags);
      else
	mem_flags = flag;
    }
  return {value, undef, mem_flags};
}

// Write value to memory. No UB checks etc. are done, and memory flags
// are not updated.
void Converter::store_value(Basic_block *bb, Instruction *orig_ptr, Instruction *value, Instruction *undef)
{
  if ((value->bitsize & 7) != 0)
    throw Not_implemented("store_value: not byte aligned");
  uint64_t size = value->bitsize / 8;
  for (uint64_t i = 0; i < size; i++)
    {
      Instruction *offset = bb->value_inst(i, orig_ptr->bitsize);
      Instruction *ptr = bb->build_inst(Op::ADD, orig_ptr, offset);
      Instruction *high = bb->value_inst(i * 8 + 7, 32);
      Instruction *low = bb->value_inst(i * 8, 32);
      Instruction *byte = bb->build_inst(Op::EXTRACT, value, high, low);
      bb->build_inst(Op::STORE, ptr, byte);
      if (undef)
	{
	  byte = bb->build_inst(Op::EXTRACT, undef, high, low);
	  bb->build_inst(Op::SET_MEM_UNDEF, ptr, byte);
	}
    }
}

void Converter::process_store(tree addr_expr, tree value_expr, Basic_block *bb)
{
  if (TREE_CODE(value_expr) == STRING_CST)
    {
      uint64_t str_len = TREE_STRING_LENGTH(value_expr);
      uint64_t size = bytesize_for_type(TREE_TYPE(addr_expr));
      assert(str_len <= size);
      const char *p = TREE_STRING_POINTER(value_expr);
      Addr addr = process_address(bb, addr_expr, true);
      assert(!addr.bitoffset);
      Instruction *memory_flag = bb->value_inst(0, 1);
      Instruction *undef = bb->value_inst(0, 8);
      if (size > MAX_MEMORY_UNROLL_LIMIT)
	throw Not_implemented("process_store: too large string");

      store_ub_check(bb, addr.ptr, addr.provenance, size);
      for (uint64_t i = 0; i < size; i++)
	{
	  Instruction *offset = bb->value_inst(i, addr.ptr->bitsize);
	  Instruction *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);
	  uint8_t byte = (i < str_len) ? p[i] : 0;
	  Instruction *value = bb->value_inst(byte, 8);
	  bb->build_inst(Op::STORE, ptr, value);
	  bb->build_inst(Op::SET_MEM_FLAG, ptr, memory_flag);
	  bb->build_inst(Op::SET_MEM_UNDEF, ptr, undef);
	}
      return;
    }

  tree value_type = TREE_TYPE(value_expr);
  bool is_bitfield = is_bit_field(addr_expr);
  Addr addr = process_address(bb, addr_expr, true);
  assert(is_bitfield || addr.bitoffset == 0);
  assert(addr.bitoffset < 8);
  auto [value, undef, provenance] = tree2inst_undef_prov(bb, value_expr);
  if (!undef)
    undef = bb->value_inst(0, value->bitsize);

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
      assert(provenance);
      Instruction *value_mem_id = bb->build_extract_id(value);
      Instruction *is_ub = bb->build_inst(Op::NE, provenance, value_mem_id);
      bb->build_inst(Op::UB, is_ub);
    }

  uint64_t size;
  if (is_bitfield)
    {
      uint64_t bitsize = bitsize_for_type(value_type);
      size = (bitsize + addr.bitoffset + 7) / 8;

      if (addr.bitoffset)
	{
	  Instruction *first_byte = bb->build_inst(Op::LOAD, addr.ptr);
	  Instruction *bits = bb->build_trunc(first_byte, addr.bitoffset);
	  value = bb->build_inst(Op::CONCAT, value, bits);

	  first_byte = bb->build_inst(Op::GET_MEM_UNDEF, addr.ptr);
	  bits = bb->build_trunc(first_byte, addr.bitoffset);
	  undef = bb->build_inst(Op::CONCAT, undef, bits);
	}

      if (bitsize + addr.bitoffset != size * 8)
	{
	  Instruction *offset = bb->value_inst(size - 1, addr.ptr->bitsize);
	  Instruction *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);

	  uint64_t remaining = size * 8 - (bitsize + addr.bitoffset);
	  assert(remaining < 8);
	  Instruction *high = bb->value_inst(7, 32);
	  Instruction *low = bb->value_inst(8 - remaining, 32);

	  Instruction *last_byte = bb->build_inst(Op::LOAD, ptr);
	  Instruction *bits = bb->build_inst(Op::EXTRACT, last_byte, high, low);
	  value = bb->build_inst(Op::CONCAT, bits, value);

	  last_byte = bb->build_inst(Op::GET_MEM_UNDEF, ptr);
	  bits = bb->build_inst(Op::EXTRACT, last_byte, high, low);
	  undef = bb->build_inst(Op::CONCAT, bits, undef);
	}
    }
  else
    {
      size = bytesize_for_type(value_type);
      std::tie(value, undef) = to_mem_repr(bb, value, undef, value_type);
    }

  // TODO: Adjust for bitfield?
  Instruction *memory_flagsx = nullptr;
  if (inst2memory_flagsx.contains(value))
    memory_flagsx = inst2memory_flagsx.at(value);

  for (uint64_t i = 0; i < size; i++)
    {
      Instruction *offset = bb->value_inst(i, addr.ptr->bitsize);
      Instruction *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);

      Instruction *high = bb->value_inst(i * 8 + 7, 32);
      Instruction *low = bb->value_inst(i * 8, 32);

      uint8_t padding = padding_at_offset(value_type, i);
      if (padding == 255)
	{
	  // No need to store if this is padding as it will be marked as
	  // undefined anyway.
	  bb->build_inst(Op::SET_MEM_UNDEF, ptr, bb->value_inst(255, 8));
	}
      else
	{
	  Instruction *byte = bb->build_inst(Op::EXTRACT, value, high, low);
	  bb->build_inst(Op::STORE, ptr, byte);

	  byte = bb->build_inst(Op::EXTRACT, undef, high, low);
	  if (padding != 0)
	    {
	      Instruction *padding_inst = bb->value_inst(padding, 8);
	      byte = bb->build_inst(Op::OR, byte, padding_inst);
	    }
	  bb->build_inst(Op::SET_MEM_UNDEF, ptr, byte);
	}

      Instruction *memory_flag;
      if (memory_flagsx)
	{
	  memory_flag = bb->build_inst(Op::EXTRACT, memory_flagsx, high, low);
	  Instruction *zero = bb->value_inst(0, memory_flag->bitsize);
	  memory_flag = bb->build_inst(Op::NE, memory_flag, zero);
	}
      else
	memory_flag = bb->value_inst(1, 1);
      bb->build_inst(Op::SET_MEM_FLAG, ptr, memory_flag);
    }

  // It is UB to write to constant memory.
  Instruction *is_const = bb->build_inst(Op::IS_CONST_MEM, addr.provenance);
  bb->build_inst(Op::UB, is_const);
}

// Convert a scalar value of src_type to dest_type.
std::tuple<Instruction *, Instruction *, Instruction *> Converter::type_convert(Instruction *inst, Instruction *undef, Instruction *provenance, tree src_type, tree dest_type, Basic_block *bb)
{
  auto [res_undef, res_def] = get_res_undef(undef, dest_type, bb);

  if (INTEGRAL_TYPE_P(src_type) || POINTER_TYPE_P(src_type) || TREE_CODE(src_type) == OFFSET_TYPE)
    {
      if (INTEGRAL_TYPE_P(dest_type) || POINTER_TYPE_P(dest_type) || TREE_CODE(dest_type) == OFFSET_TYPE)
	{
	  unsigned src_prec = inst->bitsize;
	  unsigned dest_prec;
	  if (TREE_CODE(dest_type) == BOOLEAN_TYPE)
	    {
	      dest_prec = 1;
	      if (res_undef && res_undef->bitsize > dest_prec)
		res_undef = bb->build_trunc(res_undef, dest_prec);
	    }
	  else
	    dest_prec = bitsize_for_type(dest_type);
	  if (src_prec > dest_prec)
	    inst = bb->build_trunc(inst, dest_prec);
	  else if (src_prec < dest_prec)
	    {
	      Op op = TYPE_UNSIGNED(src_type) ? Op::ZEXT : Op::SEXT;
	      Instruction *dest_prec_inst = bb->value_inst(dest_prec, 32);
	      inst =  bb->build_inst(op, inst, dest_prec_inst);
	      provenance = nullptr;
	    }
	  if (POINTER_TYPE_P(dest_type))
	    {
	      assert(!POINTER_TYPE_P(src_type) || provenance);
	      if (!provenance)
		provenance = bb->build_extract_id(inst);
	    }
	  return {inst, res_undef, provenance};
	}
      if (FLOAT_TYPE_P(dest_type))
	{
	  unsigned dest_prec = TYPE_PRECISION(dest_type);
	  Instruction *dest_prec_inst = bb->value_inst(dest_prec, 32);
	  Op op = TYPE_UNSIGNED(src_type) ? Op::U2F : Op::S2F;
	  Instruction *res = bb->build_inst(op, inst, dest_prec_inst);
	  return {res, res_undef, nullptr};
	}
    }

  if (FLOAT_TYPE_P(src_type))
    {
      if (TREE_CODE(dest_type) == INTEGER_TYPE
	  || TREE_CODE(dest_type) == BITINT_TYPE
	  || TREE_CODE(dest_type) == ENUMERAL_TYPE)
	{
	  // The result is UB if the floating point value is out of range
	  // for the integer.
	  // TODO: This is OK for float precsion <= dest precision.
	  // But for float precision > dest we currently mark as UB cases
	  // that round into range.
	  Instruction *min = tree2inst(bb, TYPE_MIN_VALUE(dest_type));
	  Instruction *max = tree2inst(bb, TYPE_MAX_VALUE(dest_type));
	  Op op = TYPE_UNSIGNED(dest_type) ? Op::U2F : Op::S2F;
	  int src_bitsize = TYPE_PRECISION(src_type);
	  Instruction *src_bitsize_inst = bb->value_inst(src_bitsize, 32);
	  Instruction *fmin = bb->build_inst(op, min, src_bitsize_inst);
	  Instruction *fmax = bb->build_inst(op, max, src_bitsize_inst);
	  Instruction *clow = bb->build_inst(Op::FGE, inst, fmin);
	  Instruction *chigh = bb->build_inst(Op::FLE, inst, fmax);
	  Instruction *is_in_range = bb->build_inst(Op::AND, clow, chigh);
	  Instruction *is_ub = bb->build_inst(Op::NOT, is_in_range);
	  if (res_def)
	    is_ub = bb->build_inst(Op::AND, is_ub, res_def);
	  bb->build_inst(Op::UB, is_ub);

	  int dest_bitsize = bitsize_for_type(dest_type);
	  op = TYPE_UNSIGNED(dest_type) ? Op::F2U : Op::F2S;
	  Instruction *dest_bitsize_inst = bb->value_inst(dest_bitsize, 32);
	  Instruction *res = bb->build_inst(op, inst, dest_bitsize_inst);
	  return {res, res_undef, nullptr};
	}
      if (FLOAT_TYPE_P(dest_type))
	{
	  unsigned src_prec = TYPE_PRECISION(src_type);
	  unsigned dest_prec = TYPE_PRECISION(dest_type);
	  if (src_prec == dest_prec)
	    return {inst, res_undef, nullptr};
	  Instruction *dest_prec_inst = bb->value_inst(dest_prec, 32);
	  Instruction *res = bb->build_inst(Op::FCHPREC, inst, dest_prec_inst);
	  return {res, res_undef, nullptr};
	}
    }

  throw Not_implemented("type_convert: unknown type");
}

Instruction *Converter::type_convert(Instruction *inst, tree src_type, tree dest_type, Basic_block *bb)
{
  return std::get<0>(type_convert(inst, nullptr, nullptr, src_type, dest_type, bb));
}

void check_wide_bool(Instruction *inst, tree type, Basic_block *bb)
{
  Instruction *false_inst = bb->value_inst(0, inst->bitsize);
  Instruction *true_inst = bb->value_inst(1, inst->bitsize);
  if (!TYPE_UNSIGNED(type))
    true_inst = bb->build_inst(Op::NEG, true_inst);
  Instruction *cond0 = bb->build_inst(Op::NE, inst, true_inst);
  Instruction *cond1 = bb->build_inst(Op::NE, inst, false_inst);
  Instruction *cond = bb->build_inst(Op::AND, cond0, cond1);
  bb->build_inst(Op::UB, cond);
}

std::pair<Instruction *, Instruction *> Converter::process_unary_bool(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, tree lhs_type, tree arg1_type, Basic_block *bb)
{
  assert(TREE_CODE(lhs_type) == BOOLEAN_TYPE);

  if (TREE_CODE(arg1_type) == BOOLEAN_TYPE && arg1->bitsize > 1)
    {
      arg1 = bb->build_trunc(arg1, 1);
      if (arg1_undef)
	{
	  Instruction *zero = bb->value_inst(0, arg1_undef->bitsize);
	  arg1_undef = bb->build_inst(Op::NE, arg1_undef, zero);
	}
    }

  auto [lhs, lhs_undef, _] =
    process_unary_int(code, arg1, arg1_undef, nullptr, lhs_type, arg1_type, bb);

  // GCC may use non-standard Boolean types (such as signed-boolean:8), so
  // we may need to extend the value if we have generated a standard 1-bit
  // Boolean for a comparison.
  uint64_t precision = TYPE_PRECISION(lhs_type);
  if (lhs->bitsize == 1 && precision > 1)
    {
      Instruction *bitsize_inst = bb->value_inst(precision, 32);
      Op op = TYPE_UNSIGNED(lhs_type) ? Op::ZEXT : Op::SEXT;
      lhs = bb->build_inst(op, lhs, bitsize_inst);
      if (lhs_undef)
	lhs_undef = bb->build_inst(Op::SEXT, lhs_undef, bitsize_inst);
    }
  if (lhs->bitsize > 1)
    check_wide_bool(lhs, lhs_type, bb);

  assert(lhs->bitsize == precision);
  assert(!lhs_undef || lhs_undef->bitsize == precision);
  return {lhs, lhs_undef};
}

std::tuple<Instruction *, Instruction *, Instruction *> Converter::process_unary_int(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg1_prov, tree lhs_type, tree arg1_type, Basic_block *bb)
{
  // Handle instructions that have special requirements for the propagation
  // of undef bits.
  switch (code)
    {
    case BIT_NOT_EXPR:
      return {bb->build_inst(Op::NOT, arg1), arg1_undef, arg1_prov};
    case CONVERT_EXPR:
    case NOP_EXPR:
      return type_convert(arg1, arg1_undef, arg1_prov, arg1_type, lhs_type, bb);
    default:
      break;
    }

  // Handle instructions where the result is undef if any input bit is undef.
  auto [res_undef, res_def] = get_res_undef(arg1_undef, lhs_type, bb);
  switch (code)
    {
    case ABS_EXPR:
      {
	if (!TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Instruction *min_int_inst = build_min_int(bb, arg1->bitsize);
	    Instruction *cond = bb->build_inst(Op::EQ, arg1, min_int_inst);
	    if (res_def)
	      cond = bb->build_inst(Op::AND, cond, res_def);
	    bb->build_inst(Op::UB, cond);
	  }
	assert(!TYPE_UNSIGNED(arg1_type));
	Instruction *neg = bb->build_inst(Op::NEG, arg1);
	Instruction *zero = bb->value_inst(0, arg1->bitsize);
	Instruction *cond = bb->build_inst(Op::SGE, arg1, zero);
	return {bb->build_inst(Op::ITE, cond, arg1, neg), res_undef, nullptr};
      }
    case ABSU_EXPR:
      {
	assert(!TYPE_UNSIGNED(arg1_type));
	Instruction *neg = bb->build_inst(Op::NEG, arg1);
	Instruction *zero = bb->value_inst(0, arg1->bitsize);
	Instruction *cond = bb->build_inst(Op::SGE, arg1, zero);
	return {bb->build_inst(Op::ITE, cond, arg1, neg), res_undef, nullptr};
      }
    case FIX_TRUNC_EXPR:
      return type_convert(arg1, arg1_undef, arg1_prov, arg1_type, lhs_type, bb);
    case NEGATE_EXPR:
      if (!TYPE_OVERFLOW_WRAPS(lhs_type))
	{
	  Instruction *min_int_inst = build_min_int(bb, arg1->bitsize);
	  Instruction *cond = bb->build_inst(Op::EQ, arg1, min_int_inst);
	  if (res_def)
	    cond = bb->build_inst(Op::AND, cond, res_def);
	  bb->build_inst(Op::UB, cond);
	}
      return {bb->build_inst(Op::NEG, arg1), res_undef, nullptr};
    default:
      break;
    }

  throw Not_implemented("process_unary_int: "s + get_tree_code_name(code));
}

std::pair<Instruction *, Instruction *> Converter::process_unary_float(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, tree lhs_type, tree arg1_type, Basic_block *bb)
{
  // Handle instructions that have special requirements for the propagation
  // of undef bits.
  switch (code)
    {
    case FLOAT_EXPR:
    case CONVERT_EXPR:
    case NOP_EXPR:
      {
	auto [inst, undef, prov] =
	  type_convert(arg1, arg1_undef, nullptr, arg1_type, lhs_type, bb);
	return {inst, undef};
      }
    default:
      break;
    }

  // Handle instructions where the result is undef if any input bit is undef.
  auto [res_undef, res_def] = get_res_undef(arg1_undef, lhs_type, bb);
  switch (code)
    {
    case ABS_EXPR:
      return {bb->build_inst(Op::FABS, arg1), res_undef};
    case NEGATE_EXPR:
      return {bb->build_inst(Op::FNEG, arg1), res_undef};
    case PAREN_EXPR:
      return {arg1, res_undef};
    default:
      break;
    }

  throw Not_implemented("process_unary_float: "s + get_tree_code_name(code));
}

Instruction *Converter::process_unary_complex(enum tree_code code, Instruction *arg1, tree lhs_type, Basic_block *bb)
{
  tree elem_type = TREE_TYPE(lhs_type);
  uint64_t bitsize = arg1->bitsize;
  uint64_t elem_bitsize = bitsize / 2;
  Instruction *real_high = bb->value_inst(elem_bitsize - 1, 32);
  Instruction *real_low = bb->value_inst(0, 32);
  Instruction *imag_high = bb->value_inst(bitsize - 1, 32);
  Instruction *imag_low = bb->value_inst(elem_bitsize, 32);
  Instruction *arg1_real =
    bb->build_inst(Op::EXTRACT, arg1, real_high, real_low);
  arg1_real = from_mem_repr(bb, arg1_real, elem_type);
  Instruction *arg1_imag =
    bb->build_inst(Op::EXTRACT, arg1, imag_high, imag_low);
  arg1_imag = from_mem_repr(bb, arg1_imag, elem_type);

  switch (code)
    {
    case CONJ_EXPR:
      {
	Instruction *inst_imag;
	inst_imag = process_unary_scalar(NEGATE_EXPR, arg1_imag,
					 elem_type, elem_type, bb);
	arg1_real = to_mem_repr(bb, arg1_real, elem_type);
	inst_imag = to_mem_repr(bb, inst_imag, elem_type);
	return bb->build_inst(Op::CONCAT, inst_imag, arg1_real);
      }
    case NEGATE_EXPR:
      {
	Instruction * inst_real =
	  process_unary_scalar(code, arg1_real, elem_type, elem_type, bb);
	Instruction *inst_imag =
	  process_unary_scalar(code, arg1_imag, elem_type, elem_type, bb);
	inst_real = to_mem_repr(bb, inst_real, elem_type);
	inst_imag = to_mem_repr(bb, inst_imag, elem_type);
	return bb->build_inst(Op::CONCAT, inst_imag, inst_real);
      }
    default:
      break;
    }

  throw Not_implemented("process_unary_complex: "s + get_tree_code_name(code));
}

Instruction *Converter::process_unary_scalar(enum tree_code code, Instruction *arg1, tree lhs_type, tree arg1_type, Basic_block *bb)
{
  auto [inst, undef, prov] =
    process_unary_scalar(code, arg1, nullptr, nullptr, lhs_type, arg1_type, bb);
  assert(!undef);
  assert(!prov);
  return inst;
}

std::tuple<Instruction *, Instruction *, Instruction *> Converter::process_unary_scalar(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg1_prov, tree lhs_type, tree arg1_type, Basic_block *bb)
{
  Instruction *inst;
  Instruction *undef = nullptr;
  Instruction *provenance = nullptr;
  if (TREE_CODE(lhs_type) == BOOLEAN_TYPE)
    {
      std::tie(inst, undef) =
	process_unary_bool(code, arg1, arg1_undef, lhs_type, arg1_type, bb);
    }
  else if (FLOAT_TYPE_P(lhs_type))
    {
      std::tie(inst, undef) =
	process_unary_float(code, arg1, arg1_undef, lhs_type, arg1_type, bb);
    }
  else
    {
      std::tie(inst, undef, provenance) =
	process_unary_int(code, arg1, arg1_undef, arg1_prov, lhs_type,
			  arg1_type, bb);
    }
  return {inst, undef, provenance};
}

std::pair<Instruction *, Instruction *> Converter::process_unary_vec(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, tree lhs_elem_type, tree arg1_elem_type, Basic_block *bb)
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

  Instruction *res = nullptr;
  Instruction *res_undef = nullptr;
  for (uint64_t i = start_idx; i < nof_elt; i++)
    {
      Instruction *a1_undef = nullptr;
      Instruction *a1 = extract_vec_elem(bb, arg1, elem_bitsize, i);
      if (arg1_undef)
	a1_undef = extract_vec_elem(bb, arg1_undef, elem_bitsize, i);
      auto [inst, inst_undef, _] =
	process_unary_scalar(code, a1, a1_undef, nullptr, lhs_elem_type,
			     arg1_elem_type, bb);

      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg1_undef)
	{
	  if (res_undef)
	    res_undef = bb->build_inst(Op::CONCAT, inst_undef, res_undef);
	  else
	    res_undef = inst_undef;
	}
    }
  return {res, res_undef};
}

std::pair<Instruction *, Instruction *> Converter::process_binary_float(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg2, Instruction *arg2_undef, tree lhs_type, Basic_block *bb)
{
  auto [res_undef, res_def] = get_res_undef(arg1_undef, arg2_undef, lhs_type,
					    bb);
  switch (code)
    {
    case EQ_EXPR:
      return {bb->build_inst(Op::FEQ, arg1, arg2), res_undef};
    case NE_EXPR:
      return {bb->build_inst(Op::FNE, arg1, arg2), res_undef};
    case GE_EXPR:
      return {bb->build_inst(Op::FGE, arg1, arg2), res_undef};
    case GT_EXPR:
      return {bb->build_inst(Op::FGT, arg1, arg2), res_undef};
    case LE_EXPR:
      return {bb->build_inst(Op::FLE, arg1, arg2), res_undef};
    case LT_EXPR:
      return {bb->build_inst(Op::FLT, arg1, arg2), res_undef};
    case UNEQ_EXPR:
      {
	Instruction *isnan1 = bb->build_inst(Op::FNE, arg1, arg1);
	Instruction *isnan2 = bb->build_inst(Op::FNE, arg2, arg2);
	Instruction *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Instruction *cmp = bb->build_inst(Op::FEQ, arg1, arg2);
	return {bb->build_inst(Op::OR, isnan, cmp), res_undef};
      }
    case UNLT_EXPR:
      {
	Instruction *isnan1 = bb->build_inst(Op::FNE, arg1, arg1);
	Instruction *isnan2 = bb->build_inst(Op::FNE, arg2, arg2);
	Instruction *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Instruction *cmp = bb->build_inst(Op::FLT, arg1, arg2);
	return {bb->build_inst(Op::OR, isnan, cmp), res_undef};
      }
    case UNLE_EXPR:
      {
	Instruction *isnan1 = bb->build_inst(Op::FNE, arg1, arg1);
	Instruction *isnan2 = bb->build_inst(Op::FNE, arg2, arg2);
	Instruction *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Instruction *cmp = bb->build_inst(Op::FLE, arg1, arg2);
	return {bb->build_inst(Op::OR, isnan, cmp), res_undef};
      }
    case UNGT_EXPR:
      {
	Instruction *isnan1 = bb->build_inst(Op::FNE, arg1, arg1);
	Instruction *isnan2 = bb->build_inst(Op::FNE, arg2, arg2);
	Instruction *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Instruction *cmp = bb->build_inst(Op::FGT, arg1, arg2);
	return {bb->build_inst(Op::OR, isnan, cmp), res_undef};
      }
    case UNGE_EXPR:
      {
	Instruction *isnan1 = bb->build_inst(Op::FNE, arg1, arg1);
	Instruction *isnan2 = bb->build_inst(Op::FNE, arg2, arg2);
	Instruction *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	Instruction *cmp = bb->build_inst(Op::FGE, arg1, arg2);
	return {bb->build_inst(Op::OR, isnan, cmp), res_undef};
      }
    case UNORDERED_EXPR:
      {
	Instruction *isnan1 = bb->build_inst(Op::FNE, arg1, arg1);
	Instruction *isnan2 = bb->build_inst(Op::FNE, arg2, arg2);
	return {bb->build_inst(Op::OR, isnan1, isnan2), res_undef};
      }
    case ORDERED_EXPR:
      {
	Instruction *isnan1 = bb->build_inst(Op::FNE, arg1, arg1);
	Instruction *isnan2 = bb->build_inst(Op::FNE, arg2, arg2);
	Instruction *isnan = bb->build_inst(Op::OR, isnan1, isnan2);
	return {bb->build_inst(Op::NOT, isnan), res_undef};
      }
    case LTGT_EXPR:
      {
	Instruction *lt = bb->build_inst(Op::FLT, arg1, arg2);
	Instruction *gt = bb->build_inst(Op::FGT, arg1, arg2);
	return {bb->build_inst(Op::OR, lt, gt), res_undef};
      }
    case RDIV_EXPR:
      return {bb->build_inst(Op::FDIV, arg1, arg2), res_undef};
    case MINUS_EXPR:
      return {bb->build_inst(Op::FSUB, arg1, arg2), res_undef};
    case MULT_EXPR:
      return {bb->build_inst(Op::FMUL, arg1, arg2), res_undef};
    case PLUS_EXPR:
      return {bb->build_inst(Op::FADD, arg1, arg2), res_undef};
    default:
      break;
    }

  throw Not_implemented("process_binary_float: "s + get_tree_code_name(code));
}

Instruction *Converter::process_binary_complex(enum tree_code code, Instruction *arg1, Instruction *arg2, tree lhs_type, Basic_block *bb)
{
  tree elem_type = TREE_TYPE(lhs_type);
  uint64_t bitsize = arg1->bitsize;
  uint64_t elem_bitsize = bitsize / 2;
  Instruction *real_high = bb->value_inst(elem_bitsize - 1, 32);
  Instruction *real_low = bb->value_inst(0, 32);
  Instruction *imag_high = bb->value_inst(bitsize - 1, 32);
  Instruction *imag_low = bb->value_inst(elem_bitsize, 32);
  Instruction *arg1_real =
    bb->build_inst(Op::EXTRACT, arg1, real_high, real_low);
  arg1_real = from_mem_repr(bb, arg1_real, elem_type);
  Instruction *arg1_imag =
    bb->build_inst(Op::EXTRACT, arg1, imag_high, imag_low);
  arg1_imag = from_mem_repr(bb, arg1_imag, elem_type);
  Instruction *arg2_real =
    bb->build_inst(Op::EXTRACT, arg2, real_high, real_low);
  arg2_real = from_mem_repr(bb, arg2_real, elem_type);
  Instruction *arg2_imag =
    bb->build_inst(Op::EXTRACT, arg2, imag_high, imag_low);
  arg2_imag = from_mem_repr(bb, arg2_imag, elem_type);

  switch (code)
    {
    case MINUS_EXPR:
    case PLUS_EXPR:
      {
	Instruction *inst_real =
	  process_binary_scalar(code, arg1_real, arg2_real,
				elem_type, elem_type, elem_type, bb);
	Instruction *inst_imag =
	  process_binary_scalar(code, arg1_imag, arg2_imag,
				elem_type, elem_type, elem_type, bb);
	inst_real = to_mem_repr(bb, inst_real, elem_type);
	inst_imag = to_mem_repr(bb, inst_imag, elem_type);
	return bb->build_inst(Op::CONCAT, inst_imag, inst_real);
      }
    default:
      break;
    }

  throw Not_implemented("process_binary_complex: "s + get_tree_code_name(code));
}

Instruction *Converter::process_binary_complex_cmp(enum tree_code code, Instruction *arg1, Instruction *arg2, tree lhs_type, tree arg1_type, Basic_block *bb)
{
  tree elem_type = TREE_TYPE(arg1_type);
  uint64_t bitsize = arg1->bitsize;
  uint64_t elem_bitsize = bitsize / 2;
  Instruction *real_high = bb->value_inst(elem_bitsize - 1, 32);
  Instruction *real_low = bb->value_inst(0, 32);
  Instruction *imag_high = bb->value_inst(bitsize - 1, 32);
  Instruction *imag_low = bb->value_inst(elem_bitsize, 32);
  Instruction *arg1_real =
    bb->build_inst(Op::EXTRACT, arg1, real_high, real_low);
  arg1_real = from_mem_repr(bb, arg1_real, elem_type);
  Instruction *arg1_imag =
    bb->build_inst(Op::EXTRACT, arg1, imag_high, imag_low);
  arg1_imag = from_mem_repr(bb, arg1_imag, elem_type);
  Instruction *arg2_real =
    bb->build_inst(Op::EXTRACT, arg2, real_high, real_low);
  arg2_real = from_mem_repr(bb, arg2_real, elem_type);
  Instruction *arg2_imag =
    bb->build_inst(Op::EXTRACT, arg2, imag_high, imag_low);
  arg2_imag = from_mem_repr(bb, arg2_imag, elem_type);

  switch (code)
    {
    case EQ_EXPR:
    case NE_EXPR:
      {
	Instruction *cmp_real =
	  process_binary_scalar(code, arg1_real, arg2_real,
				lhs_type, elem_type, elem_type, bb);
	Instruction *cmp_imag =
	  process_binary_scalar(code, arg1_imag, arg2_imag,
				lhs_type, elem_type, elem_type, bb);
	Instruction *cmp;
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

std::pair<Instruction *, Instruction *> Converter::process_binary_bool(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg2, Instruction *arg2_undef, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb)
{
  assert(TREE_CODE(lhs_type) == BOOLEAN_TYPE);

  if (TREE_CODE(arg1_type) == BOOLEAN_TYPE && arg1->bitsize > 1)
    {
      arg1 = bb->build_trunc(arg1, 1);
      if (arg1_undef)
	{
	  Instruction *zero = bb->value_inst(0, arg1_undef->bitsize);
	  arg1_undef = bb->build_inst(Op::NE, arg1_undef, zero);
	}
    }
  if (TREE_CODE(arg2_type) == BOOLEAN_TYPE && arg2->bitsize > 1)
    {
      arg2 = bb->build_trunc(arg2, 1);
      if (arg2_undef)
	{
	  Instruction *zero = bb->value_inst(0, arg2_undef->bitsize);
	  arg2_undef = bb->build_inst(Op::NE, arg2_undef, zero);
	}
    }

  Instruction *lhs;
  Instruction *lhs_undef = nullptr;
  if (FLOAT_TYPE_P(arg1_type))
    {
      std::tie(lhs, lhs_undef) =
	process_binary_float(code, arg1, arg1_undef, arg2, arg2_undef,
			     lhs_type, bb);
    }
  else
    {
      Instruction *lhs_prov;
      std::tie(lhs, lhs_undef, lhs_prov) =
	process_binary_int(code, TYPE_UNSIGNED(arg1_type), arg1, arg1_undef,
			   nullptr, arg2, arg2_undef, nullptr, lhs_type,
			   arg1_type, arg2_type, bb);
    }

  // GCC may use non-standard Boolean types (such as signed-boolean:8), so
  // we may need to extend the value if we have generated a standard 1-bit
  // Boolean for a comparison.
  uint64_t precision = TYPE_PRECISION(lhs_type);
  if (lhs->bitsize == 1 && precision > 1)
    {
      Instruction *bitsize_inst = bb->value_inst(precision, 32);
      Op op = TYPE_UNSIGNED(lhs_type) ? Op::ZEXT : Op::SEXT;
      lhs = bb->build_inst(op, lhs, bitsize_inst);
      if (lhs_undef)
	lhs_undef = bb->build_inst(Op::SEXT, lhs_undef, bitsize_inst);
    }
  if (lhs->bitsize > 1)
    check_wide_bool(lhs, lhs_type, bb);

  assert(lhs->bitsize == precision);
  assert(!lhs_undef || lhs_undef->bitsize == precision);
  return {lhs, lhs_undef};
}

Instruction *Converter::process_binary_int(enum tree_code code, bool is_unsigned, Instruction *arg1, Instruction *arg2, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow)
{
  return std::get<0>(process_binary_int(code, is_unsigned, arg1, nullptr, nullptr, arg2, nullptr, nullptr, lhs_type, arg1_type, arg2_type, bb, ignore_overflow));
}

std::tuple<Instruction *, Instruction *, Instruction *> Converter::process_binary_int(enum tree_code code, bool is_unsigned, Instruction *arg1, Instruction *arg1_undef, Instruction *arg1_prov, Instruction *arg2, Instruction *arg2_undef, Instruction *arg2_prov, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow)
{
  // Handle instructions that have special requirements for the propagation
  // of undef bits.
  switch (code)
    {
    case BIT_AND_EXPR:
      {
	Instruction *res_undef = nullptr;
	if (arg1_undef || arg2_undef)
	  {
	    if (!arg1_undef)
	      arg1_undef = bb->value_inst(0, arg1->bitsize);
	    if (!arg2_undef)
	      arg2_undef = bb->value_inst(0, arg2->bitsize);

	    // (0 & uninitialized) is 0.
	    // (1 & uninitialized) is uninitialized.
	    Instruction *mask =
	      bb->build_inst(Op::AND,
			     bb->build_inst(Op::OR, arg1, arg1_undef),
			     bb->build_inst(Op::OR, arg2, arg2_undef));
	    res_undef =
	      bb->build_inst(Op::AND,
			     bb->build_inst(Op::OR, arg1_undef, arg2_undef),
			     mask);
	  }

	Instruction *prov = nullptr;
	if (arg1_prov && arg2_prov && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in BIT_AND_EXPR");
	if (arg1_prov)
	  prov = arg1_prov;
	if (arg2_prov)
	  prov = arg2_prov;

	return {bb->build_inst(Op::AND, arg1, arg2), res_undef, prov};
      }
    case BIT_IOR_EXPR:
      {
	Instruction *res_undef = nullptr;
	if (arg1_undef || arg2_undef)
	  {
	    if (!arg1_undef)
	      arg1_undef = bb->value_inst(0, arg1->bitsize);
	    if (!arg2_undef)
	      arg2_undef = bb->value_inst(0, arg2->bitsize);

	    // (0 | uninitialized) is uninitialized.
	    // (1 | uninitialized) is 1.
	    Instruction *mask =
	      bb->build_inst(Op::AND,
			     bb->build_inst(Op::OR,
					    bb->build_inst(Op::NOT, arg1),
					    arg1_undef),
			     bb->build_inst(Op::OR,
					    bb->build_inst(Op::NOT, arg2),
					    arg2_undef));
	    res_undef =
	      bb->build_inst(Op::AND,
			     bb->build_inst(Op::OR, arg1_undef, arg2_undef),
			     mask);
	  }

	Instruction *prov = nullptr;
	if (arg1_prov && arg2_prov && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in BIT_IOR_EXPR");
	if (arg1_prov)
	  prov = arg1_prov;
	if (arg2_prov)
	  prov = arg2_prov;

	return {bb->build_inst(Op::OR, arg1, arg2), res_undef, prov};
      }
    case BIT_XOR_EXPR:
      {
	Instruction *res_undef = nullptr;
	if (arg1_undef || arg2_undef)
	  {
	    if (!arg1_undef)
	      arg1_undef = bb->value_inst(0, arg1->bitsize);
	    if (!arg2_undef)
	      arg2_undef = bb->value_inst(0, arg2->bitsize);
	    res_undef = bb->build_inst(Op::OR, arg1_undef, arg2_undef);
	  }
	return {bb->build_inst(Op::XOR, arg1, arg2), res_undef, nullptr};
      }
    case MULT_EXPR:
      {
	auto [res_undef, res_def] =
	  get_res_undef(arg1_undef, arg2_undef, lhs_type, bb);
	res_undef = nullptr;
	if (arg1_undef || arg2_undef)
	  {
	    Instruction *zero = bb->value_inst(0, arg1->bitsize);
	    if (!arg1_undef)
	      arg1_undef = zero;
	    if (!arg2_undef)
	      arg2_undef = zero;

	    // The result is defined if no input is uninitizled, or if one of
	    // that arguments is a initialized zero.
	    Instruction *arg1_unini = bb->build_inst(Op::NE, arg1_undef, zero);
	    Instruction *arg1_nonzero = bb->build_inst(Op::NE, arg1, zero);
	    Instruction *arg2_unini = bb->build_inst(Op::NE, arg2_undef, zero);
	    Instruction *arg2_nonzero = bb->build_inst(Op::NE, arg2, zero);
	    Instruction *ub =
	      bb->build_inst(Op::OR,
			     bb->build_inst(Op::AND,
					    arg1_unini,
					    bb->build_inst(Op::OR, arg2_unini,
							   arg2_nonzero)),
			     bb->build_inst(Op::AND,
					    arg2_unini,
					    bb->build_inst(Op::OR, arg1_unini,
							   arg1_nonzero)));
	    res_undef =
	      bb->build_inst(Op::SEXT, ub, bb->value_inst(arg1->bitsize, 32));
	  }

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Instruction *cond = bb->build_inst(Op::SMUL_WRAPS, arg1, arg2);
	    if (res_def)
	      cond = bb->build_inst(Op::AND, cond, res_def);
	    bb->build_inst(Op::UB, cond);
	  }
	Instruction *res = bb->build_inst(Op::MUL, arg1, arg2);
	return {res, res_undef, nullptr};
      }
    case EXACT_DIV_EXPR:
      {
	if (arg2_undef)
	  build_ub_if_not_zero(bb, arg2_undef);
	auto [res_undef, res_def] = get_res_undef(arg1_undef, lhs_type, bb);

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Instruction *min_int_inst = build_min_int(bb, arg1->bitsize);
	    Instruction *minus1_inst = bb->value_inst(-1, arg1->bitsize);
	    Instruction *cond1 = bb->build_inst(Op::EQ, arg1, min_int_inst);
	    Instruction *cond2 = bb->build_inst(Op::EQ, arg2, minus1_inst);
	    Instruction *ub_cond = bb->build_inst(Op::AND, cond1, cond2);
	    if (res_def)
	      ub_cond = bb->build_inst(Op::AND, ub_cond, res_def);
	    bb->build_inst(Op::UB, ub_cond);
	  }
	Instruction *zero = bb->value_inst(0, arg1->bitsize);
	Op rem_op = is_unsigned ? Op::UREM : Op::SREM;
	Instruction *rem = bb->build_inst(rem_op, arg1, arg2);
	Instruction *ub_cond = bb->build_inst(Op::NE, rem, zero);
	if (res_def)
	  ub_cond = bb->build_inst(Op::AND, ub_cond, res_def);
	bb->build_inst(Op::UB, ub_cond);
	Instruction *ub_cond2 = bb->build_inst(Op::EQ, arg2, zero);
	if (res_def)
	  ub_cond2 = bb->build_inst(Op::AND, ub_cond2, res_def);
	bb->build_inst(Op::UB, ub_cond2);
	Op div_op = is_unsigned ? Op::UDIV : Op::SDIV;
	return {bb->build_inst(div_op, arg1, arg2), res_undef, nullptr};
      }
    case TRUNC_DIV_EXPR:
      {
	if (arg2_undef)
	  build_ub_if_not_zero(bb, arg2_undef);
	auto [res_undef, res_def] = get_res_undef(arg1_undef, lhs_type, bb);

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Instruction *min_int_inst = build_min_int(bb, arg1->bitsize);
	    Instruction *minus1_inst = bb->value_inst(-1, arg1->bitsize);
	    Instruction *cond1 = bb->build_inst(Op::EQ, arg1, min_int_inst);
	    Instruction *cond2 = bb->build_inst(Op::EQ, arg2, minus1_inst);
	    Instruction *cond = bb->build_inst(Op::AND, cond1, cond2);
	    if (res_def)
	      cond = bb->build_inst(Op::AND, cond, res_def);
	    bb->build_inst(Op::UB, cond);
	  }
	Instruction *zero_inst = bb->value_inst(0, arg1->bitsize);
	Instruction *cond = bb->build_inst(Op::EQ, arg2, zero_inst);
	if (res_def)
	  cond = bb->build_inst(Op::AND, cond, res_def);
	bb->build_inst(Op::UB, cond);
	Op op = is_unsigned ? Op::UDIV : Op::SDIV;
	return {bb->build_inst(op, arg1, arg2), res_undef, nullptr};
      }
    case TRUNC_MOD_EXPR:
      {
	if (arg2_undef)
	  build_ub_if_not_zero(bb, arg2_undef);
	auto [res_undef, res_def] = get_res_undef(arg1_undef, lhs_type, bb);

	if (!TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Instruction *min_int_inst = build_min_int(bb, arg1->bitsize);
	    Instruction *minus1_inst = bb->value_inst(-1, arg1->bitsize);
	    Instruction *cond1 = bb->build_inst(Op::EQ, arg1, min_int_inst);
	    Instruction *cond2 = bb->build_inst(Op::EQ, arg2, minus1_inst);
	    Instruction *cond = bb->build_inst(Op::AND, cond1, cond2);
	    if (res_def)
	      cond = bb->build_inst(Op::AND, cond, res_def);
	    bb->build_inst(Op::UB, cond);
	  }
	Instruction *zero_inst = bb->value_inst(0, arg1->bitsize);
	Instruction *cond = bb->build_inst(Op::EQ, arg2, zero_inst);
	if (res_def)
	  cond = bb->build_inst(Op::AND, cond, res_def);
	bb->build_inst(Op::UB, cond);
	Op op = is_unsigned ? Op::UREM : Op::SREM;
	return {bb->build_inst(op, arg1, arg2), res_undef, nullptr};
      }
    case NE_EXPR:
      {
	auto [res_undef, res_def] =
	  get_res_undef(arg1_undef, arg2_undef, lhs_type, bb);
	if (res_undef)
	  {
	    // The result is defined if the value of the undefined bits
	    // does not matter. That is, if there are bits that are defined
	    // in both arg1 and arg2, and at least one of those bits differs
	    // between arg1 and arg2, then the result is defined.
	    if (!arg1_undef)
	      arg1_undef = bb->value_inst(0, arg1->bitsize);
	    if (!arg2_undef)
	      arg2_undef = bb->value_inst(0, arg1->bitsize);
	    Instruction *arg1_mask = bb->build_inst(Op::NOT, arg1_undef);
	    Instruction *arg2_mask = bb->build_inst(Op::NOT, arg2_undef);
	    Instruction *mask = bb->build_inst(Op::AND, arg1_mask, arg2_mask);
	    Instruction *zero = bb->value_m1_inst(arg1->bitsize);
	    Instruction *c1 = bb->build_inst(Op::EQ, mask, zero);
	    Instruction *a1 = bb->build_inst(Op::AND, arg1, arg1_mask);
	    Instruction *a2 = bb->build_inst(Op::AND, arg2, arg2_mask);
	    Instruction *c2 = bb->build_inst(Op::EQ, a1, a2);
	    Instruction *o = bb->build_inst(Op::OR, c1, c2);
	    res_undef = bb->build_inst(Op::AND, o, res_undef);
	  }
	return {bb->build_inst(Op::NE, arg1, arg2), res_undef, nullptr};
      }
    default:
      break;
    }

  // Handle instructions where the result is undef if any input bit is undef.
  auto [res_undef, res_def] = get_res_undef(arg1_undef, arg2_undef, lhs_type,
					    bb);
  switch (code)
    {
    case MAX_EXPR:
      {
	if ((arg1_prov || arg2_prov) && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in MAX_EXPR");
	Op op = is_unsigned ? Op::UMAX : Op::SMAX;
	return {bb->build_inst(op, arg1, arg2), res_undef, arg1_prov};
      }
    case MIN_EXPR:
      {
	if ((arg1_prov || arg2_prov) && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in MIN_EXPR");
	Op op = is_unsigned ? Op::UMIN : Op::SMIN;
	return {bb->build_inst(op, arg1, arg2), res_undef, arg1_prov};
      }
    case POINTER_PLUS_EXPR:
      {
	assert(arg1_prov);
	arg2 = type_convert(arg2, arg2_type, arg1_type, bb);
	Instruction *ptr = bb->build_inst(Op::ADD, arg1, arg2);

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Instruction *sub_overflow = bb->build_inst(Op::UGT, ptr, arg1);
	    Instruction *add_overflow = bb->build_inst(Op::ULT, ptr, arg1);
	    Instruction *zero = bb->value_inst(0, arg2->bitsize);
	    Instruction *is_sub = bb->build_inst(Op::SLT, arg2, zero);
	    Instruction *is_ub =
	      bb->build_inst(Op::ITE, is_sub, sub_overflow, add_overflow);
	    if (res_def)
	      is_ub = bb->build_inst(Op::AND, is_ub, res_def);
	    bb->build_inst(Op::UB, is_ub);
	  }

	// The resulting pointer cannot be NULL if arg1 is non-zero.
	// (However, GIMPLE allows NULL + 0).
	{
	  Instruction *zero = bb->value_inst(0, ptr->bitsize);
	  Instruction *cond1 = bb->build_inst(Op::EQ, ptr, zero);
	  Instruction *cond2 = bb->build_inst(Op::NE, arg1, zero);
	  Instruction *is_ub = bb->build_inst(Op::AND, cond1, cond2);
	  if (res_def)
	    is_ub = bb->build_inst(Op::AND, is_ub, res_def);
	  bb->build_inst(Op::UB, is_ub);
	}

	return {ptr, res_undef, arg1_prov};
      }
    case MINUS_EXPR:
      {
	Instruction *prov = nullptr;
	if (arg1_prov && arg2_prov && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in MINUS_EXPR");
	if (arg1_prov)
	  prov = arg1_prov;
	if (arg2_prov)
	  prov = arg2_prov;

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Instruction *cond = bb->build_inst(Op::SSUB_WRAPS, arg1, arg2);
	    if (res_def)
	      cond = bb->build_inst(Op::AND, cond, res_def);
	    bb->build_inst(Op::UB, cond);
	  }
	return {bb->build_inst(Op::SUB, arg1, arg2), res_undef, prov};
      }
    case PLUS_EXPR:
      {
	Instruction *prov = nullptr;
	if (arg1_prov && arg2_prov && arg1_prov != arg2_prov)
	  throw Not_implemented("two different provenance in PLUS_EXPR");
	if (arg1_prov)
	  prov = arg1_prov;
	if (arg2_prov)
	  prov = arg2_prov;

	if (!ignore_overflow && !TYPE_OVERFLOW_WRAPS(lhs_type))
	  {
	    Instruction *cond = bb->build_inst(Op::SADD_WRAPS, arg1, arg2);
	    if (res_def)
	      cond = bb->build_inst(Op::AND, cond, res_def);
	    bb->build_inst(Op::UB, cond);
	  }
	return {bb->build_inst(Op::ADD, arg1, arg2), res_undef, prov};
      }
    case EQ_EXPR:
      return {bb->build_inst(Op::EQ, arg1, arg2), res_undef, nullptr};
    case GE_EXPR:
      {
	Op op = is_unsigned ? Op::UGE : Op::SGE;
	return {bb->build_inst(op, arg1, arg2), res_undef, nullptr};
      }
    case GT_EXPR:
      {
	Op op = is_unsigned ? Op::UGT : Op::SGT;
	return {bb->build_inst(op, arg1, arg2), res_undef, nullptr};
      }
    case LE_EXPR:
      {
	Op op = is_unsigned ? Op::ULE : Op::SLE;
	return {bb->build_inst(op, arg1, arg2), res_undef, nullptr};
      }
    case LT_EXPR:
      {
	Op op = is_unsigned ? Op::ULT : Op::SLT;
	return {bb->build_inst(op, arg1, arg2), res_undef, nullptr};
      }
    case LSHIFT_EXPR:
      {
	Instruction *bitsize = bb->value_inst(arg1->bitsize, arg2->bitsize);
	Instruction *cond = bb->build_inst(Op::UGE, arg2, bitsize);
	if (res_def)
	  cond = bb->build_inst(Op::AND, cond, res_def);
	bb->build_inst(Op::UB, cond);
      }
      arg2 = type_convert(arg2, arg2_type, arg1_type, bb);
      return {bb->build_inst(Op::SHL, arg1, arg2), res_undef, nullptr};
    case POINTER_DIFF_EXPR:
      {
	// Pointers are treated as unsigned, and the result must fit in
	// a signed integer of the same width.
	assert(arg1->bitsize == arg2->bitsize);
	Instruction *ext_bitsize_inst = bb->value_inst(arg1->bitsize + 1, 32);
	Instruction *earg1 = bb->build_inst(Op::ZEXT, arg1, ext_bitsize_inst);
	Instruction *earg2 = bb->build_inst(Op::ZEXT, arg2, ext_bitsize_inst);
	Instruction *eres = bb->build_inst(Op::SUB, earg1, earg2);
	int bitsize = arg1->bitsize;
	Instruction *etop_bit_idx = bb->value_inst(bitsize, 32);
	Instruction *etop_bit =
	  bb->build_inst(Op::EXTRACT, eres, etop_bit_idx, etop_bit_idx);
	Instruction *top_bit_idx = bb->value_inst(bitsize - 1, 32);
	Instruction *top_bit =
	  bb->build_inst(Op::EXTRACT, eres, top_bit_idx, top_bit_idx);
	Instruction *cmp = bb->build_inst(Op::NE, top_bit, etop_bit);
	if (res_def)
	  cmp = bb->build_inst(Op::AND, cmp, res_def);
	bb->build_inst(Op::UB, cmp);
	return {bb->build_trunc(eres, bitsize), res_undef, nullptr};
      }
    case RROTATE_EXPR:
      {
	Instruction *bitsize = bb->value_inst(arg1->bitsize, arg2->bitsize);
	Instruction *cond = bb->build_inst(Op::UGE, arg2, bitsize);
	if (res_def)
	  cond = bb->build_inst(Op::AND, cond, res_def);
	bb->build_inst(Op::UB, cond);
	arg2 = type_convert(arg2, arg2_type, arg1_type, bb);
	Instruction *concat = bb->build_inst(Op::CONCAT, arg1, arg1);
	Instruction *bitsize_inst = bb->value_inst(concat->bitsize, 32);
	Instruction *shift = bb->build_inst(Op::ZEXT, arg2, bitsize_inst);
	Instruction *shifted = bb->build_inst(Op::LSHR, concat, shift);
	return {bb->build_trunc(shifted, arg1->bitsize), res_undef, nullptr};
      }
    case LROTATE_EXPR:
      {
	Instruction *bitsize = bb->value_inst(arg1->bitsize, arg2->bitsize);
	Instruction *cond = bb->build_inst(Op::UGE, arg2, bitsize);
	if (res_def)
	  cond = bb->build_inst(Op::AND, cond, res_def);
	bb->build_inst(Op::UB, cond);
	arg2 = type_convert(arg2, arg2_type, arg1_type, bb);
	Instruction *concat = bb->build_inst(Op::CONCAT, arg1, arg1);
	Instruction *bitsize_inst = bb->value_inst(concat->bitsize, 32);
	Instruction *shift = bb->build_inst(Op::ZEXT, arg2, bitsize_inst);
	Instruction *shifted = bb->build_inst(Op::SHL, concat, shift);
	Instruction *high = bb->value_inst(2 * arg1->bitsize - 1, 32);
	Instruction *low = bb->value_inst(arg1->bitsize, 32);
	Instruction *ret = bb->build_inst(Op::EXTRACT, shifted, high, low);
	return {ret, res_undef, nullptr};
      }
    case RSHIFT_EXPR:
      {
	Instruction *bitsize = bb->value_inst(arg1->bitsize, arg2->bitsize);
	Instruction *cond = bb->build_inst(Op::UGE, arg2, bitsize);
	if (res_def)
	  cond = bb->build_inst(Op::AND, cond, res_def);
	bb->build_inst(Op::UB, cond);
	Op op = is_unsigned ? Op::LSHR : Op::ASHR;
	arg2 = type_convert(arg2, arg2_type, arg1_type, bb);
	return {bb->build_inst(op, arg1, arg2), res_undef, nullptr};
      }
    case WIDEN_MULT_EXPR:
      {
	assert(arg1->bitsize == arg2->bitsize);
	Instruction *new_bitsize_inst =
	  bb->value_inst(bitsize_for_type(lhs_type), 32);
	Op op1 = TYPE_UNSIGNED(arg1_type) ? Op::ZEXT : Op::SEXT;
	arg1 = bb->build_inst(op1, arg1, new_bitsize_inst);
	Op op2 = TYPE_UNSIGNED(arg2_type) ? Op::ZEXT : Op::SEXT;
	arg2 = bb->build_inst(op2, arg2, new_bitsize_inst);
	return {bb->build_inst(Op::MUL, arg1, arg2), res_undef, nullptr};
      }
    case MULT_HIGHPART_EXPR:
      {
	assert(arg1->bitsize == arg2->bitsize);
	assert(TYPE_UNSIGNED(arg1_type) == TYPE_UNSIGNED(arg2_type));
	Instruction *new_bitsize_inst = bb->value_inst(2 * arg1->bitsize, 32);
	Op op = is_unsigned ? Op::ZEXT : Op::SEXT;
	arg1 = bb->build_inst(op, arg1, new_bitsize_inst);
	arg2 = bb->build_inst(op, arg2, new_bitsize_inst);
	Instruction *mul = bb->build_inst(Op::MUL, arg1, arg2);
	Instruction *high = bb->value_inst(mul->bitsize - 1, 32);
	Instruction *low = bb->value_inst(mul->bitsize / 2, 32);
	Instruction *res = bb->build_inst(Op::EXTRACT, mul, high, low);
	return {res, res_undef, nullptr};
      }
    default:
      break;
    }

  throw Not_implemented("process_binary_int: "s + get_tree_code_name(code));
}

Instruction *Converter::process_binary_scalar(enum tree_code code, Instruction *arg1, Instruction *arg2, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow)
{
  auto [inst, undef, prov] =
    process_binary_scalar(code, arg1, nullptr, nullptr, arg2, nullptr, nullptr, lhs_type, arg1_type, arg2_type, bb, ignore_overflow);
  assert(!undef);
  assert(!prov);
  return inst;
}

std::tuple<Instruction *, Instruction *, Instruction *> Converter::process_binary_scalar(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg1_prov, Instruction *arg2, Instruction *arg2_undef, Instruction *arg2_prov, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow)
{
  if (TREE_CODE(lhs_type) == BOOLEAN_TYPE)
    {
      auto [inst, undef] =
	process_binary_bool(code, arg1, arg1_undef, arg2, arg2_undef,
			    lhs_type, arg1_type, arg2_type, bb);
      return {inst, undef, nullptr};
    }
  else if (FLOAT_TYPE_P(lhs_type))
    {
      auto [inst, undef] =
	process_binary_float(code, arg1, arg1_undef, arg2, arg2_undef,
			     lhs_type, bb);
      return {inst, undef, nullptr};
    }
  else
    return process_binary_int(code, TYPE_UNSIGNED(arg1_type),
			      arg1, arg1_undef, arg1_prov, arg2, arg2_undef,
			      arg2_prov, lhs_type, arg1_type, arg2_type, bb,
			      ignore_overflow);
}

std::pair<Instruction *, Instruction *> Converter::process_binary_vec(enum tree_code code, Instruction *arg1, Instruction *arg1_undef, Instruction *arg2, Instruction *arg2_undef, tree lhs_type, tree arg1_type, tree arg2_type, Basic_block *bb, bool ignore_overflow)
{
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
      Instruction *arg = bb->build_inst(Op::CONCAT, arg2, arg1);
      Instruction *arg_undef = nullptr;
      if (arg1_undef || arg2_undef)
	{
	  if (!arg1_undef)
	    arg1_undef = bb->value_inst(0, arg1->bitsize);
	  if (!arg2_undef)
	    arg2_undef = bb->value_inst(0, arg2->bitsize);
	  arg_undef = bb->build_inst(Op::CONCAT, arg2_undef, arg1_undef);
	}
      return process_unary_vec(CONVERT_EXPR, arg, arg_undef, lhs_elem_type,
			       arg1_elem_type, bb);
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

  Instruction *res = nullptr;
  Instruction *res_undef = nullptr;
  for (uint64_t i = start_idx; i < nof_elt; i++)
    {
      Instruction *a1_undef = nullptr;
      Instruction *a2_undef = nullptr;
      Instruction *a1 = extract_vec_elem(bb, arg1, elem_bitsize, i);
      if (arg1_undef)
	a1_undef = extract_vec_elem(bb, arg1_undef, elem_bitsize, i);
      Instruction *a2;
      if (VECTOR_TYPE_P(arg2_type))
	{
	  a2 = extract_vec_elem(bb, arg2, elem_bitsize, i);
	  if (arg2_undef)
	    a2_undef = extract_vec_elem(bb, arg2_undef, elem_bitsize, i);
	}
      else
	{
	  a2 = arg2;
	  if (arg2_undef)
	    a2_undef = arg2_undef;
	}
      auto [inst, inst_undef, _] =
	process_binary_scalar(code, a1, a1_undef, nullptr, a2, a2_undef,
			      nullptr, lhs_elem_type, arg1_elem_type,
			      arg2_elem_type, bb, ignore_overflow);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg1_undef || arg2_undef)
	{
	  if (res_undef)
	    res_undef = bb->build_inst(Op::CONCAT, inst_undef, res_undef);
	  else
	    res_undef = inst_undef;
	}
    }
  return {res, res_undef};
}

Instruction *Converter::process_ternary(enum tree_code code, Instruction *arg1, Instruction *arg2, Instruction *arg3, tree arg1_type, tree arg2_type, tree arg3_type, Basic_block *bb)
{
  switch (code)
    {
    case SAD_EXPR:
      {
	arg1 = type_convert(arg1, arg1_type, arg3_type, bb);
	arg2 = type_convert(arg2, arg2_type, arg3_type, bb);
	Instruction *inst = bb->build_inst(Op::SUB, arg1, arg2);
	Instruction *zero = bb->value_inst(0, inst->bitsize);
	Instruction *cmp = bb->build_inst(Op::SGE, inst, zero);
	Instruction *neg = bb->build_inst(Op::NEG, inst);
	inst = bb->build_inst(Op::ITE, cmp, inst, neg);
	return bb->build_inst(Op::ADD, inst, arg3);
      }
    case DOT_PROD_EXPR:
      {
	arg1 = type_convert(arg1, arg1_type, arg3_type, bb);
	arg2 = type_convert(arg2, arg2_type, arg3_type, bb);
	Instruction *inst = bb->build_inst(Op::MUL, arg1, arg2);
	return bb->build_inst(Op::ADD, inst, arg3);
      }
    default:
      throw Not_implemented("process_ternary: "s + get_tree_code_name(code));
    }
}

Instruction *Converter::process_ternary_vec(enum tree_code code, Instruction *arg1, Instruction *arg2, Instruction *arg3, tree lhs_type, tree arg1_type, tree arg2_type, tree arg3_type, Basic_block *bb)
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
  Instruction *res = nullptr;
  for (uint64_t i = 0; i < nof_elt; i++)
    {
      Instruction *a1 = extract_vec_elem(bb, arg1, arg1_elem_bitsize, i);
      Instruction *a2 = extract_vec_elem(bb, arg2, arg2_elem_bitsize, i);
      // Instructions such as SAD_EXPR has fewer elements in the arg3,
      // and it iterates multiple times and updates that.
      uint32_t i3 = i % nof_elt3;
      if (!i3 && res)
	{
	  arg3 = res;
	  res = nullptr;
	}
      Instruction *a3 = extract_vec_elem(bb, arg3, arg3_elem_bitsize, i3);
      Instruction *inst = process_ternary(code, a1, a2, a3, arg1_elem_type,
					  arg2_elem_type, arg3_elem_type, bb);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  return res;
}

std::pair<Instruction *, Instruction *> Converter::process_vec_cond(Instruction *arg1, Instruction *arg2, Instruction *arg2_undef, Instruction *arg3, Instruction *arg3_undef, tree arg1_type, tree arg2_type, Basic_block *bb)
{
  assert(VECTOR_TYPE_P(arg1_type));
  assert(VECTOR_TYPE_P(arg2_type));
  assert(arg2->bitsize == arg3->bitsize);

  if (arg2_undef || arg3_undef)
    {
      if (!arg2_undef)
	arg2_undef = bb->value_inst(0, arg2->bitsize);
      if (!arg3_undef)
	arg3_undef = bb->value_inst(0, arg3->bitsize);
    }

  tree arg1_elem_type = TREE_TYPE(arg1_type);
  assert(TREE_CODE(arg1_elem_type) == BOOLEAN_TYPE);
  tree arg2_elem_type = TREE_TYPE(arg2_type);

  uint32_t elem_bitsize1 = bitsize_for_type(arg1_elem_type);
  uint32_t elem_bitsize2 = bitsize_for_type(arg2_elem_type);

  Instruction *res = nullptr;
  Instruction *res_undef = nullptr;
  uint32_t nof_elt = bitsize_for_type(arg1_type) / elem_bitsize1;
  for (uint64_t i = 0; i < nof_elt; i++)
    {
      Instruction *a1 = extract_vec_elem(bb, arg1, elem_bitsize1, i);
      if (TYPE_PRECISION(arg1_elem_type) != 1)
	a1 = bb->build_extract_bit(a1, 0);
      Instruction *a2 = extract_vec_elem(bb, arg2, elem_bitsize2, i);
      Instruction *a3 = extract_vec_elem(bb, arg3, elem_bitsize2, i);

      if (arg2_undef)
	{
	  Instruction *a2_undef =
	    extract_vec_elem(bb, arg2_undef, elem_bitsize2, i);
	  Instruction *a3_undef =
	    extract_vec_elem(bb, arg3_undef, elem_bitsize2, i);
	  Instruction *undef =
	    bb->build_inst(Op::ITE, a1, a2_undef, a3_undef);

	  if (res_undef)
	    res_undef = bb->build_inst(Op::CONCAT, undef, res_undef);
	  else
	    res_undef = undef;
	}

      Instruction *inst = bb->build_inst(Op::ITE, a1, a2, a3);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }

  return {res, res_undef};
}

std::pair<Instruction *, Instruction *> Converter::process_vec_perm_expr(gimple *stmt, Basic_block *bb)
{
  auto [arg1, arg1_undef] = tree2inst_undef(bb, gimple_assign_rhs1(stmt));
  auto [arg2, arg2_undef] = tree2inst_undef(bb, gimple_assign_rhs2(stmt));
  Instruction *arg3 = tree2inst(bb, gimple_assign_rhs3(stmt));
  assert(arg1->bitsize == arg2->bitsize);
  tree arg1_type = TREE_TYPE(gimple_assign_rhs1(stmt));
  tree arg1_elem_type = TREE_TYPE(arg1_type);
  tree arg3_type = TREE_TYPE(gimple_assign_rhs3(stmt));
  tree arg3_elem_type = TREE_TYPE(arg3_type);
  uint32_t elem_bitsize1 = bitsize_for_type(arg1_elem_type);
  uint32_t elem_bitsize3 = bitsize_for_type(arg3_elem_type);
  uint32_t nof_elt1 = bitsize_for_type(arg1_type) / elem_bitsize1;
  uint32_t nof_elt3 = bitsize_for_type(arg3_type) / elem_bitsize3;

  if (arg1_undef || arg2_undef)
    {
      if (!arg1_undef)
	arg1_undef = bb->value_inst(0, arg1->bitsize);
      if (!arg2_undef)
	arg2_undef = bb->value_inst(0, arg2->bitsize);
    }

  Instruction *mask1 = bb->value_inst(nof_elt1 * 2 - 1, elem_bitsize3);
  Instruction *mask2 = bb->value_inst(nof_elt1 - 1, elem_bitsize3);
  Instruction *nof_elt_inst = bb->value_inst(nof_elt1, elem_bitsize3);
  Instruction *res = nullptr;
  Instruction *res_undef = nullptr;
  for (uint64_t i = 0; i < nof_elt3; i++)
    {
      Instruction *idx1 = extract_vec_elem(bb, arg3, elem_bitsize3, i);
      idx1 = bb->build_inst(Op::AND, idx1, mask1);
      Instruction *idx2 = bb->build_inst(Op::AND, idx1, mask2);
      Instruction *cmp = bb->build_inst(Op::ULT, idx1,  nof_elt_inst);
      Instruction *elt1 = extract_elem(bb, arg1, elem_bitsize1, idx2);
      Instruction *elt2 = extract_elem(bb, arg2, elem_bitsize1, idx2);
      Instruction *inst = bb->build_inst(Op::ITE, cmp, elt1, elt2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg1_undef)
	{
	  Instruction *undef1 =
	    extract_elem(bb, arg1_undef, elem_bitsize1, idx2);
	  Instruction *undef2 =
	    extract_elem(bb, arg2_undef, elem_bitsize1, idx2);
	  Instruction *undef = bb->build_inst(Op::ITE, cmp, undef1, undef2);
	  if (res_undef)
	    res_undef = bb->build_inst(Op::CONCAT, undef, res_undef);
	  else
	    res_undef = undef;
	}
    }
  return {res, res_undef};
}

std::tuple<Instruction *, Instruction *, Instruction *> Converter::vector_constructor(Basic_block *bb, tree expr)
{
  assert(TREE_CODE(expr) == CONSTRUCTOR);
  assert(VECTOR_TYPE_P(TREE_TYPE(expr)));
  unsigned HOST_WIDE_INT idx;
  tree value;
  uint32_t vector_size = bytesize_for_type(TREE_TYPE(expr)) * 8;
  Instruction *res = nullptr;
  Instruction *undef = nullptr;
  bool any_elem_has_undef = false;
  // Note: The contstuctor elements may have different sizes. For example,
  // we may create a vector by concatenating a scalar with a vector.
  FOR_EACH_CONSTRUCTOR_VALUE(CONSTRUCTOR_ELTS(expr), idx, value)
    {
      auto [elem, elem_undef] = tree2inst_undef(bb, value);
      if (elem_undef)
	{
	  any_elem_has_undef = true;
	}
      else
	elem_undef = bb->value_inst(0, elem->bitsize);
      if (res)
	{
	  res = bb->build_inst(Op::CONCAT, elem, res);
	  undef = bb->build_inst(Op::CONCAT, elem_undef, undef);
	}
      else
	{
	  assert(idx == 0);
	  res = elem;
	  undef = elem_undef;
	}
    }
  if (CONSTRUCTOR_NO_CLEARING(expr))
    throw Not_implemented("vector_constructor: CONSTRUCTOR_NO_CLEARING");
  if (!res)
    res = bb->value_inst(0, vector_size);
  else if (res->bitsize != vector_size)
    {
      assert(res->bitsize < vector_size);
      Instruction *zero = bb->value_inst(0, vector_size - res->bitsize);
      res = bb->build_inst(Op::CONCAT, zero, res);
      undef = bb->build_inst(Op::CONCAT, zero, undef);
    }
  if (!any_elem_has_undef)
    {
      // No element had undef information, so `undef` only consists of the
      // zero values we creates. Change it to the `nullptr` so that later
      // code does not need to add UB comparisions each place the result
      // is used.
      undef = nullptr;
    }
  return {res, undef, nullptr};
}

void Converter::process_constructor(tree lhs, tree rhs, Basic_block *bb)
{
  Addr addr = process_address(bb, lhs, true);
  assert(!addr.bitoffset);

  if (TREE_CLOBBER_P(rhs) && CLOBBER_KIND(rhs) == CLOBBER_STORAGE_END)
    {
      bb->build_inst(Op::FREE, bb->build_extract_id(addr.ptr));
      return;
    }

  assert(!CONSTRUCTOR_NO_CLEARING(rhs));
  uint64_t size = bytesize_for_type(TREE_TYPE(rhs));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("process_constructor: too large constructor");
  store_ub_check(bb, addr.ptr, addr.provenance, size);

  if (TREE_CLOBBER_P(rhs))
    make_uninit(bb, addr.ptr, size);
  else
    {
      Instruction *zero = bb->value_inst(0, 8);
      Instruction *memory_flag = bb->value_inst(0, 1);
      for (uint64_t i = 0; i < size; i++)
	{
	  Instruction *offset = bb->value_inst(i, addr.ptr->bitsize);
	  Instruction *ptr = bb->build_inst(Op::ADD, addr.ptr, offset);
	  uint8_t padding = padding_at_offset(TREE_TYPE(rhs), i);
	  Instruction *undef = bb->value_inst(padding, 8);
	  bb->build_inst(Op::STORE, ptr, zero);
	  bb->build_inst(Op::SET_MEM_UNDEF, ptr, undef);
	  bb->build_inst(Op::SET_MEM_FLAG, ptr, memory_flag);
	}
    }

  assert(!CONSTRUCTOR_NELTS(rhs));
}

void Converter::process_gimple_assign(gimple *stmt, Basic_block *bb)
{
  tree lhs = gimple_assign_lhs(stmt);
  check_type(TREE_TYPE(lhs));
  enum tree_code code = gimple_assign_rhs_code(stmt);

  if (TREE_CODE(lhs) != SSA_NAME)
    {
      assert(get_gimple_rhs_class(code) == GIMPLE_SINGLE_RHS);
      tree rhs = gimple_assign_rhs1(stmt);
      if (TREE_CODE(rhs) == CONSTRUCTOR)
	process_constructor(lhs, rhs, bb);
      else
	process_store(lhs, rhs, bb);
      return;
    }

  tree rhs1 = gimple_assign_rhs1(stmt);
  check_type(TREE_TYPE(rhs1));
  Instruction *inst;
  Instruction *undef = nullptr;
  Instruction *provenance = nullptr;
  switch (get_gimple_rhs_class(code))
    {
    case GIMPLE_TERNARY_RHS:
      {
	if (code == SAD_EXPR || code == DOT_PROD_EXPR)
	  {
	    Instruction *arg1 = tree2inst(bb, gimple_assign_rhs1(stmt));
	    Instruction *arg2 = tree2inst(bb, gimple_assign_rhs2(stmt));
	    Instruction *arg3 = tree2inst(bb, gimple_assign_rhs3(stmt));
	    tree lhs_type = TREE_TYPE(gimple_assign_lhs(stmt));
	    tree arg1_type = TREE_TYPE(gimple_assign_rhs1(stmt));
	    tree arg2_type = TREE_TYPE(gimple_assign_rhs2(stmt));
	    tree arg3_type = TREE_TYPE(gimple_assign_rhs3(stmt));
	    if (VECTOR_TYPE_P(lhs_type))
	      inst = process_ternary_vec(code, arg1, arg2, arg3, lhs_type, arg1_type, arg2_type, arg3_type, bb);
	    else
	      inst = process_ternary(code, arg1, arg2, arg3, arg1_type, arg2_type, arg3_type, bb);
	  }
	else if (code == VEC_PERM_EXPR)
	  std::tie(inst, undef) = process_vec_perm_expr(stmt, bb);
	else if (code == VEC_COND_EXPR)
	  {
	    Instruction *arg1 = tree2inst(bb, gimple_assign_rhs1(stmt));
	    auto [arg2, arg2_undef] =
	      tree2inst_undef(bb, gimple_assign_rhs2(stmt));
	    auto [arg3, arg3_undef] =
	      tree2inst_undef(bb, gimple_assign_rhs3(stmt));
	    tree arg1_type = TREE_TYPE(gimple_assign_rhs1(stmt));
	    tree arg2_type = TREE_TYPE(gimple_assign_rhs2(stmt));
	    std::tie(inst, undef) =
	      process_vec_cond(arg1, arg2, arg2_undef, arg3, arg3_undef,
			       arg1_type, arg2_type, bb);
	  }
	else if (code == COND_EXPR)
	  {
	    tree rhs1 = gimple_assign_rhs1(stmt);
	    assert(TREE_CODE(TREE_TYPE(rhs1)) == BOOLEAN_TYPE);
	    Instruction *arg1 = tree2inst(bb, rhs1);
	    if (TYPE_PRECISION(TREE_TYPE(rhs1)) != 1)
	      arg1 = bb->build_extract_bit(arg1, 0);
	    auto [arg2, arg2_undef, arg2_prov] =
	      tree2inst_undef_prov(bb, gimple_assign_rhs2(stmt));
	    auto [arg3, arg3_undef, arg3_prov] =
	      tree2inst_undef_prov(bb, gimple_assign_rhs3(stmt));
	    if (arg2_undef || arg3_undef)
	      {
		if (!arg2_undef)
		  arg2_undef = bb->value_inst(0, arg2->bitsize);
		if (!arg3_undef)
		  arg3_undef = bb->value_inst(0, arg3->bitsize);
		undef =
		  bb->build_inst(Op::ITE, arg1, arg2_undef, arg3_undef);
	      }
	    if (arg2_prov && arg3_prov)
	      provenance = bb->build_inst(Op::ITE, arg1, arg2_prov, arg3_prov);
	    inst = bb->build_inst(Op::ITE, arg1, arg2, arg3);
	  }
	else if (code == BIT_INSERT_EXPR)
	  {
	    auto [arg1, arg1_undef] =
	      tree2inst_undef(bb, gimple_assign_rhs1(stmt));
	    tree arg2_expr = gimple_assign_rhs2(stmt);
	    auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
	    bool has_undef = arg1_undef || arg2_undef;
	    if (has_undef)
	      {
		if (!arg1_undef)
		  arg1_undef = bb->value_inst(0, arg1->bitsize);
		if (!arg2_undef)
		  arg2_undef = bb->value_inst(0, arg2->bitsize);
	      }
	    uint64_t bit_pos = get_int_cst_val(gimple_assign_rhs3(stmt));
	    if (bit_pos > 0)
	      {
		Instruction *extract = bb->build_trunc(arg1, bit_pos);
		inst = bb->build_inst(Op::CONCAT, arg2, extract);
		if (has_undef)
		  {
		    Instruction *extract_undef =
		      bb->build_trunc(arg1_undef, bit_pos);
		    undef =
		      bb->build_inst(Op::CONCAT, arg2_undef, extract_undef);
		  }
	      }
	    else
	      {
		inst = arg2;
		if (has_undef)
		  undef = arg2_undef;
	      }
	    if (bit_pos + arg2->bitsize != arg1->bitsize)
	      {
		Instruction *high = bb->value_inst(arg1->bitsize - 1, 32);
		Instruction *low = bb->value_inst(bit_pos + arg2->bitsize, 32);
		Instruction *extract =
		  bb->build_inst(Op::EXTRACT, arg1, high, low);
		inst = bb->build_inst(Op::CONCAT, extract, inst);
		if (has_undef)
		  {
		    Instruction *extract_undef =
		      bb->build_inst(Op::EXTRACT, arg1_undef, high, low);
		    undef = bb->build_inst(Op::CONCAT, extract_undef, undef);
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
	    auto [arg1, arg1_undef] = tree2inst_undef(bb, rhs1);
	    auto [arg2, arg2_undef] = tree2inst_undef(bb, rhs2);
	    arg1 = to_mem_repr(bb, arg1, TREE_TYPE(rhs1));
	    arg2 = to_mem_repr(bb, arg2, TREE_TYPE(rhs2));
	    inst = bb->build_inst(Op::CONCAT, arg2, arg1);
	    if (arg1_undef || arg2_undef)
	      {
		if (!arg1_undef)
		  arg1_undef = bb->value_inst(0, arg1->bitsize);
		if (!arg2_undef)
		  arg2_undef = bb->value_inst(0, arg2->bitsize);
		undef =
		  bb->build_inst(Op::CONCAT, arg2_undef, arg1_undef);
	      }
	  }
	else
	  {
	    if (TREE_CODE(lhs_type) == COMPLEX_TYPE)
	      {
		Instruction *arg1 = tree2inst(bb, rhs1);
		Instruction *arg2 = tree2inst(bb, rhs2);
		inst = process_binary_complex(code, arg1, arg2, lhs_type, bb);
	      }
	    else if (TREE_CODE(arg1_type) == COMPLEX_TYPE)
	      {
		Instruction *arg1 = tree2inst(bb, rhs1);
		Instruction *arg2 = tree2inst(bb, rhs2);
		inst = process_binary_complex_cmp(code, arg1, arg2, lhs_type,
						  arg1_type, bb);
	      }
	    else if (VECTOR_TYPE_P(lhs_type))
	      {
		auto [arg1, arg1_undef] = tree2inst_undef(bb, rhs1);
		auto [arg2, arg2_undef] = tree2inst_undef(bb, rhs2);
		std::tie(inst, undef) =
		  process_binary_vec(code, arg1, arg1_undef, arg2,
				     arg2_undef, lhs_type, arg1_type,
				     arg2_type, bb);
	      }
	    else
	      {
		auto [arg1, arg1_undef, arg1_prov] =
		  tree2inst_undef_prov(bb, rhs1);
		auto [arg2, arg2_undef, arg2_prov] =
		  tree2inst_undef_prov(bb, rhs2);
		std::tie(inst, undef, provenance) =
		  process_binary_scalar(code, arg1, arg1_undef, arg1_prov,
					arg2, arg2_undef, arg2_prov, lhs_type,
					arg1_type, arg2_type, bb);
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
	    Instruction *arg1 = tree2inst(bb, rhs1);
	    inst = process_unary_complex(code, arg1, lhs_type, bb);
	  }
	else if (VECTOR_TYPE_P(lhs_type))
	  {
	    auto [arg1, arg1_undef] = tree2inst_undef(bb, rhs1);
	    tree lhs_elem_type = TREE_TYPE(lhs_type);
	    tree arg1_elem_type = TREE_TYPE(arg1_type);
	    std::tie(inst, undef) =
	      process_unary_vec(code, arg1, arg1_undef, lhs_elem_type,
				arg1_elem_type, bb);
	  }
	else
	  {
	    auto [arg1, arg1_undef, arg1_prov] = tree2inst_undef_prov(bb, rhs1);
	    std::tie(inst, undef, provenance) =
	      process_unary_scalar(code, arg1, arg1_undef, arg1_prov, lhs_type,
				   arg1_type, bb);
	  }
      }
      break;
    case GIMPLE_SINGLE_RHS:
      std::tie(inst, undef, provenance) =
	tree2inst_undef_prov(bb, gimple_assign_rhs1(stmt));
      break;
    default:
      throw Not_implemented("unknown get_gimple_rhs_class");
    }

  constrain_range(bb, lhs, inst, undef);

  assert(TREE_CODE(lhs) == SSA_NAME);
  tree2instruction.insert({lhs, inst});
  if (undef)
    tree2undef.insert({lhs, undef});
  if (provenance)
    tree2provenance.insert({lhs, provenance});
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
  // number of EGHE_COUNT preds/succs. This is easy to fix, but does
  // not give us any benefit until we have real asm handling.
  if (gimple_asm_nlabels(asm_stmt))
    throw Not_implemented("process_function: gimple_asm");

  // TODO: This is not completely correct for asm having output, such as
  //   asm volatile ("" : "+rm" (p));
  // This will create a new SSA value for the output, which we will then
  // treat as an uninitialized variable. We should instead do something,
  // such as creating a new symbolic value. But it is annoying to maintain
  // this in sync between src and tgt.
}

void Converter::process_cfn_add_overflow(gimple *stmt, Basic_block *bb)
{
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
  auto [arg1, arg1_undef] = tree2inst_undef(bb, arg1_expr);
  auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  auto [res_undef, res_def] =
    get_res_undef(arg1_undef, arg2_undef, TREE_TYPE(lhs), bb);
  unsigned lhs_elem_bitsize = bitsize_for_type(lhs_elem_type);
  unsigned bitsize = 1 + std::max(arg1->bitsize, arg2->bitsize);
  bitsize = 1 + std::max(bitsize, lhs_elem_bitsize);
  Instruction *bitsize_inst = bb->value_inst(bitsize, 32);
  if (TYPE_UNSIGNED(arg1_type))
    arg1 = bb->build_inst(Op::ZEXT, arg1, bitsize_inst);
  else
    arg1 = bb->build_inst(Op::SEXT, arg1, bitsize_inst);
  if (TYPE_UNSIGNED(arg2_type))
    arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize_inst);
  else
    arg2 = bb->build_inst(Op::SEXT, arg2, bitsize_inst);
  Instruction *inst = bb->build_inst(Op::ADD, arg1, arg2);
  Instruction *res = bb->build_trunc(inst, lhs_elem_bitsize);
  Instruction *eres;
  if (TYPE_UNSIGNED(lhs_elem_type))
    eres = bb->build_inst(Op::ZEXT, res, bitsize_inst);
  else
    eres = bb->build_inst(Op::SEXT, res, bitsize_inst);
  Instruction *overflow = bb->build_inst(Op::NE, inst, eres);

  res = to_mem_repr(bb, res, lhs_elem_type);
  Instruction *res_bitsize_inst = bb->value_inst(res->bitsize, 32);
  overflow = bb->build_inst(Op::ZEXT, overflow, res_bitsize_inst);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_undef)
    tree2undef.insert({lhs, res_undef});
}

void Converter::process_cfn_assume_aligned(gimple *stmt, Basic_block *bb)
{
  auto [arg1, arg1_prov] = tree2inst_prov(bb, gimple_call_arg(stmt, 0));
  Instruction *arg2 = tree2inst(bb, gimple_call_arg(stmt, 1));
  assert(arg1->bitsize == arg2->bitsize);
  Instruction *one = bb->value_inst(1, arg2->bitsize);
  Instruction *mask = bb->build_inst(Op::SUB, arg2, one);
  Instruction *val = bb->build_inst(Op::AND, arg1, mask);
  Instruction *zero = bb->value_inst(0, val->bitsize);
  Instruction *cond = bb->build_inst(Op::NE, val, zero);
  bb->build_inst(Op::UB, cond);
  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      constrain_range(bb, lhs, arg1);
      tree2instruction.insert({lhs, arg1});
      tree2provenance.insert({lhs, arg1_prov});
    }
}

void Converter::process_cfn_bswap(gimple *stmt, Basic_block *bb)
{
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  auto [arg, arg_undef, arg_prov] =
    tree2inst_undef_prov(bb, gimple_call_arg(stmt, 0));
  // Determine the width from lhs as bswap16 has 32-bit arg.
  int bitwidth = TYPE_PRECISION(TREE_TYPE(lhs));
  Instruction *inst = bb->build_trunc(arg, 8);
  Instruction *inst_undef = nullptr;
  if (arg_undef)
    inst_undef = bb->build_trunc(arg_undef, 8);
  for (int i = 8; i < bitwidth; i += 8)
    {
      Instruction *high = bb->value_inst(i + 7, 32);
      Instruction *low = bb->value_inst(i, 32);
      Instruction *byte = bb->build_inst(Op::EXTRACT, arg, high, low);
      inst = bb->build_inst(Op::CONCAT, inst, byte);
      if (arg_undef)
	{
	  Instruction *byte_undef =
	    bb->build_inst(Op::EXTRACT, arg_undef, high, low);
	  inst_undef = bb->build_inst(Op::CONCAT, inst_undef, byte_undef);
	}
    }
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  if (inst_undef)
    tree2undef.insert({lhs, inst_undef});
  if (arg_prov)
    tree2provenance.insert({lhs, arg_prov});
}

void Converter::process_cfn_clrsb(gimple *stmt, Basic_block *bb)
{
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_clrsb: vector type");
  auto [arg, arg_undef] = tree2inst_undef(bb, gimple_call_arg(stmt, 0));
  auto [res_undef, res_def] = get_res_undef(arg_undef, TREE_TYPE(lhs), bb);
  assert(arg->bitsize > 1);
  int bitsize = bitsize_for_type(TREE_TYPE(lhs));
  Instruction *signbit = bb->build_extract_bit(arg, arg->bitsize - 1);
  Instruction *inst = bb->value_inst(arg->bitsize - 1, bitsize);
  for (unsigned i = 0; i < arg->bitsize - 1; i++)
    {
      Instruction *bit = bb->build_extract_bit(arg, i);
      Instruction *cmp = bb->build_inst(Op::NE, bit, signbit);
      Instruction *val = bb->value_inst(arg->bitsize - i - 2, bitsize);
      inst = bb->build_inst(Op::ITE, cmp, val, inst);
    }
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  if (res_undef)
    tree2undef.insert({lhs, res_undef});
}

void Converter::process_cfn_clz(gimple *stmt, Basic_block *bb)
{
  Instruction *arg = tree2inst(bb, gimple_call_arg(stmt, 0));
  int nargs = gimple_call_num_args(stmt);
  if (nargs == 1)
    {
      Instruction *zero = bb->value_inst(0, arg->bitsize);
      Instruction *ub = bb->build_inst(Op::EQ, arg, zero);
      bb->build_inst(Op::UB, ub);
    }
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_clz: vector type");
  int bitsize = bitsize_for_type(TREE_TYPE(lhs));
  Instruction *inst;
  if (nargs == 1)
    inst = bb->value_inst(arg->bitsize, bitsize);
  else
    inst = tree2inst(bb, gimple_call_arg(stmt, 1));
  for (unsigned i = 0; i < arg->bitsize; i++)
    {
      Instruction *bit = bb->build_extract_bit(arg, i);
      Instruction *val = bb->value_inst(arg->bitsize - i - 1, bitsize);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_cond(gimple *stmt, Basic_block *bb)
{
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  Instruction *arg1 = tree2inst(bb, arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  auto[arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  tree arg3_type = TREE_TYPE(arg3_expr);
  auto[arg3, arg3_undef] = tree2inst_undef(bb, arg3_expr);
  tree arg4_expr = gimple_call_arg(stmt, 3);
  tree arg4_type = TREE_TYPE(arg4_expr);
  auto[arg4, arg4_undef] = tree2inst_undef(bb, arg4_expr);
  tree lhs = gimple_call_lhs(stmt);

  tree_code code;
  switch (gimple_call_combined_fn(stmt))
    {
    case CFN_COND_ADD:
      code = PLUS_EXPR;
      break;
    case CFN_COND_AND:
      code = BIT_AND_EXPR;
      break;
    case CFN_COND_IOR:
      code = BIT_IOR_EXPR;
      break;
    case CFN_COND_MUL:
      code = MULT_EXPR;
      break;
    case CFN_COND_RDIV:
      code = RDIV_EXPR;
      break;
    case CFN_COND_SHL:
      code = LSHIFT_EXPR;
      break;
    case CFN_COND_SHR:
      code = RSHIFT_EXPR;
      break;
    case CFN_COND_SUB:
      code = MINUS_EXPR;
      break;
    default:
      {
	const char *name = internal_fn_name(gimple_call_internal_fn(stmt));
	throw Not_implemented("process_cfn_cond: "s + name);
      }
    }

  Instruction *op_inst;
  Instruction *op_undef = nullptr;
  // TODO: We ignore overflow for now, but we may need to modify this to
  // check for oveflow when the condition is true.
  if (VECTOR_TYPE_P(arg2_type))
    {
      std::tie(op_inst, op_undef) =
	process_binary_vec(code, arg2, arg2_undef, arg3, arg3_undef,
			   arg2_type, arg2_type, arg3_type, bb, true);
    }
  else
    {
      Instruction *op_prov;
      std::tie(op_inst, op_undef, op_prov) =
	process_binary_scalar(code, arg2, arg2_undef, nullptr,
			      arg3, arg3_undef, nullptr,
			      arg2_type, arg2_type, arg3_type, bb, true);
    }

  Instruction *ret_inst;
  Instruction *ret_undef = nullptr;
  if (VECTOR_TYPE_P(arg1_type))
    {
      std::tie(ret_inst, ret_undef) =
	process_vec_cond(arg1, op_inst, op_undef, arg4, arg4_undef,
			 arg1_type, arg4_type, bb);
    }
  else
    {
      assert(TREE_CODE(arg1_type) == BOOLEAN_TYPE);
      if (TYPE_PRECISION(arg1_type) != 1)
	arg1 = bb->build_extract_bit(arg1, 0);
      ret_inst = bb->build_inst(Op::ITE, arg1, op_inst, arg4);
      if (op_undef || arg4_undef)
	{
	  if (!op_undef)
	    op_undef = bb->value_inst(0, op_inst->bitsize);
	  if (!arg4_undef)
	    arg4_undef = bb->value_inst(0, arg4->bitsize);
	  ret_undef = bb->build_inst(Op::ITE, arg1, op_undef, arg4_undef);
	}
    }

  if (lhs)
    {
      tree2instruction.insert({lhs, ret_inst});
      if (ret_undef)
	tree2undef.insert({lhs, ret_undef});
    }
}

void Converter::process_cfn_copysign(gimple *stmt, Basic_block *bb)
{
  Instruction *arg1 = tree2inst(bb, gimple_call_arg(stmt, 0));
  Instruction *arg2 = tree2inst(bb, gimple_call_arg(stmt, 1));
  Instruction *signbit = bb->build_extract_bit(arg2, arg2->bitsize - 1);
  Instruction *res = bb->build_trunc(arg1, arg1->bitsize - 1);
  res = bb->build_inst(Op::CONCAT, signbit, res);

  tree lhs = gimple_call_lhs(stmt);
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_copysign: vector type");

  // SMT solvers has only one NaN value, so NEGATE_EXPR of NaN does not
  // change the value. This leads to incorrect reports of miscompilations
  // for transformations like -ABS_EXPR(x) -> .COPYSIGN(x, -1.0) because
  // copysign has introduceed a non-canonical NaN.
  // For now, treat copying the sign to NaN as always produce the original
  // canonical NaN.
  // TODO: Remove this when Op::IS_NONCANONICAL_NAN is removed.
  Instruction *is_nan = bb->build_inst(Op::IS_NAN, arg1);
  res = bb->build_inst(Op::ITE, is_nan, arg1, res);
  if (lhs)
    {
      constrain_range(bb, lhs, res);
      tree2instruction.insert({lhs, res});
    }
}

void Converter::process_cfn_ctz(gimple *stmt, Basic_block *bb)
{
  Instruction *arg = tree2inst(bb, gimple_call_arg(stmt, 0));
  int nargs = gimple_call_num_args(stmt);
  if (nargs == 1)
    {
      Instruction *zero = bb->value_inst(0, arg->bitsize);
      Instruction *ub = bb->build_inst(Op::EQ, arg, zero);
      bb->build_inst(Op::UB, ub);
    }
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_ctz: vector type");
  int bitsize = bitsize_for_type(TREE_TYPE(lhs));
  Instruction *inst;
  if (nargs == 1)
    inst = bb->value_inst(arg->bitsize, bitsize);
  else
    inst = tree2inst(bb, gimple_call_arg(stmt, 1));
  for (int i = arg->bitsize - 1; i >= 0; i--)
    {
      Instruction *bit = bb->build_extract_bit(arg, i);
      Instruction *val = bb->value_inst(i, bitsize);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_divmod(gimple *stmt, Basic_block *bb)
{
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
  Instruction *arg1 = tree2inst(bb, arg1_expr);
  Instruction *arg2 = tree2inst(bb, arg2_expr);
  Instruction *mod = process_binary_scalar(TRUNC_MOD_EXPR, arg1, arg2,
					   lhs_elem_type, arg1_type,
					   arg2_type, bb);
  mod = to_mem_repr(bb, mod, lhs_elem_type);
  Instruction *div = process_binary_scalar(TRUNC_DIV_EXPR, arg1, arg2,
					   lhs_elem_type, arg1_type,
					   arg2_type, bb);
  div = to_mem_repr(bb, div, lhs_elem_type);
  Instruction *inst = bb->build_inst(Op::CONCAT, mod, div);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_expect(gimple *stmt, Basic_block *bb)
{
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  Instruction *arg = tree2inst(bb, gimple_call_arg(stmt, 0));
  constrain_range(bb, lhs, arg);
  tree2instruction.insert({lhs, arg});
}

void Converter::process_cfn_ffs(gimple *stmt, Basic_block *bb)
{
  Instruction *arg = tree2inst(bb, gimple_call_arg(stmt, 0));
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_ffs: vector type");
  int bitsize = bitsize_for_type(TREE_TYPE(lhs));
  Instruction *inst;
  inst = bb->value_inst(0, bitsize);
  for (int i = arg->bitsize - 1; i >= 0; i--)
    {
      Instruction *bit = bb->build_extract_bit(arg, i);
      Instruction *val = bb->value_inst(i + 1, bitsize);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_fmax(gimple *stmt, Basic_block *bb)
{
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_fmax: vector type");
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  auto [arg1, arg1_undef] = tree2inst_undef(bb, arg1_expr);
  auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  auto [res_undef, res_def] =
    get_res_undef(arg1_undef, arg2_undef, TREE_TYPE(lhs), bb);
  Instruction *is_nan = bb->build_inst(Op::IS_NAN, arg2);
  Instruction *cmp = bb->build_inst(Op::FGT, arg1, arg2);
  Instruction *max1 = bb->build_inst(Op::ITE, cmp, arg1, arg2);
  Instruction *max2 = bb->build_inst(Op::ITE, is_nan, arg1, max1);

  // 0.0 and -0.0 is equal as floating point values, and fmax(0.0, -0.0)
  // may return eiter of them. But we treat them as 0.0 > -0.0 here,
  // otherwise we will report miscompilations when GCC switch the order
  // of the arguments.
  Instruction *zero = bb->value_inst(0, arg1->bitsize);
  Instruction *is_zero1 = bb->build_inst(Op::FEQ, arg1, zero);
  Instruction *is_zero2 = bb->build_inst(Op::FEQ, arg2, zero);
  Instruction *is_zero = bb->build_inst(Op::AND, is_zero1, is_zero2);
  Instruction *cmp2 = bb->build_inst(Op::SGT, arg1, arg2);
  Instruction *max3 = bb->build_inst(Op::ITE, cmp2, arg1, arg2);
  tree2instruction.insert({lhs, bb->build_inst(Op::ITE, is_zero, max3, max2)});
  if (res_undef)
    tree2undef.insert({lhs, res_undef});
}

void Converter::process_cfn_fmin(gimple *stmt, Basic_block *bb)
{
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_fmin: vector type");
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  auto [arg1, arg1_undef] = tree2inst_undef(bb, arg1_expr);
  auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  auto [res_undef, res_def] =
    get_res_undef(arg1_undef, arg2_undef, TREE_TYPE(lhs), bb);
  Instruction *is_nan = bb->build_inst(Op::IS_NAN, arg2);
  Instruction *cmp = bb->build_inst(Op::FLT, arg1, arg2);
  Instruction *min1 = bb->build_inst(Op::ITE, cmp, arg1, arg2);
  Instruction *min2 = bb->build_inst(Op::ITE, is_nan, arg1, min1);

  // 0.0 and -0.0 is equal as floating point values, and fmin(0.0, -0.0)
  // may return eiter of them. But we treat them as 0.0 > -0.0 here,
  // otherwise we will report miscompilations when GCC switch the order
  // of the arguments.
  Instruction *zero = bb->value_inst(0, arg1->bitsize);
  Instruction *is_zero1 = bb->build_inst(Op::FEQ, arg1, zero);
  Instruction *is_zero2 = bb->build_inst(Op::FEQ, arg2, zero);
  Instruction *is_zero = bb->build_inst(Op::AND, is_zero1, is_zero2);
  Instruction *cmp2 = bb->build_inst(Op::SLT, arg1, arg2);
  Instruction *min3 = bb->build_inst(Op::ITE, cmp2, arg1, arg2);
  tree2instruction.insert({lhs, bb->build_inst(Op::ITE, is_zero, min3, min2)});
  if (res_undef)
    tree2undef.insert({lhs, res_undef});
}

void Converter::process_cfn_loop_vectorized(gimple *stmt)
{
  tree lhs = gimple_call_lhs(stmt);
  assert(lhs);
  Basic_block *entry_bb = func->bbs[0];
  Instruction *cond;
  if (check_loop_vectorized)
    {
      if (!loop_vect_sym)
	{
	  Instruction *idx_inst = entry_bb->value_inst(LOOP_VECT_SYM_IDX, 32);
	  Instruction *bs_inst = entry_bb->value_inst(1, 32);
	  loop_vect_sym = entry_bb->build_inst(Op::SYMBOLIC, idx_inst, bs_inst);
	}
      cond = loop_vect_sym;
    }
  else
    cond = entry_bb->value_inst(0, 1);
  tree2instruction.insert({lhs, cond});
}

void Converter::process_cfn_mask_load(gimple *stmt, Basic_block *bb)
{
  tree ptr_expr = gimple_call_arg(stmt, 0);
  tree alignment_expr = gimple_call_arg(stmt, 1);
  tree mask_expr = gimple_call_arg(stmt, 2);
  tree mask_type = TREE_TYPE(mask_expr);
  tree lhs = gimple_call_lhs(stmt);
  assert(lhs);
  tree lhs_type = TREE_TYPE(lhs);

  auto [ptr, ptr_prov] = tree2inst_prov(bb, ptr_expr);
  Instruction *mask = tree2inst(bb, mask_expr);
  uint64_t size = bytesize_for_type(lhs_type);

  uint64_t alignment = get_int_cst_val(alignment_expr) / 8;
  if (alignment > 1)
    {
      uint32_t high_val = 0;
      for (;;)
	{
	  high_val++;
	  if (alignment == (1u << high_val))
	    break;
	}

      Instruction *extract = bb->build_trunc(ptr, high_val);
      Instruction *zero = bb->value_inst(0, high_val);
      Instruction *cond = bb->build_inst(Op::NE, extract, zero);
      bb->build_inst(Op::UB, cond);
    }

  assert(VECTOR_TYPE_P(lhs_type) == VECTOR_TYPE_P(mask_type));
  tree elem_type = VECTOR_TYPE_P(lhs_type) ? TREE_TYPE(lhs_type) : lhs_type;
  tree mask_elem_type =
    VECTOR_TYPE_P(mask_type) ? TREE_TYPE(mask_type) : mask_type;
  uint64_t elem_size = bytesize_for_type(elem_type);
  assert(TREE_CODE(mask_elem_type) == BOOLEAN_TYPE);
  uint64_t mask_elem_bitsize = bitsize_for_type(mask_elem_type);

  uint64_t nof_elem = size / elem_size;
  assert((size % elem_size) == 0);
  Instruction *inst = nullptr;
  Instruction *undef = nullptr;
  Instruction *mem_flags = nullptr;
  for (uint64_t i = 0; i < nof_elem; i++)
    {
      Instruction *cond = extract_vec_elem(bb, mask, mask_elem_bitsize, i);
      if (cond->bitsize != 1)
	cond = bb->build_trunc(cond, 1);

      Instruction *offset = bb->value_inst(i * elem_size, ptr->bitsize);
      Instruction *src_ptr = bb->build_inst(Op::ADD, ptr, offset);
      load_ub_check(bb, src_ptr, ptr_prov, elem_size, cond);
      auto [elem, elem_undef, elem_flags] = load_value(bb, src_ptr, elem_size);
      constrain_src_value(bb, elem, elem_type);
      Instruction *zero = bb->value_inst(0, elem->bitsize);
      Instruction *m1 = bb->value_inst(-1, elem->bitsize);
      elem = bb->build_inst(Op::ITE, cond, elem, zero);
      elem_undef = bb->build_inst(Op::ITE, cond, elem_undef, m1);

      if (inst)
	inst = bb->build_inst(Op::CONCAT, elem, inst);
      else
	inst = elem;
      if (undef)
	undef = bb->build_inst(Op::CONCAT, elem_undef, undef);
      else
	undef = elem_undef;
      if (mem_flags)
	mem_flags = bb->build_inst(Op::CONCAT, elem_flags, mem_flags);
      else
	mem_flags = elem_flags;
    }

  constrain_src_value(bb, inst, lhs_type, mem_flags);
  std::tie(inst, undef) = from_mem_repr(bb, inst, undef, lhs_type);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  tree2undef.insert({lhs, undef});
  if (POINTER_TYPE_P(lhs_type))
    tree2provenance.insert({lhs, bb->build_extract_id(inst)});
}

void Converter::process_cfn_mask_store(gimple *stmt, Basic_block *bb)
{
  tree ptr_expr = gimple_call_arg(stmt, 0);
  tree alignment_expr = gimple_call_arg(stmt, 1);
  tree mask_expr = gimple_call_arg(stmt, 2);
  tree mask_type = TREE_TYPE(mask_expr);
  tree value_expr = gimple_call_arg(stmt, 3);
  tree value_type = TREE_TYPE(value_expr);

  auto [ptr, ptr_prov] = tree2inst_prov(bb, ptr_expr);
  Instruction *mask = tree2inst(bb, mask_expr);
  auto [value, undef] = tree2inst_undef(bb, value_expr);
  assert((value->bitsize & 7) == 0);
  uint64_t size = value->bitsize / 8;

  uint64_t alignment = get_int_cst_val(alignment_expr) / 8;
  if (alignment > 1)
    {
      uint32_t high_val = 0;
      for (;;)
	{
	  high_val++;
	  if (alignment == (1u << high_val))
	    break;
	}

      Instruction *extract = bb->build_trunc(ptr, high_val);
      Instruction *zero = bb->value_inst(0, high_val);
      Instruction *cond = bb->build_inst(Op::NE, extract, zero);
      bb->build_inst(Op::UB, cond);
    }

  assert(VECTOR_TYPE_P(value_type) == VECTOR_TYPE_P(mask_type));
  tree value_elem_type =
    VECTOR_TYPE_P(value_type) ? TREE_TYPE(value_type) : value_type;
  tree mask_elem_type =
    VECTOR_TYPE_P(mask_type) ? TREE_TYPE(mask_type) : mask_type;
  uint64_t elem_size = bytesize_for_type(value_elem_type);
  assert(TREE_CODE(mask_elem_type) == BOOLEAN_TYPE);
  uint64_t mask_elem_bitsize = bitsize_for_type(mask_elem_type);

  if (!undef)
    undef = bb->value_inst(0, value->bitsize);

  auto [orig, orig_undef, _] = load_value(bb, ptr, size);
  uint64_t nof_elem = size / elem_size;
  assert((size % elem_size) == 0);
  for (uint64_t i = 0; i < nof_elem; i++)
    {
      Instruction *cond = extract_vec_elem(bb, mask, mask_elem_bitsize, i);
      if (cond->bitsize != 1)
	cond = bb->build_trunc(cond, 1);
      Instruction *orig_elem = extract_vec_elem(bb, orig, elem_size * 8, i);
      Instruction *elem = extract_vec_elem(bb, value, elem_size * 8, i);
      Instruction *new_value = bb->build_inst(Op::ITE, cond, elem, orig_elem);
      orig_elem = extract_vec_elem(bb, orig_undef, elem_size * 8, i);
      elem = extract_vec_elem(bb, undef, elem_size * 8, i);
      Instruction *new_undef = bb->build_inst(Op::ITE, cond, elem, orig_elem);
      Instruction *offset = bb->value_inst(i * elem_size, ptr->bitsize);
      Instruction *dst_ptr = bb->build_inst(Op::ADD, ptr, offset);
      store_ub_check(bb, dst_ptr, ptr_prov, elem_size, cond);
      store_value(bb, dst_ptr, new_value, new_undef);
    }
}

void Converter::process_cfn_memcpy(gimple *stmt, Basic_block *bb)
{
  if (TREE_CODE(gimple_call_arg(stmt, 2)) != INTEGER_CST)
    throw Not_implemented("non-constant memcpy size");
  auto [orig_dest_ptr, dest_prov] =
    tree2inst_prov(bb, gimple_call_arg(stmt, 0));
  auto [orig_src_ptr, src_prov] = tree2inst_prov(bb, gimple_call_arg(stmt, 1));
  unsigned __int128 size = get_int_cst_val(gimple_call_arg(stmt, 2));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("too large memcpy");

  store_ub_check(bb, orig_dest_ptr, dest_prov, size);
  load_ub_check(bb, orig_src_ptr, src_prov, size);

  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      constrain_range(bb, lhs, orig_dest_ptr);
      tree2instruction.insert({lhs, orig_dest_ptr});
      tree2provenance.insert({lhs, dest_prov});
    }

  for (size_t i = 0; i < size; i++)
    {
      Instruction *offset = bb->value_inst(i, orig_src_ptr->bitsize);
      Instruction *src_ptr = bb->build_inst(Op::ADD, orig_src_ptr, offset);
      Instruction *dest_ptr = bb->build_inst(Op::ADD, orig_dest_ptr, offset);

      Instruction *byte = bb->build_inst(Op::LOAD, src_ptr);
      bb->build_inst(Op::STORE, dest_ptr, byte);

      Instruction *mem_flag = bb->build_inst(Op::GET_MEM_FLAG, src_ptr);
      bb->build_inst(Op::SET_MEM_FLAG, dest_ptr, mem_flag);

      Instruction *undef = bb->build_inst(Op::GET_MEM_UNDEF, src_ptr);
      bb->build_inst(Op::SET_MEM_UNDEF, dest_ptr, undef);
    }
}

void Converter::process_cfn_memset(gimple *stmt, Basic_block *bb)
{
  if (TREE_CODE(gimple_call_arg(stmt, 2)) != INTEGER_CST)
    throw Not_implemented("non-constant memset size");
  auto [orig_ptr, ptr_prov] = tree2inst_prov(bb, gimple_call_arg(stmt, 0));
  Instruction *value = tree2inst(bb, gimple_call_arg(stmt, 1));
  unsigned __int128 size = get_int_cst_val(gimple_call_arg(stmt, 2));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("too large memset");

  store_ub_check(bb, orig_ptr, ptr_prov, size);

  tree lhs = gimple_call_lhs(stmt);
  if (lhs)
    {
      constrain_range(bb, lhs, orig_ptr);
      tree2instruction.insert({lhs, orig_ptr});
      tree2provenance.insert({lhs, ptr_prov});
    }

  assert(value->bitsize >= 8);
  if (value->bitsize > 8)
    value = bb->build_trunc(value, 8);
  Instruction *mem_flag = bb->value_inst(0, 1);
  Instruction *undef = bb->value_inst(0, 8);
  for (size_t i = 0; i < size; i++)
    {
      Instruction *offset = bb->value_inst(i, orig_ptr->bitsize);
      Instruction *ptr = bb->build_inst(Op::ADD, orig_ptr, offset);
      bb->build_inst(Op::STORE, ptr, value);
      bb->build_inst(Op::SET_MEM_FLAG, ptr, mem_flag);
      bb->build_inst(Op::SET_MEM_UNDEF, ptr, undef);
    }
}

void Converter::process_cfn_mul_overflow(gimple *stmt, Basic_block *bb)
{
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
  auto [arg1, arg1_undef] = tree2inst_undef(bb, arg1_expr);
  auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  auto [res_undef, res_def] =
    get_res_undef(arg1_undef, arg2_undef, TREE_TYPE(lhs), bb);
  unsigned lhs_elem_bitsize = bitsize_for_type(lhs_elem_type);
  unsigned bitsize =
    1 + std::max(arg1->bitsize + arg2->bitsize, lhs_elem_bitsize);
  Instruction *bitsize_inst = bb->value_inst(bitsize, 32);
  if (TYPE_UNSIGNED(arg1_type))
    arg1 = bb->build_inst(Op::ZEXT, arg1, bitsize_inst);
  else
    arg1 = bb->build_inst(Op::SEXT, arg1, bitsize_inst);
  if (TYPE_UNSIGNED(arg2_type))
    arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize_inst);
  else
    arg2 = bb->build_inst(Op::SEXT, arg2, bitsize_inst);
  Instruction *inst = bb->build_inst(Op::MUL, arg1, arg2);
  Instruction *res = bb->build_trunc(inst, lhs_elem_bitsize);
  Instruction *eres;
  if (TYPE_UNSIGNED(lhs_elem_type))
    eres = bb->build_inst(Op::ZEXT, res, bitsize_inst);
  else
    eres = bb->build_inst(Op::SEXT, res, bitsize_inst);
  Instruction *overflow = bb->build_inst(Op::NE, inst, eres);

  res = to_mem_repr(bb, res, lhs_elem_type);
  Instruction *res_bitsize_inst = bb->value_inst(res->bitsize, 32);
  overflow = bb->build_inst(Op::ZEXT, overflow, res_bitsize_inst);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_undef)
    tree2undef.insert({lhs, res_undef});
}

void Converter::process_cfn_nan(gimple *stmt, Basic_block *bb)
{
  // TODO: Implement the argument setting NaN payload when support for
  // noncanonical NaNs is implemented in the SMT solvers.
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_nan: vector type");

  Instruction *bs = bb->value_inst(bitsize_for_type(TREE_TYPE(lhs)), 32);
  tree2instruction.insert({lhs, bb->build_inst(Op::NAN, bs)});
}

void Converter::process_cfn_parity(gimple *stmt, Basic_block *bb)
{
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_parity: vector type");
  Instruction *arg = tree2inst(bb, gimple_call_arg(stmt, 0));
  int bitwidth = arg->bitsize;
  Instruction *inst = bb->build_extract_bit(arg, 0);
  for (int i = 1; i < bitwidth; i++)
    {
      Instruction *bit = bb->build_extract_bit(arg, i);
      inst = bb->build_inst(Op::XOR, inst, bit);
    }
  bitwidth = TYPE_PRECISION(TREE_TYPE(lhs));
  Instruction *bitwidth_inst = bb->value_inst(bitwidth, 32);
  inst = bb->build_inst(Op::ZEXT, inst, bitwidth_inst);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_popcount(gimple *stmt, Basic_block *bb)
{
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_popcount: vector type");
  Instruction *arg = tree2inst(bb, gimple_call_arg(stmt, 0));
  int bitwidth = arg->bitsize;
  Instruction *eight = bb->value_inst(8, 32);
  Instruction *bit = bb->build_extract_bit(arg, 0);
  Instruction *res = bb->build_inst(Op::ZEXT, bit, eight);
  for (int i = 1; i < bitwidth; i++)
    {
      bit = bb->build_extract_bit(arg, i);
      Instruction *ext = bb->build_inst(Op::ZEXT, bit, eight);
      res = bb->build_inst(Op::ADD, res, ext);
    }
  int lhs_bitwidth = TYPE_PRECISION(TREE_TYPE(lhs));
  Instruction *lhs_bitwidth_inst = bb->value_inst(lhs_bitwidth, 32);
  res = bb->build_inst(Op::ZEXT, res, lhs_bitwidth_inst);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
}

void Converter::process_cfn_signbit(gimple *stmt, Basic_block *bb)
{
  Instruction *arg1 = tree2inst(bb, gimple_call_arg(stmt, 0));
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_signbit: vector type");
  Instruction *signbit = bb->build_extract_bit(arg1, arg1->bitsize - 1);
  uint32_t bitsize = bitsize_for_type(TREE_TYPE(lhs));
  Instruction *lhs_bitsize_inst = bb->value_inst(bitsize, 32);
  Instruction *inst = bb->build_inst(Op::ZEXT, signbit, lhs_bitsize_inst);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_sub_overflow(gimple *stmt, Basic_block *bb)
{
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
  auto [arg1, arg1_undef] = tree2inst_undef(bb, arg1_expr);
  auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  auto [res_undef, res_def] =
    get_res_undef(arg1_undef, arg2_undef, TREE_TYPE(lhs), bb);
  unsigned lhs_elem_bitsize = bitsize_for_type(lhs_elem_type);
  unsigned bitsize = 1 + std::max(arg1->bitsize, arg2->bitsize);
  bitsize = 1 + std::max(bitsize, lhs_elem_bitsize);
  Instruction *bitsize_inst = bb->value_inst(bitsize, 32);
  if (TYPE_UNSIGNED(arg1_type))
    arg1 = bb->build_inst(Op::ZEXT, arg1, bitsize_inst);
  else
    arg1 = bb->build_inst(Op::SEXT, arg1, bitsize_inst);
  if (TYPE_UNSIGNED(arg2_type))
    arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize_inst);
  else
    arg2 = bb->build_inst(Op::SEXT, arg2, bitsize_inst);
  Instruction *inst = bb->build_inst(Op::SUB, arg1, arg2);
  Instruction *res = bb->build_trunc(inst, lhs_elem_bitsize);
  Instruction *eres;
  if (TYPE_UNSIGNED(lhs_elem_type))
    eres = bb->build_inst(Op::ZEXT, res, bitsize_inst);
  else
    eres = bb->build_inst(Op::SEXT, res, bitsize_inst);
  Instruction *overflow = bb->build_inst(Op::NE, inst, eres);

  res = to_mem_repr(bb, res, lhs_elem_type);
  Instruction *res_bitsize_inst = bb->value_inst(res->bitsize, 32);
  overflow = bb->build_inst(Op::ZEXT, overflow, res_bitsize_inst);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_undef)
    tree2undef.insert({lhs, res_undef});
}

void Converter::process_cfn_reduc(gimple *stmt, Basic_block *bb)
{
  tree arg_expr = gimple_call_arg(stmt, 0);
  tree arg_type = TREE_TYPE(arg_expr);
  assert(VECTOR_TYPE_P(arg_type));
  tree elem_type = TREE_TYPE(arg_type);
  auto[arg, arg_undef, arg_prov] = tree2inst_undef_prov(bb, arg_expr);
  tree lhs = gimple_call_lhs(stmt);

  tree_code code;
  switch (gimple_call_combined_fn(stmt))
    {
    case CFN_REDUC_AND:
      code = BIT_AND_EXPR;
      break;
    case CFN_REDUC_IOR:
      code = BIT_IOR_EXPR;
      break;
    case CFN_REDUC_MAX:
      code = MAX_EXPR;
      break;
    case CFN_REDUC_MIN:
      code = MIN_EXPR;
      break;
    case CFN_REDUC_PLUS:
      code = PLUS_EXPR;
      break;
    case CFN_REDUC_XOR:
      code = BIT_XOR_EXPR;
      break;
    default:
      {
	const char *name = internal_fn_name(gimple_call_internal_fn(stmt));
	throw Not_implemented("process_cfn_reduc: "s + name);
      }
    }

  uint32_t elem_bitsize = bitsize_for_type(elem_type);
  uint32_t nof_elt = bitsize_for_type(arg_type) / elem_bitsize;
  auto [inst, undef, prov] =
    extract_vec_elem(bb, arg, arg_undef, arg_prov, elem_bitsize, 0);
  for (uint64_t i = 1; i < nof_elt; i++)
    {
      auto [elem, elem_undef, elem_prov] =
	extract_vec_elem(bb, arg, arg_undef, arg_prov, elem_bitsize, i);
      std::tie(inst, undef, prov) =
	process_binary_scalar(code, inst, undef, prov, elem, elem_undef,
			      elem_prov, elem_type, elem_type, elem_type, bb);
    }

  if (lhs)
    {
      tree2instruction.insert({lhs, inst});
      if (undef)
	tree2undef.insert({lhs, undef});
      if (prov)
	tree2provenance.insert({lhs, prov});
    }
}

void Converter::process_cfn_trap(gimple *, Basic_block *bb)
{
  // TODO: Some passes add __builtin_trap for cases that are UB (so that
  // the program terminates instead of continuing in a random state).
  // We threat these as UB for now, but they should arguably be handled
  // in a special way to verify that we actually are termininating.
  bb->build_inst(Op::UB, bb->value_inst(1, 1));
}

void Converter::process_cfn_vcond(gimple *stmt, Basic_block *bb)
{
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg1_elem_type = TREE_TYPE(arg1_type);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree arg2_elem_type = TREE_TYPE(arg2_type);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  tree arg3_type = TREE_TYPE(arg3_expr);
  tree arg3_elem_type = TREE_TYPE(arg3_type);
  tree arg4_expr = gimple_call_arg(stmt, 3);
  tree arg5_expr = gimple_call_arg(stmt, 4);
  tree lhs = gimple_call_lhs(stmt);

  Instruction *arg1 = tree2inst(bb, arg1_expr);
  Instruction *arg2 = tree2inst(bb, arg2_expr);
  auto [arg3, arg3_undef] = tree2inst_undef(bb, arg3_expr);
  auto [arg4, arg4_undef] = tree2inst_undef(bb, arg4_expr);
  if (arg3_undef || arg4_undef)
    {
      if (!arg3_undef)
	arg3_undef = bb->value_inst(0, arg3->bitsize);
      if (!arg4_undef)
	arg4_undef = bb->value_inst(0, arg4->bitsize);
    }
  assert(arg1->bitsize == arg2->bitsize);
  assert(arg3->bitsize == arg4->bitsize);

  enum tree_code code = (enum tree_code)get_int_cst_val(arg5_expr);
  bool is_unsigned = gimple_call_combined_fn(stmt) == CFN_VCONDU;

  uint32_t elem_bitsize1 = bitsize_for_type(arg1_elem_type);
  uint32_t elem_bitsize3 = bitsize_for_type(arg3_elem_type);

  Instruction *res = nullptr;
  uint32_t nof_elt = bitsize_for_type(arg1_type) / elem_bitsize1;
  for (uint64_t i = 0; i < nof_elt; i++)
    {
      Instruction *a1 = extract_vec_elem(bb, arg1, elem_bitsize1, i);
      Instruction *a2 = extract_vec_elem(bb, arg2, elem_bitsize1, i);
      Instruction *a3 = extract_vec_elem(bb, arg3, elem_bitsize3, i);
      Instruction *a4 = extract_vec_elem(bb, arg4, elem_bitsize3, i);

      Instruction *cond;
      if (FLOAT_TYPE_P(arg1_elem_type))
	{
	  Instruction *cond_undef;
	  std::tie(cond, cond_undef) =
	    process_binary_float(code, a1, nullptr, a2, nullptr,
				 boolean_type_node, bb);
	}
      else
	cond = process_binary_int(code, is_unsigned, a1, a2,
				  boolean_type_node, arg1_elem_type,
				  arg2_elem_type, bb);
      Instruction *inst = bb->build_inst(Op::ITE, cond, a3, a4);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;

      if (arg3_undef)
	{
	  Instruction *a3_undef =
	    extract_vec_elem(bb, arg3_undef, elem_bitsize3, i);
	  Instruction *a4_undef =
	    extract_vec_elem(bb, arg4_undef, elem_bitsize3, i);
	  Instruction *undef =
	    bb->build_inst(Op::ITE, cond, a3_undef, a4_undef);
	  Instruction *zero = bb->value_inst(0, undef->bitsize);
	  Instruction *cmp = bb->build_inst(Op::NE, undef, zero);
	  bb->build_inst(Op::UB, cmp);
	}
    }
  if (lhs)
    {
      constrain_range(bb, lhs, res);
      tree2instruction.insert({lhs, res});
    }
}

void Converter::process_cfn_vcond_mask(gimple *stmt, Basic_block *bb)
{
  tree arg1_expr = gimple_call_arg(stmt, 0);
  tree arg1_type = TREE_TYPE(arg1_expr);
  tree arg2_expr = gimple_call_arg(stmt, 1);
  tree arg2_type = TREE_TYPE(arg2_expr);
  tree arg3_expr = gimple_call_arg(stmt, 2);
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;

  Instruction *arg1 = tree2inst(bb, arg1_expr);
  auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  auto [arg3, arg3_undef] = tree2inst_undef(bb, arg3_expr);
  auto [inst, undef] = process_vec_cond(arg1, arg2, arg2_undef,
					arg3, arg3_undef, arg1_type,
					arg2_type, bb);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
  if (undef)
    tree2undef.insert({lhs, undef});
}

void Converter::process_cfn_vec_convert(gimple *stmt, Basic_block *bb)
{
  tree arg1_expr = gimple_call_arg(stmt, 0);
  Instruction *arg1 = tree2inst(bb, arg1_expr);
  tree arg1_elem_type = TREE_TYPE(TREE_TYPE(arg1_expr));
  tree lhs = gimple_call_lhs(stmt);
  if (!lhs)
    return;
  tree lhs_elem_type = TREE_TYPE(TREE_TYPE(lhs));
  auto [inst, undef] =
    process_unary_vec(CONVERT_EXPR, arg1, nullptr, lhs_elem_type,
		      arg1_elem_type, bb);
  assert(!undef);
  constrain_range(bb, lhs, inst);
  tree2instruction.insert({lhs, inst});
}

void Converter::process_cfn_uaddc(gimple *stmt, Basic_block *bb)
{
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
  auto [arg1, arg1_undef] = tree2inst_undef(bb, arg1_expr);
  auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  auto [arg3, arg3_undef] = tree2inst_undef(bb, arg3_expr);
  auto [res_undef, res_def] =
    get_res_undef(arg1_undef, arg2_undef, arg3_undef, TREE_TYPE(lhs), bb);
  assert(arg1->bitsize == arg2->bitsize);
  assert(arg1->bitsize == arg3->bitsize);
  assert(lhs_elem_bitsize == arg1->bitsize);

  Instruction *bitsize_inst = bb->value_inst(arg1->bitsize + 2, 32);
  arg1 = bb->build_inst(Op::ZEXT, arg1, bitsize_inst);
  arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize_inst);
  arg3 = bb->build_inst(Op::ZEXT, arg3, bitsize_inst);
  Instruction *sum = bb->build_inst(Op::ADD, arg1, arg2);
  sum = bb->build_inst(Op::ADD, sum, arg3);
  Instruction *res = bb->build_trunc(sum, lhs_elem_bitsize);

  Instruction *high = bb->value_inst(sum->bitsize - 1, 32);
  Instruction *low = bb->value_inst(sum->bitsize - 2, 32);
  Instruction *overflow = bb->build_inst(Op::EXTRACT, sum, high, low);
  Instruction *zero = bb->value_inst(0, overflow->bitsize);
  overflow = bb->build_inst(Op::NE, overflow, zero);
  Instruction *lhs_elem_bitsize_inst = bb->value_inst(lhs_elem_bitsize, 32);
  overflow = bb->build_inst(Op::ZEXT, overflow, lhs_elem_bitsize_inst);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_undef)
    tree2undef.insert({lhs, res_undef});
}

void Converter::process_cfn_unreachable(gimple *, Basic_block *bb)
{
  bb->build_inst(Op::UB, bb->value_inst(1, 1));
}

void Converter::process_cfn_usubc(gimple *stmt, Basic_block *bb)
{
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
  auto [arg1, arg1_undef] = tree2inst_undef(bb, arg1_expr);
  auto [arg2, arg2_undef] = tree2inst_undef(bb, arg2_expr);
  auto [arg3, arg3_undef] = tree2inst_undef(bb, arg3_expr);
  assert(arg1->bitsize == arg2->bitsize);
  assert(arg1->bitsize == arg3->bitsize);
  assert(lhs_elem_bitsize == arg1->bitsize);
  auto [res_undef, res_def] =
    get_res_undef(arg1_undef, arg2_undef, arg3_undef, TREE_TYPE(lhs), bb);

  Instruction *bitsize_inst = bb->value_inst(arg1->bitsize + 2, 32);
  arg1 = bb->build_inst(Op::ZEXT, arg1, bitsize_inst);
  arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize_inst);
  arg3 = bb->build_inst(Op::ZEXT, arg3, bitsize_inst);
  Instruction *sum = bb->build_inst(Op::SUB, arg1, arg2);
  sum = bb->build_inst(Op::SUB, sum, arg3);
  Instruction *res = bb->build_trunc(sum, lhs_elem_bitsize);

  Instruction *high = bb->value_inst(sum->bitsize - 1, 32);
  Instruction *low = bb->value_inst(sum->bitsize - 2, 32);
  Instruction *overflow = bb->build_inst(Op::EXTRACT, sum, high, low);
  Instruction *zero = bb->value_inst(0, overflow->bitsize);
  overflow = bb->build_inst(Op::NE, overflow, zero);
  Instruction *lhs_elem_bitsize_inst = bb->value_inst(lhs_elem_bitsize, 32);
  overflow = bb->build_inst(Op::ZEXT, overflow, lhs_elem_bitsize_inst);
  res = bb->build_inst(Op::CONCAT, overflow, res);
  constrain_range(bb, lhs, res);
  tree2instruction.insert({lhs, res});
  if (res_undef)
    tree2undef.insert({lhs, res_undef});
}

void Converter::process_cfn_xorsign(gimple *stmt, Basic_block *bb)
{
  Instruction *arg1 = tree2inst(bb, gimple_call_arg(stmt, 0));
  Instruction *arg2 = tree2inst(bb, gimple_call_arg(stmt, 1));
  Instruction *signbit1 = bb->build_extract_bit(arg1, arg1->bitsize - 1);
  Instruction *signbit2 = bb->build_extract_bit(arg2, arg2->bitsize - 1);
  Instruction *signbit = bb->build_inst(Op::XOR, signbit1, signbit2);
  Instruction *res = bb->build_trunc(arg1, arg1->bitsize - 1);
  res = bb->build_inst(Op::CONCAT, signbit, res);

  tree lhs = gimple_call_lhs(stmt);
  if (VECTOR_TYPE_P(TREE_TYPE(lhs)))
    throw Not_implemented("process_cfn_: vector type");

  // For now, treat copying the sign to NaN as always produce the original
  // canonical NaN.
  // TODO: Remove this when Op::IS_NONCANONICAL_NAN is removed.
  Instruction *is_nan = bb->build_inst(Op::IS_NAN, arg1);
  res = bb->build_inst(Op::ITE, is_nan, arg1, res);
  if (lhs)
    {
      constrain_range(bb, lhs, res);
      tree2instruction.insert({lhs, res});
    }
}

void Converter::process_gimple_call_combined_fn(gimple *stmt, Basic_block *bb)
{
  switch (gimple_call_combined_fn(stmt))
    {
    case CFN_ADD_OVERFLOW:
      process_cfn_add_overflow(stmt, bb);
      break;
    case CFN_BUILT_IN_ASSUME_ALIGNED:
      process_cfn_assume_aligned(stmt, bb);
      break;
    case CFN_BUILT_IN_BSWAP16:
    case CFN_BUILT_IN_BSWAP32:
    case CFN_BUILT_IN_BSWAP64:
    case CFN_BUILT_IN_BSWAP128:
      process_cfn_bswap(stmt, bb);
      break;
    case CFN_BUILT_IN_CLRSB:
    case CFN_BUILT_IN_CLRSBL:
    case CFN_BUILT_IN_CLRSBLL:
      process_cfn_clrsb(stmt, bb);
      break;
    case CFN_BUILT_IN_CLZ:
    case CFN_BUILT_IN_CLZL:
    case CFN_BUILT_IN_CLZLL:
    case CFN_CLZ:
      process_cfn_clz(stmt, bb);
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
      process_cfn_copysign(stmt, bb);
      break;
    case CFN_BUILT_IN_CTZ:
    case CFN_BUILT_IN_CTZL:
    case CFN_BUILT_IN_CTZLL:
    case CFN_CTZ:
      process_cfn_ctz(stmt, bb);
      break;
    case CFN_BUILT_IN_EXPECT:
    case CFN_BUILT_IN_EXPECT_WITH_PROBABILITY:
      process_cfn_expect(stmt, bb);
      break;
    case CFN_BUILT_IN_FFS:
    case CFN_BUILT_IN_FFSL:
    case CFN_BUILT_IN_FFSLL:
    case CFN_FFS:
      process_cfn_ffs(stmt, bb);
      break;
    case CFN_BUILT_IN_FMAX:
    case CFN_BUILT_IN_FMAXF:
    case CFN_BUILT_IN_FMAXL:
      process_cfn_fmax(stmt, bb);
      break;
    case CFN_BUILT_IN_FMIN:
    case CFN_BUILT_IN_FMINF:
    case CFN_BUILT_IN_FMINL:
      process_cfn_fmin(stmt, bb);
      break;
    case CFN_BUILT_IN_MEMCPY:
      process_cfn_memcpy(stmt, bb);
      break;
    case CFN_BUILT_IN_MEMSET:
      process_cfn_memset(stmt, bb);
      break;
    case CFN_BUILT_IN_NAN:
    case CFN_BUILT_IN_NANF:
    case CFN_BUILT_IN_NANL:
      process_cfn_nan(stmt, bb);
      break;
    case CFN_BUILT_IN_PARITY:
    case CFN_BUILT_IN_PARITYL:
    case CFN_BUILT_IN_PARITYLL:
    case CFN_PARITY:
      process_cfn_parity(stmt, bb);
      break;
    case CFN_BUILT_IN_POPCOUNT:
    case CFN_BUILT_IN_POPCOUNTL:
    case CFN_BUILT_IN_POPCOUNTLL:
    case CFN_POPCOUNT:
      process_cfn_popcount(stmt, bb);
      break;
    case CFN_BUILT_IN_SIGNBIT:
    case CFN_BUILT_IN_SIGNBITF:
    case CFN_BUILT_IN_SIGNBITL:
      process_cfn_signbit(stmt, bb);
      break;
    case CFN_BUILT_IN_TRAP:
      process_cfn_trap(stmt, bb);
      break;
    case CFN_BUILT_IN_UNREACHABLE:
    case CFN_BUILT_IN_UNREACHABLE_TRAP:
      process_cfn_unreachable(stmt, bb);
      break;
    case CFN_COND_ADD:
    case CFN_COND_AND:
    case CFN_COND_IOR:
    case CFN_COND_MUL:
    case CFN_COND_RDIV:
    case CFN_COND_SHL:
    case CFN_COND_SHR:
    case CFN_COND_SUB:
      process_cfn_cond(stmt, bb);
      break;
    case CFN_DIVMOD:
      process_cfn_divmod(stmt, bb);
      break;
    case CFN_FALLTHROUGH:
      break;
    case CFN_LOOP_VECTORIZED:
      process_cfn_loop_vectorized(stmt);
      break;
    case CFN_MASK_LOAD:
      process_cfn_mask_load(stmt, bb);
      break;
    case CFN_MASK_STORE:
      process_cfn_mask_store(stmt, bb);
      break;
    case CFN_MUL_OVERFLOW:
      process_cfn_mul_overflow(stmt, bb);
      break;
    case CFN_REDUC_AND:
    case CFN_REDUC_IOR:
    case CFN_REDUC_MAX:
    case CFN_REDUC_MIN:
    case CFN_REDUC_PLUS:
    case CFN_REDUC_XOR:
      process_cfn_reduc(stmt, bb);
      break;
    case CFN_SUB_OVERFLOW:
      process_cfn_sub_overflow(stmt, bb);
      break;
    case CFN_UADDC:
      process_cfn_uaddc(stmt, bb);
      break;
    case CFN_USUBC:
      process_cfn_usubc(stmt, bb);
      break;
    case CFN_VCOND:
    case CFN_VCONDU:
      process_cfn_vcond(stmt, bb);
      break;
    case CFN_VCOND_MASK:
      process_cfn_vcond_mask(stmt, bb);
      break;
    case CFN_VEC_CONVERT:
      process_cfn_vec_convert(stmt, bb);
      break;
    case CFN_XORSIGN:
      process_cfn_xorsign(stmt, bb);
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

void Converter::process_gimple_call(gimple *stmt, Basic_block *bb)
{
  if (gimple_call_builtin_p(stmt) || gimple_call_internal_p(stmt))
    process_gimple_call_combined_fn(stmt, bb);
  else
    throw Not_implemented("gimple_call");
}

Instruction *Converter::build_label_cond(tree index_expr, tree label, Basic_block *bb)
{
  tree index_type = TREE_TYPE(index_expr);
  Instruction *index = tree2inst(bb, index_expr);
  tree low_expr = CASE_LOW(label);
  Instruction *low = tree2inst(bb, low_expr);
  low = type_convert(low, TREE_TYPE(low_expr), index_type, bb);
  tree high_expr = CASE_HIGH(label);
  Instruction *cond;
  if (high_expr)
    {
      Instruction *high = tree2inst(bb, high_expr);
      high = type_convert(high, TREE_TYPE(high_expr), index_type, bb);
      Op op = TYPE_UNSIGNED(index_type) ?  Op::UGE: Op::SGE;
      Instruction *cond_low = bb->build_inst(op, index, low);
      Instruction *cond_high = bb->build_inst(op, high, index);
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
      bb->build_br_inst(gccbb2bb.at(default_block));
      return;
    }

  n = cases.size();
  for (size_t i = 0; i < n; i++)
    {
      Instruction *cond = nullptr;
      basic_block block = cases[i];
      const std::vector<tree>& labels = block2labels.at(block);
      for (auto label : labels)
	{
	  Instruction *label_cond = build_label_cond(index_expr, label, bb);
	  if (cond)
	    cond = bb->build_inst(Op::OR, cond, label_cond);
	  else
	    cond = label_cond;
	}

      Basic_block *true_bb = gccbb2bb.at(block);
      Basic_block *false_bb;
      if (i != n - 1)
	{
	  false_bb = func->build_bb();
	  bbset.insert(false_bb);
	}
      else
	false_bb = gccbb2bb.at(default_block);
      bb->build_br_inst(cond, true_bb, false_bb);
      bb = false_bb;
    }
}

// Get the BB corresponding to the source of the phi argument i.
Basic_block *Converter::get_phi_arg_bb(gphi *phi, int i)
{
  edge e = gimple_phi_arg_edge(phi, i);
  Basic_block *arg_bb = gccbb2bb.at(e->src);
  Basic_block *phi_bb = gccbb2bb.at(e->dest);
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

void Converter::process_gimple_return(gimple *stmt, Basic_block *bb)
{
  greturn *return_stmt = dyn_cast<greturn *>(stmt);
  tree expr = gimple_return_retval(return_stmt);
  if (expr)
    bb2retval.insert({bb, tree2inst_undef(bb, expr)});
  // TODO: Add assert that the successor goes to the exit block. We will
  // miscompile otherwise...
}

Instruction *split_phi(Instruction *phi, uint64_t elem_bitsize, std::map<std::pair<Instruction *, uint64_t>, std::vector<Instruction *>>& cache)
{
  assert(phi->opcode == Op::PHI);
  assert(phi->bitsize % elem_bitsize == 0);
  if (phi->bitsize == elem_bitsize)
    return phi;
  Instruction *res = nullptr;
  uint32_t nof_elem = phi->bitsize / elem_bitsize;
  std::vector<Instruction *> phis;
  phis.reserve(nof_elem);
  for (uint64_t i = 0; i < nof_elem; i++)
    {
      Instruction *inst = phi->bb->build_phi_inst(elem_bitsize);
      phis.push_back(inst);
      if (res)
	{
	  Instruction *concat = create_inst(Op::CONCAT, inst, res);
	  if (res->opcode == Op::PHI)
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
      std::vector<Instruction *>& split = cache[{arg_inst, elem_bitsize}];
      if (split.empty())
	{
	  for (uint64_t i = 0; i < nof_elem; i++)
	    {
	      Instruction *inst =
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

void Converter::generate_return_inst(Basic_block *bb)
{
  if (!retval_bitsize)
    {
      bb->build_ret_inst();
      return;
    }

  // Some predecessors to the exit block may not have a return value;
  // They may have a return without value, or the predecessor may be
  // a builtin_unreachable, etc. We therefore creates a dummy value,
  // marked as undefined, for these predecessors to make the IR valid.
  {
    Instruction *retval = nullptr;
    Instruction *undef = nullptr;
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
		    Instruction *inst = entry_bb->value_inst(-1, bs);
		    if (undef)
		      undef = entry_bb->build_inst(Op::CONCAT, inst, undef);
		    else
		      undef = inst;
		  }
	      }
	    bb2retval.insert({pred_bb, {retval, undef}});
	  }
      }
  }

  Instruction *retval;
  Instruction *retval_undef;
  if (bb->preds.size() == 1)
    std::tie(retval, retval_undef) = bb2retval.at(bb->preds[0]);
  else
    {
      Instruction *phi = bb->build_phi_inst(retval_bitsize);
      Instruction *phi_undef = bb->build_phi_inst(retval_bitsize);
      bool need_undef_phi = false;
      for (Basic_block *pred_bb : bb->preds)
	{
	  auto [ret, ret_undef] = bb2retval.at(pred_bb);
	  phi->add_phi_arg(ret, pred_bb);
	  need_undef_phi = need_undef_phi || ret_undef;
	  if (!ret_undef)
	    ret_undef = pred_bb->value_inst(0, retval_bitsize);
	  phi_undef->add_phi_arg(ret_undef, pred_bb);
	}
      retval = phi;
      retval_undef = need_undef_phi ? phi_undef : nullptr;

      std::map<std::pair<Instruction *, uint64_t>, std::vector<Instruction *>> cache;
      if (VECTOR_TYPE_P(retval_type) || TREE_CODE(retval_type) == COMPLEX_TYPE)
	{
	  uint32_t elem_bitsize = bitsize_for_type(TREE_TYPE(retval_type));
	  retval = split_phi(retval, elem_bitsize, cache);
	  if (retval_undef)
	    retval_undef = split_phi(retval_undef, elem_bitsize, cache);
	}
    }

  // GCC treats it as UB to return the address of a local variable.
  if (POINTER_TYPE_P(retval_type))
    {
      uint32_t ptr_id_bits = func->module->ptr_id_bits;
      Instruction *mem_id = bb->build_extract_id(retval);
      Instruction *zero = bb->value_inst(0, ptr_id_bits);
      Instruction *cond = bb->build_inst(Op::SLT, mem_id, zero);
      if (retval_undef)
	{
	  Instruction *zero2 = bb->value_inst(0, retval_undef->bitsize);
	  Instruction *cond2 = bb->build_inst(Op::EQ, retval_undef, zero2);
	  cond = bb->build_inst(Op::AND, cond, cond2);
	}
      bb->build_inst(Op::UB, cond);
    }

  if (retval_undef)
    bb->build_ret_inst(retval, retval_undef);
  else
    bb->build_ret_inst(retval);
}

// Write the values to initialized variables.
void Converter::init_var_values(tree initial, Instruction *mem_inst)
{
  if (TREE_CODE(initial) == ERROR_MARK)
    throw Not_implemented("init_var_values: ERROR_MARK");

  Basic_block *bb = mem_inst->bb;
  tree type = TREE_TYPE(initial);
  uint64_t size = bytesize_for_type(TREE_TYPE(initial));

  if (TREE_CODE(initial) == STRING_CST)
    {
      uint64_t len = TREE_STRING_LENGTH(initial);
      const char *p = TREE_STRING_POINTER(initial);
      for (uint64_t i = 0; i < len; i++)
	{
	  Instruction *offset = bb->value_inst(i, mem_inst->bitsize);
	  Instruction *ptr = bb->build_inst(Op::ADD, mem_inst, offset);
	  Instruction *byte = bb->value_inst(p[i], 8);
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
      auto [value, undef, prov] = tree2inst_init_var(bb, initial);
      value = to_mem_repr(bb, value, type);
      store_value(bb, mem_inst, value);
      return;
    }

  if (TREE_CODE(type) == ARRAY_TYPE)
    {
      tree elem_type = TREE_TYPE(type);
      uint64_t elem_size = bytesize_for_type(elem_type);
      unsigned HOST_WIDE_INT idx;
      tree index;
      tree value;
      FOR_EACH_CONSTRUCTOR_ELT(CONSTRUCTOR_ELTS(initial), idx, index, value)
	{
	  if (index && TREE_CODE(index) == RANGE_EXPR)
	    throw Not_implemented("init_var: RANGE_EXPR");
	  uint64_t offset = idx * elem_size;
	  Instruction *off = bb->value_inst(offset, mem_inst->bitsize);
	  Instruction *ptr = bb->build_inst(Op::ADD, mem_inst, off);
	  init_var_values(value, ptr);
	}
      return;
    }

  if (TREE_CODE(type) == RECORD_TYPE || TREE_CODE(type) == UNION_TYPE)
    {
      unsigned HOST_WIDE_INT idx;
      tree index;
      tree value;
      FOR_EACH_CONSTRUCTOR_ELT(CONSTRUCTOR_ELTS(initial), idx, index, value)
	{
	  uint64_t offset = get_int_cst_val(DECL_FIELD_OFFSET(index));
	  uint64_t bit_offset = get_int_cst_val(DECL_FIELD_BIT_OFFSET(index));
	  offset += bit_offset / 8;
	  bit_offset &= 7;
	  Instruction *off = bb->value_inst(offset, mem_inst->bitsize);
	  Instruction *ptr = bb->build_inst(Op::ADD, mem_inst, off);
	  tree elem_type = TREE_TYPE(value);
	  if (TREE_CODE(elem_type) == ARRAY_TYPE
	      || TREE_CODE(elem_type) == RECORD_TYPE
	      || TREE_CODE(elem_type) == UNION_TYPE)
	    init_var_values(value, ptr);
	  else
	    {
	      uint64_t bitsize = bitsize_for_type(elem_type);
	      auto [value_inst, undef, prov] = tree2inst_init_var(bb, value);
	      size = (bitsize + bit_offset + 7) / 8;
	      if (DECL_BIT_FIELD_TYPE(index))
		{
		  if (bit_offset)
		    {
		      Instruction *first_byte = bb->build_inst(Op::LOAD, ptr);
		      Instruction *bits = bb->build_trunc(first_byte, bit_offset);
		      value_inst = bb->build_inst(Op::CONCAT, value_inst, bits);
		    }
		  if (bitsize + bit_offset != size * 8)
		    {
		      Instruction *offset =
			bb->value_inst(size - 1, ptr->bitsize);
		      Instruction *ptr3 = bb->build_inst(Op::ADD, ptr, offset);

		      uint64_t remaining = size * 8 - (bitsize + bit_offset);
		      assert(remaining < 8);
		      Instruction *high = bb->value_inst(7, 32);
		      Instruction *low = bb->value_inst(8 - remaining, 32);

		      Instruction *last_byte =
			bb->build_inst(Op::LOAD, ptr3);
		      Instruction *bits =
			bb->build_inst(Op::EXTRACT, last_byte, high, low);
		      value_inst = bb->build_inst(Op::CONCAT, bits, value_inst);
		    }
		}
	      else
		{
		  value_inst = to_mem_repr(bb, value_inst, elem_type);
		}
	      store_value(bb, ptr, value_inst);
	    }
	}
      return;
    }

  throw Not_implemented("init_var: unknown constructor");
}

void Converter::init_var(tree decl, Instruction *mem_inst)
{
  uint64_t size = bytesize_for_type(TREE_TYPE(decl));
  if (size > MAX_MEMORY_UNROLL_LIMIT)
    throw Not_implemented("init_var: too large constructor");
  check_type(TREE_TYPE(decl));

  Basic_block *bb = mem_inst->bb;

  tree initial = DECL_INITIAL(decl);
  if (!initial)
    {
      if (!TREE_STATIC(decl))
	return;

      // Uninitializied static variables are guaranted to be initialized to 0.
      Instruction *zero = bb->value_inst(0, 8);
      uint64_t size = bytesize_for_type(TREE_TYPE(decl));
      for (uint64_t i = 0; i < size; i++)
	{
	  Instruction *offset = bb->value_inst(i, mem_inst->bitsize);
	  Instruction *ptr = bb->build_inst(Op::ADD, mem_inst, offset);
	  bb->build_inst(Op::STORE, ptr, zero);
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

      Instruction *zero = bb->value_inst(0, 8);
      if (size > MAX_MEMORY_UNROLL_LIMIT)
	throw Not_implemented("init_var: too large constructor");
      for (uint64_t i = 0; i < size; i++)
	{
	  Instruction *offset = bb->value_inst(i, mem_inst->bitsize);
	  Instruction *ptr = bb->build_inst(Op::ADD, mem_inst, offset);
	  uint8_t padding = padding_at_offset(type, i);
	  if (padding)
	    bb->build_inst(Op::SET_MEM_UNDEF, ptr, bb->value_inst(padding, 8));
	  if (padding != 255)
	    bb->build_inst(Op::STORE, ptr, zero);
	}
    }

  init_var_values(initial, mem_inst);
}

void Converter::make_uninit(Basic_block *bb, Instruction *orig_ptr, uint64_t size)
{
  Instruction *byte_m1 = bb->value_inst(255, 8);
  for (uint64_t i = 0; i < size; i++)
    {
      Instruction *offset = bb->value_inst(i, orig_ptr->bitsize);
      Instruction *ptr = bb->build_inst(Op::ADD, orig_ptr, offset);
      bb->build_inst(Op::SET_MEM_UNDEF, ptr, byte_m1);
    }
}

void Converter::process_variables()
{
  tree retval_decl = DECL_RESULT(fun->decl);
  retval_type = TREE_TYPE(retval_decl);
  if (VOID_TYPE_P(retval_type))
    retval_bitsize = 0;
  else
    {
      retval_bitsize = bitsize_for_type(TREE_TYPE(DECL_RESULT(fun->decl)));

      uint64_t id;
      if (state->decl2id.contains(retval_decl))
	{
	  id = state->decl2id.at(retval_decl);
	}
      else
	{
	  id = --state->id_local;
	  state->decl2id.insert({retval_decl, id});
	}
      uint64_t size = bytesize_for_type(retval_type);
      Instruction *memory_inst = build_memory_inst(id, size, MEM_UNINIT);
      decl2instruction.insert({retval_decl, memory_inst});
    }

  // Add an anonymous memory as first global.
  // TODO: Should only be done if we have unconstrained pointers?
  build_memory_inst(2, ANON_MEM_SIZE, MEM_KEEP);

  // Global variables.
  {
    varpool_node *var;
    std::map<std::string, tree> name2decl;
    FOR_EACH_VARIABLE(var)
      {
	tree decl = var->decl;
	if (lookup_attribute("alias", DECL_ATTRIBUTES(decl)))
	  continue;
	uint64_t size = bytesize_for_type(TREE_TYPE(decl));
	if (size >= ((uint64_t)1 << module->ptr_offset_bits))
	  throw Not_implemented("process_function: too large global variable");
	// TODO: Implement.
	if (size == 0)
	  throw Not_implemented("process_function: unknown size");

	uint64_t id;
	if (state->decl2id.contains(decl))
	  {
	    id = state->decl2id.at(decl);
	  }
	else
	  {
	    uint32_t ptr_id_bits = func->module->ptr_id_bits;

	    // Artificial decls are used for data introduced by the compiler
	    // (such as switch tables), so normal, unconstrained, pointers
	    // cannot point to them. Give these a local ID.
	    if (DECL_ARTIFICIAL(decl))
	      {
		if (state->id_local <= -(1 << ((ptr_id_bits - 1))))
		  throw Not_implemented("too many local variables");
		id = --state->id_local;
	      }
	    else
	      {
		if (state->id_global >= (1 << ((ptr_id_bits - 1) - 1)))
		  throw Not_implemented("too many global variables");
		id = ++state->id_global;
	      }
	    state->decl2id.insert({decl, id});
	  }
	uint64_t flags = 0;
	if (TREE_READONLY(decl))
	  flags |= MEM_CONST;
	Instruction *memory_inst = build_memory_inst(id, size, flags);
	decl2instruction.insert({decl, memory_inst});
	if (DECL_NAME(decl))
	  {
	    const char *name = IDENTIFIER_POINTER(DECL_NAME(decl));
	    name2decl.insert({name, decl});
	  }
      }

    FOR_EACH_VARIABLE(var)
      {
	tree decl = var->decl;
	tree alias = lookup_attribute("alias", DECL_ATTRIBUTES(decl));
	if (alias)
	  {
	    const char *name = IDENTIFIER_POINTER(DECL_NAME(decl));
	    const char *alias_name =
	      TREE_STRING_POINTER(TREE_VALUE(TREE_VALUE(alias)));
	    if (!name2decl.contains(alias_name))
	      throw Not_implemented("unknown alias");
	    tree alias_decl = name2decl.at(alias_name);
	    decl2instruction.insert({decl, decl2instruction.at(alias_decl)});
	    name2decl.insert({name, alias_decl});
	  }
      }

    // Must do this after creating all variables as a pointer may need to
    // be initialized by an address of a later variable.
    FOR_EACH_VARIABLE(var)
      {
	tree decl = var->decl;
	if (TREE_READONLY(decl))
	  init_var(decl, decl2instruction.at(decl));
      }
  }

  // Local variables.
  {
    tree decl;
    unsigned ix;
    FOR_EACH_LOCAL_DECL(fun, ix, decl)
      {
	// Local static decls are included in the global decls, so their
	// memory objects have already been created.
	if (decl2instruction.contains(decl))
	  {
	    assert(TREE_STATIC(decl));
	    continue;
	  }

	assert(!DECL_INITIAL(decl));

	uint64_t size = bytesize_for_type(TREE_TYPE(decl));
	if (size > MAX_MEMORY_UNROLL_LIMIT)
	  throw Not_implemented("process_function: too large local variable");

	int64_t id;
	if (state->decl2id.contains(decl))
	  {
	    id = state->decl2id.at(decl);
	  }
	else
	  {
	    uint32_t ptr_id_bits = func->module->ptr_id_bits;
	    if (state->id_local <= -(1 << ((ptr_id_bits - 1))))
	      throw Not_implemented("too many local variables");
	    id = --state->id_local;
	    state->decl2id.insert({decl, id});
	  }
	uint64_t flags = MEM_UNINIT;
	if (TREE_READONLY(decl))
	  flags |= MEM_CONST;
	Instruction *memory_inst = build_memory_inst(id, size, flags);
	decl2instruction.insert({decl, memory_inst});
      }
  }
}

void Converter::process_func_args()
{
  tree fntype = TREE_TYPE(fun->decl);
  bitmap nonnullargs = get_nonnull_args(fntype);
  tree decl;
  int param_number = 0;
  Basic_block *bb = func->bbs[0];
  const char *decl_name = IDENTIFIER_POINTER(DECL_NAME(fun->decl));
  for (decl = DECL_ARGUMENTS(fun->decl); decl; decl = DECL_CHAIN(decl))
    {
      check_type(TREE_TYPE(decl));
      uint32_t bitsize = bitsize_for_type(TREE_TYPE(decl));
      if (bitsize <= 0)
	throw Not_implemented("Parameter size == 0");

      bool type_is_unsigned =
	TREE_CODE(TREE_TYPE(decl)) == INTEGER_TYPE
	&& TYPE_UNSIGNED(TREE_TYPE(decl));
      state->params.push_back({bitsize, type_is_unsigned, 0, 0});

      // TODO: There must be better ways to determine if this is the "this"
      // pointer of a C++ constructor.
      if (param_number == 0 && !strcmp(decl_name, "__ct_base "))
	{
	  assert(POINTER_TYPE_P(TREE_TYPE(decl)));

	  // We use constant ID as it must be the same between src and tgt.
	  int64_t id = 1;
	  uint64_t flags = MEM_UNINIT | MEM_KEEP;
	  uint64_t size = bytesize_for_type(TREE_TYPE(TREE_TYPE(decl)));

	  Instruction *param_inst = build_memory_inst(id, size, flags);
	  tree2provenance.insert({decl, param_inst->arguments[0]});
	  tree2instruction.insert({decl, param_inst});
	}
      else
	{
	  Instruction *param_nbr = bb->value_inst(param_number, 32);
	  Instruction *param_bitsize = bb->value_inst(bitsize, 32);
	  Instruction *param_inst =
	    bb->build_inst(Op::PARAM, param_nbr, param_bitsize);
	  tree2instruction.insert({decl, param_inst});

	  // Pointers cannot point to local variables or to the this pointer
	  // in constructors.
	  // TODO: Update all pointer UB checks for this.
	  if (POINTER_TYPE_P(TREE_TYPE(decl)))
	    {
	      uint32_t ptr_id_bits = func->module->ptr_id_bits;
	      Instruction *id = bb->build_extract_id(param_inst);
	      Instruction *zero = bb->value_inst(0, ptr_id_bits);
	      Instruction *one = bb->value_inst(1, ptr_id_bits);
	      bb->build_inst(Op::UB, bb->build_inst(Op::SLT, id, zero));
	      bb->build_inst(Op::UB, bb->build_inst(Op::EQ, id, one));
	    }

	  constrain_src_value(bb, param_inst, TREE_TYPE(decl));

	  // Params marked "nonnull" is UB if NULL.
	  if (POINTER_TYPE_P(TREE_TYPE(decl))
	      && nonnullargs
	      && (bitmap_empty_p(nonnullargs)
		  || bitmap_bit_p(nonnullargs, param_number)))
	    {
	      Instruction *zero = bb->value_inst(0, param_inst->bitsize);
	      Instruction *cond = bb->build_inst(Op::EQ, param_inst, zero);
	      bb->build_inst(Op::UB, cond);
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

	      Instruction *m_inst = bb->value_inst(~m, param_inst->bitsize);
	      Instruction *v_inst = bb->value_inst(v, param_inst->bitsize);
	      Instruction *and_inst =
		bb->build_inst(Op::AND, param_inst, m_inst);
	      Instruction *cond = bb->build_inst(Op::NE, v_inst, and_inst);
	      bb->build_inst(Op::UB, cond);
	    }
	}

      if (POINTER_TYPE_P(TREE_TYPE(decl)))
	{
	  Instruction *param_inst = tree2instruction.at(decl);
	  Instruction *id = bb->build_extract_id(param_inst);
	  tree2provenance.insert({decl, id});
	}

      param_number++;
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
      if (tree2provenance.contains(arg))
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
      Basic_block *bb = gccbb2bb.at(gcc_bb);
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
	  Instruction *phi_inst = bb->build_phi_inst(bitwidth);
	  Instruction *phi_undef = bb->build_phi_inst(bitwidth);
	  constrain_range(bb, phi_result, phi_inst, phi_undef);
	  if (need_prov_phi(phi))
	    {
	      uint32_t ptr_id_bits = bb->func->module->ptr_id_bits;
	      Instruction *phi_prov = bb->build_phi_inst(ptr_id_bits);
	      tree2provenance.insert({phi_result, phi_prov});
	    }
	  tree2instruction.insert({phi_result, phi_inst});
	  tree2undef.insert({phi_result, phi_undef});
	}
      for (gsi = gsi_start_bb(gcc_bb); !gsi_end_p(gsi); gsi_next(&gsi))
	{
	  gimple *stmt = gsi_stmt(gsi);
	  switch (gimple_code(stmt))
	    {
	    case GIMPLE_ASSIGN:
	      process_gimple_assign(stmt, bb);
	      break;
	    case GIMPLE_ASM:
	      process_gimple_asm(stmt);
	      break;
	    case GIMPLE_CALL:
	      process_gimple_call(stmt, bb);
	      break;
	    case GIMPLE_COND:
	      assert(!cond_stmt);
	      assert(!switch_stmt);
	      cond_stmt = stmt;
	      break;
	    case GIMPLE_RETURN:
	      process_gimple_return(stmt, bb);
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
	      bb->build_br_inst(gccbb2bb.at(gcc_exit_block));
	    }
	  else
	    generate_return_inst(bb);
	}
      else if (cond_stmt)
	{
	  tree_code code = gimple_cond_code(cond_stmt);
	  tree arg1_expr = gimple_cond_lhs(cond_stmt);
	  tree arg2_expr = gimple_cond_rhs(cond_stmt);
	  tree arg1_type = TREE_TYPE(arg1_expr);
	  tree arg2_type = TREE_TYPE(arg2_expr);
	  Instruction *arg1 = tree2inst(bb, arg1_expr);
	  Instruction *arg2 = tree2inst(bb, arg2_expr);
	  Instruction *cond;
	  if (TREE_CODE(arg1_type) == COMPLEX_TYPE)
	    cond = process_binary_complex_cmp(code, arg1, arg2,
					      boolean_type_node,
					      arg1_type, bb);
	  else
	    cond = process_binary_scalar(code, arg1, arg2, boolean_type_node,
					 arg1_type, arg2_type, bb);
	  edge true_edge, false_edge;
	  extract_true_false_edges_from_block(gcc_bb, &true_edge, &false_edge);
	  Basic_block *true_bb = gccbb2bb.at(true_edge->dest);
	  Basic_block *false_bb = gccbb2bb.at(false_edge->dest);
	  bb->build_br_inst(cond, true_bb, false_bb);
	}
      else
	{
	  assert(EDGE_COUNT(gcc_bb->succs) == 1);
	  Basic_block *succ_bb = gccbb2bb.at(single_succ_edge(gcc_bb)->dest);
	  bb->build_br_inst(succ_bb);
	}
    }

  // We have created all instructions, so it is now safe to add the phi
  // arguments (as they must have been created now).
  std::map<std::pair<Instruction *, uint64_t>, std::vector<Instruction *>> cache;
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
	  Instruction *phi_inst = tree2instruction.at(phi_result);
	  Instruction *phi_undef = tree2undef.at(phi_result);
	  Instruction *phi_prov = nullptr;
	  if (tree2provenance.contains(phi_result))
	    phi_prov = tree2provenance.at(phi_result);
	  for (unsigned i = 0; i < gimple_phi_num_args(phi); i++)
	    {
	      Basic_block *arg_bb = get_phi_arg_bb(phi, i);
	      tree arg = gimple_phi_arg_def(phi, i);
	      auto [arg_inst, arg_undef, arg_prov] =
		tree2inst_undef_prov(arg_bb, arg);
	      phi_inst->add_phi_arg(arg_inst, arg_bb);
	      if (!arg_undef)
		arg_undef = arg_bb->value_inst(0, arg_inst->bitsize);
	      phi_undef->add_phi_arg(arg_undef, arg_bb);
	      if (phi_prov)
		{
		  assert(!POINTER_TYPE_P(phi_type) || arg_prov);
		  if (!arg_prov)
		    arg_prov = arg_bb->value_inst(0, module->ptr_id_bits);
		  phi_prov->add_phi_arg(arg_prov, arg_bb);
		}
	    }

	  if (VECTOR_TYPE_P(phi_type) || TREE_CODE(phi_type) == COMPLEX_TYPE)
	    {
	      uint32_t bs = bitsize_for_type(TREE_TYPE(phi_type));
	      tree2instruction[phi_result] = split_phi(phi_inst, bs, cache);
	      tree2undef[phi_result] = split_phi(phi_undef, bs, cache);
	    }
	}
    }
}

Function *Converter::process_function()
{
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
	gccbb2bb.insert({gcc_bb, func->build_bb()});
      }

    process_variables();
    process_func_args();

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

  validate(func);

  reverse_post_order(func);

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
      simplify_insts(func);
      dead_code_elimination(func);
      simplify_cfg(func);
    }
  dead_code_elimination(func);
  validate(func);
}

void unroll_and_optimize(Module *module)
{
  for (auto func : module->functions)
    unroll_and_optimize(func);
}

Module *create_module()
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
      ptr_bits = 64;
      ptr_id_bits = 16;
      ptr_offset_bits = 48;
    }
  return create_module(ptr_bits, ptr_id_bits, ptr_offset_bits);
}

// We emit CFN_LOOP_VECTORIZED as a symbolic value, which ensures that we
// verify tgt is a refinement of src regardless of its value. However, this
// approach can lead to false positives in cases where gcc decides not to
// vectorize the loop. The issue arises because the 'true' branch of
// CFN_LOOP_VECTORIZED may have changed some integer variables to unsigned
// to prevent overflow. When CFN_LOOP_VECTORIZED is eliminated, we then
// detect a possible overflow when this is compared to tgt, which now only
// contains the original, overflowing calculations.
//
// We address this by setting CFN_LOOP_VECTORIZED to `false` in src if tgt
// does not include any CFN_LOOP_VECTORIZED.
void adjust_loop_vectorized(smtgcc::Module *module)
{
  assert(module->functions.size() == 2);
  Function *src = module->functions[0];
  Function *tgt = module->functions[1];
  if (src->name != "src")
    std::swap(src, tgt);
  assert(src->name == "src" && tgt->name == "tgt");
  Instruction *src_lv_inst = get_lv_inst(src);
  Instruction *tgt_lv_inst = get_lv_inst(tgt);
  if (src_lv_inst && !tgt_lv_inst)
    src_lv_inst->replace_all_uses_with(src_lv_inst->bb->value_inst(0, 1));
}
