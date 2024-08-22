#ifndef SMTGCC_H
#define SMTGCC_H

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#define MEM_KEEP    1
#define MEM_CONST   2
#define MEM_UNINIT  4

namespace smtgcc {

const int unroll_limit = 12;

const uint32_t max_nof_bb = 50000;
const uint32_t max_nof_inst = 1000000;

struct Not_implemented
{
  Not_implemented(const std::string& msg) : msg{msg} {}
  const std::string msg;
};

struct Parse_error
{
  Parse_error(const std::string& msg, int line) : msg{msg}, line{line} {}
  const std::string msg;
  int line;
};

enum class Op : uint8_t {
  // Integer comparison
  EQ,
  NE,
  SGE,
  SGT,
  SLE,
  SLT,
  UGE,
  UGT,
  ULE,
  ULT,

  // Floating-point comparison
  FEQ,
  FGE,
  FGT,
  FLE,
  FLT,
  FNE,

  // Nullary
  MEM_ARRAY,
  MEM_FLAG_ARRAY,
  MEM_SIZE_ARRAY,
  MEM_INDEF_ARRAY,

  // Integer unary
  ASSERT,
  FREE,
  GET_MEM_FLAG,
  GET_MEM_SIZE,
  GET_MEM_INDEF,
  IS_CONST_MEM,
  IS_NAN,
  IS_NONCANONICAL_NAN,
  LOAD,
  MOV,
  NEG,
  NOT,
  READ,
  REGISTER,
  SRC_ASSERT,
  TGT_ASSERT,
  UB,

  // Floating-point unary
  FABS,
  FNEG,
  NAN,

  // Integer binary
  ADD,
  AND,
  ARRAY_GET_FLAG,
  ARRAY_GET_SIZE,
  ARRAY_GET_INDEF,
  ARRAY_LOAD,
  ASHR,
  CONCAT,
  LSHR,
  MUL,
  OR,
  PARAM,
  PRINT,
  SADD_WRAPS,
  SDIV,
  SET_MEM_FLAG,
  SET_MEM_INDEF,
  SHL,
  SMUL_WRAPS,
  SRC_RETVAL,
  SRC_UB,
  SREM,
  SSUB_WRAPS,
  STORE,
  SUB,
  SYMBOLIC,
  TGT_RETVAL,
  TGT_UB,
  UDIV,
  UREM,
  WRITE,
  XOR,

  // Floating-point binary
  FADD,
  FDIV,
  FMUL,
  FSUB,

  // Ternary
  ARRAY_SET_FLAG,
  ARRAY_SET_SIZE,
  ARRAY_SET_INDEF,
  ARRAY_STORE,
  EXTRACT,
  ITE,
  MEMORY,
  SRC_MEM,
  TGT_MEM,

  // Conversions
  F2S,
  F2U,
  FCHPREC,
  S2F,
  SEXT,
  U2F,
  ZEXT,

  // Special
  BR,
  PHI,
  RET,
  VALUE,
};

enum class Inst_class : uint8_t {
  // Nullary operations
  nullary,

  // Unary operations
  iunary,
  funary,

  // Binary operations
  ibinary,
  fbinary,
  icomparison,
  fcomparison,
  conv,

  // Ternary operations
  ternary,

  // Misc
  special
};

struct Inst_info {
  const char *name;
  Op op;
  Inst_class iclass;
  bool has_lhs;
  bool is_commutative;
};

extern const std::array<Inst_info, 96> inst_info;

struct Module;
struct Function;
struct Basic_block;
struct Inst;

struct Phi_arg {
  Inst *inst;
  Basic_block *bb;
};

struct Inst {
  uint32_t bitsize = 0;
  Op op;
  uint16_t nof_args = 0;
  Inst *args[3];
  Basic_block *bb = nullptr;
  Inst *prev = nullptr;
  Inst *next = nullptr;
  uint32_t id;
  std::set<Inst *> used_by;
  std::vector<Phi_arg> phi_args;

  union {
    struct {
      Basic_block *dest_bb;
    } br1;
    struct {
      Basic_block *true_bb;
      Basic_block *false_bb;
    } br3;
    struct {
      unsigned __int128 value;
    } value;
  } u;

  Inst_class iclass() const
  {
    return inst_info[(int)op].iclass;
  }
  const char *name() const
  {
    return inst_info[(int)op].name;
  }
  bool has_lhs() const
  {
    return inst_info[(int)op].has_lhs;
  }
  bool is_commutative() const
  {
    return inst_info[(int)op].is_commutative;
  }
  unsigned __int128 value() const;
  __int128 signed_value() const;
  void insert_after(Inst *inst);
  void insert_before(Inst *inst);
  void move_after(Inst *inst);
  void move_before(Inst *inst);
  void replace_use_with(Inst *use, Inst *new_inst);
  void replace_all_uses_with(Inst *inst);
  void update_uses();
  Inst *get_phi_arg(Basic_block *bb);
  void update_phi_arg(Inst *inst, Basic_block *bb);
  void add_phi_arg(Inst *inst, Basic_block *bb);
  void remove_phi_arg(Basic_block *bb);
  void remove_phi_args();
  void print(FILE *stream) const;

  Inst();
};

struct Basic_block {
  std::vector<Inst *> phis;
  std::vector<Basic_block *> preds;
  std::vector<Basic_block *> succs;

  Inst *first_inst = nullptr;
  Inst *last_inst = nullptr;
  Function *func;
  int id;

  void insert_last(Inst *inst);
  void insert_phi(Inst *inst);
  Inst *build_inst(Op op);
  Inst *build_inst(Op op, Inst *arg);
  Inst *build_inst(Op op, Inst *arg1, Inst *arg2);
  Inst *build_inst(Op op, Inst *arg1, Inst *arg2, Inst *arg3);
  Inst *build_phi_inst(int bitsize);
  Inst *build_ret_inst();
  Inst *build_ret_inst(Inst *arg);
  Inst *build_ret_inst(Inst *arg1, Inst *arg2);
  Inst *build_br_inst(Basic_block *dest_bb);
  Inst *build_br_inst(Inst *cond, Basic_block *true_bb, Basic_block *false_bb);
  Inst *build_extract_id(Inst *arg);
  Inst *build_extract_offset(Inst *arg);
  Inst *build_extract_bit(Inst *arg, uint32_t bit_idx);
  Inst *build_trunc(Inst *arg, uint32_t nof_bits);
  Inst *value_inst(unsigned __int128 value, uint32_t bitsize);
  Inst *value_m1_inst(uint32_t bitsize);
  void print(FILE *stream) const;
};

struct Function {
public:
  std::string name;
  std::vector<Basic_block *> bbs;
  std::map<std::pair<uint32_t, unsigned __int128>, Inst *> values;
  Inst *last_value_inst = nullptr;

  // Data for dominance calculations.
  bool has_dominance = false;
  std::map<const Basic_block *, Basic_block *> nearest_dom;
  std::map<const Basic_block *, Basic_block *> nearest_postdom;

  Basic_block *build_bb();
  Inst *value_inst(unsigned __int128 value, uint32_t bitsize);
  void rename(const std::string& str);
  void canonicalize();
  void reset_ir_id();
  Function *clone(Module *dest_module);
  void print(FILE *stream) const;
  Module *module;
private:
  int next_bb_id = 0;
};

struct Module {
  std::vector<Function *> functions;
  Function *build_function(const std::string& name);
  void canonicalize();
  Module *clone();
  void print(FILE *stream) const;
  uint32_t ptr_bits;
  uint32_t ptr_id_bits;
  uint32_t ptr_id_high;
  uint32_t ptr_id_low;
  uint32_t ptr_offset_bits;
  uint32_t ptr_offset_high;
  uint32_t ptr_offset_low;
};

struct Config
{
  Config();
  int verbose;

  // SMT solver timeout in ms.
  int timeout;

  // SMT solver memory limit in megabytes.
  int memory_limit;
};

extern Config config;

enum class Result_status {
  correct, incorrect, unknown
};

struct Solver_result {
  Result_status status;
  std::optional<std::string> message;
};

Module *create_module(uint32_t ptr_bits, uint32_t id_bits, uint32_t offset_bits);
void destroy_module(Module *);
void destroy_function(Function *);
void destroy_basic_block(Basic_block *);
void destroy_instruction(Inst *);

Inst *create_inst(Op op);
Inst *create_inst(Op op, Inst *arg);
Inst *create_inst(Op op, Inst *arg1, Inst *arg2);
Inst *create_inst(Op op, Inst *arg1, Inst *arg2, Inst *arg3);
Inst *create_phi_inst(int bitsize);
Inst *create_ret_inst();
Inst *create_ret_inst(Inst *arg);
Inst *create_ret_inst(Inst *arg1, Inst *arg2);
Inst *create_br_inst(Basic_block *dest_bb);
Inst *create_br_inst(Inst *cond, Basic_block *true_bb, Basic_block *false_bb);

/* Opt level runs all optimization <= the level:
 *  0: Dead code elimination
 *  1: Registers are eliminated. Simple peephole optimizations. */
void optimize_func(Function *func, int opt_level);
void optimize_module(Module *module, int opt_level);

struct SStats {
  std::array<uint64_t, 3> time = {0, 0, 0};
  bool skipped = true;
};

uint64_t get_time();

// cfg.cpp
void clear_dominance(Function *func);
void calculate_dominance(Function *func);
void reverse_post_order(Function *func);
bool has_loops(Function *func);
void simplify_cfg(Function *func);
void simplify_cfg(Module *module);
Basic_block *nearest_dominator(const Basic_block *bb);
bool dominates(const Basic_block *bb1, const Basic_block *bb2);
bool postdominates(const Basic_block *bb1, const Basic_block *bb2);

// check.cpp
bool identical(Function *func1, Function *func2);
Solver_result check_refine(Module *module, bool run_simplify_inst = true);
Solver_result check_assert(Function *func);
Solver_result check_ub(Function *func);
void convert(Module *module);

// dead_code_elimination.cpp
void dead_code_elimination(Function *func);
void dead_code_elimination(Module *module);

// loop_unroll.cpp
bool loop_unroll(Function *func);
bool loop_unroll(Module *module);

// memory_opt.cpp
void canonicalize_memory(Module *module);
void ls_elim(Function *func);
void ls_elim(Module *module);

// read_ir.cpp
Module *parse_ir(std::string const& file_name);

struct MemoryObject {
  std::string sym_name;
  uint64_t id;
  uint64_t size;
  uint64_t flags;
};

// read_riscv.cpp
struct riscv_state {
  std::vector<Inst *> registers;
  std::vector<Inst *> fregisters;

  // The memory instruction corresponding to each symbol.
  std::map<std::string, Inst *> sym_name2mem;

  std::string func_name;
  Module *module;
  Basic_block *entry_bb;
  Basic_block *exit_bb;
  uint32_t reg_bitsize;
  uint32_t freg_bitsize;
  std::vector<MemoryObject> memory_objects;
};
Function *parse_riscv(std::string const& file_name, riscv_state *state);

// simplify_insts.cpp
Inst *constant_fold_inst(Inst *inst);
Inst *simplify_inst(Inst *inst);
void simplify_insts(Function *func);
void simplify_insts(Module *module);
void simplify_mem(Function *func);
void simplify_mem(Module *module);

// smt_cvc5.cpp
std::pair<SStats, Solver_result> check_refine_cvc5(Function *func);
std::pair<SStats, Solver_result> check_assert_cvc5(Function *func);
std::pair<SStats, Solver_result> check_ub_cvc5(Function *func);

// smt_z3.cpp
std::pair<SStats, Solver_result> check_refine_z3(Function *func);
std::pair<SStats, Solver_result> check_assert_z3(Function *func);
std::pair<SStats, Solver_result> check_ub_z3(Function *func);

// util.cpp
uint32_t popcount(unsigned __int128 x);
uint32_t clz(unsigned __int128 x);
uint32_t ctz(unsigned __int128 x);
bool is_value_zero(Inst *inst);
bool is_value_one(Inst *inst);
bool is_value_signed_min(Inst *inst);
bool is_value_signed_max(Inst *inst);
bool is_value_m1(Inst *inst);
bool is_value_pow2(Inst *inst);

// validate_ir.cpp
void validate(Module *module);
void validate(Function *func);

// vrp.cpp
void vrp(Function *func);
void vrp(Module *module);

} // end namespace smtgcc

#endif
