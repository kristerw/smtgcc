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

  // Integer unary
  ASSERT,
  FREE,
  GET_MEM_FLAG,
  GET_MEM_UNDEF,
  IS_CONST_MEM,
  IS_NONCANONICAL_NAN,
  LOAD,
  MEM_SIZE,
  MOV,
  NEG,
  NOT,
  READ,
  REGISTER,
  SYMBOLIC,
  UB,

  // Floating-point unary
  FABS,
  FNEG,

  // Integer binary
  ADD,
  AND,
  ASHR,
  CONCAT,
  LSHR,
  MUL,
  OR,
  PARAM,
  SADD_WRAPS,
  SDIV,
  SET_MEM_FLAG,
  SET_MEM_UNDEF,
  SHL,
  SMAX,
  SMIN,
  SMUL_WRAPS,
  SREM,
  SSUB_WRAPS,
  STORE,
  SUB,
  UDIV,
  UMAX,
  UMIN,
  UREM,
  WRITE,
  XOR,

  // Floating-point binary
  FADD,
  FDIV,
  FMUL,
  FSUB,

  // Ternary
  EXTRACT,
  ITE,
  MEMORY,

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

struct Instruction_info {
  const char *name;
  Op opcode;
  Inst_class iclass;
  bool has_lhs;
  bool is_commutative;
};

extern const std::array<Instruction_info, 77> inst_info;

struct Module;
struct Function;
struct Basic_block;
struct Instruction;

struct Phi_arg {
  Instruction *inst;
  Basic_block *bb;
};

struct Instruction {
  uint32_t bitsize = 0;
  Op opcode;
  uint16_t nof_args = 0;
  Instruction *arguments[3];
  Basic_block *bb = nullptr;
  Instruction *prev = nullptr;
  Instruction *next = nullptr;
  uint32_t id;
  std::set<Instruction *> used_by;
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
    return inst_info[(int)opcode].iclass;
  }
  const char *name() const
  {
    return inst_info[(int)opcode].name;
  }
  bool has_lhs() const
  {
    return inst_info[(int)opcode].has_lhs;
  }
  bool is_commutative() const
  {
    return inst_info[(int)opcode].is_commutative;
  }
  unsigned __int128 value() const;
  void insert_after(Instruction *inst);
  void insert_before(Instruction *inst);
  void move_before(Instruction *inst);
  void replace_use_with(Instruction *use, Instruction *new_inst);
  void replace_all_uses_with(Instruction *inst);
  void update_uses();
  Instruction *get_phi_arg(Basic_block *bb);
  void add_phi_arg(Instruction *inst, Basic_block *bb);
  void remove_phi_arg(Basic_block *bb);
  void remove_phi_args();
  void print(FILE *stream) const;

  Instruction();
};

struct Basic_block {
  std::vector<Instruction *> phis;
  std::vector<Basic_block *> preds;
  std::vector<Basic_block *> succs;
  std::set<Basic_block *> dom;
  std::set<Basic_block *> post_dom;

  Instruction *first_inst = nullptr;
  Instruction *last_inst = nullptr;
  Function *func;
  int id;

  void insert_last(Instruction *inst);
  void insert_phi(Instruction *inst);
  Instruction *build_inst(Op opcode, Instruction *arg);
  Instruction *build_inst(Op opcode, Instruction *arg1, Instruction *arg2);
  Instruction *build_inst(Op opcode, Instruction *arg1, Instruction *arg2,
			  Instruction *arg3);
  Instruction *build_phi_inst(int bitsize);
  Instruction *build_ret_inst();
  Instruction *build_ret_inst(Instruction *arg);
  Instruction *build_ret_inst(Instruction *arg1, Instruction *arg2);
  Instruction *build_br_inst(Basic_block *dest_bb);
  Instruction *build_br_inst(Instruction *cond, Basic_block *true_bb,
			     Basic_block *false_bb);
  Instruction *build_extract_id(Instruction *arg);
  Instruction *build_extract_offset(Instruction *arg);
  Instruction *build_extract_bit(Instruction *arg, uint32_t bit_idx);
  Instruction *build_trunc(Instruction *arg, uint32_t nof_bits);
  Instruction *value_inst(unsigned __int128 value, uint32_t bitsize);
  Instruction *value_m1_inst(uint32_t bitsize);
  void print(FILE *stream) const;
};

struct Function {
public:
  std::string name;
  std::vector<Basic_block *> bbs;
  std::map<std::pair<unsigned __int128, uint32_t>, Instruction *> values;
  Instruction *last_value_inst = nullptr;

  Basic_block *build_bb();
  Instruction *value_inst(unsigned __int128 value, uint32_t bitsize);
  void rename(const std::string& str);
  void canonicalize();
  void reset_ir_id();
  void print(FILE *stream) const;
  Module *module;
private:
  int next_bb_id = 0;
};

struct Module {
  std::vector<Function *> functions;
  Function *build_function(const std::string& name);
  Function *clone(Function *func);
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
void destroy_instruction(Instruction *);

Instruction *create_inst(Op opcode, Instruction *arg);
Instruction *create_inst(Op opcode, Instruction *arg1, Instruction *arg2);
Instruction *create_inst(Op opcode, Instruction *arg1, Instruction *arg2,
			 Instruction *arg3);
Instruction *create_phi_inst(int bitsize);
Instruction *create_ret_inst();
Instruction *create_ret_inst(Instruction *arg);
Instruction *create_ret_inst(Instruction *arg1, Instruction *arg2);
Instruction *create_br_inst(Basic_block *dest_bb);
Instruction *create_br_inst(Instruction *cond, Basic_block *true_bb,
			    Basic_block *false_bb);

bool identical(Function *func1, Function *func2);

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
Solver_result check_refine(Module *module);
Solver_result check_assert(Function *func);
Solver_result check_ub(Function *func);

// cfg.cpp
void reverse_post_order(Function *func);
bool has_loops(Function *func);
void simplify_cfg(Function *func);
Basic_block *nearest_dominator(const Basic_block *bb);
bool dominates(const Basic_block *bb1, const Basic_block *bb2);
bool post_dominates(const Basic_block *bb1, const Basic_block *bb2);

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

// read_riscv.cpp
struct riscv_state {
  Module *module;
  std::vector<bool> param_is_unsigned;
};
Function *parse_riscv(std::string const& file_name, riscv_state *state);

// simplify_insts.cpp
void simplify_insts(Function *func);
void simplify_insts(Module *module);
void simplify_mem(Function *func);
void simplify_mem(Module *module);

// smt_cvc5.cpp
std::pair<SStats, Solver_result> check_refine_cvc5(Function *src, Function *tgt);
std::pair<SStats, Solver_result> check_assert_cvc5(Function *func);
std::pair<SStats, Solver_result> check_ub_cvc5(Function *func);

// smt_z3.cpp
std::pair<SStats, Solver_result> check_refine_z3(Function *src, Function *tgt);
std::pair<SStats, Solver_result> check_assert_z3(Function *func);
std::pair<SStats, Solver_result> check_ub_z3(Function *func);

// validate_ir.cpp
void validate(Module *module);
void validate(Function *func);

} // end namespace smtgcc

#endif
