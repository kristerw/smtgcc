#include <fstream>
#include <cassert>
#include <cstring>
#include <limits>
#include <string>

#include "smtgcc.h"

using namespace std::string_literals;

using namespace smtgcc;

namespace smtgcc {
namespace {

enum class SIMD_cond {
  FEQ, FGE, FGT, FLE, FLT, EQ, GE, GT, HI, HS, LE, LT
};

struct Parser {
  Parser(aarch64_state *rstate) : rstate{rstate} {}

  enum class Cond_code {
    EQ, NE, CS, CC, MI, PL, VS, VC, HI, LS, GE, LT, GT, LE
  };

  enum class Lexeme {
    label,
    label_def,
    name,
    integer,
    hex,
    lo12,
    comma,
    plus,
    minus,
    exclamation,
    left_bracket,
    right_bracket,
    left_brace,
    right_brace,
    left_paren,
    right_paren
  };

  struct Token {
    Lexeme kind;
    int pos;
    int size;
  };
  std::vector<Token> tokens;
  std::map<std::string, std::vector<unsigned char>> sym_name2data;
  std::map<std::string, std::pair<std::string, uint64_t>> sym_alias;

  int line_number = 0;
  size_t pos;

  std::vector<char> buf;

  std::optional<std::string_view> parse_label_def();
  std::string_view parse_cmd();
  void parse_data(std::vector<unsigned char>& data);
  void skip_line();
  void skip_whitespace();
  void parse_rodata();
  Function *parse(std::string const& file_name);

  aarch64_state *rstate;
  Module *module;
  uint32_t reg_bitsize;
  uint32_t freg_bitsize;
  Function *src_func;

private:
  Function *func = nullptr;
  Basic_block *bb = nullptr;
  std::map<std::string_view, Basic_block *> label2bb;
  std::map<uint32_t, Inst *> id2inst;

  // Dummy register used for instructions that write to the zero register.
  Inst *zero_reg;

  void lex_line(void);
  void lex_label_or_label_def(void);
  void lex_hex(void);
  void lex_integer(void);
  void lex_hex_or_integer(void);
  void lex_name(void);
  bool is_lo12(void);

  std::string_view token_string(const Token& tok);

  uint64_t get_u64(const char *p);
  unsigned __int128 get_hex(const char *p);
  unsigned __int128 get_hex_or_integer(unsigned idx);
  Inst *get_reg(unsigned idx);
  Inst *get_freg(unsigned idx);
  std::tuple<uint64_t, uint32_t, uint32_t> get_vreg_(unsigned idx);
  std::tuple<Inst *, uint32_t, uint32_t> get_vreg(unsigned idx);
  std::tuple<Inst *, uint32_t, uint32_t> get_scalar_vreg(unsigned idx);
  uint32_t get_reg_size(unsigned idx);
  bool is_vector_op();
  bool is_freg(unsigned idx);
  bool is_vreg(unsigned idx);
  Inst *get_imm(unsigned idx);
  Inst *get_reg_value(unsigned idx);
  Inst *get_reg_or_imm_value(unsigned idx, uint32_t bitsize);
  Inst *get_freg_value(unsigned idx);
  Inst *get_vreg_value(unsigned idx, uint32_t nof_elem, uint32_t elem_bitsize);
  uint64_t get_vreg_idx(unsigned idx, uint32_t nof_elem, uint32_t elem_bitsize);
  Basic_block *get_bb(unsigned idx);
  Basic_block *get_bb_def(unsigned idx);
  std::string_view get_name(unsigned idx);
  Inst *get_sym_addr(unsigned idx);
  Cond_code get_cc(unsigned idx);
  void get_comma(unsigned idx);
  void get_exclamation(unsigned idx);
  void get_minus(unsigned idx);
  void get_left_bracket(unsigned idx);
  void get_right_bracket(unsigned idx);
  void get_left_brace(unsigned idx);
  void get_right_brace(unsigned idx);
  void get_end_of_line(unsigned idx);
  void write_reg(Inst *reg, Inst *value);
  Inst *build_cond(Cond_code cc);
  void process_cond_branch(Cond_code cc);
  void process_cbz(bool is_cbnz = false);
  void process_tbz(bool is_cbnz = false);
  void process_csel(Op op = Op::MOV);
  void process_fcsel();
  void process_cset(Op op = Op::ZEXT);
  void process_cinc();
  void process_call();
  void store_ub_check(Inst *ptr, uint64_t size);
  void load_ub_check(Inst *ptr, uint64_t size);
  Inst *process_address(unsigned idx);
  void process_load(uint32_t trunc_size = 0, Op op = Op::ZEXT);
  void process_ldp();
  void process_ldpsw();
  void process_store(uint32_t trunc_size = 0);
  void process_stp();
  void process_fcmp();
  void process_fccmp();
  void process_i2f(bool is_unsigned);
  void process_f2i(bool is_unsigned);
  void process_f2f();
  void process_mul_op(Op op);
  void process_maddl(Op op);
  void process_msubl(Op op);
  void process_mnegl(Op op);
  void process_mull(Op op);
  void process_mulh(Op op);
  void process_adrp();
  void process_adc();
  void process_adcs();
  void process_sbc();
  void process_sbcs();
  void process_ngc();
  void process_ngcs();
  void process_movk();
  void process_fmov();
  void process_smov();
  void process_umov();
  void process_unary(Op op);
  void process_unary(Inst*(*gen_elem)(Basic_block*, Inst*));
  Inst *process_arg_shift(unsigned idx, Inst *arg);
  Inst *process_arg_ext(unsigned idx, Inst *arg, uint32_t bitsize);
  Inst *process_last_arg(unsigned idx, uint32_t bitsize);
  Inst *process_last_scalar_vec_arg(unsigned idx);
  void process_binary(Op op, bool perform_not = false);
  void process_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  void process_simd_compare(SIMD_cond op);
  void process_simd_shift(Op op);
  void process_rev();
  void process_rev(uint32_t bitsize);
  Inst *gen_sub_cond_flags(Inst *arg1, Inst *arg2);
  Inst *gen_add_cond_flags(Inst *arg1, Inst *arg2);
  Inst *gen_and_cond_flags(Inst *arg1, Inst *arg2);
  void process_cmn();
  void process_cmp();
  void process_subs();
  void process_negs();
  void process_adds();
  void process_mneg();
  void process_ands(bool perform_not = false);
  void process_tst();
  void process_ccmp(bool is_ccmn = false);
  void process_ext(Op op, uint32_t src_bitsize);
  void process_shift(Op op);
  void process_ror();
  void process_dup();
  void process_extr();
  void process_bfi();
  void process_bfxil();
  void process_ubfx(Op op);
  void process_ubfiz(Op op);
  Inst *extract_vec_elem(Inst *inst, uint32_t elem_bitsize, uint32_t idx);
  void process_vec_unary(Op op);
  void process_vec_unary(Inst*(*gen_elem)(Basic_block*, Inst*));
  void process_vec_binary(Op op, bool perform_not = false);
  void process_vec_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  void process_vec_widen_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op widen_op, bool high);
  void process_vec_widen_binary_add(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op widen_op, bool high);
  void process_vec_widen_pairwise_add(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op widen_op);
  void process_vec_widen_pairwise(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op widen_op);
  void process_vec_widen2_binary(Op op, Op widen_op, bool high);
  void process_vec_reduc(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  void process_vec_reducl(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op op);
  void process_vec_pairwise(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  void process_vec_widen(Op, bool high = false);
  void process_vec_bic();
  void process_vec_bif();
  void process_vec_bit();
  void process_vec_bsl();
  void process_vec_ext();
  void process_vec_shrn(bool high);
  void process_vec_shift(Op op);
  void process_vec_shift_acc(Op op);
  void process_vec_widen_shift(Op op, Op widen_op, bool high);
  void process_vec_narrow(Op op, bool high);
  void process_vec_binary_high_narrow(Op op, bool high);
  void process_vec_xtn(bool high);
  void process_vec_mla(Op op = Op::ADD);
  void process_vec_uzp(bool odd);
  void process_vec_simd_compare(SIMD_cond op);
  void process_vec_ins();
  void process_vec_rev(uint32_t bitsize);
  void process_vec_dup();
  void process_vec_mov();
  void process_vec_movi(bool invert = false);
  void process_vec_orr();
  void process_vec_zip1();
  void process_vec_zip2();
  void process_vec_trn1();
  void process_vec_trn2();
  void process_vec_tbl();
  void parse_vector_op();
  void parse_function();

  void skip_space_and_comments();
};

void Parser::skip_space_and_comments()
{
  while (isspace(buf[pos]))
    pos++;
  if (buf[pos] == '#' && (!isdigit(buf[pos + 1]) && buf[pos + 1] != ':'))
    {
      while (buf[pos] != '\n')
	pos++;
    }
}

void Parser::lex_label_or_label_def(void)
{
  assert(buf[pos] == '.');
  int start_pos = pos;
  pos++;
  if (buf[pos] != 'L')
    throw Parse_error("expected 'L' after '.'", line_number);
  pos++;
  if (!isalnum(buf[pos]))
    throw Parse_error("expected a digit after \".L\"", line_number);
  pos++;
  while (isalnum(buf[pos]))
    pos++;
  if (buf[pos] == ':')
    {
      pos++;
      tokens.emplace_back(Lexeme::label_def, start_pos, pos - start_pos);
    }
  else
    tokens.emplace_back(Lexeme::label, start_pos, pos - start_pos);
}

void Parser::lex_hex(void)
{
  assert(buf[pos] == '0');
  int start_pos = pos;
  pos++;
  assert(buf[pos] == 'x' || buf[pos] == 'X');
  pos++;
  if (!isxdigit(buf[pos]))
    throw Parse_error("expected a hex digit after 0x", line_number);
  while (isxdigit(buf[pos]))
    pos++;
  tokens.emplace_back(Lexeme::hex, start_pos, pos - start_pos);
}

void Parser::lex_integer(void)
{
  int start_pos = pos;
  if (buf[pos] == '-')
    pos++;
  assert(isdigit(buf[pos]));
  pos++;
  if (isdigit(buf[pos]) && buf[pos - 1] == '0')
    throw Parse_error("octal numbers are not supported", line_number);
  while (isdigit(buf[pos]))
    pos++;
  if (buf[pos] == '.')
    throw Parse_error("fp constants are not supported yet", line_number);
  tokens.emplace_back(Lexeme::integer, start_pos, pos - start_pos);
}

void Parser::lex_hex_or_integer(void)
{
  assert(isdigit(buf[pos]) || buf[pos] == '-');
  if (buf[pos] == '0' && (buf[pos + 1] == 'x' || buf[pos + 1] == 'X'))
    lex_hex();
  else
    lex_integer();
}

void Parser::lex_name(void)
{
  assert(isalpha(buf[pos])
	 || buf[pos] == '_'
	 || buf[pos] == '.'
	 || buf[pos] == '$');
  int start_pos = pos;
  pos++;
  while (isalnum(buf[pos])
	 || buf[pos] == '_'
	 || buf[pos] == '.'
	 || buf[pos] == '$')
    pos++;
  tokens.emplace_back(Lexeme::name, start_pos, pos - start_pos);
}

bool Parser::is_lo12(void)
{
  if (buf[pos] == ':'
      && buf[pos + 1] == 'l'
      && buf[pos + 2] == 'o'
      && buf[pos + 3] == '1'
      && buf[pos + 4] == '2'
      && buf[pos + 5] == ':')
    return true;
  return false;
}

std::string_view Parser::token_string(const Token& tok)
{
  return std::string_view(&buf[tok.pos], tok.size);
}

std::string_view Parser::get_name(unsigned idx)
{
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::name)
    throw Parse_error("expected a name after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
  return std::string_view(&buf[tokens[idx].pos], tokens[idx].size);
}

Inst *Parser::get_sym_addr(unsigned idx)
{
  std::string name = std::string(get_name(idx));
  uint64_t offset = 0;
  if (sym_alias.contains(name))
    std::tie(name, offset) = sym_alias[name];
  if (!rstate->sym_name2mem.contains(name))
    throw Parse_error("unknown symbol " + name, line_number);
  Inst *inst = rstate->sym_name2mem[name];
  if (offset)
    inst = bb->build_inst(Op::ADD, inst, bb->value_inst(offset, inst->bitsize));
  return inst;
}

Parser::Cond_code Parser::get_cc(unsigned idx)
{
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::name)
    throw Parse_error("expected a condition code after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
  std::string_view name =
    std::string_view(&buf[tokens[idx].pos], tokens[idx].size);
  if (name == "eq")
    return Cond_code::EQ;
  else if (name == "ne")
    return Cond_code::NE;
  else if (name == "cs")
    return Cond_code::CS;
  else if (name == "cc")
    return Cond_code::CC;
  else if (name == "mi")
    return Cond_code::MI;
  else if (name == "pl")
    return Cond_code::PL;
  else if (name == "vs")
    return Cond_code::VS;
  else if (name == "vc")
    return Cond_code::VC;
  else if (name == "hi")
    return Cond_code::HI;
  else if (name == "ls")
    return Cond_code::LS;
  else if (name == "ge")
    return Cond_code::GE;
  else if (name == "lt")
    return Cond_code::LT;
  else if (name == "gt")
    return Cond_code::GT;
  else if (name == "le")
    return Cond_code::LE;
  throw Parse_error("unknown condition code " + std::string(name),
		    line_number);
}

uint64_t Parser::get_u64(const char *p)
{
  assert(isdigit(*p));
  unsigned __int128 value = 0;
  while (isdigit(*p))
    {
      value = value * 10 + (*p++ - '0');
      if (value > std::numeric_limits<uint64_t>::max())
	throw Parse_error("too large decimal integer value", line_number);
    }
  return value;
}

unsigned __int128 Parser::get_hex(const char *p)
{
  const unsigned __int128 max_val = -1;
  unsigned __int128 value = 0;
  p += 2;
  while (isxdigit(*p))
    {
      if (value > (max_val >> 4))
	throw Parse_error("too large hexadecimal value", line_number);
      unsigned nibble;
      if (isdigit(*p))
	nibble = *p - '0';
      else if (isupper(*p))
	nibble = 10 + (*p - 'A');
      else
	nibble = 10 + (*p - 'a');
      value = (value << 4) | nibble;
      p++;
    }
  return value;
}

unsigned __int128 Parser::get_hex_or_integer(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::hex && tokens[idx].kind != Lexeme::integer)
    throw Parse_error("expected a hexadecimal or decimal integer instead of "
		      + std::string(token_string(tokens[idx])), line_number);

  int pos = tokens[idx].pos;
  if (buf[pos] == '-')
    pos++;
  unsigned __int128 val;
  if (tokens[idx].kind == Lexeme::integer)
    val = get_u64(&buf[pos]);
  else
    val = get_hex(&buf[pos]);
  if (buf[tokens[idx].pos] == '-')
    val = -val;
  return val;
}

Inst *Parser::get_reg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].size == 3
      && buf[tokens[idx].pos + 0] == 'x'
      && buf[tokens[idx].pos + 1] == 'z'
      && buf[tokens[idx].pos + 2] == 'r')
    return zero_reg;
  if (tokens[idx].size == 3
      && buf[tokens[idx].pos + 0] == 'w'
      && buf[tokens[idx].pos + 1] == 'z'
      && buf[tokens[idx].pos + 2] == 'r')
    return zero_reg;
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 's'
      && buf[tokens[idx].pos + 1] == 'p')
    return rstate->registers[Aarch64RegIdx::sp];
  if (tokens[idx].size == 3
      && buf[tokens[idx].pos + 0] == 'w'
      && buf[tokens[idx].pos + 1] == 's'
      && buf[tokens[idx].pos + 2] == 'p')
    return rstate->registers[Aarch64RegIdx::sp];

  if (is_freg(idx))
    return get_freg(idx);

  if (tokens[idx].kind != Lexeme::name
      || (buf[tokens[idx].pos] != 'x' && buf[tokens[idx].pos] != 'w')
      || !isdigit(buf[tokens[idx].pos + 1])
      || (tokens[idx].size == 3 && !isdigit(buf[tokens[idx].pos + 2]))
      || tokens[idx].size > 3)
    throw Parse_error("expected a register instead of "
		      + std::string(token_string(tokens[idx])), line_number);

  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  if (tokens[idx].size == 3)
    value = value * 10 + (buf[tokens[idx].pos + 2] - '0');
  return rstate->registers[Aarch64RegIdx::x0 + value];
}

Inst *Parser::get_freg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);

  if (tokens[idx].kind != Lexeme::name
      || (buf[tokens[idx].pos] != 'q'
	  && buf[tokens[idx].pos] != 'd'
	  && buf[tokens[idx].pos] != 's'
	  && buf[tokens[idx].pos] != 'h'
	  && buf[tokens[idx].pos] != 'b')
      || !isdigit(buf[tokens[idx].pos + 1])
      || (tokens[idx].size == 3 && !isdigit(buf[tokens[idx].pos + 2]))
      || tokens[idx].size > 3)
    throw Parse_error("expected a floating-point register instead of "
		      + std::string(token_string(tokens[idx])), line_number);

  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  if (tokens[idx].size == 3)
    value = value * 10 + (buf[tokens[idx].pos + 2] - '0');
  return rstate->registers[Aarch64RegIdx::v0 + value];
}

Inst *Parser::get_freg_value(unsigned idx)
{
  Inst *inst = bb->build_inst(Op::READ, get_freg(idx));
  return bb->build_trunc(inst, get_reg_size(idx));
}

std::tuple<uint64_t, uint32_t, uint32_t> Parser::get_vreg_(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);

  if (tokens[idx].kind != Lexeme::name
      || tokens[idx].size < 4
      || buf[tokens[idx].pos] != 'v'
      || !isdigit(buf[tokens[idx].pos + 1]))
    throw Parse_error("expected a vector register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  uint32_t pos = 2;
  if (isdigit(buf[tokens[idx].pos + pos]))
    value = value * 10 + (buf[tokens[idx].pos + pos++] - '0');
  if (value > 31)
    throw Parse_error("expected a vector register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  uint64_t reg_idx = Aarch64RegIdx::v0 + value;
  std::string_view suffix(&buf[tokens[idx].pos + pos], tokens[idx].size - pos);
  if (suffix == ".2d")
    return {reg_idx, 2, 64};
  if (suffix == ".2s")
    return {reg_idx, 2, 32};
  if (suffix == ".4s")
    return {reg_idx, 4, 32};
  if (suffix == ".4h")
    return {reg_idx, 4, 16};
  if (suffix == ".8h")
    return {reg_idx, 8, 16};
  if (suffix == ".8b")
    return {reg_idx, 8, 8};
  if (suffix == ".16b")
    return {reg_idx, 16, 8};

  throw Parse_error("expected a vector register instead of "
		    + std::string(token_string(tokens[idx])), line_number);
}

std::tuple<Inst *, uint32_t, uint32_t> Parser::get_vreg(unsigned idx)
{
  auto [reg_idx, nof_elem, elem_bitsize] = get_vreg_(idx);
  return {rstate->registers[reg_idx], nof_elem, elem_bitsize};
}

std::tuple<Inst *, uint32_t, uint32_t> Parser::get_scalar_vreg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);

  if (tokens[idx].kind != Lexeme::name
      || tokens[idx].size < 4
      || buf[tokens[idx].pos] != 'v'
      || !isdigit(buf[tokens[idx].pos + 1]))
    throw Parse_error("expected a vector register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  uint32_t pos = 2;
  if (isdigit(buf[tokens[idx].pos + pos]))
    value = value * 10 + (buf[tokens[idx].pos + pos++] - '0');
  if (value > 31)
    throw Parse_error("expected a vector register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  Inst *reg = rstate->registers[Aarch64RegIdx::v0 + value];
  std::string_view suffix(&buf[tokens[idx].pos + pos], tokens[idx].size - pos);
  uint32_t elem_bitsize;
  if (suffix == ".d")
    elem_bitsize = 64;
  if (suffix == ".s")
    elem_bitsize = 32;
  if (suffix == ".h")
    elem_bitsize = 16;
  if (suffix == ".b")
    elem_bitsize = 8;
  uint32_t nof_elem = 128 / elem_bitsize;
  idx++;

  get_left_bracket(idx++);
  uint32_t elem_idx = get_imm(idx++)->value();
  if (elem_idx >= nof_elem)
    throw Parse_error("elem index out of range", line_number);
  get_right_bracket(idx);

  return {reg, elem_bitsize, elem_idx};
}

Inst *Parser::get_vreg_value(unsigned idx, uint32_t nof_elem,
			     uint32_t elem_bitsize)
{
  auto [dest, dest_nof_elem, dest_elem_bitsize] = get_vreg(idx);
  if (nof_elem != dest_nof_elem || elem_bitsize != dest_elem_bitsize)
    throw Parse_error("expected same arg vector size as dest", line_number);
  Inst *inst = bb->build_inst(Op::READ, dest);
  return bb->build_trunc(inst, nof_elem * elem_bitsize);
}

uint64_t Parser::get_vreg_idx(unsigned idx, uint32_t nof_elem,
			      uint32_t elem_bitsize)
{
  auto [reg_idx, dest_nof_elem, dest_elem_bitsize] = get_vreg_(idx);
  if (nof_elem != dest_nof_elem || elem_bitsize != dest_elem_bitsize)
    throw Parse_error("expected same arg vector size as dest", line_number);
  return reg_idx;
}

uint32_t Parser::get_reg_size(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);

  if (tokens[idx].kind == Lexeme::name)
    {
      if (tokens[idx].size == 3
	  && buf[tokens[idx].pos + 0] == 'x'
	  && buf[tokens[idx].pos + 1] == 'z'
	  && buf[tokens[idx].pos + 2] == 'r')
	return 64;
      if (tokens[idx].size == 3
	  && buf[tokens[idx].pos + 0] == 'w'
	  && buf[tokens[idx].pos + 1] == 'z'
	  && buf[tokens[idx].pos + 2] == 'r')
	return 32;
      if (tokens[idx].size == 2
	  && buf[tokens[idx].pos + 0] == 's'
	  && buf[tokens[idx].pos + 1] == 'p')
	return 64;
      if (tokens[idx].size == 3
	  && buf[tokens[idx].pos + 0] == 'w'
	  && buf[tokens[idx].pos + 1] == 's'
	  && buf[tokens[idx].pos + 2] == 'p')
	return 32;

      if (buf[tokens[idx].pos] == 'x')
	return 64;
      if (buf[tokens[idx].pos] == 'w')
	return 32;
      if (buf[tokens[idx].pos] == 'q')
	return 128;
      if (buf[tokens[idx].pos] == 'd')
	return 64;
      if (buf[tokens[idx].pos] == 's')
	return 32;
      if (buf[tokens[idx].pos] == 'h')
	return 16;
      if (buf[tokens[idx].pos] == 'b')
	return 8;
    }

  throw Parse_error("expected a floating-point register", line_number);
}

bool Parser::is_vector_op()
{
  return is_vreg(1);
}

bool Parser::is_freg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);

  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 's'
      && buf[tokens[idx].pos + 1] == 'p')
    return false;

  if (tokens[idx].kind == Lexeme::name
      && (buf[tokens[idx].pos] == 'q'
	  || buf[tokens[idx].pos] == 'd'
	  || buf[tokens[idx].pos] == 's'
	  || buf[tokens[idx].pos] == 'h'
	  || buf[tokens[idx].pos] == 'b'))
      return true;

  return false;
}

bool Parser::is_vreg(unsigned idx)
{
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::name)
    return false;
  if (buf[tokens[idx].pos] != 'v')
    return false;

  if (isdigit(buf[tokens[idx].pos + 1]))
    {
      if ((buf[tokens[idx].pos + 2] == '.')
	  || (isdigit(buf[tokens[idx].pos + 2])
	      && buf[tokens[idx].pos + 3] == '.'))
	return true;
    }

  return false;
}

Inst *Parser::get_imm(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  return bb->value_inst(get_hex_or_integer(idx), reg_bitsize);
}

Inst *Parser::get_reg_value(unsigned idx)
{
  if (is_freg(idx))
    return get_freg_value(idx);

  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].size == 3
      && buf[tokens[idx].pos + 0] == 'x'
      && buf[tokens[idx].pos + 1] == 'z'
      && buf[tokens[idx].pos + 2] == 'r')
    return bb->value_inst(0, reg_bitsize);
  if (tokens[idx].size == 3
      && buf[tokens[idx].pos + 0] == 'w'
      && buf[tokens[idx].pos + 1] == 'z'
      && buf[tokens[idx].pos + 2] == 'r')
    return bb->value_inst(0, 32);
  Inst *inst = bb->build_inst(Op::READ, get_reg(idx));
  if (buf[tokens[idx].pos] == 'w')
    inst = bb->build_trunc(inst, 32);
  return inst;
}

Inst *Parser::get_reg_or_imm_value(unsigned idx, uint32_t bitsize)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind == Lexeme::name)
    return get_reg_value(idx);
  else
    return bb->build_trunc(get_imm(idx), bitsize);
}

Basic_block *Parser::get_bb(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::label)
    throw Parse_error("expected a label instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  std::string_view label(&buf[tokens[idx].pos], tokens[idx].size);
  auto I = label2bb.find(label);
  if (I != label2bb.end())
    return I->second;
  Basic_block *new_bb = func->build_bb();
  label2bb.insert({label, new_bb});
  return new_bb;
}

Basic_block *Parser::get_bb_def(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::label_def)
    throw Parse_error("expected a label instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  assert(tokens[idx].size > 0
	 && buf[tokens[idx].pos + tokens[idx].size - 1] == ':');
  std::string_view label(&buf[tokens[idx].pos], tokens[idx].size - 1);
  auto I = label2bb.find(label);
  if (I != label2bb.end())
    return I->second;
  Basic_block *new_bb = func->build_bb();
  label2bb.insert({label, new_bb});
  return new_bb;
}

void Parser::get_comma(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::comma)
    throw Parse_error("expected a ',' after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
}

void Parser::get_exclamation(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::exclamation)
    throw Parse_error("expected a '!' after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
}

void Parser::get_minus(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::minus)
    throw Parse_error("expected a '-' after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
}

void Parser::get_left_bracket(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::left_bracket)
    throw Parse_error("expected a '[' after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
}

void Parser::get_right_bracket(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::right_bracket)
    throw Parse_error("expected a ']' after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
}

void Parser::get_left_brace(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::left_brace)
    throw Parse_error("expected a '{' after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
}

void Parser::get_right_brace(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::right_brace)
    throw Parse_error("expected a '}' after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
}

void Parser::get_end_of_line(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() > idx)
    throw Parse_error("expected end of line after "
		      + std::string(token_string(tokens[idx - 1])),
				    line_number);
}

void Parser::write_reg(Inst *reg, Inst *value)
{
  assert(reg->op == Op::REGISTER);
  uint32_t reg_bitsize = reg->args[0]->value();
  if (reg_bitsize > value->bitsize)
    value = bb->build_inst(Op::ZEXT, value, reg_bitsize);
  bb->build_inst(Op::WRITE, reg, value);
}

Inst *Parser::build_cond(Cond_code cc)
{
  switch (cc)
    {
    case Cond_code::EQ:
      return bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z]);
    case Cond_code::NE:
      {
	Inst *z = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z]);
	return bb->build_inst(Op::NOT, z);
      }
    case Cond_code::CS:
      return bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
    case Cond_code::CC:
      {
	Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
	return bb->build_inst(Op::NOT, c);
      }
    case Cond_code::MI:
      return bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::n]);
    case Cond_code::PL:
      {
	Inst *n = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::n]);
	return bb->build_inst(Op::NOT, n);
      }
    case Cond_code::VS:
      return bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::v]);
    case Cond_code::VC:
      {
	Inst *v = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::v]);
	return bb->build_inst(Op::NOT, v);
      }
    case Cond_code::HI:
      {
	Inst *z = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z]);
	Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
	Inst *not_z = bb->build_inst(Op::NOT, z);
	return bb->build_inst(Op::AND, not_z, c);
      }
    case Cond_code::LS:
      {
	Inst *z = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z]);
	Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
	Inst *not_c = bb->build_inst(Op::NOT, c);
	return bb->build_inst(Op::OR, z, not_c);
      }
    case Cond_code::GE:
      {
	Inst *n = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::n]);
	Inst *v = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::v]);
	return bb->build_inst(Op::EQ, n, v);
      }
    case Cond_code::LT:
      {
	Inst *n = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::n]);
	Inst *v = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::v]);
	return bb->build_inst(Op::NE, n, v);
      }
    case Cond_code::GT:
      {
	Inst *n = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::n]);
	Inst *z = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z]);
	Inst *v = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::v]);
	Inst *not_z = bb->build_inst(Op::NOT, z);
	return bb->build_inst(Op::AND, not_z, bb->build_inst(Op::EQ, n, v));
      }
    case Cond_code::LE:
      {
	Inst *n = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::n]);
	Inst *z = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z]);
	Inst *v = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::v]);
	return bb->build_inst(Op::OR, z, bb->build_inst(Op::NE, n, v));
      }
    }

  throw Parse_error("unhandled condition code", line_number);
}

Inst *gen_s2f(Basic_block *bb, Inst *elem1)
{
  return bb->build_inst(Op::S2F, elem1, elem1->bitsize);
}

Inst *gen_u2f(Basic_block *bb, Inst *elem1)
{
  return bb->build_inst(Op::U2F, elem1, elem1->bitsize);
}

Inst *gen_f2s(Basic_block *bb, Inst *elem1)
{
  return bb->build_inst(Op::F2S, elem1, elem1->bitsize);
}

Inst *gen_f2u(Basic_block *bb, Inst *elem1)
{
  return bb->build_inst(Op::F2U, elem1, elem1->bitsize);
}

Inst *gen_abs(Basic_block *bb, Inst *elem1)
{
  Inst *neg = bb->build_inst(Op::NEG, elem1);
  Inst *zero = bb->value_inst(0, elem1->bitsize);
  Inst *cond = bb->build_inst(Op::SLE, zero, elem1);
  return bb->build_inst(Op::ITE, cond, elem1, neg);
}

Inst *gen_cmtst(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *inst = bb->build_inst(Op::AND, elem1, elem2);
  Inst *zero = bb->value_inst(0, elem1->bitsize);
  Inst *cmp = bb->build_inst(Op::NE, inst, zero);
  return bb->build_inst(Op::SEXT, cmp, elem1->bitsize);
}

Inst *gen_smin(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *cmp = bb->build_inst(Op::SLT, elem1, elem2);
  return bb->build_inst(Op::ITE, cmp, elem1, elem2);
}

Inst *gen_umin(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *cmp = bb->build_inst(Op::ULT, elem1, elem2);
  return bb->build_inst(Op::ITE, cmp, elem1, elem2);
}

Inst *gen_smax(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *cmp = bb->build_inst(Op::SLT, elem1, elem2);
  return bb->build_inst(Op::ITE, cmp, elem2, elem1);
}

Inst *gen_umax(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *cmp = bb->build_inst(Op::ULT, elem1, elem2);
  return bb->build_inst(Op::ITE, cmp, elem2, elem1);
}

Inst *gen_add(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return bb->build_inst(Op::ADD, elem1, elem2);
}

Inst *gen_sub(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return bb->build_inst(Op::SUB, elem1, elem2);
}

Inst *gen_mul(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return bb->build_inst(Op::MUL, elem1, elem2);
}

Inst *gen_abd(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *inst = bb->build_inst(Op::SUB, elem1, elem2);
  Inst *neg_inst = bb->build_inst(Op::NEG, inst);
  Inst *zero = bb->value_inst(0, inst->bitsize);
  Inst *cmp = bb->build_inst(Op::SLT, inst, zero);
  return bb->build_inst(Op::ITE, cmp, neg_inst, inst);
}

Inst *gen_fnmul(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *res = bb->build_inst(Op::FMUL, elem1, elem2);
  return bb->build_inst(Op::FNEG, res);
}

Inst *gen_fmin_fmax(Basic_block *bb, Inst *elem1, Inst *elem2, bool is_min)
{
  Inst *is_nan = bb->build_inst(Op::IS_NAN, elem2);
  Inst *cmp;
  if (is_min)
    cmp = bb->build_inst(Op::FLT, elem1, elem2);
  else
    cmp = bb->build_inst(Op::FLT, elem2, elem1);
  Inst *res1 = bb->build_inst(Op::ITE, cmp, elem1, elem2);
  Inst *res2 = bb->build_inst(Op::ITE, is_nan, elem1, res1);
  // 0.0 and -0.0 is equal as floating-point values, and fmin(0.0, -0.0)
  // may return eiter of them. But we treat them as 0.0 > -0.0 here,
  // otherwise we will report miscompilations when GCC switch the order
  // of the arguments.
  Inst *zero = bb->value_inst(0, elem1->bitsize);
  Inst *is_zero1 = bb->build_inst(Op::FEQ, elem1, zero);
  Inst *is_zero2 = bb->build_inst(Op::FEQ, elem2, zero);
  Inst *is_zero = bb->build_inst(Op::AND, is_zero1, is_zero2);
  Inst *cmp2;
  if (is_min)
    cmp2 = bb->build_inst(Op::SLT, elem1, elem2);
  else
    cmp2 = bb->build_inst(Op::SLT, elem2, elem1);
  Inst *res3 = bb->build_inst(Op::ITE, cmp2, elem1, elem2);
  Inst *res = bb->build_inst(Op::ITE, is_zero, res3, res2);
  return res;
}

Inst *gen_fmin(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return gen_fmin_fmax(bb, elem1, elem2, true);
}

Inst *gen_fmax(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return gen_fmin_fmax(bb, elem1, elem2, false);
}

Inst *gen_sshl(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *zero = bb->value_inst(0, 8);
  elem2 = bb->build_trunc(elem2, 8);
  Inst *is_rshift = bb->build_inst(Op::SLT, elem2, zero);
  Inst *lshift = elem2;
  Inst *rshift = bb->build_inst(Op::NEG, elem2);
  if (elem1->bitsize > elem2->bitsize)
    {
      lshift = bb->build_inst(Op::ZEXT, elem2, elem1->bitsize);
      rshift = bb->build_inst(Op::ZEXT, rshift, elem1->bitsize);
    }
  lshift = bb->build_inst(Op::SHL, elem1, lshift);
  rshift = bb->build_inst(Op::ASHR, elem1, rshift);
  return bb->build_inst(Op::ITE, is_rshift, rshift, lshift);
}

Inst *gen_ushl(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  Inst *zero = bb->value_inst(0, 8);
  elem2 = bb->build_trunc(elem2, 8);
  Inst *is_rshift = bb->build_inst(Op::SLT, elem2, zero);
  Inst *lshift = elem2;
  Inst *rshift = bb->build_inst(Op::NEG, elem2);
  if (elem1->bitsize > elem2->bitsize)
    {
      lshift = bb->build_inst(Op::ZEXT, elem2, elem1->bitsize);
      rshift = bb->build_inst(Op::ZEXT, rshift, elem1->bitsize);
    }
  lshift = bb->build_inst(Op::SHL, elem1, lshift);
  rshift = bb->build_inst(Op::LSHR, elem1, rshift);
  return bb->build_inst(Op::ITE, is_rshift, rshift, lshift);
}

Inst *gen_sqxtn(Basic_block *bb, Inst *elem1)
{
  __int128 smax_val = (((__int128)1) << (elem1->bitsize / 2 - 1)) - 1;
  __int128 smin_val = ((__int128)-1) << (elem1->bitsize / 2 - 1);
  Inst *smax1 = bb->value_inst(smax_val, elem1->bitsize);
  Inst *smax2 = bb->value_inst(smax_val, elem1->bitsize / 2);
  Inst *smin1 = bb->value_inst(smin_val, elem1->bitsize);
  Inst *smin2 = bb->value_inst(smin_val, elem1->bitsize / 2);
  Inst *res = bb->build_trunc(elem1, elem1->bitsize / 2);
  Inst *cmp1 = bb->build_inst(Op::SLT, elem1, smin1);
  Inst *cmp2 = bb->build_inst(Op::SLT, smax1, elem1);
  res = bb->build_inst(Op::ITE, cmp1, smin2, res);
  return bb->build_inst(Op::ITE, cmp2, smax2, res);
}

Inst *gen_simd_compare(Basic_block *bb, Inst *elem1, Inst *elem2, SIMD_cond op)
{
  Inst *cond;
  switch (op)
    {
    case SIMD_cond::EQ:
      cond = bb->build_inst(Op::EQ, elem1, elem2);
      break;
    case SIMD_cond::GE:
      cond = bb->build_inst(Op::SLE, elem2, elem1);
      break;
    case SIMD_cond::GT:
      cond = bb->build_inst(Op::SLT, elem2, elem1);
      break;
    case SIMD_cond::HI:
      cond = bb->build_inst(Op::ULT, elem2, elem1);
      break;
    case SIMD_cond::HS:
      cond = bb->build_inst(Op::ULE, elem2, elem1);
      break;
    case SIMD_cond::LE:
      cond = bb->build_inst(Op::SLE, elem1, elem2);
      break;
    case SIMD_cond::LT:
      cond = bb->build_inst(Op::SLT, elem1, elem2);
      break;
    case SIMD_cond::FEQ:
      cond = bb->build_inst(Op::FEQ, elem1, elem2);
      break;
    case SIMD_cond::FGE:
      cond = bb->build_inst(Op::FLE, elem2, elem1);
      break;
    case SIMD_cond::FGT:
      cond = bb->build_inst(Op::FLT, elem2, elem1);
      break;
    case SIMD_cond::FLE:
      cond = bb->build_inst(Op::FLE, elem1, elem2);
      break;
    case SIMD_cond::FLT:
      cond = bb->build_inst(Op::FLT, elem1, elem2);
      break;
    }
  return bb->build_inst(Op::SEXT, cond, elem1->bitsize);
}

void Parser::process_cond_branch(Cond_code cc)
{
  Basic_block *true_bb = get_bb(1);
  get_end_of_line(2);

  Inst *cond = build_cond(cc);
  Basic_block *false_bb = func->build_bb();
  bb->build_br_inst(cond, true_bb, false_bb);
  bb = false_bb;
}

void Parser::process_cbz(bool is_cbnz)
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Basic_block *true_bb = get_bb(3);
  get_end_of_line(4);

  Basic_block *false_bb = func->build_bb();
  Inst *zero = bb->value_inst(0, arg1->bitsize);
  Inst *cond = bb->build_inst(is_cbnz ? Op::NE : Op::EQ, arg1, zero);
  bb->build_br_inst(cond, true_bb, false_bb);
  bb = false_bb;
}

void Parser::process_tbz(bool is_cbnz)
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = get_imm(3);
  get_comma(4);
  Basic_block *true_bb = get_bb(5);
  get_end_of_line(6);

  Basic_block *false_bb = func->build_bb();
  Inst *cond = bb->build_extract_bit(arg1, arg2->value());
  if (!is_cbnz)
    cond = bb->build_inst(Op::NOT, cond);
  bb->build_br_inst(cond, true_bb, false_bb);
  bb = false_bb;
}

void Parser::process_csel(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_comma(6);
  Cond_code cc = get_cc(7);
  get_end_of_line(8);

  Inst *cond = build_cond(cc);
  if (op == Op::NEG || op == Op::NOT)
    arg2 = bb->build_inst(op, arg2);
  else if (op == Op::ADD)
    {
      Inst *one = bb->value_inst(1, arg2->bitsize);
      arg2 = bb->build_inst(op, arg2, one);
    }
  else
    assert(op == Op::MOV);
  Inst *res = bb->build_inst(Op::ITE, cond, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_cset(Op op)
{
  Inst *dest = get_reg(1);
  uint32_t dest_bitsize = get_reg_size(1);
  get_comma(2);
  Cond_code cc = get_cc(3);
  get_end_of_line(4);

  Inst *cond = build_cond(cc);
  Inst *res = bb->build_inst(op, cond, dest_bitsize);
  write_reg(dest, res);
}

void Parser::process_cinc()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Cond_code cc = get_cc(5);
  get_end_of_line(6);

  Inst *cond = build_cond(cc);
  Inst *inst = bb->build_inst(Op::ADD, arg1, bb->value_inst(1, arg1->bitsize));
  Inst *res = bb->build_inst(Op::ITE, cond, inst, arg1);
  write_reg(dest, res);
}

void Parser::process_call()
{
  std::string_view name = get_name(1);
  get_end_of_line(2);

  throw Not_implemented("call " + std::string(name));
}

Inst *Parser::process_address(unsigned idx)
{
  get_left_bracket(idx++);
  Inst *ptr_reg = get_reg(idx);
  Inst *ptr = get_reg_value(idx++);
  if (tokens.size() > idx && tokens[idx].kind == Lexeme::comma)
    {
      get_comma(idx++);
      Inst *offset;
      if (tokens.size() > idx && tokens[idx].kind == Lexeme::lo12)
	{
	  idx++;
	  offset = get_sym_addr(idx++);
	  if (tokens.size() > idx && tokens[idx].kind == Lexeme::plus)
	    {
	      idx++;
	      Inst *offset2 = get_reg_or_imm_value(idx++, ptr->bitsize);
	      offset = bb->build_inst(Op::ADD, offset, offset2);
	    }
	  else if (tokens.size() > idx && tokens[idx].kind == Lexeme::minus)
	    {
	      idx++;
	      Inst *offset2 = get_reg_or_imm_value(idx++, ptr->bitsize);
	      offset = bb->build_inst(Op::SUB, offset, offset2);
	    }
	  offset = bb->build_trunc(offset, 12);
	  offset = bb->build_inst(Op::ZEXT, offset, ptr->bitsize);
	}
      else
	offset = get_reg_or_imm_value(idx++, ptr->bitsize);
      if (tokens.size() > idx && tokens[idx].kind == Lexeme::comma)
	{
	  get_comma(idx++);
	  std::string_view cmd = get_name(idx++);
	  if (cmd == "sxtw")
	    offset = bb->build_inst(Op::SEXT, offset, ptr->bitsize);
	  else if (cmd == "uxtw")
	    offset = bb->build_inst(Op::ZEXT, offset, ptr->bitsize);
	  else if (cmd == "lsl")
	    ;  // Nothing to do.
	  else
	    throw Parse_error("unknown extension " + std::string(cmd),
			      line_number);

	  if (tokens.size() > idx && tokens[idx].kind != Lexeme::right_bracket)
	    {
	      Inst *shift = get_imm(idx++);
	      offset = bb->build_inst(Op::SHL, offset, shift);
	    }
	}
      get_right_bracket(idx++);
      ptr = bb->build_inst(Op::ADD, ptr, offset);
      if (tokens.size() > idx)
	{
	  get_exclamation(idx++);
	  bb->build_inst(Op::WRITE, ptr_reg, ptr);
	}
    }
  else
    {
      get_right_bracket(idx++);

      if (tokens.size() > idx && tokens[idx].kind == Lexeme::comma)
	{
	  get_comma(idx++);
	  Inst *offset = get_imm(idx++);

	  Inst *updated_ptr = bb->build_inst(Op::ADD, ptr, offset);
	  bb->build_inst(Op::WRITE, ptr_reg, updated_ptr);
	}
    }
  get_end_of_line(idx);

  return ptr;
}

void Parser::store_ub_check(Inst *ptr, uint64_t size)
{
  Inst *ptr_mem_id = bb->build_extract_id(ptr);

  // It is UB to write to constant memory.
  Inst *is_const = bb->build_inst(Op::IS_CONST_MEM, ptr_mem_id);
  bb->build_inst(Op::UB, is_const);

  // It is UB if the store overflows into a different memory object.
  Inst *size_inst = bb->value_inst(size - 1, ptr->bitsize);
  Inst *last_addr = bb->build_inst(Op::ADD, ptr, size_inst);
  Inst *last_mem_id = bb->build_extract_id(last_addr);
  Inst *is_ub = bb->build_inst(Op::NE, ptr_mem_id, last_mem_id);
  bb->build_inst(Op::UB, is_ub);

  // It is UB if the end is outside the memory object -- the start is
  // obviously in the memory object if the end is within the object.
  // Otherwise, the  previous overflow check would have failed.
  Inst *mem_size = bb->build_inst(Op::GET_MEM_SIZE, ptr_mem_id);
  Inst *offset = bb->build_extract_offset(last_addr);
  Inst *out_of_bound = bb->build_inst(Op::ULE, mem_size, offset);
  bb->build_inst(Op::UB, out_of_bound);
}

void Parser::load_ub_check(Inst *ptr, uint64_t size)
{
  Inst *ptr_mem_id = bb->build_extract_id(ptr);

  // It is UB if the store overflows into a different memory object.
  Inst *size_inst = bb->value_inst(size - 1, ptr->bitsize);
  Inst *last_addr = bb->build_inst(Op::ADD, ptr, size_inst);
  Inst *last_mem_id = bb->build_extract_id(last_addr);
  Inst *is_ub = bb->build_inst(Op::NE, ptr_mem_id, last_mem_id);
  bb->build_inst(Op::UB, is_ub);

  // It is UB if the start is outside the memory object.
  // The RISC-V backend sometimes reads where the end is outside the memory
  // object. For example, consider the variable `c` defined as follows:
  //
  //      .globl  c
  //      .align  2
  //      .type   c, @object
  //      .size   c, 3
  //   c:
  //      .zero   3
  //
  // This variable may be read using a four-byte load, which is acceptable
  // because the alignment ensures at least one byte is available beyond
  // the object.
  //
  // TODO: We should improve this to verify that it does not read more bytes
  // than are guaranteed to be available.
  Inst *mem_size = bb->build_inst(Op::GET_MEM_SIZE, ptr_mem_id);
  Inst *offset = bb->build_extract_offset(ptr);
  Inst *out_of_bound = bb->build_inst(Op::ULE, mem_size, offset);
  bb->build_inst(Op::UB, out_of_bound);
}

void Parser::process_load(uint32_t trunc_size, Op op)
{
  uint32_t size;
  Inst *dest;
  if (is_freg(1))
    {
      dest = get_freg(1);
      size = get_reg_size(1) / 8;
    }
  else
    {
      dest = get_reg(1);
      size = buf[tokens[1].pos] == 'w' ? 4 : 8;
    }
  get_comma(2);
  Inst *ptr = process_address(3);

  if (trunc_size)
    size = trunc_size;
  load_ub_check(ptr, size);
  Inst *value = bb->build_inst(Op::LOAD, ptr);
  for (uint32_t i = 1; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, offset);
      Inst *byte = bb->build_inst(Op::LOAD, addr);
      value = bb->build_inst(Op::CONCAT, byte, value);
    }
  if (op != Op::ZEXT)
    value = bb->build_inst(op, value, reg_bitsize);
  write_reg(dest, value);
}

void Parser::process_ldp()
{
  Inst *dest1;
  uint32_t size;
  if (is_freg(1))
    {
      dest1 = get_freg(1);
      size = get_reg_size(1) / 8;
    }
  else
    {
      dest1 = get_reg(1);
      size = buf[tokens[1].pos] == 'w' ? 4 : 8;
    }
  get_comma(2);
  Inst *dest2 = get_reg(3);
  get_comma(4);
  Inst *ptr = process_address(5);

  load_ub_check(ptr, size * 2);
  Inst *value = bb->build_inst(Op::LOAD, ptr);
  for (uint32_t i = 1; i < size * 2; i++)
    {
      Inst *offset = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, offset);
      Inst *byte = bb->build_inst(Op::LOAD, addr);
      value = bb->build_inst(Op::CONCAT, byte, value);
    }
  uint32_t bitsize = size * 8;
  Inst *value1 = bb->build_trunc(value, bitsize);
  Inst *value2 = bb->build_inst(Op::EXTRACT, value, bitsize * 2 - 1, bitsize);

  write_reg(dest1, value1);
  write_reg(dest2, value2);
}

void Parser::process_ldpsw()
{
  Inst *dest1 = get_reg(1);
  get_comma(2);
  Inst *dest2 = get_reg(3);
  get_comma(4);
  Inst *ptr = process_address(5);

  uint32_t size = 4;
  load_ub_check(ptr, size * 2);
  Inst *value = bb->build_inst(Op::LOAD, ptr);
  for (uint32_t i = 1; i < size * 2; i++)
    {
      Inst *offset = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, offset);
      Inst *byte = bb->build_inst(Op::LOAD, addr);
      value = bb->build_inst(Op::CONCAT, byte, value);
    }
  uint32_t bitsize = size * 8;
  Inst *value1 = bb->build_trunc(value, bitsize);
  value1 = bb->build_inst(Op::SEXT, value1, 64);
  Inst *value2 = bb->build_inst(Op::EXTRACT, value, bitsize * 2 - 1, bitsize);
  value2 = bb->build_inst(Op::SEXT, value2, 64);

  write_reg(dest1, value1);
  write_reg(dest2, value2);
}

void Parser::process_store(uint32_t trunc_size)
{
  Inst *value = get_reg_value(1);
  get_comma(2);
  Inst *ptr = process_address(3);

  if (trunc_size)
    value = bb->build_trunc(value, trunc_size * 8);
  uint32_t size = value->bitsize / 8;
  store_ub_check(ptr, size);
  for (uint32_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, offset);
      Inst *byte = bb->build_inst(Op::EXTRACT, value, i * 8 + 7, i * 8);
      bb->build_inst(Op::STORE, addr, byte);
    }
}

void Parser::process_stp()
{
  Inst *value1 = get_reg_value(1);
  get_comma(2);
  Inst *value2 = get_reg_value(3);
  get_comma(4);
  Inst *ptr = process_address(5);

  Inst *value = bb->build_inst(Op::CONCAT, value2, value1);
  uint32_t size = value->bitsize / 8;
  store_ub_check(ptr, size);
  for (uint32_t i = 0; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, offset);
      Inst *byte = bb->build_inst(Op::EXTRACT, value, i * 8 + 7, i * 8);
      bb->build_inst(Op::STORE, addr, byte);
    }
}

void Parser::process_fcmp()
{
  Inst *arg1 = get_freg_value(1);
  get_comma(2);
  Inst *arg2 = get_freg_value(3);
  get_end_of_line(4);

  Inst *b0 = bb->value_inst(0, 1);
  Inst *b1 = bb->value_inst(1, 1);

  Inst *is_inf1 = bb->build_inst(Op::IS_INF, arg1);
  Inst *is_inf2 = bb->build_inst(Op::IS_INF, arg2);
  Inst *is_inf = bb->build_inst(Op::AND, is_inf1, is_inf2);

  Inst *is_nan1 = bb->build_inst(Op::IS_NAN, arg1);
  Inst *is_nan2 = bb->build_inst(Op::IS_NAN, arg2);
  Inst *any_is_nan = bb->build_inst(Op::OR, is_nan1, is_nan2);

  Inst *cmp_n = bb->build_inst(Op::FLT, arg1, arg2);
  Inst *n = bb->build_inst(Op::ITE, any_is_nan, b0, cmp_n);

  Inst *cmp_z = bb->build_inst(Op::FEQ, arg1, arg2);
  Inst *cmp_z_inf = bb->build_inst(Op::EQ, arg1, arg2);
  Inst *z = bb->build_inst(Op::ITE, is_inf, cmp_z_inf, cmp_z);
  z = bb->build_inst(Op::ITE, any_is_nan, b0, z);

  Inst *cmp_c = bb->build_inst(Op::FLE, arg2, arg1);
  Inst *cmp_c_inf = bb->build_inst(Op::NOT, cmp_n);
  Inst *c = bb->build_inst(Op::ITE, is_inf, cmp_c_inf, cmp_c);
  c = bb->build_inst(Op::ITE, any_is_nan, b1, c);

  Inst *v = any_is_nan;

  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::n], n);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::z], z);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::c], c);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::v], v);
}

void Parser::process_fccmp()
{
  Inst *arg1 = get_freg_value(1);
  get_comma(2);
  Inst *arg2 = get_freg_value(3);
  get_comma(4);
  Inst *arg3 = get_imm(5);
  get_comma(6);
  Cond_code cc = get_cc(7);
  get_end_of_line(8);

  Inst *b0 = bb->value_inst(0, 1);
  Inst *b1 = bb->value_inst(1, 1);

  Inst *is_inf1 = bb->build_inst(Op::IS_INF, arg1);
  Inst *is_inf2 = bb->build_inst(Op::IS_INF, arg2);
  Inst *is_inf = bb->build_inst(Op::AND, is_inf1, is_inf2);

  Inst *is_nan1 = bb->build_inst(Op::IS_NAN, arg1);
  Inst *is_nan2 = bb->build_inst(Op::IS_NAN, arg2);
  Inst *any_is_nan = bb->build_inst(Op::OR, is_nan1, is_nan2);

  Inst *cmp_n = bb->build_inst(Op::FLT, arg1, arg2);
  Inst *n1 = bb->build_inst(Op::ITE, any_is_nan, b0, cmp_n);

  Inst *cmp_z = bb->build_inst(Op::FEQ, arg1, arg2);
  Inst *cmp_z_inf = bb->build_inst(Op::EQ, arg1, arg2);
  Inst *z1 = bb->build_inst(Op::ITE, is_inf, cmp_z_inf, cmp_z);
  z1 = bb->build_inst(Op::ITE, any_is_nan, b0, z1);

  Inst *cmp_c = bb->build_inst(Op::FLE, arg2, arg1);
  Inst *cmp_c_inf = bb->build_inst(Op::NOT, cmp_n);
  Inst *c1 = bb->build_inst(Op::ITE, is_inf, cmp_c_inf, cmp_c);
  c1 = bb->build_inst(Op::ITE, any_is_nan, b1, c1);

  Inst *v1 = any_is_nan;

  uint32_t flags = arg3->value();
  assert(flags < 16);

  Inst *n2 = bb->value_inst((flags & 8) != 0, 1);
  Inst *z2 = bb->value_inst((flags & 4) != 0, 1);
  Inst *c2 = bb->value_inst((flags & 2) != 0, 1);
  Inst *v2 = bb->value_inst((flags & 1) != 0, 1);

  Inst *cond = build_cond(cc);
  Inst *n = bb->build_inst(Op::ITE, cond, n1, n2);
  Inst *z = bb->build_inst(Op::ITE, cond, z1, z2);
  Inst *c = bb->build_inst(Op::ITE, cond, c1, c2);
  Inst *v = bb->build_inst(Op::ITE, cond, v1, v2);

  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::n], n);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::z], z);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::c], c);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::v], v);
}

void Parser::process_i2f(bool is_unsigned)
{
  Inst *dest = get_freg(1);
  uint32_t dest_bitsize = get_reg_size(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  Inst *arg2 = nullptr;
  if (tokens.size() > 4)
    {
      get_comma(4);
      arg2 = get_imm(5);
      get_end_of_line(6);
    }
  else
    get_end_of_line(4);

  Op op = is_unsigned ? Op::U2F : Op::S2F;
  Inst *res = bb->build_inst(op, arg1, dest_bitsize);
  if (arg2)
    {
      Inst *scale = bb->value_inst(1ull << arg2->value(), 64);
      scale = bb->build_inst(Op::U2F, scale, dest_bitsize);
      res = bb->build_inst(Op::FDIV, res, scale);
    }
  write_reg(dest, res);
}

void Parser::process_f2i(bool is_unsigned)
{
  Inst *dest = get_reg(1);
  uint32_t dest_bitsize = get_reg_size(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  Inst *arg2 = nullptr;
  if (tokens.size() > 4)
    {
      get_comma(4);
      arg2 = get_imm(5);
      get_end_of_line(6);
    }
  else
    get_end_of_line(4);

  if (arg2)
    {
      Inst *scale = bb->value_inst(1ull << arg2->value(), 64);
      scale = bb->build_inst(Op::U2F, scale, arg1->bitsize);
      arg1 = bb->build_inst(Op::FMUL, arg1, scale);
    }
  Op op = is_unsigned ? Op::F2U : Op::F2S;
  Inst *res = bb->build_inst(op, arg1, dest_bitsize);
  write_reg(dest, res);
}

void Parser::process_f2f()
{
  Inst *dest = get_freg(1);
  uint32_t dest_bitsize = get_reg_size(1);
  get_comma(2);
  Inst *arg1 = get_freg_value(3);
  get_end_of_line(4);

  Inst *res = bb->build_inst(Op::FCHPREC, arg1, dest_bitsize);
  write_reg(dest, res);
}

void Parser::process_mul_op(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_comma(6);
  Inst *arg3 = get_reg_value(7);
  get_end_of_line(8);

  Inst *mul = bb->build_inst(Op::MUL, arg1, arg2);
  Inst *res = bb->build_inst(op, arg3, mul);
  write_reg(dest, res);
}

void Parser::process_maddl(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_comma(6);
  Inst *arg3 = get_reg_value(7);
  get_end_of_line(8);

  arg1 = bb->build_inst(op, arg1, 64);
  arg2 = bb->build_inst(op, arg2, 64);
  Inst *res = bb->build_inst(Op::MUL, arg1, arg2);
  res = bb->build_inst(Op::ADD, res, arg3);
  write_reg(dest, res);
}

void Parser::process_msubl(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_comma(6);
  Inst *arg3 = get_reg_value(7);
  get_end_of_line(8);

  arg1 = bb->build_inst(op, arg1, 64);
  arg2 = bb->build_inst(op, arg2, 64);
  Inst *res = bb->build_inst(Op::MUL, arg1, arg2);
  res = bb->build_inst(Op::SUB, arg3, res);
  write_reg(dest, res);
}

void Parser::process_mnegl(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_end_of_line(6);

  arg1 = bb->build_inst(op, arg1, 64);
  arg2 = bb->build_inst(op, arg2, 64);
  Inst *res = bb->build_inst(Op::MUL, arg1, arg2);
  res = bb->build_inst(Op::NEG, res);
  write_reg(dest, res);
}

void Parser::process_mull(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_end_of_line(6);

  arg1 = bb->build_inst(op, arg1, 64);
  arg2 = bb->build_inst(op, arg2, 64);
  Inst *res = bb->build_inst(Op::MUL, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_mulh(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_end_of_line(6);

  arg1 = bb->build_inst(op, arg1, 128);
  arg2 = bb->build_inst(op, arg2, 128);
  Inst *res = bb->build_inst(Op::MUL, arg1, arg2);
  res = bb->build_inst(Op::EXTRACT, res, 127, 64);
  write_reg(dest, res);
}

void Parser::process_adrp()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *ptr = get_sym_addr(3);
  if (tokens.size() > 4 && tokens[4].kind == Lexeme::plus)
    {
      Inst *offset = get_reg_or_imm_value(5, ptr->bitsize);
      ptr = bb->build_inst(Op::ADD, ptr, offset);
      get_end_of_line(6);
    }
  else if (tokens.size() > 4 && tokens[4].kind == Lexeme::minus)
    {
      Inst *offset = get_reg_or_imm_value(5, ptr->bitsize);
      ptr = bb->build_inst(Op::SUB, ptr, offset);
      get_end_of_line(6);
    }
  else
    get_end_of_line(4);

  Inst *shift = bb->value_inst(12, ptr->bitsize);
  Inst *res = bb->build_inst(Op::LSHR, ptr, shift);
  res = bb->build_inst(Op::SHL, res, shift);
  write_reg(dest, res);
}

void Parser::process_adc()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_end_of_line(6);

  Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
  c = bb->build_inst(Op::ZEXT, c, arg1->bitsize);
  Inst *res = bb->build_inst(Op::ADD, arg1, arg2);
  res = bb->build_inst(Op::ADD, res, c);
  write_reg(dest, res);
}

void Parser::process_adcs()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = process_last_arg(5, arg1->bitsize);

  Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
  c = bb->build_inst(Op::ZEXT, c, arg2->bitsize);
  arg2 = bb->build_inst(Op::ADD, arg2, c);
  Inst *res = gen_add_cond_flags(arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_sbc()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_end_of_line(6);

  Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
  c = bb->build_inst(Op::NOT, c);
  c = bb->build_inst(Op::ZEXT, c, arg1->bitsize);
  Inst *res = bb->build_inst(Op::SUB, arg1, arg2);
  res = bb->build_inst(Op::SUB, res, c);
  write_reg(dest, res);
}

void Parser::process_sbcs()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_end_of_line(6);

  Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
  c = bb->build_inst(Op::NOT, c);
  c = bb->build_inst(Op::ZEXT, c, arg2->bitsize);
  arg2 = bb->build_inst(Op::ADD, arg2, c);
  Inst *res = gen_sub_cond_flags(arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_ngc()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
  c = bb->build_inst(Op::NOT, c);
  c = bb->build_inst(Op::ZEXT, c, arg1->bitsize);
  Inst *res = bb->build_inst(Op::NEG, arg1);
  res = bb->build_inst(Op::SUB, res, c);
  write_reg(dest, res);
}

void Parser::process_ngcs()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *c = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
  c = bb->build_inst(Op::NOT, c);
  c = bb->build_inst(Op::ZEXT, c, arg1->bitsize);
  arg1 = bb->build_inst(Op::ADD, arg1, c);
  Inst *zero = bb->value_inst(0, arg1->bitsize);
  Inst *res = gen_sub_cond_flags(zero, arg1);
  write_reg(dest, res);
}

void Parser::process_movk()
{
  Inst *dest = get_reg(1);
  Inst *orig = get_reg_value(1);
  get_comma(2);
  Inst *arg1 = get_imm(3);
  assert(tokens.size() > 4 && tokens[4].kind == Lexeme::comma);
  get_comma(4);
  std::string_view lsl = get_name(5);
  if (lsl != "lsl")
    throw Parse_error("expected lsl for shift", line_number);
  uint32_t shift = get_imm(6)->value();
  if (shift != 0 && shift != 16 && shift != 32 && shift != 48)
    throw Parse_error("invalid shift value for movk", line_number);
  get_end_of_line(7);

  Inst *res = bb->build_trunc(arg1, 16);
  if (shift + 16 != orig->bitsize)
    {
      Inst *inst =
	bb->build_inst(Op::EXTRACT, orig, orig->bitsize - 1, shift + 16);
      res = bb->build_inst(Op::CONCAT, inst, res);
    }
  if (shift != 0)
    {
      Inst *inst = bb->build_inst(Op::EXTRACT, orig, shift - 1, 0);
      res = bb->build_inst(Op::CONCAT, res, inst);
    }
  write_reg(dest, res);
}

void Parser::process_fmov()
{
  if (tokens.size() > 4 && tokens[4].kind == Lexeme::left_bracket)
    process_umov();
  else
    process_unary(Op::MOV);
}

void Parser::process_smov()
{
  Inst *dest = get_reg(1);
  uint32_t dest_bitsize = get_reg_size(1);
  get_comma(2);
  Inst *arg1 = process_last_scalar_vec_arg(3);

  if (arg1->bitsize != dest_bitsize)
    arg1 = bb->build_inst(Op::SEXT, arg1, dest_bitsize);
  write_reg(dest, arg1);
}

void Parser::process_umov()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = process_last_scalar_vec_arg(3);

  write_reg(dest, arg1);
}

void Parser::process_unary(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  bool is_w_reg = buf[tokens[1].pos] == 'w';
  Inst *arg1 = process_last_arg(3, is_w_reg ? 32 : 64);

  Inst *res = bb->build_inst(op, arg1);
  write_reg(dest, res);
}

void Parser::process_unary(Inst*(*gen_elem)(Basic_block*, Inst*))
{
  Inst *dest = get_reg(1);
  get_comma(2);
  bool is_w_reg = buf[tokens[1].pos] == 'w';
  Inst *arg1 = process_last_arg(3, is_w_reg ? 32 : 64);

  Inst *res = gen_elem(bb, arg1);
  write_reg(dest, res);
}

Inst *Parser::process_arg_shift(unsigned idx, Inst *arg)
{
  std::string_view cmd = get_name(idx);
  uint32_t shift_value = get_imm(idx + 1)->value() & 0x3f;
  Inst *shift = bb->value_inst(shift_value, arg->bitsize);
  if (cmd == "lsl")
    arg = bb->build_inst(Op::SHL, arg, shift);
  else if (cmd == "lsr")
    arg = bb->build_inst(Op::LSHR, arg, shift);
  else if (cmd == "asr")
    arg = bb->build_inst(Op::ASHR, arg, shift);
  else if (cmd == "ror")
    {
      Inst *lshr = bb->build_inst(Op::LSHR, arg, shift);
      Inst *lshift = bb->value_inst(arg->bitsize - shift_value, arg->bitsize);
      Inst *shl = bb->build_inst(Op::SHL, arg, lshift);
      arg = bb->build_inst(Op::OR, lshr, shl);
    }
  else
    throw Parse_error("unknown shift " + std::string(cmd), line_number);
  get_end_of_line(idx + 2);
  return arg;
}

Inst *Parser::process_arg_ext(unsigned idx, Inst *arg2, uint32_t bitsize)
{
  std::string_view cmd = get_name(idx);
  if (cmd == "sxtw")
    arg2 = bb->build_inst(Op::SEXT, arg2, bitsize);
  else if (cmd == "sxth")
    {
      arg2 = bb->build_trunc(arg2, 16);
      arg2 = bb->build_inst(Op::SEXT, arg2, bitsize);
    }
  else if (cmd == "sxtb")
    {
      arg2 = bb->build_trunc(arg2, 8);
      arg2 = bb->build_inst(Op::SEXT, arg2, bitsize);
    }
  else if (cmd == "uxtw")
    arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize);
  else if (cmd == "uxth")
    {
      arg2 = bb->build_trunc(arg2, 16);
      arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize);
    }
  else if (cmd == "uxtb")
    {
      arg2 = bb->build_trunc(arg2, 8);
      arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize);
    }
  else if (cmd == "lsl")
    ;
  else
    throw Parse_error("unknown extension " + std::string(cmd),
		      line_number);

  if (tokens.size() > idx + 1)
    {
      Inst *shift = get_imm(idx + 1);
      shift = bb->build_trunc(shift, arg2->bitsize);
      arg2 = bb->build_inst(Op::SHL, arg2, shift);
      get_end_of_line(idx + 2);
    }
  else
    get_end_of_line(idx + 1);

  return arg2;
}

// Many instructions treat the last argument in a special way; it may
// be a register or a constant, and allows modifiers for shift and sign
// extension.
Inst *Parser::process_last_arg(unsigned idx, uint32_t bitsize)
{
  if (tokens.size() > idx && tokens[idx].kind == Lexeme::lo12)
    {
      idx++;
      Inst *arg = get_sym_addr(idx++);
      if (tokens.size() > idx && tokens[idx].kind == Lexeme::plus)
	{
	  idx++;
	  Inst *offset = get_reg_or_imm_value(idx++, arg->bitsize);
	  arg = bb->build_inst(Op::ADD, arg, offset);
	}
      else if (tokens.size() > idx && tokens[idx].kind == Lexeme::minus)
	{
	  idx++;
	  Inst *offset = get_reg_or_imm_value(idx++, arg->bitsize);
	  arg = bb->build_inst(Op::SUB, arg, offset);
	}
      arg = bb->build_trunc(arg, 12);
      arg = bb->build_inst(Op::ZEXT, arg, bitsize);
      get_end_of_line(idx);
      return arg;
    }
  Inst *arg = get_reg_or_imm_value(idx++, bitsize);
  if (tokens.size() > idx && tokens[idx].kind == Lexeme::comma)
    {
      idx++;
      std::string_view cmd = get_name(idx);
      if (cmd == "lsl" || cmd == "lsr" || cmd == "asr" || cmd == "ror")
	arg = process_arg_shift(idx, arg);
      else if (cmd == "sxtw" || cmd == "sxth" || cmd == "sxtb"
	       || cmd == "uxtw" || cmd == "uxth" || cmd == "uxtb")
	arg = process_arg_ext(idx, arg, bitsize);
      else
	get_end_of_line(idx);  // Generate parse error.
    }
  else
    get_end_of_line(idx);
  return arg;
}

Inst *Parser::process_last_scalar_vec_arg(unsigned idx)
{
  auto [reg, elem_bitsize, elem_idx] = get_scalar_vreg(idx);
  get_end_of_line(idx + 4);

  Inst *inst = bb->build_inst(Op::READ, reg);
  return extract_vec_elem(inst, elem_bitsize, elem_idx);
}

void Parser::process_binary(Op op, bool perform_not)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = process_last_arg(5, arg1->bitsize);

  if (perform_not)
    arg2 = bb->build_inst(Op::NOT, arg2);
  Inst *res = bb->build_inst(op, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*))
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = process_last_arg(5, arg1->bitsize);

  Inst *res = gen_elem(bb, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_simd_compare(SIMD_cond op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = process_last_arg(5, arg1->bitsize);

  Inst *res = gen_simd_compare(bb, arg1, arg2, op);
  write_reg(dest, res);
}

void Parser::process_simd_shift(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_imm(5);

  arg2 = bb->build_trunc(arg2, arg1->bitsize);
  Inst *res = bb->build_inst(op, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_ext(Op op, uint32_t src_bitsize)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *res = bb->build_trunc(arg1, src_bitsize);
  res = bb->build_inst(op, res, reg_bitsize);
  write_reg(dest, res);
}

void Parser::process_ubfx(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_imm(5);
  get_comma(6);
  Inst *arg3 = get_imm(7);
  get_end_of_line(8);

  uint32_t lo = arg2->value();
  uint32_t hi = lo + arg3->value() - 1;
  Inst *res = bb->build_inst(Op::EXTRACT, arg1, hi, lo);
  res = bb->build_inst(op, res, arg1->bitsize);
  write_reg(dest, res);
}

void Parser::process_bfi()
{
  Inst *dest = get_reg(1);
  Inst *orig = get_reg_value(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_imm(5);
  get_comma(6);
  Inst *arg3 = get_imm(7);
  get_end_of_line(8);

  uint32_t lo = arg2->value();
  uint32_t hi = lo + arg3->value() - 1;
  Inst *res = bb->build_trunc(arg1, arg3->value());
  if (hi != orig->bitsize - 1)
    {
      Inst *inst = bb->build_inst(Op::EXTRACT, orig,
				  orig->bitsize - 1, lo + arg3->value());
      res = bb->build_inst(Op::CONCAT, inst, res);
    }
  if (lo != 0)
    {
      Inst *inst = bb->build_inst(Op::EXTRACT, orig, arg2->value() - 1, 0);
      res = bb->build_inst(Op::CONCAT, res, inst);
    }
  write_reg(dest, res);
}

void Parser::process_bfxil()
{
  Inst *dest = get_reg(1);
  Inst *orig = get_reg_value(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_imm(5);
  get_comma(6);
  Inst *arg3 = get_imm(7);
  get_end_of_line(8);

  uint32_t lo = arg2->value();
  uint32_t hi = lo + arg3->value() - 1;
  Inst *res = bb->build_inst(Op::EXTRACT, arg1, hi, lo);
  Inst *inst =
    bb->build_inst(Op::EXTRACT, orig, orig->bitsize - 1, res->bitsize);
  res = bb->build_inst(Op::CONCAT, inst, res);
  write_reg(dest, res);
}

void Parser::process_ubfiz(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_imm(5);
  get_comma(6);
  Inst *arg3 = get_imm(7);
  get_end_of_line(8);

  Inst *res = bb->build_trunc(arg1, arg3->value());
  res = bb->build_inst(op, res, arg1->bitsize);
  res = bb->build_inst(Op::SHL, res, bb->build_trunc(arg2, arg1->bitsize));
  write_reg(dest, res);
}

void Parser::process_shift(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_or_imm_value(5, arg1->bitsize);
  get_end_of_line(6);

  arg2 = bb->build_trunc(arg2, arg1->bitsize == 32 ? 5 : 6);
  arg2 = bb->build_inst(Op::ZEXT, arg2, arg1->bitsize);
  Inst *res = bb->build_inst(op, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_ror()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_or_imm_value(5, arg1->bitsize);
  get_end_of_line(6);

  Inst *shift = bb->build_trunc(arg2, arg1->bitsize == 32 ? 5 : 6);
  shift = bb->build_inst(Op::ZEXT, shift, arg1->bitsize);
  Inst *lshr = bb->build_inst(Op::LSHR, arg1, shift);
  Inst *bs = bb->value_inst(arg1->bitsize, arg1->bitsize);
  Inst *lshift = bb->build_inst(Op::SUB, bs, shift);
  Inst *shl = bb->build_inst(Op::SHL, arg1, lshift);
  Inst *res = bb->build_inst(Op::OR, lshr, shl);
  write_reg(dest, res);
}

void Parser::process_extr()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_comma(6);
  Inst *arg3 = get_imm(7);
  get_end_of_line(8);

  Inst *shift = bb->build_trunc(arg3, arg1->bitsize == 32 ? 5 : 6);
  shift = bb->build_inst(Op::ZEXT, shift, arg1->bitsize);
  Inst *lshr = bb->build_inst(Op::LSHR, arg2, shift);
  Inst *bs = bb->value_inst(arg1->bitsize, arg1->bitsize);
  Inst *lshift = bb->build_inst(Op::SUB, bs, shift);
  Inst *shl = bb->build_inst(Op::SHL, arg1, lshift);
  Inst *res = bb->build_inst(Op::OR, lshr, shl);
  write_reg(dest, res);
}

void Parser::process_rev()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *res = gen_bswap(bb, arg1);
  write_reg(dest, res);
}

// Reverse bytes in each bitsize-part of the register.
void Parser::process_rev(uint32_t bitsize)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *res = gen_bswap(bb, bb->build_trunc(arg1, bitsize));
  for (uint32_t i = bitsize; i < arg1->bitsize; i += bitsize)
    {
      Inst *inst = bb->build_inst(Op::EXTRACT, arg1, i + bitsize - 1, i);
      res = bb->build_inst(Op::CONCAT, gen_bswap(bb, inst), res);
    }
  write_reg(dest, res);
}

Inst *Parser::gen_sub_cond_flags(Inst *arg1, Inst *arg2)
{
  Inst *res = bb->build_inst(Op::SUB, arg1, arg2);

  Inst *zero = bb->value_inst(0, res->bitsize);
  Inst *is_neg_arg1 = bb->build_inst(Op::SLT, arg1, zero);
  Inst *is_neg_arg2 = bb->build_inst(Op::SLT, arg2, zero);
  Inst *is_neg_res = bb->build_inst(Op::SLT, res, zero);
  Inst *is_pos_arg1 = bb->build_inst(Op::NOT, is_neg_arg1);
  Inst *is_pos_arg2 = bb->build_inst(Op::NOT, is_neg_arg2);
  Inst *is_pos_res = bb->build_inst(Op::NOT, is_neg_res);

  Inst *n = is_neg_res;
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::n], n);

  Inst *z = bb->build_inst(Op::EQ, arg1, arg2);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::z], z);

  Inst *c1 = bb->build_inst(Op::AND, is_neg_arg1, is_pos_arg2);
  Inst *c2 = bb->build_inst(Op::AND, is_neg_arg1, is_pos_res);
  Inst *c3 = bb->build_inst(Op::AND, is_pos_arg2, is_pos_res);
  Inst *c = bb->build_inst(Op::OR, c1, c2);
  c = bb->build_inst(Op::OR, c, c3);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::c], c);

  Inst *v1 = bb->build_inst(Op::AND, is_neg_arg1, is_pos_arg2);
  v1 = bb->build_inst(Op::AND, v1, is_pos_res);
  Inst *v2 = bb->build_inst(Op::AND, is_pos_arg1, is_neg_arg2);
  v2 = bb->build_inst(Op::AND, v2, is_neg_res);
  Inst *v = bb->build_inst(Op::OR, v1, v2);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::v], v);

  return res;
}

Inst *Parser::gen_add_cond_flags(Inst *arg1, Inst *arg2)
{
  Inst *res = bb->build_inst(Op::ADD, arg1, arg2);

  Inst *zero = bb->value_inst(0, res->bitsize);
  Inst *is_neg_arg1 = bb->build_inst(Op::SLT, arg1, zero);
  Inst *is_neg_arg2 = bb->build_inst(Op::SLT, arg2, zero);
  Inst *is_neg_res = bb->build_inst(Op::SLT, res, zero);
  Inst *is_pos_arg1 = bb->build_inst(Op::NOT, is_neg_arg1);
  Inst *is_pos_arg2 = bb->build_inst(Op::NOT, is_neg_arg2);
  Inst *is_pos_res = bb->build_inst(Op::NOT, is_neg_res);

  Inst *n = is_neg_res;
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::n], n);

  Inst *z = bb->build_inst(Op::EQ, res, zero);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::z], z);

  Inst *c1 = bb->build_inst(Op::AND, is_neg_arg1, is_neg_arg2);
  Inst *c2 = bb->build_inst(Op::AND, is_neg_arg1, is_pos_res);
  Inst *c3 = bb->build_inst(Op::AND, is_neg_arg2, is_pos_res);
  Inst *c = bb->build_inst(Op::OR, c1, c2);
  c = bb->build_inst(Op::OR, c, c3);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::c], c);

  Inst *v1 = bb->build_inst(Op::AND, is_neg_arg1, is_neg_arg2);
  v1 = bb->build_inst(Op::AND, v1, is_pos_res);
  Inst *v2 = bb->build_inst(Op::AND, is_pos_arg1, is_pos_arg2);
  v2 = bb->build_inst(Op::AND, v2, is_neg_res);
  Inst *v = bb->build_inst(Op::OR, v1, v2);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::v], v);

  return res;
}

Inst *Parser::gen_and_cond_flags(Inst *arg1, Inst *arg2)
{
  Inst *res = bb->build_inst(Op::AND, arg1, arg2);

  Inst *zero = bb->value_inst(0, res->bitsize);
  Inst *n = bb->build_inst(Op::SLT, res, zero);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::n], n);
  Inst *z = bb->build_inst(Op::EQ, res, zero);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::z], z);
  Inst *c = bb->value_inst(0, 1);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::c], c);
  Inst *v = bb->value_inst(0, 1);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::v], v);

  return res;
}

void Parser::process_cmn()
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = process_last_arg(3, arg1->bitsize);

  gen_add_cond_flags(arg1, arg2);
}

void Parser::process_cmp()
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = process_last_arg(3, arg1->bitsize);

  gen_sub_cond_flags(arg1, arg2);
}

void Parser::process_subs()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = process_last_arg(5, arg1->bitsize);

  Inst *res = gen_sub_cond_flags(arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_negs()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  bool is_w_reg = buf[tokens[1].pos] == 'w';
  Inst *arg1 = process_last_arg(3, is_w_reg ? 32 : 64);

  Inst *zero = bb->value_inst(0, arg1->bitsize);
  Inst *res = gen_sub_cond_flags(zero, arg1);
  write_reg(dest, res);
}

void Parser::process_adds()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = process_last_arg(5, arg1->bitsize);

  Inst *res = gen_add_cond_flags(arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_mneg()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_or_imm_value(5, arg1->bitsize);
  get_end_of_line(6);

  Inst *res = bb->build_inst(Op::MUL, arg1, arg2);
  res = bb->build_inst(Op::NEG, res);
  write_reg(dest, res);
}

void Parser::process_ands(bool perform_not)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = process_last_arg(5, arg1->bitsize);

  if (perform_not)
    arg2 = bb->build_inst(Op::NOT, arg2);
  Inst *res = gen_and_cond_flags(arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_tst()
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = process_last_arg(3, arg1->bitsize);

  gen_and_cond_flags(arg1, arg2);
}

void Parser::process_ccmp(bool is_ccmn)
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = get_reg_or_imm_value(3, arg1->bitsize);
  get_comma(4);
  Inst *arg3 = get_imm(5);
  get_comma(6);
  Cond_code cc = get_cc(7);
  get_end_of_line(8);

  Inst *cond = build_cond(cc);
  if (is_ccmn)
    gen_add_cond_flags(arg1, arg2);
  else
    gen_sub_cond_flags(arg1, arg2);
  Inst *n1 = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::n]);
  Inst *z1 = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::z]);
  Inst *c1 = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::c]);
  Inst *v1 = bb->build_inst(Op::READ, rstate->registers[Aarch64RegIdx::v]);
  Inst *n2 = bb->build_extract_bit(arg3, 3);
  Inst *z2 = bb->build_extract_bit(arg3, 2);
  Inst *c2 = bb->build_extract_bit(arg3, 1);
  Inst *v2 = bb->build_extract_bit(arg3, 0);
  Inst *n = bb->build_inst(Op::ITE, cond, n1, n2);
  Inst *z = bb->build_inst(Op::ITE, cond, z1, z2);
  Inst *c = bb->build_inst(Op::ITE, cond, c1, c2);
  Inst *v = bb->build_inst(Op::ITE, cond, v1, v2);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::n], n);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::z], z);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::c], c);
  bb->build_inst(Op::WRITE, rstate->registers[Aarch64RegIdx::v], v);
}

Inst *Parser::extract_vec_elem(Inst *inst, uint32_t elem_bitsize, uint32_t idx)
{
  if (idx == 0 && inst->bitsize == elem_bitsize)
    return inst;
  assert(inst->bitsize % elem_bitsize == 0);
  Inst *high = bb->value_inst(idx * elem_bitsize + elem_bitsize - 1, 32);
  Inst *low = bb->value_inst(idx * elem_bitsize, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
}

void Parser::process_vec_unary(Op op)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_end_of_line(4);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *inst = bb->build_inst(op, elem1);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_unary(Inst*(*gen_elem)(Basic_block*, Inst*))
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_end_of_line(4);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *inst = gen_elem(bb, elem1);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_binary(Op op, bool perform_not)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2;
  if (tokens.size() > 6)
    arg2 = process_last_scalar_vec_arg(5);
  else
    {
      arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
      get_end_of_line(6);
    }

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      if (perform_not)
	elem2 = bb->build_inst(Op::NOT, elem2);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*))
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = gen_elem(bb, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_widen_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op widen_op, bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  uint32_t src_elem_bitsize = elem_bitsize / 2;
  uint32_t src_nof_elem = high ? 2 * nof_elem : nof_elem;
  Inst *arg1 = get_vreg_value(3, src_nof_elem, src_elem_bitsize);
  get_comma(4);
  Inst *arg2;
  if (tokens.size() > 6)
    arg2 = process_last_scalar_vec_arg(5);
  else
    {
      arg2 = get_vreg_value(5, src_nof_elem, src_elem_bitsize);
      get_end_of_line(6);
    }

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      uint32_t idx = high ? i + nof_elem : i;
      Inst *elem1 = extract_vec_elem(arg1, src_elem_bitsize, idx);
      Inst *elem2;
      if (arg2->bitsize == src_elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, src_elem_bitsize, idx);
      elem1 = bb->build_inst(widen_op, elem1, elem_bitsize);
      elem2 = bb->build_inst(widen_op, elem2, elem_bitsize);
      Inst *inst = gen_elem(bb, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_widen_binary_add(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op widen_op, bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  uint32_t src_elem_bitsize = elem_bitsize / 2;
  uint32_t src_nof_elem = high ? 2 * nof_elem : nof_elem;
  Inst *arg1 = get_vreg_value(3, src_nof_elem, src_elem_bitsize);
  get_comma(4);
  Inst *arg2;
  if (tokens.size() > 6)
    arg2 = process_last_scalar_vec_arg(5);
  else
    {
      arg2 = get_vreg_value(5, src_nof_elem, src_elem_bitsize);
      get_end_of_line(6);
    }

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      uint32_t idx = high ? i + nof_elem : i;
      Inst *elem0 = extract_vec_elem(orig, elem_bitsize, i);
      Inst *elem1 = extract_vec_elem(arg1, src_elem_bitsize, idx);
      Inst *elem2;
      if (arg2->bitsize == src_elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, src_elem_bitsize, idx);
      elem1 = bb->build_inst(widen_op, elem1, elem_bitsize);
      elem2 = bb->build_inst(widen_op, elem2, elem_bitsize);
      Inst *inst = gen_elem(bb, elem1, elem2);
      inst = bb->build_inst(Op::ADD, inst, elem0);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_widen_pairwise_add(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op widen_op)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, 2 * nof_elem, elem_bitsize / 2);
  get_end_of_line(4);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem0 = extract_vec_elem(orig, elem_bitsize, i);
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize / 2, 2 * i);
      Inst *elem2 = extract_vec_elem(arg1, elem_bitsize / 2, 2 * i + 1);
      elem1 = bb->build_inst(widen_op, elem1, elem_bitsize);
      elem2 = bb->build_inst(widen_op, elem2, elem_bitsize);
      Inst *inst = gen_elem(bb, elem1, elem2);
      inst = bb->build_inst(Op::ADD, inst, elem0);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_widen_pairwise(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op widen_op)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, 2 * nof_elem, elem_bitsize / 2);
  get_end_of_line(4);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize / 2, 2 * i);
      Inst *elem2 = extract_vec_elem(arg1, elem_bitsize / 2, 2 * i + 1);
      elem1 = bb->build_inst(widen_op, elem1, elem_bitsize);
      elem2 = bb->build_inst(widen_op, elem2, elem_bitsize);
      Inst *inst = gen_elem(bb, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_widen2_binary(Op op, Op widen_op, bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  uint32_t src_elem_bitsize = elem_bitsize / 2;
  uint32_t src_nof_elem = high ? 2 * nof_elem : nof_elem;
  Inst *arg2 = get_vreg_value(5, src_nof_elem, src_elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      uint32_t idx = high ? i + nof_elem : i;
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2 = extract_vec_elem(arg2, src_elem_bitsize, idx);
      elem2 = bb->build_inst(widen_op, elem2, elem_bitsize);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_reduc(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*))
{
  Inst *dest = get_reg(1);
  get_comma(2);
  auto [_, nof_elem, elem_bitsize] = get_vreg(3);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_end_of_line(4);

  Inst *res = extract_vec_elem(arg1, elem_bitsize, 0);
  for (uint32_t i = 1; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      res = gen_elem(bb, res, elem1);
    }
  write_reg(dest, res);
}

void Parser::process_vec_reducl(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  auto [_, nof_elem, elem_bitsize] = get_vreg(3);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_end_of_line(4);

  Inst *res = extract_vec_elem(arg1, elem_bitsize, 0);
  res = bb->build_inst(op, res, 2 * elem_bitsize);
  for (uint32_t i = 1; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      elem1 = bb->build_inst(op, elem1, 2 * elem_bitsize);
      res = gen_elem(bb, res, elem1);
    }
  write_reg(dest, res);
}

void Parser::process_vec_pairwise(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*))
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i += 2)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2 = extract_vec_elem(arg1, elem_bitsize, i + 1);
      Inst *inst = gen_elem(bb, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  for (uint32_t i = 0; i < nof_elem; i += 2)
    {
      Inst *elem1 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, i + 1);
      Inst *inst = gen_elem(bb, elem1, elem2);
      res = bb->build_inst(Op::CONCAT, inst, res);
    }
  write_reg(dest, res);
}

void Parser::process_vec_widen(Op op, bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1;
  if (high)
    arg1 = get_vreg_value(3, nof_elem * 2, elem_bitsize / 2);
  else
    arg1 = get_vreg_value(3, nof_elem, elem_bitsize / 2);
  get_end_of_line(4);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      uint32_t idx = high ? i + nof_elem : i;
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize / 2, idx);
      Inst *inst = bb->build_inst(op, elem1, elem_bitsize);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_simd_compare(SIMD_cond op)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2;
  if (is_vreg(5))
    arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  else
    {
      arg2 = get_imm(5);
      if (arg2->value() != 0)
	throw Parse_error("expected 0", line_number);
      arg2 = bb->build_trunc(arg2, elem_bitsize);
    }
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = gen_simd_compare(bb, elem1, elem2, op);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_ins()
{
  auto [dest, elem_bitsize, elem_idx] = get_scalar_vreg(1);
  get_comma(5);
  Inst *arg1;
  if (is_vreg(6))
    arg1 = process_last_scalar_vec_arg(6);
  else
    {
      arg1 = get_reg_value(6);
      arg1 = bb->build_trunc(arg1, elem_bitsize);
      get_end_of_line(7);
    }

  Inst *orig = bb->build_inst(Op::READ, dest);
  uint32_t nof_elem = orig->bitsize / elem_bitsize;
  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem;
      if (i == elem_idx)
	elem = arg1;
      else
	elem = extract_vec_elem(orig, elem_bitsize, i);
      if (res)
	res = bb->build_inst(Op::CONCAT, elem, res);
      else
	res = elem;
    }
  write_reg(dest, res);
}

void Parser::process_vec_rev(uint32_t bitsize)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_end_of_line(4);

  assert(arg1->bitsize >= bitsize);
  assert(arg1->bitsize % bitsize == 0);
  assert(bitsize >= elem_bitsize);
  assert(bitsize % elem_bitsize == 0);
  uint32_t nof_elem1 = arg1->bitsize / bitsize;
  uint32_t nof_elem2 = bitsize / elem_bitsize;

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem1; i++)
    {
      Inst *elem = extract_vec_elem(arg1, bitsize, i);
      Inst *inst = bb->build_trunc(elem, elem_bitsize);
      for (uint32_t j = 1; j < nof_elem2; j++)
	{
	  Inst *inst2 = extract_vec_elem(elem, elem_bitsize, j);
	  inst = bb->build_inst(Op::CONCAT, inst, inst2);
	}
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_dup()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = process_last_scalar_vec_arg(3);

  write_reg(dest, arg1);
}

void Parser::process_vec_dup()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1;
  if (is_vreg(3))
    arg1 = process_last_scalar_vec_arg(3);
  else
    {
      arg1 = get_reg_value(3);
      arg1 = bb->build_trunc(arg1, elem_bitsize);
      get_end_of_line(4);
    }

  Inst *res = arg1;
  for (uint32_t i = 1; i < nof_elem; i++)
    {
      res = bb->build_inst(Op::CONCAT, arg1, res);
    }
  write_reg(dest, res);
}

void Parser::process_vec_mov()
{
  if (tokens.size() > 3 && tokens[2].kind == Lexeme::left_bracket)
    process_vec_ins();
  else
    process_vec_unary(Op::MOV);
}

void Parser::process_vec_movi(bool invert)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  uint64_t value = get_imm(3)->value();
  uint64_t shift = 0;
  uint64_t m = 0;
  if (tokens.size() > 4)
    {
      get_comma(4);
      std::string_view lsl = get_name(5);
      shift = get_imm(6)->value();
      if (lsl == "lsl")
	{
	  if (shift != 0 && shift != 8 && shift != 16 && shift != 24)
	    throw Parse_error("invalid shift value for movi", line_number);
	}
      else if (lsl == "msl")
	{
	  if (shift == 8)
	    m = 0xff;
	  else if (shift == 16)
	    m = 0xffff;
	  else
	    throw Parse_error("invalid shift value for movi", line_number);
	}
      else
	throw Parse_error("expected lsl/msl for shift", line_number);
      get_end_of_line(7);
    }
  else
    get_end_of_line(4);
  value = (value << shift) | m;
  if (invert)
    value = ~value;
  Inst *arg1 = bb->value_inst(value, elem_bitsize);

  Inst *res = arg1;
  for (uint32_t i = 1; i < nof_elem; i++)
    {
      res = bb->build_inst(Op::CONCAT, arg1, res);
    }
  write_reg(dest, res);
}

void Parser::process_vec_orr()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1;
  Inst *arg2;
  if (is_vreg(3))
    {
      arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
      get_comma(4);
      arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
      get_end_of_line(6);
    }
  else
    {
      arg1 = get_vreg_value(1, nof_elem, elem_bitsize);
      uint64_t value = get_imm(3)->value();
      uint64_t shift = 0;
      if (tokens.size() > 4)
	{
	  get_comma(4);
	  std::string_view lsl = get_name(5);
	  if (lsl != "lsl")
	    throw Parse_error("expected lsl for shift", line_number);
	  shift = get_imm(6)->value();
	  if (shift != 0 && shift != 8 && shift != 16 && shift != 24)
	    throw Parse_error("invalid shift value for orr", line_number);
	  get_end_of_line(7);
	}
      else
	get_end_of_line(4);
      arg2 = bb->value_inst(value << shift, elem_bitsize);
    }

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = bb->build_inst(Op::OR, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_zip1()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem/2; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      if (res)
	res = bb->build_inst(Op::CONCAT, elem1, res);
      else
	res = elem1;
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      res = bb->build_inst(Op::CONCAT, elem2, res);
    }
  write_reg(dest, res);
}

void Parser::process_vec_zip2()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = nof_elem/2; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      if (res)
	res = bb->build_inst(Op::CONCAT, elem1, res);
      else
	res = elem1;
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      res = bb->build_inst(Op::CONCAT, elem2, res);
    }
  write_reg(dest, res);
}

void Parser::process_vec_trn1()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem/2; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, 2 * i);
      if (res)
	res = bb->build_inst(Op::CONCAT, elem1, res);
      else
	res = elem1;
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, 2 * i);
      res = bb->build_inst(Op::CONCAT, elem2, res);
    }
  write_reg(dest, res);
}

void Parser::process_vec_trn2()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem/2; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, 2 * i + 1);
      if (res)
	res = bb->build_inst(Op::CONCAT, elem1, res);
      else
	res = elem1;
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, 2 * i + 1);
      res = bb->build_inst(Op::CONCAT, elem2, res);
    }
  write_reg(dest, res);
}

void Parser::process_vec_tbl()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  if (elem_bitsize != 8 || nof_elem != 16)
    throw Parse_error("expected element size 8", line_number);
  get_comma(2);
  Inst *arg1, *arg2;
  if (tokens.size() == 8)
    {
      get_left_brace(3);
      arg1 = get_vreg_value(4, nof_elem, elem_bitsize);
      get_right_brace(5);
      get_comma(6);
      arg2 = get_vreg_value(7, nof_elem, elem_bitsize);
      get_end_of_line(8);
    }
  else
    {
      get_left_brace(3);
      uint64_t start_idx = get_vreg_idx(4, nof_elem, elem_bitsize);
      get_minus(5);
      uint64_t end_idx = get_vreg_idx(6, nof_elem, elem_bitsize);
      get_right_brace(7);
      get_comma(8);
      arg2 = get_vreg_value(9, nof_elem, elem_bitsize);
      get_end_of_line(10);

      assert(start_idx >= Aarch64RegIdx::v0);
      assert(start_idx < end_idx);
      assert(end_idx <= Aarch64RegIdx::v31);
      assert(nof_elem * elem_bitsize == 128);
      arg1 = bb->build_inst(Op::READ, rstate->registers[start_idx]);
      for (uint64_t i = start_idx + 1; i <= end_idx; i++)
	{
	  Inst *inst = bb->build_inst(Op::READ, rstate->registers[i]);
	  arg1 = bb->build_inst(Op::CONCAT, inst, arg1);
	}
    }

  Inst *res = nullptr;
  uint32_t arg1_nof_elem = arg1->bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *idx = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = bb->value_inst(0, elem_bitsize);
      for (uint32_t j = 0; j < arg1_nof_elem; j++)
	{
	  Inst *elem_idx = bb->value_inst(j, elem_bitsize);
	  Inst *elem = extract_vec_elem(arg1, elem_bitsize, j);
	  Inst *cmp = bb->build_inst(Op::EQ, idx, elem_idx);
	  inst = bb->build_inst(Op::ITE, cmp, elem, inst);
	}
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_bic()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1, *arg2;
  if (is_vreg(3))
    {
      arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
      get_comma(4);
      arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
      get_end_of_line(6);
    }
  else
    {
      arg1 = get_vreg_value(1, nof_elem, elem_bitsize);
      arg2 = process_last_arg(3, elem_bitsize);
    }

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      elem2 = bb->build_inst(Op::NOT, elem2);
      Inst *inst = bb->build_inst(Op::AND, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_bif()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem0 = extract_vec_elem(orig, elem_bitsize, i);
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      elem2 = bb->build_inst(Op::NOT, elem2);
      Inst *inst = bb->build_inst(Op::XOR, elem0, elem1);
      inst = bb->build_inst(Op::AND, inst, elem2);
      inst = bb->build_inst(Op::XOR, inst, elem0);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_bit()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem0 = extract_vec_elem(orig, elem_bitsize, i);
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = bb->build_inst(Op::XOR, elem0, elem1);
      inst = bb->build_inst(Op::AND, inst, elem2);
      inst = bb->build_inst(Op::XOR, inst, elem0);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_bsl()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem0 = extract_vec_elem(orig, elem_bitsize, i);
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = bb->build_inst(Op::XOR, elem1, elem2);
      inst = bb->build_inst(Op::AND, inst, elem0);
      inst = bb->build_inst(Op::XOR, inst, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_ext()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_comma(6);
  Inst *arg3 = get_imm(7);
  get_end_of_line(8);

  Inst *res = nullptr;
  assert((8 * arg3->value()) % elem_bitsize  == 0);
  uint32_t offset = 8 * arg3->value() / elem_bitsize;
  for (uint32_t i = offset; i < nof_elem; i++)
    {
      Inst *inst = extract_vec_elem(arg1, elem_bitsize, i);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  for (uint32_t i = 0; i < offset; i++)
    {
      Inst *inst = extract_vec_elem(arg2, elem_bitsize, i);
      res = bb->build_inst(Op::CONCAT, inst, res);
    }

  write_reg(dest, res);
}

void Parser::process_vec_shrn(bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  uint32_t src_elem_bitsize = elem_bitsize * 2;
  uint32_t src_nof_elem = high ? nof_elem / 2 : nof_elem;
  Inst *arg1 = get_vreg_value(3, src_nof_elem, src_elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_imm(5);
  get_end_of_line(6);

  Inst *elem2 = bb->build_trunc(arg2, src_elem_bitsize);
  Inst *res = nullptr;
  if (high)
    {
      for (uint32_t i = 0; i < src_nof_elem; i++)
	{
	  Inst *inst = extract_vec_elem(orig, elem_bitsize, i);
	  if (res)
	    res = bb->build_inst(Op::CONCAT, inst, res);
	  else
	    res = inst;
	}
    }
  for (uint32_t i = 0; i < src_nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, src_elem_bitsize, i);
      Inst *inst = bb->build_inst(Op::LSHR, elem1, elem2);
      inst = bb->build_trunc(inst, elem_bitsize);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  if (!high)
    {
      for (uint32_t i = 0; i < src_nof_elem; i++)
	{
	  Inst *inst = bb->value_inst(0, elem_bitsize);
	  res = bb->build_inst(Op::CONCAT, inst, res);
	}
    }
  write_reg(dest, res);
}

void Parser::process_vec_shift(Op op)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_imm(5);
  get_end_of_line(6);

  Inst *elem2 = bb->build_trunc(arg2, elem_bitsize);
  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_shift_acc(Op op)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_imm(5);
  get_end_of_line(6);

  Inst *elem2 = bb->build_trunc(arg2, elem_bitsize);
  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem0 = extract_vec_elem(orig, elem_bitsize, i);
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      inst = bb->build_inst(Op::ADD,elem0, inst);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_widen_shift(Op op, Op widen_op, bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  uint32_t src_elem_bitsize = elem_bitsize / 2;
  uint32_t src_nof_elem = high ? 2 * nof_elem : nof_elem;
  Inst *arg1 = get_vreg_value(3, src_nof_elem, src_elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_imm(5);
  get_end_of_line(6);

  Inst *elem2 = bb->build_trunc(arg2, elem_bitsize);
  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      uint32_t idx = high ? i + nof_elem : i;
      Inst *elem1 = extract_vec_elem(arg1, src_elem_bitsize, idx);
      elem1 = bb->build_inst(widen_op, elem1, elem_bitsize);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_narrow(Op op, bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  uint32_t src_elem_bitsize = elem_bitsize * 2;
  uint32_t src_nof_elem = high ? nof_elem / 2 : nof_elem;
  Inst *arg1 = get_vreg_value(3, src_nof_elem, src_elem_bitsize);
  get_end_of_line(4);

  Inst *res = nullptr;
  if (high)
    {
      for (uint32_t i = 0; i < src_nof_elem; i++)
	{
	  Inst *inst = extract_vec_elem(orig, elem_bitsize, i);
	  if (res)
	    res = bb->build_inst(Op::CONCAT, inst, res);
	  else
	    res = inst;
	}
    }
  for (uint32_t i = 0; i < src_nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, src_elem_bitsize, i);
      Inst *inst = bb->build_inst(op, elem1, elem_bitsize);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  if (!high)
    {
      for (uint32_t i = 0; i < src_nof_elem; i++)
	{
	  Inst *inst = bb->value_inst(0, elem_bitsize);
	  res = bb->build_inst(Op::CONCAT, inst, res);
	}
    }
  write_reg(dest, res);
}

void Parser::process_vec_binary_high_narrow(Op op, bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  uint32_t src_elem_bitsize = elem_bitsize * 2;
  uint32_t src_nof_elem = high ? nof_elem / 2 : nof_elem;
  Inst *arg1 = get_vreg_value(3, src_nof_elem, src_elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, src_nof_elem, src_elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  if (high)
    {
      for (uint32_t i = 0; i < src_nof_elem; i++)
	{
	  Inst *inst = extract_vec_elem(orig, elem_bitsize, i);
	  if (res)
	    res = bb->build_inst(Op::CONCAT, inst, res);
	  else
	    res = inst;
	}
    }
  for (uint32_t i = 0; i < src_nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, src_elem_bitsize, i);
      Inst *elem2 = extract_vec_elem(arg2, src_elem_bitsize, i);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      inst = bb->build_trunc(inst, elem_bitsize);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  if (!high)
    {
      for (uint32_t i = 0; i < src_nof_elem; i++)
	{
	  Inst *inst = bb->value_inst(0, elem_bitsize);
	  res = bb->build_inst(Op::CONCAT, inst, res);
	}
    }
  write_reg(dest, res);
}


void Parser::process_vec_xtn(bool high)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  uint32_t src_elem_bitsize = elem_bitsize * 2;
  uint32_t src_nof_elem = high ? nof_elem / 2 : nof_elem;
  Inst *arg1 = get_vreg_value(3, src_nof_elem, src_elem_bitsize);
  get_end_of_line(4);

  Inst *res = nullptr;
  if (high)
    {
      for (uint32_t i = 0; i < src_nof_elem; i++)
	{
	  Inst *inst = extract_vec_elem(orig, elem_bitsize, i);
	  if (res)
	    res = bb->build_inst(Op::CONCAT, inst, res);
	  else
	    res = inst;
	}
    }
  for (uint32_t i = 0; i < src_nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, src_elem_bitsize, i);
      Inst *inst = bb->build_trunc(elem1, elem_bitsize);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  if (!high)
    {
      for (uint32_t i = 0; i < src_nof_elem; i++)
	{
	  Inst *inst = bb->value_inst(0, elem_bitsize);
	  res = bb->build_inst(Op::CONCAT, inst, res);
	}
    }
  write_reg(dest, res);
}

void Parser::process_vec_mla(Op op)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  Inst *orig = get_vreg_value(1, nof_elem, elem_bitsize);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2;
  if (tokens.size() > 6)
    arg2 = process_last_scalar_vec_arg(5);
  else
    {
      arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
      get_end_of_line(6);
    }

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem0 = extract_vec_elem(orig, elem_bitsize, i);
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = bb->build_inst(Op::MUL, elem1, elem2);
      inst = bb->build_inst(op, elem0, inst);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_uzp(bool odd)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5, nof_elem, elem_bitsize);
  get_end_of_line(6);

  Inst *res = nullptr;
  for (uint32_t i = 0; i < nof_elem / 2; i++)
    {
      Inst *inst = extract_vec_elem(arg1, elem_bitsize, 2 * i + odd);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  for (uint32_t i = 0; i < nof_elem / 2; i++)
    {
      Inst *inst = extract_vec_elem(arg2, elem_bitsize, 2 * i + odd);
      res = bb->build_inst(Op::CONCAT, inst, res);
    }
  write_reg(dest, res);
}

void Parser::parse_vector_op()
{
  std::string_view name = get_name(0);

  if (name == "abs")
    process_vec_unary(gen_abs);
  else if (name == "add")
    process_vec_binary(Op::ADD);
  else if (name == "addhn")
    process_vec_binary_high_narrow(Op::ADD, false);
  else if (name == "addhn2")
    process_vec_binary_high_narrow(Op::ADD, true);
  else if (name == "addp")
    process_vec_pairwise(gen_add);
  else if (name == "and")
    process_vec_binary(Op::AND);
  else if (name == "bic")
    process_vec_bic();
  else if (name == "bif")
    process_vec_bif();
  else if (name == "bit")
    process_vec_bit();
  else if (name == "bsl")
    process_vec_bsl();
  else if (name == "cls")
    process_vec_unary(gen_clrsb);
  else if (name == "clz")
    process_vec_unary(gen_clz);
  else if (name == "cmeq")
    process_vec_simd_compare(SIMD_cond::EQ);
  else if (name == "cmge")
    process_vec_simd_compare(SIMD_cond::GE);
  else if (name == "cmgt")
    process_vec_simd_compare(SIMD_cond::GT);
  else if (name == "cmhi")
    process_vec_simd_compare(SIMD_cond::HI);
  else if (name == "cmhs")
    process_vec_simd_compare(SIMD_cond::HS);
  else if (name == "cmle")
    process_vec_simd_compare(SIMD_cond::LE);
  else if (name == "cmlt")
    process_vec_simd_compare(SIMD_cond::LT);
  else if (name == "cmtst")
    process_vec_binary(gen_cmtst);
  else if (name == "cnt")
    process_vec_unary(gen_popcount);
  else if (name == "dup")
    process_vec_dup();
  else if (name == "eor")
    process_vec_binary(Op::XOR);
  else if (name == "ext")
    process_vec_ext();
  else if (name == "fabs")
    process_vec_unary(Op::FABS);
  else if (name == "fadd")
    process_vec_binary(Op::FADD);
  else if (name == "fcmeq")
    process_vec_simd_compare(SIMD_cond::FEQ);
  else if (name == "fcmge")
    process_vec_simd_compare(SIMD_cond::FGE);
  else if (name == "fcmgt")
    process_vec_simd_compare(SIMD_cond::FGT);
  else if (name == "fcmle")
    process_vec_simd_compare(SIMD_cond::FLE);
  else if (name == "fcmlt")
    process_vec_simd_compare(SIMD_cond::FLT);
  else if (name == "fcvtl")
    process_vec_widen(Op::FCHPREC, false);
  else if (name == "fcvtl2")
    process_vec_widen(Op::FCHPREC, true);
  else if (name == "fcvtn")
    process_vec_narrow(Op::FCHPREC, false);
  else if (name == "fcvtn2")
    process_vec_narrow(Op::FCHPREC, true);
  else if (name == "fcvtzs")
    process_vec_unary(gen_f2s);
  else if (name == "fcvtzu")
    process_vec_unary(gen_f2u);
  else if (name == "fdiv")
    process_vec_binary(Op::FDIV);
  else if (name == "fmaxnm")
    process_vec_binary(gen_fmax);
  else if (name == "fminnm")
    process_vec_binary(gen_fmin);
  else if (name == "fmul")
    process_vec_binary(Op::FMUL);
  else if (name == "fneg")
    process_vec_unary(Op::FNEG);
  else if (name == "fsub")
    process_vec_binary(Op::FSUB);
  else if (name == "ins")
    process_vec_ins();
  else if (name == "mla")
    process_vec_mla();
  else if (name == "mls")
    process_vec_mla(Op::SUB);
  else if (name == "mov")
    process_vec_mov();
  else if (name == "movi")
    process_vec_movi();
  else if (name == "mul")
    process_vec_binary(Op::MUL);
  else if (name == "mvn")
    process_vec_unary(Op::NOT);
  else if (name == "mvni")
    process_vec_movi(true);
  else if (name == "neg")
    process_vec_unary(Op::NEG);
  else if (name == "not")
    process_vec_unary(Op::NOT);
  else if (name == "orn")
    process_vec_binary(Op::OR, true);
  else if (name == "orr")
    process_vec_orr();
  else if (name == "rbit")
    process_vec_unary(gen_bitreverse);
  else if (name == "rev16")
    process_vec_rev(16);
  else if (name == "rev32")
    process_vec_rev(32);
  else if (name == "rev64")
    process_vec_rev(64);
  else if (name == "sabal")
    process_vec_widen_binary_add(gen_abd, Op::SEXT, false);
  else if (name == "sabal2")
    process_vec_widen_binary_add(gen_abd, Op::SEXT, true);
  else if (name == "sabdl")
    process_vec_widen_binary(gen_abd, Op::SEXT, false);
  else if (name == "sabdl2")
    process_vec_widen_binary(gen_abd, Op::SEXT, true);
  else if (name == "sadalp")
    process_vec_widen_pairwise_add(gen_add, Op::SEXT);
  else if (name == "saddl")
    process_vec_widen_binary(gen_add, Op::SEXT, false);
  else if (name == "saddl2")
    process_vec_widen_binary(gen_add, Op::SEXT, true);
  else if (name == "saddlp")
    process_vec_widen_pairwise(gen_add, Op::SEXT);
  else if (name == "saddw")
    process_vec_widen2_binary(Op::ADD, Op::SEXT, false);
  else if (name == "saddw2")
    process_vec_widen2_binary(Op::ADD, Op::SEXT, true);
  else if (name == "scvtf")
    process_vec_unary(gen_s2f);
  else if (name == "shl")
    process_vec_shift(Op::SHL);
  else if (name == "shrn")
    process_vec_shrn(false);
  else if (name == "shrn2")
    process_vec_shrn(true);
  else if (name == "smax")
    process_vec_binary(gen_smax);
  else if (name == "smaxp")
    process_vec_pairwise(gen_smax);
  else if (name == "smin")
    process_vec_binary(gen_smin);
  else if (name == "sminp")
    process_vec_pairwise(gen_smin);
  else if (name == "smlal")
    process_vec_widen_binary_add(gen_mul, Op::SEXT, false);
  else if (name == "smlal2")
    process_vec_widen_binary_add(gen_mul, Op::SEXT, true);
  else if (name == "smull")
    process_vec_widen_binary(gen_mul, Op::SEXT, false);
  else if (name == "smull2")
    process_vec_widen_binary(gen_mul, Op::SEXT, true);
  else if (name == "shll")
    process_vec_widen_shift(Op::SHL, Op::ZEXT, false);
  else if (name == "shll2")
    process_vec_widen_shift(Op::SHL, Op::ZEXT, true);
  else if (name == "sshl")
    process_vec_binary(gen_sshl);
  else if (name == "sshll")
    process_vec_widen_shift(Op::SHL, Op::SEXT, false);
  else if (name == "sshll2")
    process_vec_widen_shift(Op::SHL, Op::SEXT, true);
  else if (name == "sshr")
    process_vec_shift(Op::ASHR);
  else if (name == "ssra")
    process_vec_shift_acc(Op::ASHR);
  else if (name == "ssubl")
    process_vec_widen_binary(gen_sub, Op::SEXT, false);
  else if (name == "ssubl2")
    process_vec_widen_binary(gen_sub, Op::SEXT, true);
  else if (name == "ssubw")
    process_vec_widen2_binary(Op::SUB, Op::SEXT, false);
  else if (name == "ssubw2")
    process_vec_widen2_binary(Op::SUB, Op::SEXT, true);
  else if (name == "sub")
    process_vec_binary(Op::SUB);
  else if (name == "subhn")
    process_vec_binary_high_narrow(Op::SUB, false);
  else if (name == "subhn2")
    process_vec_binary_high_narrow(Op::SUB, true);
  else if (name == "sxtl")
    process_vec_widen(Op::SEXT);
  else if (name == "sxtl2")
    process_vec_widen(Op::SEXT, true);
  else if (name == "tbl")
    process_vec_tbl();
  else if (name == "trn1")
    process_vec_trn1();
  else if (name == "trn2")
    process_vec_trn2();
  else if (name == "uabal")
    process_vec_widen_binary_add(gen_abd, Op::ZEXT, false);
  else if (name == "uabal2")
    process_vec_widen_binary_add(gen_abd, Op::ZEXT, true);
  else if (name == "uabdl")
    process_vec_widen_binary(gen_abd, Op::ZEXT, false);
  else if (name == "uabdl2")
    process_vec_widen_binary(gen_abd, Op::ZEXT, true);
  else if (name == "uadalp")
    process_vec_widen_pairwise_add(gen_add, Op::ZEXT);
  else if (name == "uaddl")
    process_vec_widen_binary(gen_add, Op::ZEXT, false);
  else if (name == "uaddl2")
    process_vec_widen_binary(gen_add, Op::ZEXT, true);
  else if (name == "uaddlp")
    process_vec_widen_pairwise(gen_add, Op::ZEXT);
  else if (name == "uaddw")
    process_vec_widen2_binary(Op::ADD, Op::ZEXT, false);
  else if (name == "uaddw2")
    process_vec_widen2_binary(Op::ADD, Op::ZEXT, true);
  else if (name == "ucvtf")
    process_vec_unary(gen_u2f);
  else if (name == "umax")
    process_vec_binary(gen_umax);
  else if (name == "umaxp")
    process_vec_pairwise(gen_umax);
  else if (name == "umin")
    process_vec_binary(gen_umin);
  else if (name == "uminp")
    process_vec_pairwise(gen_umin);
  else if (name == "umlal")
    process_vec_widen_binary_add(gen_mul, Op::ZEXT, false);
  else if (name == "umlal2")
    process_vec_widen_binary_add(gen_mul, Op::ZEXT, true);
  else if (name == "umull")
    process_vec_widen_binary(gen_mul, Op::ZEXT, false);
  else if (name == "umull2")
    process_vec_widen_binary(gen_mul, Op::ZEXT, true);
  else if (name == "ushl")
    process_vec_binary(gen_ushl);
  else if (name == "ushll")
    process_vec_widen_shift(Op::SHL, Op::ZEXT, false);
  else if (name == "ushll2")
    process_vec_widen_shift(Op::SHL, Op::ZEXT, true);
  else if (name == "ushr")
    process_vec_shift(Op::LSHR);
  else if (name == "usra")
    process_vec_shift_acc(Op::LSHR);
  else if (name == "usubl")
    process_vec_widen_binary(gen_sub, Op::ZEXT, false);
  else if (name == "usubl2")
    process_vec_widen_binary(gen_sub, Op::ZEXT, true);
  else if (name == "usubw")
    process_vec_widen2_binary(Op::SUB, Op::ZEXT, false);
  else if (name == "usubw2")
    process_vec_widen2_binary(Op::SUB, Op::ZEXT, true);
  else if (name == "uxtl")
    process_vec_widen(Op::ZEXT);
  else if (name == "uxtl2")
    process_vec_widen(Op::ZEXT, true);
  else if (name == "uzp1")
    process_vec_uzp(false);
  else if (name == "uzp2")
    process_vec_uzp(true);
  else if (name == "xtn")
    process_vec_xtn(false);
  else if (name == "xtn2")
    process_vec_xtn(true);
  else if (name == "zip1")
    process_vec_zip1();
  else if (name == "zip2")
    process_vec_zip2();
  else
    throw Parse_error("unhandled vector instruction: "s + std::string(name),
		      line_number);
}

void Parser::parse_function()
{
  if (tokens[0].kind == Lexeme::label_def)
  {
    Basic_block *dest_bb = get_bb_def(0);
    get_end_of_line(1);

    if (bb)
      bb->build_br_inst(dest_bb);
    bb = dest_bb;
    return;
  }

  if (tokens[0].kind != Lexeme::name)
    throw Parse_error("syntax error: " + std::string(token_string(tokens[0])),
		      line_number);

  std::string_view name = get_name(0);
  if (name.starts_with(".cfi"))
    ;
  else if (name.starts_with(".LF"))
    ;
  else if (name.starts_with(".p2align"))
    ;
  else if (name == ".section")
    {
      if (tokens.size() > 1)
	{
	  if (get_name(1) == "__patchable_function_entries")
	    throw Not_implemented("attribue patchable_function_entry");
	}
      throw Parse_error(".section in the middle of a function", line_number);
    }

  else if (is_vector_op())
    parse_vector_op();

  // Branches, Exception generating, and System instructions
  else if (name == "beq")
    process_cond_branch(Cond_code::EQ);
  else if (name == "bne")
    process_cond_branch(Cond_code::NE);
  else if (name == "bcs")
    process_cond_branch(Cond_code::CS);
  else if (name == "bcc")
    process_cond_branch(Cond_code::CC);
  else if (name == "bmi")
    process_cond_branch(Cond_code::MI);
  else if (name == "bpl")
    process_cond_branch(Cond_code::PL);
  else if (name == "bvs")
    process_cond_branch(Cond_code::VS);
  else if (name == "bvc")
    process_cond_branch(Cond_code::VC);
  else if (name == "bhi")
    process_cond_branch(Cond_code::HI);
  else if (name == "bls")
    process_cond_branch(Cond_code::LS);
  else if (name == "bge")
    process_cond_branch(Cond_code::GE);
  else if (name == "blt")
    process_cond_branch(Cond_code::LT);
  else if (name == "bgt")
    process_cond_branch(Cond_code::GT);
  else if (name == "ble")
    process_cond_branch(Cond_code::LE);
  else if (name == "cbnz")
    process_cbz(true);
  else if (name == "cbz")
    process_cbz();
  else if (name == "tbnz")
    process_tbz(true);
  else if (name == "tbz")
    process_tbz();
  else if (name == "b" && tokens.size() > 1 && tokens[1].kind != Lexeme::label)
    process_call();
  else if (name == "b")
    {
      Basic_block *dest_bb = get_bb(1);
      get_end_of_line(2);

      bb->build_br_inst(dest_bb);
      bb = nullptr;
    }
  else if (name == "bl")
    process_call();
  else if (name == "ret")
    {
      get_end_of_line(1);

      bb->build_br_inst(rstate->exit_bb);
      bb = nullptr;
    }
  else if (name == "brk")
    {
      get_imm(1);
      get_end_of_line(2);

      bb->build_inst(Op::UB, bb->value_inst(1, 1));
      bb->build_br_inst(rstate->exit_bb);
      bb = nullptr;
    }

  // Hint instructions
  else if (name == "nop")
    get_end_of_line(1);

  // Loads and Stores
  else if (name == "ldr")
    process_load();
  else if (name == "ldrb")
    process_load(1);
  else if (name == "ldrsb")
    process_load(1, Op::SEXT);
  else if (name == "ldrh")
    process_load(2);
  else if (name == "ldrsh")
    process_load(2, Op::SEXT);
  else if (name == "ldrsw")
    process_load(4, Op::SEXT);
  else if (name == "str")
    process_store();
  else if (name == "strb")
    process_store(1);
  else if (name == "strh")
    process_store(2);
  else if (name == "ldp")
    process_ldp();
  else if (name == "ldpsw")
    process_ldpsw();
  else if (name == "stp")
    process_stp();

  // Data processing - arithmetic
  else if (name == "add")
    process_binary(Op::ADD);
  else if (name == "adds")
    process_adds();
  else if (name == "sub")
    process_binary(Op::SUB);
  else if (name == "subs")
    process_subs();
  else if (name == "cmn")
    process_cmn();
  else if (name == "cmp")
    process_cmp();
  else if (name == "neg")
    process_unary(Op::NEG);
  else if (name == "negs")
    process_negs();

  // Data processing - arithmetic with carry
  else if (name == "adc")
    process_adc();
  else if (name == "adcs")
    process_adcs();
  else if (name == "sbc")
    process_sbc();
  else if (name == "sbcs")
    process_sbcs();
  else if (name == "ngc")
    process_ngc();
  else if (name == "ngcs")
    process_ngcs();

  // Data processing - integer minimum and maximum
  else if (name == "smax")
    process_binary(gen_smax);
  else if (name == "smin")
    process_binary(gen_smin);
  else if (name == "umax")
    process_binary(gen_umax);
  else if (name == "umin")
    process_binary(gen_umin);

  // Data processing - logical
  else if (name == "and")
    process_binary(Op::AND);
  else if (name == "ands")
    process_ands();
  else if (name == "bic")
    process_binary(Op::AND, true);
  else if (name == "bics")
    process_ands(true);
  else if (name == "eon")
    process_binary(Op::XOR, true);
  else if (name == "eor")
    process_binary(Op::XOR);
  else if (name == "orr")
    process_binary(Op::OR);
  else if (name == "mvn")
    process_unary(Op::NOT);
  else if (name == "orn")
    process_binary(Op::OR, true);
  else if (name == "tst")
    process_tst();

  // Data processing - move
  else if (name == "mov")
    process_unary(Op::MOV);
  else if (name == "movk")
    process_movk();

  // Data processing - absolute value
  else if (name == "abs")
    process_unary(gen_abs);

  // Data processing - address calculation
  else if (name == "adrp")
    process_adrp();

  // Data processing - bitfield insert and extract
  else if (name == "bfi")
    process_bfi();
  else if (name == "bfxil")
    process_bfxil();
  else if (name == "sbfiz")
    process_ubfiz(Op::SEXT);
  else if (name == "sbfx")
    process_ubfx(Op::SEXT);
  else if (name == "ubfiz")
    process_ubfiz(Op::ZEXT);
  else if (name == "ubfx")
    process_ubfx(Op::ZEXT);

  // Data processing - extract register
  else if (name == "extr")
    process_extr();

  // Data processing - shift
  else if (name == "asr")
    process_shift(Op::ASHR);
  else if (name == "lsl")
    process_shift(Op::SHL);
  else if (name == "lsr")
    process_shift(Op::LSHR);
  else if (name == "ror")
    process_ror();

  // Data processing - multiply
  else if (name == "madd")
    process_mul_op(Op::ADD);
  else if (name == "msub")
    process_mul_op(Op::SUB);
  else if (name == "mneg")
    process_mneg();
  else if (name == "mul")
    process_binary(Op::MUL);
  else if (name == "smaddl")
    process_maddl(Op::SEXT);
  else if (name == "smsubl")
    process_msubl(Op::SEXT);
  else if (name == "smnegl")
    process_mnegl(Op::SEXT);
  else if (name == "smull")
    process_mull(Op::SEXT);
  else if (name == "smulh")
    process_mulh(Op::SEXT);
  else if (name == "umaddl")
    process_maddl(Op::ZEXT);
  else if (name == "umsubl")
    process_msubl(Op::ZEXT);
  else if (name == "umnegl")
    process_mnegl(Op::ZEXT);
  else if (name == "umull")
    process_mull(Op::ZEXT);
  else if (name == "umulh")
    process_mulh(Op::ZEXT);

  // Data processing - divide
  else if (name == "sdiv")
    process_binary(Op::SDIV);
  else if (name == "udiv")
    process_binary(Op::UDIV);

  // Data processing - bit operations
  else if (name == "cls")
    process_unary(gen_clrsb);
  else if (name == "clz")
    process_unary(gen_clz);
  else if (name == "cnt")
    process_unary(gen_popcount);
  else if (name == "ctz")
    process_unary(gen_ctz);
  else if (name == "rbit")
    process_unary(gen_bitreverse);
  else if (name == "rev")
    process_rev();
  else if (name == "rev16")
    process_rev(16);
  else if (name == "rev32")
    process_rev(32);
  else if (name == "rev64")
    process_rev(64);

  // Data processing - conditional select
  else if (name == "csel")
    process_csel();
  else if (name == "csinc")
    process_csel(Op::ADD);
  else if (name == "csinv")
    process_csel(Op::NOT);
  else if (name == "csneg")
    process_csel(Op::NEG);
  else if (name == "cset")
    process_cset();
  else if (name == "csetm")
    process_cset(Op::SEXT);
  else if (name == "cinc")
    process_cinc();

  // Data processing - conditional comparision
  else if (name == "ccmn")
    process_ccmp(true);
  else if (name == "ccmp")
    process_ccmp();

  // Data processing - sign-extend and zero-extend
  else if (name == "sxtb")
    process_ext(Op::SEXT, 8);
  else if (name == "sxth")
    process_ext(Op::SEXT, 16);
  else if (name == "sxtw")
    process_ext(Op::SEXT, 32);
  else if (name == "uxtb")
    process_ext(Op::ZEXT, 8);
  else if (name == "uxth")
    process_ext(Op::ZEXT, 16);
  else if (name == "uxtw")
    process_ext(Op::ZEXT, 32);

  // Floating-point move
  else if (name == "fmov")
    process_fmov();

  // Floating-point conversion
  else if (name == "fcvt")
    process_f2f();
  else if (name == "fcvtzs")
    process_f2i(false);
  else if (name == "fcvtzu")
    process_f2i(true);
  else if (name == "scvtf")
    process_i2f(false);
  else if (name == "ucvtf")
    process_i2f(true);

  // Floating-point arithmetic (one source)
  else if (name == "fabs")
    process_unary(Op::FABS);
  else if (name == "fneg")
    process_unary(Op::FNEG);

  // Floating-point arithmetic (two sources)
  else if (name == "fadd")
    process_binary(Op::FADD);
  else if (name == "fdiv")
    process_binary(Op::FDIV);
  else if (name == "fmul")
    process_binary(Op::FMUL);
  else if (name == "fnmul")
    process_binary(gen_fnmul);
  else if (name == "fsub")
    process_binary(Op::FSUB);

  // Floating-point minimum and maximum
  else if (name == "fmaxnm")
    process_binary(gen_fmax);
  else if (name == "fminnm")
    process_binary(gen_fmin);

  // Floating-point comparison
  else if (name == "fcmp" || name == "fcmpe")
    process_fcmp();
  else if (name == "fccmp" || name == "fccmpe")
    process_fccmp();

  // Floating-point conditional select
  else if (name == "fcsel")
    process_csel();

  // SIMD compare
  else if (name == "cmeq")
    process_simd_compare(SIMD_cond::EQ);
  else if (name == "cmhs")
    process_simd_compare(SIMD_cond::HS);
  else if (name == "cmge")
    process_simd_compare(SIMD_cond::GE);
  else if (name == "cmhi")
    process_simd_compare(SIMD_cond::HI);
  else if (name == "cmgt")
    process_simd_compare(SIMD_cond::GT);
  else if (name == "cmle")
    process_simd_compare(SIMD_cond::LE);
  else if (name == "cmlt")
    process_simd_compare(SIMD_cond::LT);
  else if (name == "fcmeq")
    process_simd_compare(SIMD_cond::FEQ);
  else if (name == "fcmge")
    process_simd_compare(SIMD_cond::FGE);
  else if (name == "fcmgt")
    process_simd_compare(SIMD_cond::FGT);
  else if (name == "fcmle")
    process_simd_compare(SIMD_cond::FLE);
  else if (name == "fcmlt")
    process_simd_compare(SIMD_cond::FLT);

  // SIMD move
  else if (name == "dup")
    process_dup();
  else if (name == "movi")
    process_unary(Op::MOV);
  else if (name == "smov")
    process_smov();
  else if (name == "umov")
    process_umov();

  // SIMD shift
  else if (name == "sshr")
    process_simd_shift(Op::ASHR);
  else if (name == "ushr")
    process_simd_shift(Op::LSHR);

  // SIMD reduce
  else if (name == "addv")
    process_vec_reduc(gen_add);
  else if (name == "saddlv")
    process_vec_reducl(gen_add, Op::SEXT);
  else if (name == "smaxv")
    process_vec_reduc(gen_smax);
  else if (name == "sminv")
    process_vec_reduc(gen_smin);
  else if (name == "uaddlv")
    process_vec_reducl(gen_add, Op::ZEXT);
  else if (name == "umaxv")
    process_vec_reduc(gen_umax);
  else if (name == "uminv")
    process_vec_reduc(gen_umin);

  // SIMD pairwise arithmetic
  else if (name == "addp")
    process_vec_reduc(gen_add);

  // SIMD unary
  else if (name == "sqxtn")
    process_unary(gen_sqxtn);

  else
    throw Parse_error("unhandled instruction: "s + std::string(name),
		      line_number);
}

void Parser::lex_line(void)
{
  tokens.clear();
  while (buf[pos] != '\n' && buf[pos] != ';')
    {
      skip_space_and_comments();
      if (buf[pos] == '\n' || buf[pos] == ';')
	break;
      if (buf[pos] == '-'
	  && !tokens.empty()
	  && tokens.back().kind != Lexeme::comma)
	{
	  tokens.emplace_back(Lexeme::minus, pos, 1);
	  pos++;
	}
      else if (isdigit(buf[pos]) || buf[pos] == '-')
	lex_hex_or_integer();
      else if (is_lo12())
	{
	  tokens.emplace_back(Lexeme::lo12, pos, 6);
	  pos += 6;
	}
      else if (buf[pos] == '#')
	{
	  pos++;
	  if (is_lo12())
	    {
	      tokens.emplace_back(Lexeme::lo12, pos, 6);
	      pos += 6;
	    }
	  else
	    lex_hex_or_integer();
	}
      else if (buf[pos] == '.' && buf[pos + 1] == 'L' && isdigit(buf[pos + 2]))
	lex_label_or_label_def();
      else if (buf[pos] == '.' && tokens.empty())
	{
	  lex_name();
	  if (get_name(0) == ".section")
	    {
	      skip_space_and_comments();
	      lex_name();
	    }

	  // The assembler directives have a different grammar than the
	  // assembler instructions. But we are not using the arguments,
	  // so just skip the content for now.
	  while (buf[pos] != '\n' && buf[pos] != ';')
	    pos++;
	  break;
	}
      else if (isalpha(buf[pos]) || buf[pos] == '_' || buf[pos] == '.')
	lex_name();
      else if (buf[pos] == ',')
	{
	  tokens.emplace_back(Lexeme::comma, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '+')
	{
	  tokens.emplace_back(Lexeme::plus, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '!')
	{
	  tokens.emplace_back(Lexeme::exclamation, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '[')
	{
	  tokens.emplace_back(Lexeme::left_bracket, pos, 1);
	  pos++;
	}
      else if (buf[pos] == ']')
	{
	  tokens.emplace_back(Lexeme::right_bracket, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '(')
	{
	  tokens.emplace_back(Lexeme::left_paren, pos, 1);
	  pos++;
	}
      else if (buf[pos] == ')')
	{
	  tokens.emplace_back(Lexeme::right_paren, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '{')
	{
	  tokens.emplace_back(Lexeme::left_brace, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '}')
	{
	  tokens.emplace_back(Lexeme::right_brace, pos, 1);
	  pos++;
	}
      else
	throw Parse_error("syntax error", line_number);
    }
  pos++;
}

std::optional<std::string_view> Parser::parse_label_def()
{
  size_t start_pos = pos;
  while (buf[pos] != ':' && buf[pos] != '\n')
    {
      pos++;
    }
  if (buf[pos] == '\n')
    {
      pos++;
      return {};
    }
  std::string_view label(&buf[start_pos], pos - start_pos);
  pos++;
  if (buf[pos] == '\n')
    {
      pos++;
      return label;
    }
  skip_line();
  return {};
}

std::string_view Parser::parse_cmd()
{
  size_t start_pos = pos;
  while (buf[pos] == '.' || isalnum(buf[pos]))
    pos++;
  assert(buf[pos] == ' '
	 || buf[pos] == '\t'
	 || buf[pos] == '\n'
	 || buf[pos] == ':');
  return std::string_view(&buf[start_pos], pos - start_pos);
}

void Parser::parse_data(std::vector<unsigned char>& data)
{
  for (;;)
    {
      size_t start_pos = pos;

      if (pos == buf.size() - 1)
	break;
      assert(pos < buf.size());

      skip_whitespace();
      if (buf[pos] != '.')
	{
	  pos = start_pos;
	  return;
	}
      std::string_view cmd = parse_cmd();
      if (cmd == ".xword"
	  || cmd == ".word"
	  || cmd == ".hword"
	  || cmd == ".2byte"
	  || cmd == ".byte")
	{
	  skip_whitespace();

	  uint64_t value = 0;
	  bool negate = false;
	  if (buf[pos] == '-')
	    {
	      pos++;
	      negate = true;
	    }
	  if (!isdigit(buf[pos]))
	    throw Parse_error(std::string(cmd) + " value is not a number",
			      line_number);
	  while (isdigit(buf[pos]))
	    {
	      value = value * 10 + (buf[pos] - '0');
	      pos++;
	    }
	  if (negate)
	    value = -value;

	  int size;
	  if (cmd == ".byte")
	    size = 1;
	  else if (cmd == ".hword" || cmd == ".2byte")
	    size = 2;
	  else if (cmd == ".word")
	    size = 4;
	  else
	    size = 8;
	  for (int i = 0; i < size; i++)
	    {
	      data.push_back(value & 0xff);
	      value = value >> 8;
	    }

	  assert(buf[pos] == '\n');

	  skip_line();
	}
      else if (cmd == ".string" || cmd == ".ascii")
	{
	  skip_whitespace();

	  if (buf[pos++] != '"')
	    throw Parse_error("expected '\"' after " + std::string(cmd),
			      line_number);

	  while (buf[pos] != '"')
	    {
	      assert(buf[pos] != '\n');
	      if (buf[pos] == '\\')
		{
		  char c = buf[++pos];
		  if (c == 'b')
		    data.push_back('\b');
		  else if (c == 'f')
		    data.push_back('\f');
		  else if (c == 'n')
		    data.push_back('\n');
		  else if (c == 'r')
		    data.push_back('\r');
		  else if (c == 't')
		    data.push_back('\t');
		  else if (c == '"')
		    data.push_back('\"');
		  else if (c == '\\')
		    data.push_back('\\');
		  else if ('0' <= c && c <= '7')
		    {
		      uint8_t val = c;
		      for (int i = 1; i < 3; i++)
			{
			  c = buf[pos + 1];
			  if (c < '0' || c > '7')
			    break;
			  val = val * 8 + (c - '0');
			  pos++;
			}
		      data.push_back(val);
		    }
		  else
		    throw Parse_error("unknown escape sequence \\"s + c,
				      line_number);
		}
	      else
		data.push_back(buf[pos]);
	      pos++;
	    }
	  pos++;
	  assert(buf[pos] == '\n');
	  skip_line();

	  if (cmd == ".string")
	    data.push_back(0);
	}
      else if (cmd == ".zero")
	{
	  skip_whitespace();
	  uint64_t size = 0;
	  if (!isdigit(buf[pos]))
	    throw Parse_error(".zero size is not a number", line_number);
	  while (isdigit(buf[pos]))
	    {
	      size = size * 10 + (buf[pos] - '0');
	      pos++;
	    }

	  for (size_t i = 0; i < size; i++)
	    data.push_back(0);

	  assert(buf[pos] == '\n');

	  skip_line();
	}
      else if (cmd == ".base64")
	throw Parse_error(".base64 not supported yet", line_number);
      else
	{
	  pos = start_pos;
	  break;
	}
    }
}

void Parser::skip_line()
{
  while (buf[pos] != '\n')
    {
      pos++;
    }
  pos++;
}

void Parser::skip_whitespace()
{
  while (buf[pos] == ' ' || buf[pos] == '\t')
    {
      pos++;
    }
}

void Parser::parse_rodata()
{
  enum class state {
    global,
    memory_section
  };

  pos = 0;
  state parser_state = state::global;
  for (;;)
    {
      size_t start_pos = pos;

      if (pos == buf.size() - 1)
	break;
      assert(pos < buf.size());

      skip_whitespace();
      if (buf[pos] == '.'
	  && buf[pos + 1] == 's'
	  && buf[pos + 2] == 'e'
	  && buf[pos + 3] == 'c'
	  && buf[pos + 4] == 't'
	  && buf[pos + 5] == 'i'
	  && buf[pos + 6] == 'o'
	  && buf[pos + 7] == 'n'
	  && (buf[pos + 8] == ' ' || buf[pos + 8] == '\t'))
	{
	  pos += 9;
	  skip_whitespace();

	  size_t first_pos = pos;
	  while (buf[pos] == '.' || isalnum(buf[pos]))
	    pos++;
	  std::string_view name(&buf[first_pos], pos - first_pos);
	  if (name.starts_with(".rodata") || name.starts_with(".srodata"))
	    parser_state = state::memory_section;
	  else
	    parser_state = state::global;
	  skip_line();
	  continue;
	}
      if (buf[pos] == '.'
	  && buf[pos + 1] == 'a'
	  && buf[pos + 2] == 'l'
	  && buf[pos + 3] == 'i'
	  && buf[pos + 4] == 'g'
	  && buf[pos + 5] == 'n'
	  && (buf[pos + 6] == ' ' || buf[pos + 6] == '\t'))
	{
	  pos += 7;
	  skip_line();
	  continue;
	}
      if (buf[pos] == '.'
	  && buf[pos + 1] == 'p'
	  && buf[pos + 2] == '2'
	  && buf[pos + 3] == 'a'
	  && buf[pos + 4] == 'l'
	  && buf[pos + 5] == 'i'
	  && buf[pos + 6] == 'g'
	  && buf[pos + 7] == 'n'
	  && (buf[pos + 8] == ' ' || buf[pos + 8] == '\t'))
	{
	  pos += 9;
	  skip_line();
	  continue;
	}
      if (buf[pos] == '.'
	  && buf[pos + 1] == 't'
	  && buf[pos + 2] == 'y'
	  && buf[pos + 3] == 'p'
	  && buf[pos + 4] == 'e'
	  && (buf[pos + 5] == ' ' || buf[pos + 5] == '\t'))
	{
	  pos += 6;
	  skip_line();
	  continue;
	}
      if (buf[pos] == '.'
	  && buf[pos + 1] == 's'
	  && buf[pos + 2] == 'i'
	  && buf[pos + 3] == 'z'
	  && buf[pos + 4] == 'e'
	  && (buf[pos + 5] == ' ' || buf[pos + 5] == '\t'))
	{
	  pos += 6;
	  skip_line();
	  continue;
	}
      if (buf[pos] == '.'
	  && buf[pos + 1] == 'l'
	  && buf[pos + 2] == 'o'
	  && buf[pos + 3] == 'c'
	  && buf[pos + 4] == 'a'
	  && buf[pos + 5] == 'l'
	  && (buf[pos + 6] == ' ' || buf[pos + 6] == '\t'))
	{
	  pos += 7;
	  skip_line();
	  continue;
	}
      if (buf[pos] == '.'
	  && buf[pos + 1] == 'g'
	  && buf[pos + 2] == 'l'
	  && buf[pos + 3] == 'o'
	  && buf[pos + 4] == 'b'
	  && buf[pos + 5] == 'a'
	  && buf[pos + 6] == 'l'
	  && (buf[pos + 7] == ' ' || buf[pos + 7] == '\t'))
	{
	  pos += 8;
	  skip_line();
	  continue;
	}
      if (buf[pos] == '.'
	  && buf[pos + 1] == 'c'
	  && buf[pos + 2] == 'o'
	  && buf[pos + 3] == 'm'
	  && buf[pos + 4] == 'm'
	  && (buf[pos + 5] == ' ' || buf[pos + 5] == '\t'))
	{
	  pos += 6;
	  skip_line();
	  continue;
	}
      if (buf[pos] == '.'
	  && buf[pos + 1] == 's'
	  && buf[pos + 2] == 'e'
	  && buf[pos + 3] == 't'
	  && (buf[pos + 4] == ' ' || buf[pos + 4] == '\t'))
	{
	  pos += 5;
	  skip_whitespace();

	  int start = pos;
	  while (isalnum(buf[pos])
		 || buf[pos] == '_'
		 || buf[pos] == '-'
		 || buf[pos] == '.'
		 || buf[pos] == '$')
	    pos++;
	  skip_whitespace();
	  std::string name1(&buf[start], pos - start);

	  if (buf[pos++] != ',')
	    continue;
	  skip_whitespace();

	  start = pos;
	  while (isalnum(buf[pos])
		 || buf[pos] == '_'
		 || buf[pos] == '-'
		 || buf[pos] == '.'
		 || buf[pos] == '$')
	    pos++;
	  skip_whitespace();
	  std::string name2(&buf[start], pos - start);

	  uint64_t offset = 0;
	  if (buf[pos] == '+')
	    {
	      pos++;
	      skip_whitespace();
	      while (isdigit(buf[pos]))
		{
		  offset = offset * 10 + (buf[pos] - '0');
		  pos++;
		}
	      skip_whitespace();
	    }

	  if (buf[pos] != '\n')
	    throw Parse_error(".set", line_number);

	  sym_alias.insert({name1, {name2, offset}});

	  skip_line();
	  continue;
	}

      pos = start_pos;

      if (parser_state == state::memory_section)
	{
	  std::optional<std::string_view> label = parse_label_def();
	  if (label)
	    {
	      std::string label_name = std::string(*label);

	      // TODO: Change to check for duplicated labels.
	      assert(!sym_name2data.contains(label_name));

	      parse_data(sym_name2data[label_name]);

	      if (sym_name2data[label_name].empty())
		throw Parse_error("Failed to parse data for " + label_name,
				  line_number);

	      continue;
	    }
	}
      else
	skip_line();

      parser_state = state::global;
    }
}

Function *Parser::parse(std::string const& file_name)
{
  enum class state {
    global,
    function,
    basic_block,
    instruction,
    done
  };

  std::ifstream file(file_name);
  if (!file)
    throw Parse_error("Could not open file.", 0);
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  buf.resize(file_size + 1);
  if (!file.read(buf.data(), file_size))
    throw Parse_error("Could not read file.", 0);
  buf[file_size] = '\n';

  // The parsing code has problems with Unicode characters in labels.
  // Report "not implemented" for all uses of non-ASCII characters for
  // now.
  for (auto c : buf)
    {
      if ((unsigned char)c > 127)
	throw Not_implemented("non-ASCII character in assembly file");
    }

  module = rstate->module;
  assert(module->functions.size() == 2);
  reg_bitsize = rstate->reg_bitsize;
  freg_bitsize = rstate->freg_bitsize;
  src_func = module->functions[0];
  zero_reg = rstate->entry_bb->build_inst(Op::REGISTER, reg_bitsize);

  parse_rodata();

  state parser_state = state::global;
  pos = 0;
  while (parser_state != state::done) {
    if (pos == file_size)
      break;
    assert(pos < file_size);

    if (pos == 0 || buf[pos] != ';')
      line_number++;

    if (parser_state == state::global)
      {
	std::optional<std::string_view> label = parse_label_def();
	if (!label)
	  continue;

	if (label == rstate->func_name)
	  {
	    func = module->functions[1];
	    Basic_block *entry_bb = rstate->entry_bb;

	    bb = func->build_bb();
	    entry_bb->build_br_inst(bb);

	    int64_t next_id = -(((int64_t)1) << (module->ptr_id_bits - 1)) + 1;
	    for (const auto& [name, data] : sym_name2data)
	      {
		Inst *mem;
		if (rstate->sym_name2mem.contains(name))
		  mem = rstate->sym_name2mem.at(name);
		else
		  {
		    if (next_id == 0)
		      throw Not_implemented("too many local variables");

		    Inst *id =
		      entry_bb->value_inst(next_id++, module->ptr_id_bits);
		    Inst *mem_size =
		      entry_bb->value_inst(data.size(), module->ptr_offset_bits);
		    Inst *flags = entry_bb->value_inst(MEM_CONST, 32);
		    mem = entry_bb->build_inst(Op::MEMORY, id, mem_size, flags);

		    assert(!rstate->sym_name2mem.contains(name));
		    rstate->sym_name2mem.insert({name, mem});
		  }
		for (size_t i = 0; i < data.size(); i++)
		  {
		    Inst *off = entry_bb->value_inst(i, mem->bitsize);
		    Inst *ptr = entry_bb->build_inst(Op::ADD, mem, off);
		    Inst *byte = entry_bb->value_inst(data[i], 8);
		    entry_bb->build_inst(Op::STORE, ptr, byte);
		  }
	      }

	    parser_state = state::function;
	  }
	continue;
      }

    lex_line();
    if (tokens.empty())
      continue;

    if (parser_state == state::function)
      {
	if (tokens[0].kind == Lexeme::name && get_name(0) == ".size")
	  {
	    // We may have extra labels after the function, such as:
	    //
	    //   foo:
	    //        ret
	    //   .L4:
	    //        .size   foo, .-foo
	    //
	    // Make this valid by branching back to bb (this is then
	    // removed when building RPO as the BB is unreachable).
	    if (bb)
	      {
		bb->build_br_inst(bb);
		bb = nullptr;
	      }

	    parser_state = state::done;
	    continue;
	  }
	parse_function();
      }
    else
      {
	throw Parse_error("Cannot happen", line_number);
      }
  }

  if (parser_state != state::done)
    throw Parse_error("EOF in the middle of a function", line_number);

  return func;
}

} // end anonymous namespace

Function *parse_aarch64(std::string const& file_name, aarch64_state *state)
{
  Parser p(state);
  Function *func = p.parse(file_name);
  reverse_post_order(func);
  return func;
}

} // end namespace smtgcc
