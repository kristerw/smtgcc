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
    exclamation,
    left_bracket,
    right_bracket,
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
  std::tuple<Inst *, uint32_t, uint32_t> get_vreg(unsigned idx);
  uint32_t get_reg_size(unsigned idx);
  bool is_vector_op();
  bool is_freg(unsigned idx);
  bool is_vreg(unsigned idx);
  Inst *get_imm(unsigned idx);
  Inst *get_reg_value(unsigned idx);
  Inst *get_reg_or_imm_value(unsigned idx, uint32_t bitsize);
  Inst *get_freg_value(unsigned idx);
  Inst *get_vreg_value(unsigned idx, uint32_t nof_elem, uint32_t elem_bitsize);
  Basic_block *get_bb(unsigned idx);
  Basic_block *get_bb_def(unsigned idx);
  std::string_view get_name(unsigned idx);
  Inst *get_sym_addr(unsigned idx);
  Cond_code get_cc(unsigned idx);
  void get_comma(unsigned idx);
  void get_exclamation(unsigned idx);
  void get_left_bracket(unsigned idx);
  void get_right_bracket(unsigned idx);
  void get_end_of_line(unsigned idx);
  void write_reg(Inst *reg, Inst *value);
  Inst *build_cond(Cond_code cc);
  void process_cond_branch(Cond_code cc);
  void process_cbz(bool is_cbnz = false);
  void process_tbz(bool is_cbnz = false);
  void process_csel(Op op = Op::MOV);
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
  void process_i2f(bool is_unsigned);
  void process_f2i(bool is_unsigned);
  void process_f2f();
  void process_fmin_fmax(bool is_min);
  void process_min_max(bool is_min, bool is_unsigned);
  void process_mul_op(Op op);
  void process_maddl(Op op);
  void process_msubl(Op op);
  void process_mnegl(Op op);
  void process_mull(Op op);
  void process_mulh(Op op);
  void process_abs();
  void process_adrp();
  void process_adc();
  void process_adcs();
  void process_sbc();
  void process_sbcs();
  void process_ngc();
  void process_ngcs();
  void process_movk();
  void process_unary(Op op);
  Inst *process_arg_shift(unsigned idx, Inst *arg);
  Inst *process_arg_ext(unsigned idx, Inst *arg, uint32_t bitsize);
  Inst *process_last_arg(unsigned idx, uint32_t bitsize);
  Inst *process_last_scalar_vec_arg(unsigned idx, uint32_t elem_bitsize);
  void process_binary(Op op, bool perform_not = false);
  void process_cls();
  void process_clz();
  void process_rbit();
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
  void process_extr();
  void process_bfi();
  void process_bfxil();
  void process_ubfx(Op op);
  void process_ubfiz(Op op);
  Inst *extract_vec_elem(Inst *inst, uint32_t elem_bitsize, uint32_t idx);
  void process_vec_unary(Op op);
  void process_vec_binary(Op op);
  void process_vec_dup();
  void process_vec_movi();
  void process_vec_orr();
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
	 || buf[pos] == '-'
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
    throw Parse_error("expected a floating point register instead of "
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

std::tuple<Inst *, uint32_t, uint32_t> Parser::get_vreg(unsigned idx)
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
  if (suffix == ".2d")
    return {reg, 2, 64};
  if (suffix == ".2s")
    return {reg, 2, 32};
  if (suffix == ".4s")
    return {reg, 4, 32};
  if (suffix == ".4h")
    return {reg, 4, 16};
  if (suffix == ".8h")
    return {reg, 8, 16};
  if (suffix == ".8b")
    return {reg, 8, 8};
  if (suffix == ".16b")
    return {reg, 16, 8};

  throw Parse_error("expected a vector register instead of "
		    + std::string(token_string(tokens[idx])), line_number);
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

  throw Parse_error("expected a floating point register", line_number);
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

void Parser::process_fmin_fmax(bool is_min)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_or_imm_value(5, arg1->bitsize);
  get_end_of_line(6);

  Inst *is_nan = bb->build_inst(Op::IS_NAN, arg2);
  Inst *cmp;
  if (is_min)
    cmp = bb->build_inst(Op::FLT, arg1, arg2);
  else
    cmp = bb->build_inst(Op::FLT, arg2, arg1);
  Inst *res1 = bb->build_inst(Op::ITE, cmp, arg1, arg2);
  Inst *res2 = bb->build_inst(Op::ITE, is_nan, arg1, res1);
  // 0.0 and -0.0 is equal as floating point values, and fmin(0.0, -0.0)
  // may return eiter of them. But we treat them as 0.0 > -0.0 here,
  // otherwise we will report miscompilations when GCC switch the order
  // of the arguments.
  Inst *zero = bb->value_inst(0, arg1->bitsize);
  Inst *is_zero1 = bb->build_inst(Op::FEQ, arg1, zero);
  Inst *is_zero2 = bb->build_inst(Op::FEQ, arg2, zero);
  Inst *is_zero = bb->build_inst(Op::AND, is_zero1, is_zero2);
  Inst *cmp2;
  if (is_min)
    cmp2 = bb->build_inst(Op::SLT, arg1, arg2);
  else
    cmp2 = bb->build_inst(Op::SLT, arg2, arg1);
  Inst *res3 = bb->build_inst(Op::ITE, cmp2, arg1, arg2);
  Inst *res = bb->build_inst(Op::ITE, is_zero, res3, res2);
  write_reg(dest, res);
}

void Parser::process_min_max(bool is_min, bool is_unsigned)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_or_imm_value(5, arg1->bitsize);
  get_end_of_line(6);

  Inst *cmp = bb->build_inst(is_unsigned ? Op::ULT : Op::SLT, arg1, arg2);
  Inst *res;
  if (is_min)
    res = bb->build_inst(Op::ITE, cmp, arg1, arg2);
  else
    res = bb->build_inst(Op::ITE, cmp, arg2, arg1);
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

void Parser::process_abs()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *neg = bb->build_inst(Op::NEG, arg1);
  Inst *zero = bb->value_inst(0, arg1->bitsize);
  Inst *cond = bb->build_inst(Op::SLE, zero, arg1);
  Inst *res = bb->build_inst(Op::ITE, cond, arg1, neg);
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

void Parser::process_unary(Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  bool is_w_reg = buf[tokens[1].pos] == 'w';
  Inst *arg1 = process_last_arg(3, is_w_reg ? 32 : 64);

  Inst *res = bb->build_inst(op, arg1);
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

Inst *Parser::process_last_scalar_vec_arg(unsigned idx, uint32_t elem_bitsize)
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
  uint32_t bitsize;
  if (suffix == ".d")
    bitsize = 64;
  if (suffix == ".s")
    bitsize = 32;
  if (suffix == ".h")
    bitsize = 16;
  if (suffix == ".b")
    bitsize = 8;
  else if (elem_bitsize != bitsize)
    throw Parse_error("expected same arg vector size as dest", line_number);
  uint32_t nof_elem = 128 / elem_bitsize;
  idx++;

  get_left_bracket(idx++);
  uint32_t elem_idx = get_imm(idx++)->value();
  if (elem_idx >= nof_elem)
    throw Parse_error("elem index out of range", line_number);
  get_right_bracket(idx++);
  get_end_of_line(idx);

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

void Parser::process_cls()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *res = gen_clrsb(bb, arg1);
  write_reg(dest, res);
}

void Parser::process_clz()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *res = gen_clz(bb, arg1);
  write_reg(dest, res);
}

void Parser::process_rbit()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *res = gen_bitreverse(bb, arg1);
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

void Parser::process_vec_binary(Op op)
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3, nof_elem, elem_bitsize);
  get_comma(4);
  Inst *arg2;
  if (tokens.size() > 6)
    arg2 = process_last_scalar_vec_arg(5, elem_bitsize);
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
      Inst *inst = bb->build_inst(op, elem1, elem2);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  write_reg(dest, res);
}

void Parser::process_vec_dup()
{
  auto [dest, nof_elem, elem_bitsize] = get_vreg(1);
  get_comma(2);
  Inst *arg1;
  if (is_vreg(3))
    arg1 = process_last_scalar_vec_arg(3, elem_bitsize);
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

void Parser::process_vec_movi()
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
	    throw Parse_error("invalid shift value for orr", line_number);
	}
      else if (lsl == "msl")
	{
	  if (shift == 8)
	    m = 0xff;
	  else if (shift == 16)
	    m = 0xffff;
	  else
	    throw Parse_error("invalid shift value for orr", line_number);
	}
      else
	throw Parse_error("expected lsl/msl for shift", line_number);
      get_end_of_line(7);
    }
  else
    get_end_of_line(4);
  Inst *arg1 = bb->value_inst((value << shift) | m, elem_bitsize);

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

void Parser::parse_vector_op()
{
  std::string_view name = get_name(0);

  if (name == "add")
    process_vec_binary(Op::ADD);
  else if (name == "and")
    process_vec_binary(Op::AND);
  else if (name == "dup")
    process_vec_dup();
  else if (name == "eor")
    process_vec_binary(Op::XOR);
  else if (name == "fadd")
    process_vec_binary(Op::FADD);
  else if (name == "fdiv")
    process_vec_binary(Op::FDIV);
  else if (name == "fmul")
    process_vec_binary(Op::FMUL);
  else if (name == "fneg")
    process_vec_unary(Op::FNEG);
  else if (name == "fsub")
    process_vec_binary(Op::FSUB);
  else if (name == "movi")
    process_vec_movi();
  else if (name == "mul")
    process_vec_binary(Op::MUL);
  else if (name == "neg")
    process_vec_unary(Op::NEG);
  else if (name == "not")
    process_vec_unary(Op::NOT);
  else if (name == "orr")
    process_vec_orr();
  else if (name == "sub")
    process_vec_binary(Op::SUB);
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
  else if (name.starts_with(".p2align"))
    ;

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
    process_min_max(false, false);
  else if (name == "smin")
    process_min_max(true, false);
  else if (name == "umax")
    process_min_max(false, true);
  else if (name == "umin")
    process_min_max(true, true);

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
    process_abs();

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
    process_cls();
  else if (name == "clz")
    process_clz();
  // cnt
  else if (name == "rbit")
    process_rbit();
  else if (name == "rev")
    process_rev();
  else if (name == "rev16")
    process_rev(16);
  else if (name == "rev32")
    process_rev(32);

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

  // Data processing - SIMD and floating point
  // TODO:
  else if (name == "fmov")
    process_unary(Op::MOV);
  else if (name == "fneg")
    process_unary(Op::FNEG);
  else if (name == "fadd")
    process_binary(Op::FADD);
  else if (name == "fsub")
    process_binary(Op::FSUB);
  else if (name == "fmul")
    process_binary(Op::FMUL);
  else if (name == "fdiv")
    process_binary(Op::FDIV);
  else if (name == "scvtf")
    process_i2f(false);
  else if (name == "ucvtf")
    process_i2f(true);
  else if (name == "fcvtzs")
    process_f2i(false);
  else if (name == "fcvtzu")
    process_f2i(true);
  else if (name == "fcvt")
    process_f2f();
  else if (name == "fcmp" || name == "fcmpe")
    process_fcmp();
  else if (name == "fminnm")
    process_fmin_fmax(true);
  else if (name == "fmaxnm")
    process_fmin_fmax(false);

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
      if (isdigit(buf[pos]) || buf[pos] == '-')
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

	      // TODO: Change to check.
	      assert(!sym_name2data[label_name].empty());

	      continue;
	    }
	}

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
