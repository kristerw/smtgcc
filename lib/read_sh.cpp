#include <fstream>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "smtgcc.h"

using namespace std::string_literals;

using namespace smtgcc;

namespace smtgcc {
namespace {

struct Parser : public ParserBase {
  Parser(sh_state *rstate)
    : ParserBase(rstate->sym_name2mem)
    , rstate{rstate} {}

  enum class Cond_code {
    EQ, HS, GE, HI, GT, PZ, PL
  };

  enum class Lexeme {
    label,
    label_def,
    name,
    integer,
    hex,
    comma,
    at,
    plus,
    minus,
    left_paren,
    right_paren,
    string
  };

  struct Token {
    Lexeme kind;
    int pos;
    int size;
  };
  std::vector<Token> tokens;
  std::map<std::string, size_t, std::less<>> label_name2offset;
  Inst *function_data_mem = nullptr;

  Function *parse(std::string const& file_name);

  sh_state *rstate;
  Module *module;
  Function *src_func;

private:
  Function *func = nullptr;
  Basic_block *bb = nullptr;
  std::map<std::string_view, Basic_block *> label2bb;
  std::map<uint32_t, Inst *> id2inst;

  // Is this instruction in a delay slot? If so, branch_bb contains
  // the basic block where the branch instruction lives.
  bool delay_slot = false;
  Basic_block *branch_bb;

  void skip_space_and_comments();
  void lex_label_or_label_def();
  void lex_hex();
  void lex_integer();
  void lex_imm();
  void lex_string();
  void lex_name();

  uint64_t get_u64(const char *p);
  unsigned __int128 get_hex(const char *p);
  unsigned __int128 get_hex_or_integer(unsigned idx);
  Inst *get_imm(unsigned idx);
  bool is_reg(unsigned idx);
  bool is_freg(unsigned idx);
  bool is_dreg(unsigned idx);
  Inst *get_system_register_reg(unsigned idx);
  void get_fpul(unsigned idx);
  Inst *get_fpul_value(unsigned idx);
  Inst *get_reg(unsigned idx);
  Inst *get_reg_value(unsigned idx);
  Inst *get_reg_or_imm_value(unsigned idx);
  Inst *get_freg(unsigned idx);
  Inst *get_freg_value(unsigned idx);
  std::pair<Inst *, Inst *> get_dreg(unsigned idx);
  Inst *get_dreg_value(unsigned idx);
  void get_right_paren(unsigned idx);

  void write_reg(Inst *reg, Inst *value);
  void write_reg(std::pair<Inst *, Inst *> reg, Inst *value);
  std::string_view token_string(const Token& tok);
  std::string_view get_name(unsigned idx);
  Basic_block *get_bb(unsigned idx);
  Basic_block *get_bb_def(unsigned idx);
  void get_comma(unsigned idx);
  void get_at(unsigned idx);
  void get_end_of_line(unsigned idx);
  Inst *extract_vec_elem(Inst *inst, uint32_t elem_bitsize, uint32_t idx);
  void validate_fpscr_pr(bool expected_value);
  void validate_fpscr_sz(bool expected_value);
  Inst *load_value(Inst *ptr, uint64_t size);
  void store_value(Inst *ptr, Inst *value);
  void process_unary(Op op);
  void process_binary(Op op);
  void process_negc();
  void process_clrt();
  void process_sett();
  void process_dt();
  void process_trapa();
  void process_tst();
  void process_cmp_0(Cond_code cc);
  void process_cmp(Cond_code cc);
  void process_div0s();
  void process_shift(Op op, int shift);
  void process_shift_left1();
  void process_shift_right1(Op op);
  void process_rotl();
  void process_rotr();
  void process_rotcl();
  void process_rotcr();
  void process_shad();
  void process_shld();
  void process_lds();
  void process_lds_l();
  void process_sts();
  void process_sts_l();
  void process_mul_l();
  void process_mul_w(Op op);
  void process_dmul_l(Op op);
  void process_addc();
  void process_subc();
  void process_ext(Op op, int bitsize);
  void process_mov();
  void process_mov(uint64_t size);
  void process_mova();
  void process_movt();
  void process_fmov();
  void process_fmov_s();
  void process_fp_unary(Op op);
  void process_fp_binary(Op op);
  void process_fcmp_eq();
  void process_fcmp_gt();
  void process_float();
  void process_fcnvds();
  void process_fcnvsd();
  void process_fldi0();
  void process_fldi1();
  void process_ftrc();
  void process_flds();
  void process_fsts();
  void process_bf(bool delayed);
  void process_bt(bool delayed);
  void process_bra();
  void process_rts();
  void process_swap_b();
  void process_swap_w();
  void process_xtrct();
  Inst *process_address(unsigned& idx, uint64_t size);
  void process_store(uint64_t size);
  void process_load(uint64_t size);
  void process_fp_store();
  void process_fp_load();

  void parse_function();
  void lex_line();
  void parse_function_data();
};

void Parser::skip_space_and_comments()
{
  while (isspace(buf[pos]) && buf[pos] != '\n')
    pos++;
}

void Parser::lex_label_or_label_def()
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

void Parser::lex_hex()
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

void Parser::lex_integer()
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
  tokens.emplace_back(Lexeme::integer, start_pos, pos - start_pos);
}

void Parser::lex_imm()
{
  if (buf[pos] == '#')
    pos++;
  assert(isdigit(buf[pos]) || buf[pos] == '-');
  if (buf[pos] == '0' && (buf[pos + 1] == 'x' || buf[pos + 1] == 'X'))
    lex_hex();
  else
    lex_integer();
}

void Parser::lex_string()
{
  assert(buf[pos] == '"');
  int start_pos = pos;
  pos++;
  while (buf[pos] != '"')
    pos++;
  pos++;
  tokens.emplace_back(Lexeme::string, start_pos, pos - start_pos);
}

void Parser::lex_name()
{
  assert(isalpha(buf[pos])
	 || buf[pos] == '_'
	 || buf[pos] == '.'
	 || buf[pos] == '$');
  int start_pos = pos;
  pos++;
  while (isalnum(buf[pos])
	 || buf[pos] == '/'
	 || buf[pos] == '_'
	 || buf[pos] == '.'
	 || buf[pos] == '$')
    pos++;
  tokens.emplace_back(Lexeme::name, start_pos, pos - start_pos);
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

Inst *Parser::get_imm(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  return bb->value_inst(get_hex_or_integer(idx), 32);
}

bool Parser::is_reg(unsigned idx)
{
  if (tokens[idx].kind != Lexeme::name)
    return false;
  if (tokens[idx].size < 2 || tokens[idx].size > 3)
    return false;
  if (buf[tokens[idx].pos] != 'r')
    return false;
  if (!isdigit(buf[tokens[idx].pos + 1]))
    return false;
  if (tokens[idx].size == 3 && !isdigit(buf[tokens[idx].pos + 2]))
    return false;
  return true;
}

bool Parser::is_freg(unsigned idx)
{
  if (tokens[idx].kind != Lexeme::name)
    return false;
  if (tokens[idx].size < 3 || tokens[idx].size > 4)
    return false;
  if (buf[tokens[idx].pos] != 'f')
    return false;
  if (buf[tokens[idx].pos + 1] != 'r')
    return false;
  if (!isdigit(buf[tokens[idx].pos + 2]))
    return false;
  if (tokens[idx].size == 4 && !isdigit(buf[tokens[idx].pos + 3]))
    return false;
  return true;
}

bool Parser::is_dreg(unsigned idx)
{
  if (tokens[idx].kind != Lexeme::name)
    return false;
  if (tokens[idx].size < 3 || tokens[idx].size > 4)
    return false;
  if (buf[tokens[idx].pos] != 'd')
    return false;
  if (buf[tokens[idx].pos + 1] != 'r')
    return false;
  if (!isdigit(buf[tokens[idx].pos + 2]))
    return false;
  if (tokens[idx].size == 4 && !isdigit(buf[tokens[idx].pos + 3]))
    return false;
  return true;
}

Inst *Parser::get_system_register_reg(unsigned idx)
{
  std::string_view reg_name = get_name(idx);
  if (reg_name == "mach")
    return rstate->registers[ShRegIdx::mach];
  else if (reg_name == "macl")
    return rstate->registers[ShRegIdx::macl];
  else if (reg_name == "pr")
    return rstate->registers[ShRegIdx::pr];
  else if (reg_name == "fpscr")
    return rstate->registers[ShRegIdx::fpscr];
  else if (reg_name == "fpul")
    return rstate->registers[ShRegIdx::fpul];
  else
    throw Parse_error("unknown system register" + std::string(reg_name),
		      line_number);
}

void Parser::get_fpul(unsigned idx)
{
  std::string_view reg_name = get_name(idx);
  if (reg_name != "fpul")
    throw Parse_error("expected fpul instead of " + std::string(reg_name),
		      line_number);
}

Inst *Parser::get_fpul_value(unsigned idx)
{
  get_fpul(idx);
  return bb->build_inst(Op::READ, rstate->registers[ShRegIdx::fpul]);
}

Inst *Parser::get_reg(unsigned idx)
{
  if (!is_reg(idx))
    throw Parse_error("expected a register instead of "
		      + std::string(token_string(tokens[idx])), line_number);

  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  if (tokens[idx].size == 3)
    value = value * 10 + (buf[tokens[idx].pos + 2] - '0');
  return rstate->registers[ShRegIdx::r0 + value];
}

Inst *Parser::get_reg_value(unsigned idx)
{
  return bb->build_inst(Op::READ, get_reg(idx));
}

Inst *Parser::get_reg_or_imm_value(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  Inst *value;
  if (tokens[idx].kind == Lexeme::name)
    value = get_reg_value(idx);
  else
    value = get_imm(idx);
  return value;
}

Inst *Parser::get_freg(unsigned idx)
{
  if (!is_freg(idx))
    throw Parse_error("expected a fp register instead of "
		      + std::string(token_string(tokens[idx])), line_number);

  uint32_t value = buf[tokens[idx].pos + 2] - '0';
  if (tokens[idx].size == 4)
    value = value * 10 + (buf[tokens[idx].pos + 3] - '0');
  return rstate->registers[ShRegIdx::fr0 + value];
}

Inst *Parser::get_freg_value(unsigned idx)
{
  return bb->build_inst(Op::READ, get_freg(idx));
}

std::pair<Inst *, Inst *> Parser::get_dreg(unsigned idx)
{
  if (!is_dreg(idx))
    throw Parse_error("expected a fp register instead of "
		      + std::string(token_string(tokens[idx])), line_number);

  uint32_t value = buf[tokens[idx].pos + 2] - '0';
  if (tokens[idx].size == 4)
    value = value * 10 + (buf[tokens[idx].pos + 3] - '0');
  Inst *reg1 = rstate->registers[ShRegIdx::fr0 + value];
  Inst *reg2 = rstate->registers[ShRegIdx::fr0 + value + 1];
  return {reg1, reg2};
}

Inst *Parser::get_dreg_value(unsigned idx)
{
  auto [reg1, reg2] = get_dreg(idx);
  Inst *inst1 = bb->build_inst(Op::READ, reg1);
  Inst *inst2 = bb->build_inst(Op::READ, reg2);
  return bb->build_inst(Op::CONCAT, inst1, inst2);
}

void Parser::get_right_paren(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::right_paren)
    throw Parse_error("expected a ')' after "
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

void Parser::write_reg(std::pair<Inst *, Inst *> reg, Inst *value)
{
  auto [reg1, reg2] = reg;
  assert(reg1->op == Op::REGISTER && reg2->op == Op::REGISTER);
  assert(value->bitsize == 64);
  bb->build_inst(Op::WRITE, reg1, bb->build_inst(Op::EXTRACT, value, 63, 32));
  bb->build_inst(Op::WRITE, reg2, bb->build_inst(Op::EXTRACT, value, 31, 0));
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

void Parser::get_at(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::at)
    throw Parse_error("expected a '@' after "
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

Inst *Parser::extract_vec_elem(Inst *inst, uint32_t elem_bitsize, uint32_t idx)
{
  if (idx == 0 && inst->bitsize == elem_bitsize)
    return inst;
  assert(inst->bitsize % elem_bitsize == 0);
  Inst *high = bb->value_inst(idx * elem_bitsize + elem_bitsize - 1, 32);
  Inst *low = bb->value_inst(idx * elem_bitsize, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
}

void Parser::validate_fpscr_pr(bool expected_value)
{
  Inst *value = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::fpscr]);
  Inst *pr = bb->build_extract_bit(value, 19);
  if (expected_value)
    bb->build_inst(Op::UB, bb->build_inst(Op::NOT, pr));
  else
    bb->build_inst(Op::UB, pr);
}

void Parser::validate_fpscr_sz(bool expected_value)
{
  Inst *value = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::fpscr]);
  Inst *sz = bb->build_extract_bit(value, 20);
  if (expected_value)
    bb->build_inst(Op::UB, bb->build_inst(Op::NOT, sz));
  else
    bb->build_inst(Op::UB, sz);
}

Inst *Parser::load_value(Inst *ptr, uint64_t size)
{
  Inst *value = bb->build_inst(Op::LOAD, ptr);
  for (uint64_t i = 1; i < size; i++)
    {
      Inst *offset = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, offset);
      Inst *data_byte = bb->build_inst(Op::LOAD, addr);
      value = bb->build_inst(Op::CONCAT, data_byte, value);
    }
  return value;
}

void Parser::store_value(Inst *ptr, Inst *value)
{
  for (uint64_t i = 0; i < value->bitsize / 8; i++)
    {
      Inst *offset = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, offset);
      Inst *data_byte = extract_vec_elem(value, 8, i);
      bb->build_inst(Op::STORE, addr, data_byte);
    }
}

void Parser::process_unary(Op op)
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  write_reg(rn_reg, bb->build_inst(op, rm));
}

void Parser::process_binary(Op op)
{
  Inst *rm = get_reg_or_imm_value(1);
  get_comma(2);
  Inst *rn = get_reg_value(3);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  write_reg(rn_reg, bb->build_inst(op, rn, rm));
}

void Parser::process_negc()
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  Inst *zero = bb->value_inst(0, rm->bitsize);
  Inst *tmp = bb->build_inst(Op::NEG, rm);
  Inst *t = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::t]);
  Inst *t_ext = bb->build_inst(Op::ZEXT, t, tmp->bitsize);
  Inst *res = bb->build_inst(Op::SUB, tmp, t_ext);
  t = bb->build_inst(Op::NE, tmp, zero);
  t = bb->build_inst(Op::OR, t, bb->build_inst(Op::ULT, tmp, res));
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_clrt()
{
  Inst *t = bb->value_inst(0, 1);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
}

void Parser::process_sett()
{
  Inst *t = bb->value_inst(1, 1);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
}

void Parser::process_dt()
{
  Inst *rn_reg = get_reg(1);
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *m1 = bb->value_inst(-1, rn->bitsize);
  Inst *res = bb->build_inst(Op::ADD, rn, m1);
  Inst *zero = bb->value_inst(0, res->bitsize);
  Inst *t = bb->build_inst(Op::EQ, res, zero);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_trapa()
{
  bb->build_inst(Op::UB, bb->value_inst(1,1));
}

void Parser::process_tst()
{
  Inst *rm = get_reg_or_imm_value(1);
  get_comma(2);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  Inst *inst = bb->build_inst(Op::AND, rm, rn);
  Inst *zero = bb->value_inst(0, inst->bitsize);
  Inst *t = bb->build_inst(Op::EQ, inst, zero);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
}

void Parser::process_cmp_0(Cond_code cc)
{
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *t;
  Inst *zero = bb->value_inst(0, rn->bitsize);
  switch (cc)
    {
    case Cond_code::PL:
      t = bb->build_inst(Op::SLT, zero, rn);
      break;
    case Cond_code::PZ:
      t = bb->build_inst(Op::NOT, bb->build_inst(Op::SLT, rn, zero));
      break;
    default:
      throw Parse_error("unhandled condition code", line_number);
    }
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
}

void Parser::process_cmp(Cond_code cc)
{
  Inst *rm = get_reg_or_imm_value(1);
  get_comma(2);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  Inst *t;
  switch (cc)
    {
    case Cond_code::EQ:
      t = bb->build_inst(Op::EQ, rn, rm);
      break;
    case Cond_code::GT:
      t = bb->build_inst(Op::SLT, rm, rn);
      break;
    case Cond_code::GE:
      t = bb->build_inst(Op::NOT, bb->build_inst(Op::SLT, rn, rm));
      break;
    case Cond_code::HI:
      t = bb->build_inst(Op::ULT, rm, rn);
      break;
    case Cond_code::HS:
      t = bb->build_inst(Op::NOT, bb->build_inst(Op::ULT, rn, rm));
      break;
    default:
      throw Parse_error("unhandled condition code", line_number);
    }
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
}

void Parser::process_div0s()
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  Inst *zero = bb->value_inst(0, rm->bitsize);
  Inst *q = bb->build_inst(Op::SLT, rn, zero);
  Inst *m = bb->build_inst(Op::SLT, rm, zero);
  Inst *t = bb->build_inst(Op::XOR, q, m);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::q], q);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::m], m);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
}

void Parser::process_shift(Op op, int shift)
{
  Inst *rn_reg = get_reg(1);
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *res = bb->build_inst(op, rn, bb->value_inst(shift, rn->bitsize));
  write_reg(rn_reg, res);
}

void Parser::process_shift_left1()
{
  Inst *rn_reg = get_reg(1);
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *t = bb->build_extract_bit(rn, rn->bitsize - 1);
  Inst *shift = bb->value_inst(1, rn->bitsize);
  Inst *res = bb->build_inst(Op::SHL, rn, shift);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_shift_right1(Op op)
{
  Inst *rn_reg = get_reg(1);
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *t = bb->build_extract_bit(rn, 0);
  Inst *shift = bb->value_inst(1, rn->bitsize);
  Inst *res = bb->build_inst(op, rn, shift);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_rotl()
{
  Inst *rn_reg = get_reg(1);
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *concat = bb->build_inst(Op::CONCAT, rn, rn);
  Inst *shift = bb->value_inst(1, concat->bitsize);
  Inst *shifted = bb->build_inst(Op::SHL, concat, shift);
  Inst *high = bb->value_inst(2 * rn->bitsize - 1, 32);
  Inst *low = bb->value_inst(rn->bitsize, 32);
  Inst *res = bb->build_inst(Op::EXTRACT, shifted, high, low);
  Inst *t = bb->build_extract_bit(rn, rn->bitsize - 1);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_rotr()
{
  Inst *rn_reg = get_reg(1);
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *concat = bb->build_inst(Op::CONCAT, rn, rn);
  Inst *shift = bb->value_inst(1, concat->bitsize);
  Inst *shifted = bb->build_inst(Op::LSHR, concat, shift);
  Inst *res = bb->build_trunc(shifted, rn->bitsize);
  Inst *t = bb->build_extract_bit(rn, 0);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_rotcl()
{
  Inst *rn_reg = get_reg(1);
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *t = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::t]);
  Inst *res = bb->build_inst(Op::EXTRACT, rn, 30, 0);
  res = bb->build_inst(Op::CONCAT, res, t);
  t = bb->build_extract_bit(rn, rn->bitsize - 1);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_rotcr()
{
  Inst *rn_reg = get_reg(1);
  Inst *rn = get_reg_value(1);
  get_end_of_line(2);

  Inst *t = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::t]);
  Inst *res = bb->build_inst(Op::EXTRACT, rn, 31, 1);
  res = bb->build_inst(Op::CONCAT, t, res);
  t = bb->build_extract_bit(rn, 0);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_shad()
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  Inst *zero = bb->value_inst(0, rm->bitsize);
  Inst *is_neg = bb->build_inst(Op::SLT, rm, zero);
  Inst *shift_left = bb->build_trunc(rm, 5);
  Inst *shift_right = bb->build_inst(Op::NEG, shift_left);
  shift_left = bb->build_inst(Op::ZEXT, shift_left, rn->bitsize);
  shift_right = bb->build_inst(Op::ZEXT, shift_right, rn->bitsize);
  Inst *is_shift_right_zero = bb->build_inst(Op::EQ, shift_right, zero);
  Inst *res_left = bb->build_inst(Op::SHL, rn, shift_left);
  Inst *res_right = bb->build_inst(Op::ASHR, rn, shift_right);
  Inst *tmp = bb->value_inst(rn->bitsize, rn->bitsize);
  tmp = bb->build_inst(Op::ASHR, rn, tmp);
  res_right = bb->build_inst(Op::ITE, is_shift_right_zero, tmp, res_right);
  Inst *res = bb->build_inst(Op::ITE, is_neg, res_right, res_left);
  write_reg(rn_reg, res);
}

void Parser::process_shld()
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  Inst *zero = bb->value_inst(0, rm->bitsize);
  Inst *is_neg = bb->build_inst(Op::SLT, rm, zero);
  Inst *shift_left = bb->build_trunc(rm, 5);
  Inst *shift_right = bb->build_inst(Op::NEG, shift_left);
  shift_left = bb->build_inst(Op::ZEXT, shift_left, rn->bitsize);
  shift_right = bb->build_inst(Op::ZEXT, shift_right, rn->bitsize);
  Inst *is_shift_right_zero = bb->build_inst(Op::EQ, shift_right, zero);
  Inst *res_left = bb->build_inst(Op::SHL, rn, shift_left);
  Inst *res_right = bb->build_inst(Op::LSHR, rn, shift_right);
  res_right = bb->build_inst(Op::ITE, is_shift_right_zero, zero, res_right);
  Inst *res = bb->build_inst(Op::ITE, is_neg, res_right, res_left);
  write_reg(rn_reg, res);
}

void Parser::process_lds()
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *sr_reg = get_system_register_reg(3);
  get_end_of_line(4);

  bb->build_inst(Op::WRITE, sr_reg, rm);
}

void Parser::process_lds_l()
{
  unsigned idx = 1;
  Inst *ptr = process_address(idx, 4);
  get_comma(idx++);
  Inst *sr_reg = get_system_register_reg(idx++);
  get_end_of_line(idx);

  Inst *value = load_value(ptr, 4);
  bb->build_inst(Op::WRITE, sr_reg, value);
}

void Parser::process_sts()
{
  Inst *sr_reg = get_system_register_reg(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  Inst *value = bb->build_inst(Op::READ, sr_reg);
  write_reg(rn_reg, value);
}

void Parser::process_sts_l()
{
  Inst *sr_reg = get_system_register_reg(1);
  get_comma(2);
  unsigned idx = 3;
  Inst *ptr = process_address(idx, 4);
  get_end_of_line(idx);

  Inst *value = bb->build_inst(Op::READ, sr_reg);
  store_value(ptr, value);
}

void Parser::process_mul_l()
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  Inst *res = bb->build_inst(Op::MUL, rn, rm);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::macl], res);
}

void Parser::process_mul_w(Op op)
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  rm = bb->build_inst(op, bb->build_trunc(rm, 16), rm->bitsize);
  rn = bb->build_inst(op, bb->build_trunc(rn, 16), rn->bitsize);
  Inst *res = bb->build_inst(Op::MUL, rn, rm);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::macl], res);
}

void Parser::process_dmul_l(Op op)
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  rm = bb->build_inst(op, rm, 64);
  rn = bb->build_inst(op, rn, 64);
  Inst *res = bb->build_inst(Op::MUL, rn, rm);
  Inst *mach = bb->build_inst(Op::EXTRACT, res, 63, 32);
  Inst *macl = bb->build_trunc(res, 32);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::mach], mach);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::macl], macl);
}

void Parser::process_addc()
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  Inst *tmp = bb->build_inst(Op::ADD, rn, rm);
  Inst *t = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::t]);
  Inst *t_ext = bb->build_inst(Op::ZEXT, t, tmp->bitsize);
  Inst *res = bb->build_inst(Op::ADD, tmp, t_ext);
  t = bb->build_inst(Op::ULT, tmp, rn);
  t = bb->build_inst(Op::OR, t, bb->build_inst(Op::ULT, res, tmp));
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_subc()
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  Inst *rn = get_reg_value(3);
  get_end_of_line(4);

  Inst *tmp = bb->build_inst(Op::SUB, rn, rm);
  Inst *t = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::t]);
  Inst *t_ext = bb->build_inst(Op::ZEXT, t, tmp->bitsize);
  Inst *res = bb->build_inst(Op::SUB, tmp, t_ext);
  t = bb->build_inst(Op::ULT, rn, tmp);
  t = bb->build_inst(Op::OR, t, bb->build_inst(Op::ULT, tmp, res));
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
  write_reg(rn_reg, res);
}

void Parser::process_ext(Op op, int bitsize)
{
  Inst *rm = get_reg_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  Inst *res = bb->build_inst(op, bb->build_trunc(rm, bitsize), rm->bitsize);
  write_reg(rn_reg, res);
}

void Parser::process_mov()
{
 Inst *rm = get_reg_or_imm_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  write_reg(rn_reg, rm);
}

// Used for moves that may access memory (mov.l, mov.w, mov.b).
void Parser::process_mov(uint64_t size)
{
  if (is_reg(1))
    process_store(size);
  else
    process_load(size);
}

void Parser::process_mova()
{
  get_bb(1);  // Ensure this is a label.
  unsigned idx = 1;
  Inst *ptr = process_address(idx, 0);
  get_comma(idx++);
  Inst *reg = get_reg(idx++);
  get_end_of_line(idx++);

  write_reg(reg, ptr);
}

void Parser::process_movt()
{
  Inst *rn_reg = get_reg(1);
  get_end_of_line(2);

  Inst *t = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::t]);
  write_reg(rn_reg, t);
}

void Parser::process_fmov()
{
  Inst *rm = get_freg_value(1);
  get_comma(2);
  Inst *rn_reg = get_freg(3);
  get_end_of_line(4);
  validate_fpscr_sz(false);

  write_reg(rn_reg, rm);
}

void Parser::process_fmov_s()
{
  if (is_freg(1) || is_dreg(1))
    process_fp_store();
  else
    process_fp_load();
}

void Parser::process_fp_unary(Op op)
{
  if (is_dreg(1))
    {
      Inst *rn = get_dreg_value(1);
      std::pair<Inst *, Inst *> rn_reg = get_dreg(1);
      get_end_of_line(2);

      write_reg(rn_reg, bb->build_inst(op, rn));
    }
  else
    {
      Inst *rn = get_freg_value(1);
      Inst *rn_reg = get_freg(1);
      get_end_of_line(2);

      write_reg(rn_reg, bb->build_inst(op, rn));
    }
}

void Parser::process_fp_binary(Op op)
{
  if (is_dreg(1))
    {
      Inst *rm = get_dreg_value(1);
      get_comma(2);
      Inst *rn = get_dreg_value(3);
      std::pair<Inst *, Inst *> rn_reg = get_dreg(3);
      get_end_of_line(4);
      validate_fpscr_pr(true);

      write_reg(rn_reg, bb->build_inst(op, rn, rm));
    }
  else
    {
      Inst *rm = get_freg_value(1);
      get_comma(2);
      Inst *rn = get_freg_value(3);
      Inst *rn_reg = get_freg(3);
      get_end_of_line(4);
      validate_fpscr_pr(false);

      write_reg(rn_reg, bb->build_inst(op, rn, rm));
    }
}

void Parser::process_fcmp_eq()
{
  Inst *rm;
  Inst *rn;
  if (is_dreg(1))
    {
      rm = get_dreg_value(1);
      get_comma(2);
      rn = get_dreg_value(3);
      get_end_of_line(4);
      validate_fpscr_pr(true);
    }
  else
    {
      rm = get_freg_value(1);
      get_comma(2);
      rn = get_freg_value(3);
      get_end_of_line(4);
      validate_fpscr_pr(false);
    }

  Inst *t = bb->build_inst(Op::FEQ, rn, rm);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
}

void Parser::process_fcmp_gt()
{
  Inst *rm;
  Inst *rn;
  if (is_dreg(1))
    {
      rm = get_dreg_value(1);
      get_comma(2);
      rn = get_dreg_value(3);
      get_end_of_line(4);
      validate_fpscr_pr(true);
    }
  else
    {
      rm = get_freg_value(1);
      get_comma(2);
      rn = get_freg_value(3);
      get_end_of_line(4);
      validate_fpscr_pr(false);
    }

  Inst *t = bb->build_inst(Op::FLT, rm, rn);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::t], t);
}

void Parser::process_float()
{
  Inst *fpul = get_fpul_value(1);
  get_comma(2);
  if (is_dreg(3))
    {
      std::pair<Inst *, Inst *> rn_reg = get_dreg(3);
      write_reg(rn_reg, bb->build_inst(Op::S2F, fpul, 64));
      validate_fpscr_pr(true);
    }
  else
    {
      Inst *rn_reg = get_freg(3);
      write_reg(rn_reg, bb->build_inst(Op::S2F, fpul, 32));
      validate_fpscr_pr(false);
    }
  get_end_of_line(4);
}

void Parser::process_fcnvds()
{
  Inst *rm = get_dreg_value(1);
  get_comma(2);
  get_fpul(3);
  get_end_of_line(4);

  validate_fpscr_pr(true);
  Inst *fpul = bb->build_inst(Op::FCHPREC, rm, 32);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::fpul], fpul);
}

void Parser::process_fcnvsd()
{
  Inst *fpul = get_fpul_value(1);
  get_comma(2);
  std::pair<Inst *, Inst *> rn_reg = get_dreg(3);
  get_end_of_line(4);

  validate_fpscr_pr(true);
  Inst *rn = bb->build_inst(Op::FCHPREC, fpul, 64);
  write_reg(rn_reg, rn);
}

void Parser::process_fldi0()
{
  Inst *rn_reg = get_freg(1);
  get_end_of_line(2);
  validate_fpscr_pr(false);

  write_reg(rn_reg, bb->value_inst(0, 32));
}

void Parser::process_fldi1()
{
  Inst *rn_reg = get_freg(1);
  get_end_of_line(2);
  validate_fpscr_pr(false);

  write_reg(rn_reg, bb->value_inst(0x3f800000, 32));
}

void Parser::process_ftrc()
{
  Inst *rm;
  if (is_dreg(1))
    {
      rm = get_dreg_value(1);
      validate_fpscr_pr(true);
    }
  else
    {
      rm = get_freg_value(1);
      validate_fpscr_pr(false);
    }
  get_comma(2);
  get_end_of_line(4);

  // TODO: Handle out of bound values.
  Inst *fpul = bb->build_inst(Op::F2S, rm, 32);
  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::fpul], fpul);
}

void Parser::process_flds()
{
  Inst *rm = get_freg_value(1);
  get_comma(2);
  get_fpul(3);
  get_end_of_line(4);

  bb->build_inst(Op::WRITE, rstate->registers[ShRegIdx::fpul], rm);
}

void Parser::process_fsts()
{
  Inst *fpul = get_fpul_value(1);
  get_comma(2);
  Inst *rn_reg = get_freg(3);
  get_end_of_line(4);

  write_reg(rn_reg, fpul);
}

void Parser::process_bf(bool delayed)
{
  Basic_block *true_bb = get_bb(1);
  get_end_of_line(2);

  if (delayed)
    {
      delay_slot = true;
      branch_bb = bb;
    }

  Inst *t = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::t]);
  Basic_block *false_bb = func->build_bb();
  bb->build_br_inst(t, false_bb, true_bb);
  bb = false_bb;
}

void Parser::process_bt(bool delayed)
{
  Basic_block *true_bb = get_bb(1);
  get_end_of_line(2);

  if (delayed)
    {
      delay_slot = true;
      branch_bb = bb;
    }

  Inst *t = bb->build_inst(Op::READ, rstate->registers[ShRegIdx::t]);
  Basic_block *false_bb = func->build_bb();
  bb->build_br_inst(t, true_bb, false_bb);
  bb = false_bb;
}

void Parser::process_bra()
{
  Basic_block *dest_bb = get_bb(1);
  get_end_of_line(2);

  delay_slot = true;
  branch_bb = bb;

  // We must emit this as a conditional branch with a constant condition
  // because of how our delay slot handling works.
  Basic_block *dummy_bb = func->build_bb();
  Inst *b1 = bb->value_inst(1, 1);
  bb->build_br_inst(b1, dest_bb, dummy_bb);
  bb = dummy_bb;
}

void Parser::process_rts()
{
  get_end_of_line(1);

  delay_slot = true;
  branch_bb = bb;

  bb->build_br_inst(rstate->exit_bb);
  bb = func->build_bb();
}

void Parser::process_swap_b()
{
  Inst *rm = get_reg_or_imm_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  Inst *high = bb->build_inst(Op::EXTRACT, rm, 31, 16);
  Inst *byte0 = bb->build_inst(Op::EXTRACT, rm, 15, 8);
  Inst *byte1 = bb->build_trunc(rm, 8);
  Inst *res = bb->build_inst(Op::CONCAT, byte1, byte0);
  res = bb->build_inst(Op::CONCAT, high, res);
  write_reg(rn_reg, res);
}

void Parser::process_swap_w()
{
  Inst *rm = get_reg_or_imm_value(1);
  get_comma(2);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  Inst *high = bb->build_trunc(rm, 16);
  Inst *low = bb->build_inst(Op::EXTRACT, rm, 31, 16);
  write_reg(rn_reg, bb->build_inst(Op::CONCAT, high, low));
}

void Parser::process_xtrct()
{
  Inst *rm = get_reg_or_imm_value(1);
  get_comma(2);
  Inst *rn = get_reg_value(3);
  Inst *rn_reg = get_reg(3);
  get_end_of_line(4);

  Inst *high = bb->build_trunc(rm, 16);
  Inst *low = bb->build_inst(Op::EXTRACT, rn, 31, 16);
  write_reg(rn_reg, bb->build_inst(Op::CONCAT, high, low));
}

Inst *Parser::process_address(unsigned& idx, uint64_t size)
{
  if (tokens.size() > idx && tokens[idx].kind == Lexeme::label)
    {
      std::string_view label_name = token_string(tokens[idx++]);
      auto it = label_name2offset.find(label_name);
      if (it == label_name2offset.end())
	throw Parse_error("process_address: unkown label: "
			  + std::string(label_name), line_number);
      Inst *offset = bb->value_inst(it->second, 32);
      return bb->build_inst(Op::ADD, function_data_mem, offset);
    }

  get_at(idx++);
  Inst *ptr;
  if (tokens.size() > idx && tokens[idx].kind == Lexeme::left_paren)
    {
      idx++;
      ptr = get_reg_or_imm_value(idx++);
      if (tokens.size() > idx && tokens[idx].kind == Lexeme::comma)
	{
	  idx++;
	  Inst *offset = get_reg_value(idx++);
	  ptr = bb->build_inst(Op::ADD, ptr, offset);
	}
      get_right_paren(idx++);
    }
  else if (tokens.size() > idx && tokens[idx].kind == Lexeme::minus)
    {
      idx++;
      Inst *reg = get_reg(idx);
      ptr = get_reg_value(idx++);
      Inst *offset = bb->value_inst(-size, ptr->bitsize);
      ptr = bb->build_inst(Op::ADD, ptr, offset);
      write_reg(reg, ptr);
      // TODO: Verify the register is written at the correct time. I.e. we
      // should handle
      //   mov.l   @-r5,r5
      // identically to the hardware.
    }
  else
    {
      Inst *reg = get_reg(idx);
      ptr = get_reg_value(idx++);
      if (tokens.size() > idx && tokens[idx].kind == Lexeme::plus)
	{
	  idx++;

	  // This should be done after instruction execution, so I guess
	  // we give an incorrect result for
	  //   mov.l   @r5+,r5
	  // TODO: Move this to be done after the instruction.
	  Inst *offset = bb->value_inst(size, ptr->bitsize);
	  Inst *new_value = bb->build_inst(Op::ADD, ptr, offset);
	  write_reg(reg, new_value);
	}
    }

  return ptr;
}

void Parser::process_store(uint64_t size)
{
  Inst *value = get_reg_value(1);
  get_comma(2);
  unsigned idx = 3;
  Inst *ptr = process_address(idx, size);
  get_end_of_line(idx++);

  if (size * 8 < value->bitsize)
    value = bb->build_trunc(value, size * 8);
  store_value(ptr, value);
}

void Parser::process_load(uint64_t size)
{
  unsigned idx = 1;
  Inst *ptr = process_address(idx, size);
  get_comma(idx++);
  Inst *dest_reg = get_reg(idx++);
  get_end_of_line(idx++);

  Inst *value = load_value(ptr, size);
  if (value->bitsize < 32)
    value = bb->build_inst(Op::SEXT, value, 32);
  write_reg(dest_reg, value);
}

void Parser::process_fp_store()
{
  Inst *value;
  if (is_dreg(1))
    {
      value = get_dreg_value(1);
      validate_fpscr_sz(true);
    }
  else
    {
      value = get_freg_value(1);
      validate_fpscr_sz(false);
    }

  get_comma(2);
  unsigned idx = 3;
  Inst *ptr = process_address(idx, 4);
  get_end_of_line(idx++);

  store_value(ptr, value);
}

void Parser::process_fp_load()
{
  if (is_dreg(tokens.size() - 1))
    {
      unsigned idx = 1;
      Inst *ptr = process_address(idx, 8);
      get_comma(idx++);
      std::pair<Inst *, Inst *> dest_reg = get_dreg(idx++);
      get_end_of_line(idx++);
      validate_fpscr_sz(true);

      Inst *value = load_value(ptr, 8);
      write_reg(dest_reg, value);
    }
  else
    {
      unsigned idx = 1;
      Inst *ptr = process_address(idx, 4);
      get_comma(idx++);
      Inst *dest_reg = get_freg(idx++);
      get_end_of_line(idx++);
      validate_fpscr_sz(false);

      Inst *value = load_value(ptr, 4);
      write_reg(dest_reg, value);
    }
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

  bool processing_delay_slot = delay_slot;
  delay_slot = false;

  if (tokens[0].kind != Lexeme::name)
    throw Parse_error("syntax error: " + std::string(token_string(tokens[0])),
		      line_number);

  std::string_view name = get_name(0);
  if (name.starts_with(".cfi"))
    ;
  else if (name.starts_with(".LF"))
    ;
  else if (name.starts_with(".align"))
    ;
  else if (name.starts_with(".long")
	   || name.starts_with(".short"))
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
  else if (name == "add")
    process_binary(Op::ADD);
  else if (name == "addc")
    process_addc();
  else if (name == "and")
    process_binary(Op::AND);
  else if (name == "bf")
    process_bf(false);
  else if (name == "bf/s")
    process_bf(true);
  else if (name == "bra")
    process_bra();
  else if (name == "bt")
    process_bt(false);
  else if (name == "bt/s")
    process_bt(true);
  else if (name == "cmp/pl")
    process_cmp_0(Cond_code::PL);
  else if (name == "cmp/pz")
    process_cmp_0(Cond_code::PZ);
  else if (name == "cmp/eq")
    process_cmp(Cond_code::EQ);
  else if (name == "cmp/gt")
    process_cmp(Cond_code::GT);
  else if (name == "cmp/ge")
    process_cmp(Cond_code::GE);
  else if (name == "cmp/hi")
    process_cmp(Cond_code::HI);
  else if (name == "cmp/hs")
    process_cmp(Cond_code::HS);
  else if (name == "clrt")
    process_clrt();
  else if (name == "div0s")
    process_div0s();
  else if (name == "dt")
    process_dt();
  else if (name == "dmuls.l")
    process_dmul_l(Op::SEXT);
  else if (name == "dmulu.l")
    process_dmul_l(Op::ZEXT);
  else if (name == "exts.b")
    process_ext(Op::SEXT, 8);
  else if (name == "exts.w")
    process_ext(Op::SEXT, 16);
  else if (name == "extu.b")
    process_ext(Op::ZEXT, 8);
  else if (name == "extu.w")
    process_ext(Op::ZEXT, 16);
  else if (name == "fabs")
    process_fp_unary(Op::FABS);
  else if (name == "fadd")
    process_fp_binary(Op::FADD);
  else if (name == "fcmp/eq")
    process_fcmp_eq();
  else if (name == "fcmp/gt")
    process_fcmp_gt();
  else if (name == "fcnvds")
    process_fcnvds();
  else if (name == "fcnvsd")
    process_fcnvsd();
  else if (name == "fdiv")
    process_fp_binary(Op::FDIV);
  else if (name == "fldi0")
    process_fldi0();
  else if (name == "fldi1")
    process_fldi1();
  else if (name == "flds")
    process_flds();
  else if (name == "float")
    process_float();
  else if (name == "fmov")
    process_fmov();
  else if (name == "fmov.s")
    process_fmov_s();
  else if (name == "fmul")
    process_fp_binary(Op::FMUL);
  else if (name == "fneg")
    process_fp_unary(Op::FNEG);
  else if (name == "fsts")
    process_fsts();
  else if (name == "fsub")
    process_fp_binary(Op::FSUB);
  else if (name == "ftrc")
    process_ftrc();
  else if (name == "lds")
    process_lds();
  else if (name == "lds.l")
    process_lds_l();
  else if (name == "mov")
    process_mov();
  else if (name == "mov.b")
    process_mov(1);
  else if (name == "mov.w")
    process_mov(2);
  else if (name == "mov.l")
    process_mov(4);
  else if (name == "mova")
    process_mova();
  else if (name == "movt")
    process_movt();
  else if (name == "mul.l")
    process_mul_l();
  else if (name == "muls.w")
    process_mul_w(Op::SEXT);
  else if (name == "mulu.w")
    process_mul_w(Op::ZEXT);
  else if (name == "neg")
    process_unary(Op::NEG);
  else if (name == "negc")
    process_negc();
  else if (name == "not")
    process_unary(Op::NOT);
  else if (name == "nop")
    get_end_of_line(1);
  else if (name == "or")
    process_binary(Op::OR);
  else if (name == "rotcl")
    process_rotcl();
  else if (name == "rotcr")
    process_rotcr();
  else if (name == "rotl")
    process_rotl();
  else if (name == "rotr")
    process_rotr();
  else if (name == "rts")
    process_rts();
  else if (name == "sett")
    process_sett();
  else if (name == "shad")
    process_shad();
  else if (name == "shar")
    process_shift_right1(Op::ASHR);
  else if (name == "shld")
    process_shld();
  else if (name == "shll")
    process_shift_left1();
  else if (name == "shll2")
    process_shift(Op::SHL, 2);
  else if (name == "shll8")
    process_shift(Op::SHL, 8);
  else if (name == "shll16")
    process_shift(Op::SHL, 16);
  else if (name == "shlr")
    process_shift_right1(Op::LSHR);
  else if (name == "shlr2")
    process_shift(Op::LSHR, 2);
  else if (name == "shlr8")
    process_shift(Op::LSHR, 8);
  else if (name == "shlr16")
    process_shift(Op::LSHR, 16);
  else if (name == "sts")
    process_sts();
  else if (name == "sts.l")
    process_sts_l();
  else if (name == "sub")
    process_binary(Op::SUB);
  else if (name == "subc")
    process_subc();
  else if (name == "swap.b")
    process_swap_b();
  else if (name == "swap.w")
    process_swap_w();
  else if (name == "trapa")
    process_trapa();
  else if (name == "tst")
    process_tst();
  else if (name == "xor")
    process_binary(Op::XOR);
  else if (name == "xtrct")
    process_xtrct();
  else
    throw Parse_error("unhandled instruction: "s + std::string(name),
		      line_number);

  if (processing_delay_slot)
    {
      while (bb->first_inst)
	{
	  bb->first_inst->move_before(branch_bb->last_inst);
	}
    }
}

void Parser::lex_line()
{
  tokens.clear();
  while (buf[pos] != '\n' && buf[pos] != ';')
    {
      skip_space_and_comments();
      if (buf[pos] == '\n')
	break;
      if (buf[pos] == '-'
	  && !tokens.empty()
	  && tokens.back().kind != Lexeme::comma
	  && tokens.back().kind != Lexeme::plus)
	{
	  tokens.emplace_back(Lexeme::minus, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '#')
	lex_imm();
      else if (buf[pos] == '"')
	lex_string();
      else if (isdigit(buf[pos]) || buf[pos] == '-')
	lex_imm();
      else if (buf[pos] == '.' && buf[pos + 1] == 'L')
	lex_label_or_label_def();
      else if (buf[pos] == '.' && tokens.empty())
	{
	  lex_name();
	  if (get_name(0) == ".section")
	    {
	      skip_space_and_comments();
	      lex_name();
	    }
	}
      else if (isalpha(buf[pos]) || buf[pos] == '_' || buf[pos] == '.')
	lex_name();
      else if (buf[pos] == ',')
	{
	  tokens.emplace_back(Lexeme::comma, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '@')
	{
	  tokens.emplace_back(Lexeme::at, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '+')
	{
	  tokens.emplace_back(Lexeme::plus, pos, 1);
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

void Parser::parse_function_data()
{
  size_t orig_pos = pos;

  std::vector<Inst *> data;
  for (;;)
    {
      if (pos == buf.size() - 1)
	break;

      if (buf[pos] == '.')
	{
	  std::optional<std::string_view> label = parse_label_def();
	  if (label)
	    {
	      size_t offset = data.size();
	      if (parse_data(rstate->entry_bb, data))
		label_name2offset.emplace(*label, offset);
	    }
	}
      else if (buf[pos] == '_' || isalpha(buf[pos]))
	{
	  // This is a global label, so we are done with the function.
	  break;
	}
      else
	skip_line();
    }
  pos = orig_pos;

  Basic_block *entry_bb = rstate->entry_bb;
  Inst *id = entry_bb->value_inst(rstate->next_local_id++, module->ptr_id_bits);
  Inst *mem_size = entry_bb->value_inst(data.size(), module->ptr_offset_bits);
  Inst *flags = entry_bb->value_inst(MEM_CONST, 32);
  Inst *mem = entry_bb->build_inst(Op::MEMORY, id, mem_size, flags);
  for (size_t i = 0; i < data.size(); i++)
    {
      Inst *off = entry_bb->value_inst(i, mem->bitsize);
      Inst *ptr = entry_bb->build_inst(Op::ADD, mem, off);
      entry_bb->build_inst(Op::STORE, ptr, data[i]);
    }
  function_data_mem = mem;
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
  src_func = module->functions[0];

  parse_rodata(rstate->entry_bb);

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

	    for (const auto& [name, mem] : rstate->sym_name2mem)
	      {
		if (!sym_name2data.contains(name))
		  continue;
		std::vector<Inst *>& data = sym_name2data.at(name);
		for (size_t i = 0; i < data.size(); i++)
		  {
		    Inst *off = entry_bb->value_inst(i, mem->bitsize);
		    Inst *ptr = entry_bb->build_inst(Op::ADD, mem, off);
		    entry_bb->build_inst(Op::STORE, ptr, data[i]);
		  }
	      }

	    parse_function_data();

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

Function *parse_sh(std::string const& file_name, sh_state *state)
{
  Parser p(state);
  Function *func = p.parse(file_name);
  reverse_post_order(func);
  return func;
}

} // end namespace smtgcc
