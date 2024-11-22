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

enum class Cond_code {
  EQ, NE, SLT, ULT, SLE, ULE, SGT, UGT, SGE, UGE,
  FEQ, FNE, FLT, FLE, FGT, FGE
};

struct Parser {
  Parser(riscv_state *rstate) : rstate{rstate} {}

  enum class Lexeme {
    label,
    label_def,
    name,
    integer,
    hex,
    comma,
    lo,
    hi,
    left_bracket,
    right_bracket,
    left_paren,
    right_paren
  };

  enum class LStype {
    signed_ls,
    unsigned_ls,
    float_ls
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

  riscv_state *rstate;
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
  void lex_hilo(void);

  std::string_view token_string(const Token& tok);

  uint32_t get_u32(const char *p);
  unsigned __int128 get_hex(const char *p);

  unsigned __int128 get_hex_or_integer(unsigned idx);
  bool is_reg_x0(unsigned idx);
  Inst *get_reg(unsigned idx);
  Inst *get_freg(unsigned idx);
  Inst *get_vreg(unsigned idx);
  Inst *get_hilo_addr(const Token& tok);
  Inst *get_hi(unsigned idx);
  Inst *get_lo(unsigned idx);
  Inst *get_imm(unsigned idx);
  Inst *get_reg_value(unsigned idx);
  Inst *get_freg_value(unsigned idx);
  Inst *get_vreg_value(unsigned idx);
  Basic_block *get_bb(unsigned idx);
  Basic_block *get_bb_def(unsigned idx);
  std::string_view get_name(unsigned idx);
  void get_comma(unsigned idx);
  void get_left_paren(unsigned idx);
  void get_right_paren(unsigned idx);
  void get_end_of_line(unsigned idx);
  void process_cond_branch(Op op, bool swap = false);
  Inst *gen_ffs(Inst *arg);
  Inst *gen_parity(Inst *arg);
  Inst *gen_popcount(Inst *arg);
  Inst *gen_sdiv(Inst *arg1, Inst *arg2);
  Inst *gen_udiv(Inst *arg1, Inst *arg2);
  Inst *read_arg(uint32_t reg, uint32_t bitsize);
  void write_retval(Inst *retval);
  void process_call();
  void process_tail();
  void store_ub_check(Inst *ptr, uint64_t size);
  void load_ub_check(Inst *ptr, uint64_t size);
  void process_load(int size, LStype lstype = LStype::signed_ls);
  void process_store(int size, LStype lstype = LStype::signed_ls);
  void process_funary(std::string_view name, Op op);
  void process_fbinary(std::string_view name, Op op);
  void process_fcmp(std::string_view name, Op op, bool swap = false);
  void process_fcvt_i2f(uint32_t src_bitsize, uint32_t dest_bitsize,
			bool is_unsigned);
  void process_fcvt_f2i(uint32_t src_bitsize, uint32_t dest_bitsize,
			bool is_unsigned);
  void process_fcvt_f2f(uint32_t src_bitsize, uint32_t dest_bitsize);
  void process_fmin_fmax(uint32_t bitsize, bool is_min);
  void process_iunary(std::string_view name, Op op);
  void process_ibinary(std::string_view name, Op op);
  void process_icmp(std::string_view name, Op op, bool swap = false);
  void process_ishift(std::string_view name, Op op);
  void process_zba_sh_add(uint64_t shift_val, bool truncate_arg1);
  Inst *extract_vec_elem(Inst *inst, uint32_t elem_bitsize, uint32_t idx);
  Inst *change_prec(Inst *inst, uint32_t bitsize);
  void process_vsetvli(bool arg1_is_imm);
  void process_vle(uint32_t elem_bitsize);
  void process_vse(uint32_t elem_bitsize);
  Inst *gen_vec_unary(Op op, Inst *orig, Inst *arg1, Inst *mask,
		      uint32_t elem_bitsize);
  void process_vec_unary(Op op);
  void process_vec_unary_vi(Op op);
  void process_vec_unary_vx(Op op);
  Inst *gen_vec_binary(Op op, Inst *orig, Inst *arg1, Inst *arg2, Inst *mask,
		       uint32_t elem_bitsize);
  void process_vec_binary(Op op);
  void process_vec_binary_vi(Op op);
  void process_vec_binary_vx(Op op);
  Inst *gen_vec_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Inst *orig,
		       Inst *arg1, Inst *arg2, Inst *mask,
		       uint32_t elem_bitsize);
  void process_vec_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  void process_vec_binary_vi(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  void process_vec_binary_vx(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  void process_vec_mask_unary(Op op);
  void process_vec_mask_set(bool value);
  void process_vec_mask_binary(Op op);
  void process_vec_mask_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  Inst *gen_vec_reduc(Op op, Inst *orig, Inst *arg1, Inst *arg2, Inst *mask,
		      uint32_t elem_bitsize);
  void process_vec_reduc(Op op);
  Inst *gen_vec_reduc(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*), Inst *orig,
		      Inst *arg1, Inst *arg2, Inst *mask,
		      uint32_t elem_bitsize);
  void process_vec_reduc(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*));
  Inst *gen_vmerge(Inst *orig, Inst *arg1, Inst *arg2, Inst *arg3,
		   uint32_t elem_bitsize);
  void process_vmerge();
  void process_vmerge_vi();
  void process_vmerge_vx();
  Inst *gen_vec_cmp(Cond_code ccode, Inst *orig, Inst *arg1, Inst *arg2,
		    uint32_t elem_bitsize);
  void process_vec_cmp(Cond_code ccode);
  void process_vec_cmp_vi(Cond_code ccode);
  void process_vec_cmp_vx(Cond_code ccode);
  Inst *gen_vid(Inst *orig, uint32_t elem_bitsize);
  void process_vid();
  void process_vmv_xs();
  Inst *gen_vmv_sx(Inst *orig, Inst *arg1, uint32_t elem_bitsize);
  void process_vmv_sx();

  void parse_function();

  void skip_space_and_comments();
};

void Parser::skip_space_and_comments()
{
  while (isspace(buf[pos]))
    pos++;
  if (buf[pos] == '#')
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

void Parser::lex_hilo(void)
{
  int start_pos = pos;
  bool is_lo = (buf[pos] == '%' && buf[pos + 1] == 'l' && buf[pos + 2] == 'o');
  Lexeme op = is_lo ? Lexeme::lo : Lexeme::hi;
  if (buf[pos + 3] != '(')
    throw Parse_error("expected '('", line_number);
  pos += 4;
  if (buf[pos] == ')')
    throw Parse_error("expected a name after '('", line_number);
  while (isalnum(buf[pos])
	 || buf[pos] == '_'
	 || buf[pos] == '-'
	 || buf[pos] == '.')
    pos++;
  if (buf[pos] == '+')
    {
      pos++;
      while (isdigit(buf[pos]))
	pos++;
    }
  if (buf[pos] != ')')
    throw Parse_error("expected ')'", line_number);
  pos++;
  tokens.emplace_back(op, start_pos, pos - start_pos);
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

uint32_t Parser::get_u32(const char *p)
{
  assert(isdigit(*p));
  uint64_t value = 0;
  while (isdigit(*p))
    {
      value = value * 10 + (*p++ - '0');
      if (value > std::numeric_limits<uint32_t>::max())
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

bool Parser::is_reg_x0(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].size == 4
      && buf[tokens[idx].pos + 0] == 'z'
      && buf[tokens[idx].pos + 1] == 'e'
      && buf[tokens[idx].pos + 2] == 'r'
      && buf[tokens[idx].pos + 3] == 'o')
    return true;
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 'x'
      && buf[tokens[idx].pos + 1] == '0')
    return true;
  return false;
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
    val = get_u32(&buf[pos]);
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
  if (tokens[idx].size == 4
      && buf[tokens[idx].pos + 0] == 'z'
      && buf[tokens[idx].pos + 1] == 'e'
      && buf[tokens[idx].pos + 2] == 'r'
      && buf[tokens[idx].pos + 3] == 'o')
    return zero_reg;
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 'x'
      && buf[tokens[idx].pos + 1] == '0')
    return zero_reg;
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 's'
      && buf[tokens[idx].pos + 1] == 'p')
    return rstate->registers[RiscvRegIdx::x2];
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 'r'
      && buf[tokens[idx].pos + 1] == 'a')
    return rstate->registers[RiscvRegIdx::x1];
  if (tokens[idx].kind != Lexeme::name
      || (buf[tokens[idx].pos] != 'a'
	  && buf[tokens[idx].pos] != 's'
	  && buf[tokens[idx].pos] != 't'))
    throw Parse_error("expected a register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  // TODO: Check length.
  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  if (tokens[idx].size == 3)
    value = value * 10 + (buf[tokens[idx].pos + 2] - '0');
  if (buf[tokens[idx].pos] == 'a')
    return rstate->registers[RiscvRegIdx::x10 + value];
  else if (buf[tokens[idx].pos] == 's')
    {
      if (value < 2)
	return rstate->registers[RiscvRegIdx::x8 + value];
      else
	return rstate->registers[RiscvRegIdx::x18 + (value - 2)];
    }
  else if (buf[tokens[idx].pos] == 't')
    {
      if (value < 3)
	return rstate->registers[RiscvRegIdx::x5 + value];
      else
	return rstate->registers[RiscvRegIdx::x28 + (value - 3)];
    }
  else
    throw Parse_error("expected a register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
}

Inst *Parser::get_freg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  int pos = tokens[idx].pos;
  if (buf[pos] != 'f')
    throw Parse_error("expected a floating-point register", line_number);
  pos++;
  bool is_pseudo_reg = false;
  if (!isdigit(buf[pos]))
    {
      if (buf[pos] != 'a' && buf[pos] != 's' && buf[pos] != 't')
	throw Parse_error("invalid floating-point register "
			  + std::string(token_string(tokens[idx])),
			  line_number);
      is_pseudo_reg = true;
      pos++;
    }
  uint32_t value = buf[pos] - '0';
  if (tokens[idx].size == 2 + pos - tokens[idx].pos)
    value = value * 10 + (buf[pos + 1] - '0');
  if (is_pseudo_reg)
    {
      char c = buf[tokens[idx].pos + 1];
      assert(c == 'a' || c =='s' || c =='t');
      if (c == 's')
	{
	  if (value == 0)
	    return rstate->registers[RiscvRegIdx::f8];
	  else if (value == 1)
	    return rstate->registers[RiscvRegIdx::f9];
	  else
	    return rstate->registers[RiscvRegIdx::f16 + value];
	}
      else if (c == 't')
	{
	  if (value <= 7)
	    return rstate->registers[RiscvRegIdx::f0 + value];
	  else
	    return rstate->registers[RiscvRegIdx::f20 + value];
	}
      else
	return rstate->registers[RiscvRegIdx::f10 + value];
    }
  else
    return rstate->registers[RiscvRegIdx::f0 + value];
}

Inst *Parser::get_vreg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  int pos = tokens[idx].pos;
  if (tokens[idx].size == 4
      && buf[pos] == 'v'
      && buf[pos + 1] == '0'
      && buf[pos + 2] == '.'
      && buf[pos + 3] == 't')
    return rstate->registers[RiscvRegIdx::v0];
  if (tokens[idx].size != 2 && tokens[idx].size != 3)
    throw Parse_error("expected a vector register", line_number);
  if (buf[pos] != 'v')
    throw Parse_error("expected a vector register", line_number);
  pos++;
  if (!isdigit(buf[pos]))
    throw Parse_error("expected a digit", line_number);
  uint32_t value = buf[pos] - '0';
  if (tokens[idx].size == 3)
    {
      if (!isdigit(buf[pos + 1]))
	throw Parse_error("expected a digit", line_number);
      value = value * 10 + (buf[pos + 1] - '0');
    }
  return rstate->registers[RiscvRegIdx::v0 + value];
}

Inst *Parser::get_hilo_addr(const Token& tok)
{
  assert(tok.size > 5);
  assert(buf[tok.pos + 3] == '(');
  assert(buf[tok.pos + tok.size - 1] == ')');
  int pos = tok.pos + 4;
  while (isalnum(buf[pos])
	 || buf[pos] == '_'
	 || buf[pos] == '-'
	 || buf[pos] == '.')
    pos++;
  std::string sym_name(&buf[tok.pos + 4], pos - (tok.pos + 4));
  uint64_t offset = 0;
  if (sym_alias.contains(sym_name))
    std::tie(sym_name, offset) = sym_alias[sym_name];
  if (!rstate->sym_name2mem.contains(sym_name))
    throw Parse_error("unknown symbol " + sym_name, line_number);
  Inst *addr = rstate->sym_name2mem.at(sym_name);
  if (offset)
    addr = bb->build_inst(Op::ADD, addr, bb->value_inst(offset, addr->bitsize));
  if (buf[pos] == '+')
    {
      pos++;
      assert(isdigit(buf[pos]));
      uint64_t value = 0;
      while (isdigit(buf[pos]))
	{
	  value = value * 10 + (buf[pos] - '0');
	  if (value > std::numeric_limits<uint64_t>::max())
	    throw Parse_error("too large decimal integer value", line_number);
	  pos++;
	}
      Inst *value_inst = bb->value_inst(value, addr->bitsize);
      addr = bb->build_inst(Op::ADD, addr, value_inst);
    }
  assert(buf[pos] == ')');
  return addr;
}

Inst *Parser::get_hi(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::hi)
    throw Parse_error("expected %lo instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  Inst *addr = get_hilo_addr(tokens[idx]);
  Inst *res = bb->build_inst(Op::EXTRACT, addr, 31, 12);
  Inst *lo_signbit = bb->build_extract_bit(addr, 11);
  Inst *bias = bb->build_inst(Op::ZEXT, lo_signbit, res->bitsize);
  res = bb->build_inst(Op::ADD, res, bias);
  Inst *zero = bb->value_inst(0, 12);
  return bb->build_inst(Op::CONCAT, res, zero);
}

Inst *Parser::get_lo(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::lo)
    throw Parse_error("expected %lo instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  return bb->build_trunc(get_hilo_addr(tokens[idx]), 12);
}

Inst *Parser::get_imm(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  Inst *inst;
  if (tokens[idx].kind == Lexeme::lo)
    inst = get_lo(idx);
  else
    inst = bb->value_inst(get_hex_or_integer(idx), 12);
  return bb->build_inst(Op::SEXT, inst, reg_bitsize);
}

Inst *Parser::get_reg_value(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].size == 4
      && buf[tokens[idx].pos + 0] == 'z'
      && buf[tokens[idx].pos + 1] == 'e'
      && buf[tokens[idx].pos + 2] == 'r'
      && buf[tokens[idx].pos + 3] == 'o')
    return bb->value_inst(0, reg_bitsize);
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 'x'
      && buf[tokens[idx].pos + 1] == '0')
    return bb->value_inst(0, reg_bitsize);
  return bb->build_inst(Op::READ, get_reg(idx));
}

Inst *Parser::get_freg_value(unsigned idx)
{
  return bb->build_inst(Op::READ, get_freg(idx));
}

Inst *Parser::get_vreg_value(unsigned idx)
{
  return bb->build_inst(Op::READ, get_vreg(idx));
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

void Parser::get_left_paren(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::left_paren)
    throw Parse_error("expected a '(' after "
		      + std::string(token_string(tokens[idx - 1])),
		      line_number);
}

void Parser::get_right_paren(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::right_paren)
    throw Parse_error("expected a ')' after "
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

void Parser::process_cond_branch(Op op, bool swap)
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = get_reg_value(3);
  get_comma(4);
  Basic_block *true_bb = get_bb(5);
  get_end_of_line(6);

  Basic_block *false_bb = func->build_bb();
  if (swap)
    std::swap(arg1, arg2);
  Inst *cond = bb->build_inst(op, arg1, arg2);
  bb->build_br_inst(cond, true_bb, false_bb);
  bb = false_bb;
}

Inst *Parser::gen_ffs(Inst *arg)
{
  Inst *inst = bb->value_inst(0, 32);
  for (int i = arg->bitsize - 1; i >= 0; i--)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *val = bb->value_inst(i + 1, 32);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  return inst;
}

Inst *Parser::gen_parity(Inst *arg)
{
  Inst *inst = bb->build_extract_bit(arg, 0);
  for (uint32_t i = 1; i < arg->bitsize; i++)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      inst = bb->build_inst(Op::XOR, inst, bit);
    }
  inst = bb->build_inst(Op::ZEXT, inst, reg_bitsize);

  return inst;
}

Inst *Parser::gen_popcount(Inst *arg)
{
  Inst *bit = bb->build_extract_bit(arg, 0);
  Inst *inst = bb->build_inst(Op::ZEXT, bit, 32);
  for (uint32_t i = 1; i < arg->bitsize; i++)
    {
      bit = bb->build_extract_bit(arg, i);
      Inst *ext = bb->build_inst(Op::ZEXT, bit, 32);
      inst = bb->build_inst(Op::ADD, inst, ext);
    }
  return inst;
}

Inst *Parser::gen_sdiv(Inst *arg1, Inst *arg2)
{
  Inst *zero = bb->value_inst(0, arg2->bitsize);
  bb->build_inst(Op::UB, bb->build_inst(Op::EQ, arg2, zero));
  return bb->build_inst(Op::SDIV, arg1, arg2);
}

Inst *Parser::gen_udiv(Inst *arg1, Inst *arg2)
{
  Inst *zero = bb->value_inst(0, arg2->bitsize);
  bb->build_inst(Op::UB, bb->build_inst(Op::EQ, arg2, zero));
  return bb->build_inst(Op::UDIV, arg1, arg2);
}

Inst *Parser::read_arg(uint32_t reg, uint32_t bitsize)
{
  if (reg_bitsize == 64)
    {
      Inst *inst = bb->build_inst(Op::READ, rstate->registers[reg]);
      if (bitsize == 64)
	return inst;
      return bb->build_trunc(inst, bitsize);
    }
  else
    {
      assert(reg_bitsize == 32);
      if (bitsize == 32)
	return bb->build_inst(Op::READ, rstate->registers[reg]);
      else
	{
	  assert(bitsize == 64);
	  Inst *a0 = bb->build_inst(Op::READ, rstate->registers[reg + 0]);
	  Inst *a1 = bb->build_inst(Op::READ, rstate->registers[reg + 1]);
	  return bb->build_inst(Op::CONCAT, a1, a0);
	}
    }
}

void Parser::write_retval(Inst *retval)
{
  if (reg_bitsize == 64)
    {
      if (retval->bitsize == 32)
	retval = bb->build_inst(Op::SEXT, retval, 64);
      assert(retval->bitsize == 64);
      bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::x10], retval);
    }
  else
    {
      assert(reg_bitsize == 32);
      if (retval->bitsize == 32)
	bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::x10], retval);
      else
	{
	  assert(retval->bitsize == 64);
	  Inst *a0 = bb->build_trunc(retval, reg_bitsize);
	  Inst *a1 = bb->build_inst(Op::EXTRACT, retval, 2 * reg_bitsize - 1,
				    reg_bitsize);
	  bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::x10], a0);
	  bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::x11], a1);
	}
    }
}

void Parser::process_call()
{
  std::string_view name = get_name(1);
  get_end_of_line(2);

  if (name == "__ashldi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *arg2 = read_arg(RiscvRegIdx::x12, 32);
      arg2 = bb->build_inst(Op::ZEXT, arg2, 64);
      Inst *res = bb->build_inst(Op::SHL, arg1, arg2);
      write_retval(res);
      return;
    }
  if (name == "__ashrdi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *arg2 = read_arg(RiscvRegIdx::x12, 32);
      arg2 = bb->build_inst(Op::ZEXT, arg2, 64);
      Inst *res = bb->build_inst(Op::ASHR, arg1, arg2);
      write_retval(res);
      return;
    }
  if (name == "__bswapdi2")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = gen_bswap(bb, arg);
      write_retval(res);
      return;
    }
  if (name == "__bswapsi2")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 32);
      Inst *res = gen_bswap(bb, arg);
      write_retval(res);
      return;
    }
  if (name == "__clrsbdi2")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = gen_clrsb(bb, arg);
      write_retval(res);
      return;
    }
  if (name == "__clrsbsi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 32);
      Inst *res = gen_clrsb(bb, arg);
      write_retval(res);
      return;
    }
  if (name == "__clzdi2")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 64);
      Inst *zero = bb->value_inst(0, arg->bitsize);
      Inst *ub = bb->build_inst(Op::EQ, arg, zero);
      bb->build_inst(Op::UB, ub);
      Inst *res = gen_clz(bb, arg);
      write_retval(res);
      return;
    }
  if (name == "__clzsi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 32);
      Inst *zero = bb->value_inst(0, arg->bitsize);
      Inst *ub = bb->build_inst(Op::EQ, arg, zero);
      bb->build_inst(Op::UB, ub);
      Inst *res = gen_clz(bb, arg);
      write_retval(res);
      return;
    }
  if (name == "__ctzdi2")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 64);
      Inst *zero = bb->value_inst(0, arg->bitsize);
      Inst *ub = bb->build_inst(Op::EQ, arg, zero);
      bb->build_inst(Op::UB, ub);
      Inst *res = gen_ctz(bb, arg);
      write_retval(res);
      return;
    }
  if (name == "__ctzsi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 32);
      Inst *zero = bb->value_inst(0, arg->bitsize);
      Inst *ub = bb->build_inst(Op::EQ, arg, zero);
      bb->build_inst(Op::UB, ub);
      Inst *res = gen_ctz(bb, arg);
      write_retval(res);
      return;
    }
  if (name == "__divdi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *arg2 = read_arg(RiscvRegIdx::x12, 64);
      Inst *res = gen_sdiv(arg1, arg2);
      write_retval(res);
      return;
    }
  if (name == "__udivdi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *arg2 = read_arg(RiscvRegIdx::x12, 64);
      Inst *res = gen_udiv(arg1, arg2);
      write_retval(res);
      return;
    }
  if (name == "__ffsdi2")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = gen_ffs(arg);
      res = bb->build_inst(Op::SEXT, res, 64);
      write_retval(res);
      return;
    }
  if (name == "__ffssi2" || name == "ffs")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 32);
      Inst *res = gen_ffs(arg);
      write_retval(res);
      return;
    }
  if (name == "__fixsfdi" && reg_bitsize == 32)
    {
      Inst *arg1 =
	bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f10]);
      arg1 = bb->build_trunc(arg1, 32);
      Inst *res = bb->build_inst(Op::F2S, arg1, 64);
      write_retval(res);
      return;
    }
  if (name == "__fixunssfdi" && reg_bitsize == 32)
    {
      Inst *arg1 =
	bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f10]);
      arg1 = bb->build_trunc(arg1, 32);
      Inst *res = bb->build_inst(Op::F2U, arg1, 64);
      write_retval(res);
      return;
    }
  if (name == "__fixdfdi" && reg_bitsize == 32)
    {
      Inst *arg1 =
	bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f10]);
      Inst *res = bb->build_inst(Op::F2S, arg1, 64);
      write_retval(res);
      return;
    }
  if (name == "__fixunsdfdi" && reg_bitsize == 32)
    {
      Inst *arg1 = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::f10]);
      Inst *res = bb->build_inst(Op::F2U, arg1, 64);
      write_retval(res);
      return;
    }
  if (name == "__floatdisf" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = bb->build_inst(Op::S2F, arg1, 32);
      Inst *m1 = bb->value_m1_inst(32);
      res = bb->build_inst(Op::CONCAT, m1, res);
      bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::f10], res);
      return;
    }
  if (name == "__floatundisf" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = bb->build_inst(Op::U2F, arg1, 32);
      Inst *m1 = bb->value_m1_inst(32);
      res = bb->build_inst(Op::CONCAT, m1, res);
      bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::f10], res);
      return;
    }
  if (name == "__floatdidf" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = bb->build_inst(Op::S2F, arg1, 64);
      bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::f10], res);
      return;
    }
  if (name == "__floatundidf" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = bb->build_inst(Op::U2F, arg1, 64);
      bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::f10], res);
      return;
    }
  if (name == "__lshrdi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *arg2 = read_arg(RiscvRegIdx::x12, 32);
      arg2 = bb->build_inst(Op::ZEXT, arg2, 64);
      Inst *res = bb->build_inst(Op::LSHR, arg1, arg2);
      write_retval(res);
      return;
    }
  if (name == "__moddi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *arg2 = read_arg(RiscvRegIdx::x12, 64);
      Inst *res = bb->build_inst(Op::SREM, arg1, arg2);
      write_retval(res);
      return;
    }
  if (name == "__popcountdi2")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = gen_popcount(arg);
      res = bb->build_inst(Op::SEXT, res, 64);
      write_retval(res);
      return;
    }
  if (name == "__popcountsi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 32);
      Inst *res = gen_popcount(arg);
      write_retval(res);
      return;
    }
  if (name == "__paritydi2")
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 64);
      Inst *res = gen_parity(arg);
      write_retval(res);
      return;
    }
  if (name == "__paritysi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(RiscvRegIdx::x10, 32);
      Inst *res = gen_parity(arg);
      write_retval(res);
      return;
    }
  if (name == "__umoddi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(RiscvRegIdx::x10, 64);
      Inst *arg2 = read_arg(RiscvRegIdx::x12, 64);
      Inst *res = bb->build_inst(Op::UREM, arg1, arg2);
      write_retval(res);
      return;
    }

  throw Not_implemented("call " + std::string(name));
}

void Parser::process_tail()
{
  std::string_view name = get_name(1);
  get_end_of_line(2);

  throw Not_implemented("tail " + std::string(name));
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

void Parser::process_load(int size, LStype lstype)
{
  Inst *ptr;
  Inst *dest;
  if (lstype == LStype::float_ls)
    dest = get_freg(1);
  else
    dest = get_reg(1);
  get_comma(2);
  Inst *offset = get_imm(3);
  get_left_paren(4);
  Inst *base = get_reg_value(5);
  get_right_paren(6);
  get_end_of_line(7);

  ptr = bb->build_inst(Op::ADD, base, offset);
  load_ub_check(ptr, size);
  Inst *value = nullptr;
  for (int i = 0; i < size; i++)
    {
      Inst *size_inst = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, size_inst);
      Inst *byte = bb->build_inst(Op::LOAD, addr);
      if (value)
	value = bb->build_inst(Op::CONCAT, byte, value);
      else
	value = byte;
    }
  if (lstype == LStype::float_ls && size == 4)
    {
      Inst *m1 = bb->value_m1_inst(32);
      value = bb->build_inst(Op::CONCAT, m1, value);
    }
  else if (value->bitsize < reg_bitsize)
    {
      Op op = lstype == LStype::unsigned_ls ? Op::ZEXT : Op::SEXT;
      value = bb->build_inst(op, value, reg_bitsize);
    }
  bb->build_inst(Op::WRITE, dest, value);
}

void Parser::process_store(int size, LStype lstype)
{
  Inst *ptr;
  Inst *value;
  if (lstype == LStype::float_ls)
    value = get_freg_value(1);
  else
    value = get_reg_value(1);
  get_comma(2);
  Inst *offset = get_imm(3);
  get_left_paren(4);
  Inst *base = get_reg_value(5);
  get_right_paren(6);
  get_end_of_line(7);

  ptr = bb->build_inst(Op::ADD, base, offset);
  store_ub_check(ptr, size);
  for (int i = 0; i < size; i++)
    {
      Inst *size_inst = bb->value_inst(i, ptr->bitsize);
      Inst *addr = bb->build_inst(Op::ADD, ptr, size_inst);
      Inst *byte = bb->build_inst(Op::EXTRACT, value, i * 8 + 7, i * 8);
      bb->build_inst(Op::STORE, addr, byte);
    }
}

void Parser::process_funary(std::string_view name, Op op)
{
  Inst *dest = get_freg(1);
  get_comma(2);
  Inst *arg1 = get_freg_value(3);
  get_end_of_line(4);

  bool is_single_prec =
    name[name.length() - 2] == '.' && name[name.length() - 1] == 's';
  if (is_single_prec)
    {
      arg1 = bb->build_trunc(arg1, 32);
    }
  Inst *res = bb->build_inst(op, arg1);
  if (is_single_prec)
    {
      Inst *m1 = bb->value_m1_inst(32);
      res = bb->build_inst(Op::CONCAT, m1, res);
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_fbinary(std::string_view name, Op op)
{
  Inst *dest = get_freg(1);
  get_comma(2);
  Inst *arg1 = get_freg_value(3);
  get_comma(4);
  Inst *arg2 = get_freg_value(5);
  get_end_of_line(6);

  bool is_single_prec =
    name[name.length() - 2] == '.' && name[name.length() - 1] == 's';
  if (is_single_prec)
    {
      arg1 = bb->build_trunc(arg1, 32);
      arg2 = bb->build_trunc(arg2, 32);
    }
  Inst *res = bb->build_inst(op, arg1, arg2);
  if (is_single_prec)
    {
      Inst *m1 = bb->value_m1_inst(32);
      res = bb->build_inst(Op::CONCAT, m1, res);
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_fcmp(std::string_view name, Op op, bool swap)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_freg_value(3);
  get_comma(4);
  Inst *arg2 = get_freg_value(5);
  get_end_of_line(6);

  bool is_single_prec =
    name[name.length() - 2] == '.' && name[name.length() - 1] == 's';
  if (is_single_prec)
    {
      arg1 = bb->build_trunc(arg1, 32);
      arg2 = bb->build_trunc(arg2, 32);
    }
  if (swap)
    std::swap(arg1, arg2);
  Inst *res = bb->build_inst(op, arg1, arg2);
  res = bb->build_inst(Op::ZEXT, res, reg_bitsize);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_fcvt_i2f(uint32_t src_bitsize, uint32_t dest_bitsize,
			      bool is_unsigned)
{
  Inst *dest = get_freg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  if (src_bitsize < freg_bitsize)
    arg1 = bb->build_trunc(arg1, src_bitsize);
  Op op = is_unsigned ? Op::U2F : Op::S2F;
  Inst *res = bb->build_inst(op, arg1, dest_bitsize);
  if (dest_bitsize < freg_bitsize)
    {
      Inst *m1 = bb->value_m1_inst(32);
      res = bb->build_inst(Op::CONCAT, m1, res);
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_fcvt_f2i(uint32_t src_bitsize, uint32_t dest_bitsize,
			      bool is_unsigned)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_freg_value(3);
  get_comma(4);
  std::string_view rounding_mode = get_name(5);
  if (rounding_mode != "rtz")
    throw Parse_error("expected rtz as rounding mode", line_number);
  get_end_of_line(6);

  if (src_bitsize < freg_bitsize)
    arg1 = bb->build_trunc(arg1, src_bitsize);
  Op op = is_unsigned ? Op::F2U : Op::F2S;
  Inst *res = bb->build_inst(op, arg1, dest_bitsize);
  if (dest_bitsize < reg_bitsize)
    res = bb->build_inst(is_unsigned ? Op::ZEXT : Op::SEXT, res, reg_bitsize);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_fcvt_f2f(uint32_t src_bitsize, uint32_t dest_bitsize)
{
  Inst *dest = get_freg(1);
  get_comma(2);
  Inst *arg1 = get_freg_value(3);
  get_end_of_line(4);

  if (src_bitsize == 32)
    arg1 = bb->build_trunc(arg1, 32);
  Inst *res = bb->build_inst(Op::FCHPREC, arg1, dest_bitsize);
  if (dest_bitsize == 32)
    {
      Inst *m1 = bb->value_m1_inst(32);
      res = bb->build_inst(Op::CONCAT, m1, res);
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_fmin_fmax(uint32_t bitsize, bool is_min)
{
  Inst *dest = get_freg(1);
  get_comma(2);
  Inst *arg1 = get_freg_value(3);
  get_comma(4);
  Inst *arg2 = get_freg_value(5);
  get_end_of_line(6);

  if (bitsize == 32)
    {
      arg1 = bb->build_trunc(arg1, 32);
      arg2 = bb->build_trunc(arg2, 32);
    }
  Inst *is_nan = bb->build_inst(Op::IS_NAN, arg2);
  Inst *cmp;
  if (is_min)
    cmp = bb->build_inst(Op::FLT, arg1, arg2);
  else
    cmp = bb->build_inst(Op::FLT, arg2, arg1);
  Inst *res1 = bb->build_inst(Op::ITE, cmp, arg1, arg2);
  Inst *res2 = bb->build_inst(Op::ITE, is_nan, arg1, res1);
  // 0.0 and -0.0 is equal as floating-point values, and fmin(0.0, -0.0)
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
  if (bitsize == 32)
    {
      Inst *m1 = bb->value_m1_inst(32);
      res = bb->build_inst(Op::CONCAT, m1, res);
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_iunary(std::string_view name, Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  bool has_w_suffix = name[name.length() - 1] == 'w';
  if (has_w_suffix)
    arg1 = bb->build_trunc(arg1, 32);
  Inst *res = bb->build_inst(op, arg1);
  if (has_w_suffix)
    res = bb->build_inst(Op::SEXT, res, reg_bitsize);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_ibinary(std::string_view name, Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2;
  bool is_imm =
    name[name.length() - 1] == 'i'
    || (name[name.length() - 2] == 'i' && name[name.length() - 1] == 'w');
  if (is_imm)
    arg2 = get_imm(5);
  else
    arg2 = get_reg_value(5);
  get_end_of_line(6);

  bool has_w_suffix = name[name.length() - 1] == 'w';
  if (has_w_suffix)
    {
      arg1 = bb->build_trunc(arg1, 32);
      arg2 = bb->build_trunc(arg2, 32);
    }
  Inst *res = bb->build_inst(op, arg1, arg2);
  if (has_w_suffix)
    res = bb->build_inst(Op::SEXT, res, reg_bitsize);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_ishift(std::string_view name, Op op)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2;
  bool is_imm =
    name[name.length() - 1] == 'i'
    || (name[name.length() - 2] == 'i' && name[name.length() - 1] == 'w');
  if (is_imm)
    arg2 = get_imm(5);
  else
    arg2 = get_reg_value(5);
  get_end_of_line(6);

  bool has_w_suffix = name[name.length() - 1] == 'w';
  if (reg_bitsize == 32 || has_w_suffix)
    {
      arg1 = bb->build_trunc(arg1, 32);
      arg2 = bb->build_inst(Op::ZEXT, bb->build_trunc(arg2, 5), 32);
    }
  else
    arg2 = bb->build_inst(Op::ZEXT, bb->build_trunc(arg2, 6), reg_bitsize);
  Inst *res = bb->build_inst(op, arg1, arg2);
  if (has_w_suffix)
    res = bb->build_inst(Op::SEXT, res, reg_bitsize);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_icmp(std::string_view name, Op op, bool swap)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2;
  bool is_imm =
    name[name.length() - 1] == 'i'
    || (name[name.length() - 2] == 'i' && name[name.length() - 1] == 'u')
    || (name[name.length() - 2] == 'i' && name[name.length() - 1] == 'w')
    || (name[name.length() - 3] == 'i'
	&& name[name.length() - 2] == 'u'
	&& name[name.length() - 1] == 'w');
  if (is_imm)
    arg2 = get_imm(5);
  else
    arg2 = get_reg_value(5);
  get_end_of_line(6);

  bool has_w_suffix = name[name.length() - 1] == 'w';
  if (has_w_suffix)
    {
      arg1 = bb->build_trunc(arg1, 32);
      arg2 = bb->build_trunc(arg2, 32);
    }
  if (swap)
    std::swap(arg1, arg2);
  Inst *res = bb->build_inst(op, arg1, arg2);
  res = bb->build_inst(Op::ZEXT, res, reg_bitsize);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_zba_sh_add(uint64_t shift_val, bool truncate_arg1)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_end_of_line(6);

  if (truncate_arg1)
    arg1 = bb->build_inst(Op::ZEXT, bb->build_trunc(arg1, 32), reg_bitsize);
  Inst *shift = bb->value_inst(shift_val, reg_bitsize);
  Inst *res = bb->build_inst(Op::SHL, arg1, shift);
  res = bb->build_inst(Op::ADD, res, arg2);
  bb->build_inst(Op::WRITE, dest, res);
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

Inst *Parser::change_prec(Inst *inst, uint32_t bitsize)
{
  if (inst->bitsize < bitsize)
    return bb->build_inst(Op::SEXT, inst, bitsize);
  if (inst->bitsize > bitsize)
    return bb->build_trunc(inst, bitsize);
  return inst;
}

Inst *gen_rsub(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return bb->build_inst(Op::SUB, elem2, elem1);
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

Inst *gen_mulh(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  elem1 = bb->build_inst(Op::SEXT, elem1, 2 * elem1->bitsize);
  elem2 = bb->build_inst(Op::SEXT, elem2, 2 * elem2->bitsize);
  Inst *res = bb->build_inst(Op::MUL, elem1, elem2);
  return bb->build_inst(Op::EXTRACT, res, res->bitsize - 1, res->bitsize / 2);
}

Inst *gen_mulhu(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  elem1 = bb->build_inst(Op::ZEXT, elem1, 2 * elem1->bitsize);
  elem2 = bb->build_inst(Op::ZEXT, elem2, 2 * elem2->bitsize);
  Inst *res = bb->build_inst(Op::MUL, elem1, elem2);
  return bb->build_inst(Op::EXTRACT, res, res->bitsize - 1, res->bitsize / 2);
}

Inst *gen_mulhsu(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  elem1 = bb->build_inst(Op::SEXT, elem1, 2 * elem1->bitsize);
  elem2 = bb->build_inst(Op::ZEXT, elem2, 2 * elem2->bitsize);
  Inst *res = bb->build_inst(Op::MUL, elem1, elem2);
  return bb->build_inst(Op::EXTRACT, res, res->bitsize - 1, res->bitsize / 2);
}

Inst *gen_nand(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return bb->build_inst(Op::NOT, bb->build_inst(Op::AND, elem1, elem2));
}

Inst *gen_andn(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  elem2 = bb->build_inst(Op::NOT, elem2);
  return bb->build_inst(Op::AND, elem1, elem2);
}

Inst *gen_nor(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return bb->build_inst(Op::NOT, bb->build_inst(Op::OR, elem1, elem2));
}

Inst *gen_orn(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  elem2 = bb->build_inst(Op::NOT, elem2);
  return bb->build_inst(Op::OR, elem1, elem2);
}

Inst *gen_xnor(Basic_block *bb, Inst *elem1, Inst *elem2)
{
  return bb->build_inst(Op::NOT, bb->build_inst(Op::XOR, elem1, elem2));
}

void Parser::process_vsetvli(bool arg1_is_imm)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1;
  if (arg1_is_imm)
    arg1 = get_imm(3);
  else
    arg1 = get_reg_value(3);
  get_comma(4);

  std::string_view arg2 = get_name(5);
  get_comma(6);
  std::string_view arg3 = get_name(7);
  get_comma(8);
  std::string_view arg4 = get_name(9);
  get_comma(10);
  std::string_view arg5 = get_name(11);
  get_end_of_line(12);

  Inst *vsew = nullptr;
  uint32_t nof_vec_elem;
  if (arg2 == "e8")
    {
      vsew = bb->value_inst(0, 3);
      nof_vec_elem = rstate->vreg_bitsize / 8;
    }
  else if (arg2 == "e16")
    {
      vsew = bb->value_inst(1, 3);
      nof_vec_elem = rstate->vreg_bitsize / 16;
    }
  else if (arg2 == "e32")
    {
      vsew = bb->value_inst(2, 3);
      nof_vec_elem = rstate->vreg_bitsize / 32;
    }
  else if (arg2 == "e64")
    {
      vsew = bb->value_inst(3, 3);
      nof_vec_elem = rstate->vreg_bitsize / 64;
    }
  else
    throw Parse_error("invalid SEW", line_number);

  if (arg3 != "m1" && arg3 != "mf2" && arg3 != "mf4" && arg3 != "mf8")
    throw Parse_error("vsetvli: only m1/mf* are implemented", line_number);
  if (arg4 != "ta" && arg4 != "tu")
    throw Parse_error("expected ta or tu", line_number);
  if (arg5 != "ma" && arg5 != "mu")
    throw Parse_error("expected ma or mu", line_number);

  bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::vsew], vsew);
  Inst *vlmax = bb->value_inst(nof_vec_elem, arg1->bitsize);
  if (is_reg_x0(3))
    {
      if (is_reg_x0(0))
	{
	  // Keep existing vl. I.e. nothing to do here.
	}
      else
	{
	  bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::vl], vlmax);
	  bb->build_inst(Op::WRITE, dest, vlmax);
	}
    }
  else
    {
      Inst *cmp = bb->build_inst(Op::ULT, vlmax, arg1);
      Inst *vl = bb->build_inst(Op::ITE, cmp, vlmax, arg1);
      bb->build_inst(Op::WRITE, rstate->registers[RiscvRegIdx::vl], vl);
      bb->build_inst(Op::WRITE, dest, vl);
    }
}

void Parser::process_vle(uint32_t elem_bitsize)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  unsigned __int128 offset = get_hex_or_integer(3);
  if (offset)
    throw Parse_error("only 0 is implemented", line_number);
  get_left_paren(4);
  Inst *ptr = get_reg_value(5);
  get_right_paren(6);
  get_end_of_line(7);

  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  Inst *value = nullptr;
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem_offset_inst =
	bb->value_inst(i * elem_bitsize / 8, ptr->bitsize);
      Inst *elem_ptr = bb->build_inst(Op::ADD, ptr, elem_offset_inst);
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      for (uint32_t j = 0; j < elem_bitsize / 8; j++)
	{
	  Inst *offset_inst = bb->value_inst(j, ptr->bitsize);
	  Inst *addr = bb->build_inst(Op::ADD, elem_ptr, offset_inst);
	  Inst *loaded_byte = bb->build_inst(Op::LOAD, addr);
	  Inst *orig_byte = extract_vec_elem(orig_elem, 8, j);
	  Inst *byte = bb->build_inst(Op::ITE, cmp, loaded_byte, orig_byte);
	  if (value)
	    value = bb->build_inst(Op::CONCAT, byte, value);
	  else
	    value = byte;
	}
    }
  bb->build_inst(Op::WRITE, dest, value);
}

void Parser::process_vse(uint32_t elem_bitsize)
{
  Inst *value = get_vreg_value(1);
  get_comma(2);
  unsigned __int128 offset = get_hex_or_integer(3);
  if (offset)
    throw Parse_error("only 0 is implemented", line_number);
  get_left_paren(4);
  Inst *ptr = get_reg_value(5);
  get_right_paren(6);
  get_end_of_line(7);

  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Basic_block *true_bb = func->build_bb();
      Basic_block *false_bb = func->build_bb();
      Inst *i_inst = bb->value_inst(i, vl->bitsize);
      Inst *cmp = bb->build_inst(Op::ULT, i_inst, vl);
      bb->build_br_inst(cmp, true_bb, false_bb);
      bb = true_bb;

      Inst *elem_offset_inst =
	bb->value_inst(i * elem_bitsize / 8, ptr->bitsize);
      Inst *elem_ptr = bb->build_inst(Op::ADD, ptr, elem_offset_inst);
      Inst *elem = extract_vec_elem(value, elem_bitsize, i);
      for (uint32_t j = 0; j < elem_bitsize / 8; j++)
	{
	  Inst *offset_inst = bb->value_inst(j, ptr->bitsize);
	  Inst *addr = bb->build_inst(Op::ADD, elem_ptr, offset_inst);
	  Inst *byte = extract_vec_elem(elem, 8, j);
	  bb->build_inst(Op::STORE, addr, byte);
	}
      bb->build_br_inst(false_bb);
      bb = false_bb;
    }
}

Inst *Parser::gen_vec_unary(Op op, Inst *orig, Inst *arg1, Inst *mask,
			    uint32_t elem_bitsize)
{
  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1;
      if (arg1->bitsize == elem_bitsize)
	elem1 = arg1;
      else
	elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *inst = bb->build_inst(op, elem1);
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      if (mask)
	cmp = bb->build_inst(Op::AND, cmp, extract_vec_elem(mask, 1, i));
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  return res;
}

void Parser::process_vec_unary(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  Inst *mask = nullptr;
  if (tokens.size() > 4)
    {
      get_comma(4);
      mask = get_vreg_value(5);
      get_end_of_line(6);
    }
  else
    get_end_of_line(4);

  Inst *res8 = gen_vec_unary(op, orig, arg1, mask, 8);
  Inst *res16 = gen_vec_unary(op, orig, arg1, mask, 16);
  Inst *res32 = gen_vec_unary(op, orig, arg1, mask, 32);
  Inst *res64 = gen_vec_unary(op, orig, arg1, mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_unary_vi(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  unsigned __int128 arg1 = get_hex_or_integer(3);
  Inst *mask = nullptr;
  if (tokens.size() > 4)
    {
      get_comma(4);
      mask = get_vreg_value(5);
      get_end_of_line(6);
    }
  else
    get_end_of_line(4);

  Inst *imm8 = bb->value_inst(arg1, 8);
  Inst *imm16 = bb->value_inst(arg1, 16);
  Inst *imm32 = bb->value_inst(arg1, 32);
  Inst *imm64 = bb->value_inst(arg1, 64);
  Inst *res8 = gen_vec_unary(op, orig, imm8, mask, 8);
  Inst *res16 = gen_vec_unary(op, orig, imm16, mask, 16);
  Inst *res32 = gen_vec_unary(op, orig, imm32, mask, 32);
  Inst *res64 = gen_vec_unary(op, orig, imm64, mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_unary_vx(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  Inst *mask = nullptr;
  if (tokens.size() > 4)
    {
      get_comma(4);
      mask = get_vreg_value(5);
      get_end_of_line(6);
    }
  else
    get_end_of_line(4);

  Inst *res8 = gen_vec_unary(op, orig, change_prec(arg1, 8), mask, 8);
  Inst *res16 = gen_vec_unary(op, orig, change_prec(arg1, 16), mask, 16);
  Inst *res32 = gen_vec_unary(op, orig, change_prec(arg1, 32), mask, 32);
  Inst *res64 = gen_vec_unary(op, orig, change_prec(arg1, 64), mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

Inst *Parser::gen_vec_binary(Op op, Inst *orig, Inst *arg1, Inst *arg2,
			     Inst *mask, uint32_t elem_bitsize)
{
  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      if (mask)
	cmp = bb->build_inst(Op::AND, cmp, extract_vec_elem(mask, 1, i));
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  return res;
}

void Parser::process_vec_binary(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5);
  Inst *mask = nullptr;
  if (tokens.size() > 6)
    {
      get_comma(6);
      mask = get_vreg_value(7);
      get_end_of_line(8);
    }
  else
    get_end_of_line(6);

  Inst *res8 = gen_vec_binary(op, orig, arg1, arg2, mask, 8);
  Inst *res16 = gen_vec_binary(op, orig, arg1, arg2, mask, 16);
  Inst *res32 = gen_vec_binary(op, orig, arg1, arg2, mask, 32);
  Inst *res64 = gen_vec_binary(op, orig, arg1, arg2, mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_binary_vi(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  unsigned __int128 arg2 = get_hex_or_integer(5);
  Inst *mask = nullptr;
  if (tokens.size() > 6)
    {
      get_comma(6);
      mask = get_vreg_value(7);
      get_end_of_line(8);
    }
  else
    get_end_of_line(6);

  Inst *imm8 = bb->value_inst(arg2, 8);
  Inst *imm16 = bb->value_inst(arg2, 16);
  Inst *imm32 = bb->value_inst(arg2, 32);
  Inst *imm64 = bb->value_inst(arg2, 64);
  Inst *res8 = gen_vec_binary(op, orig, arg1, imm8, mask, 8);
  Inst *res16 = gen_vec_binary(op, orig, arg1, imm16, mask, 16);
  Inst *res32 = gen_vec_binary(op, orig, arg1, imm32, mask, 32);
  Inst *res64 = gen_vec_binary(op, orig, arg1, imm64, mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_binary_vx(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  Inst *mask = nullptr;
  if (tokens.size() > 6)
    {
      get_comma(6);
      mask = get_vreg_value(7);
      get_end_of_line(8);
    }
  else
    get_end_of_line(6);

  Inst *res8 = gen_vec_binary(op, orig, arg1, change_prec(arg2, 8), mask, 8);
  Inst *res16 = gen_vec_binary(op, orig, arg1, change_prec(arg2, 16), mask, 16);
  Inst *res32 = gen_vec_binary(op, orig, arg1, change_prec(arg2, 32), mask, 32);
  Inst *res64 = gen_vec_binary(op, orig, arg1, change_prec(arg2, 64), mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

Inst *Parser::gen_vec_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*),
			     Inst *orig, Inst *arg1, Inst *arg2, Inst *mask,
			     uint32_t elem_bitsize)
{
  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *inst = gen_elem(bb, elem1, elem2);
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      if (mask)
	cmp = bb->build_inst(Op::AND, cmp, extract_vec_elem(mask, 1, i));
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  return res;
}

void Parser::process_vec_binary(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*))
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5);
  Inst *mask = nullptr;
  if (tokens.size() > 6)
    {
      get_comma(6);
      mask = get_vreg_value(7);
      get_end_of_line(8);
    }
  else
    get_end_of_line(6);

  Inst *res8 = gen_vec_binary(gen_elem, orig, arg1, arg2, mask, 8);
  Inst *res16 = gen_vec_binary(gen_elem, orig, arg1, arg2, mask, 16);
  Inst *res32 = gen_vec_binary(gen_elem, orig, arg1, arg2, mask, 32);
  Inst *res64 = gen_vec_binary(gen_elem, orig, arg1, arg2, mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_binary_vi(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*))
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  unsigned __int128 arg2 = get_hex_or_integer(5);
  Inst *mask = nullptr;
  if (tokens.size() > 6)
    {
      get_comma(6);
      mask = get_vreg_value(7);
      get_end_of_line(8);
    }
  else
    get_end_of_line(6);

  Inst *imm8 = bb->value_inst(arg2, 8);
  Inst *imm16 = bb->value_inst(arg2, 16);
  Inst *imm32 = bb->value_inst(arg2, 32);
  Inst *imm64 = bb->value_inst(arg2, 64);
  Inst *res8 = gen_vec_binary(gen_elem, orig, arg1, imm8, mask, 8);
  Inst *res16 = gen_vec_binary(gen_elem, orig, arg1, imm16, mask, 16);
  Inst *res32 = gen_vec_binary(gen_elem, orig, arg1, imm32, mask, 32);
  Inst *res64 = gen_vec_binary(gen_elem, orig, arg1, imm64, mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_binary_vx(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*))
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  Inst *mask = nullptr;
  if (tokens.size() > 6)
    {
      get_comma(6);
      mask = get_vreg_value(7);
      get_end_of_line(8);
    }
  else
    get_end_of_line(6);

  Inst *res8 =
    gen_vec_binary(gen_elem, orig, arg1, change_prec(arg2, 8), mask, 8);
  Inst *res16 =
    gen_vec_binary(gen_elem, orig, arg1, change_prec(arg2, 16), mask, 16);
  Inst *res32 =
    gen_vec_binary(gen_elem, orig, arg1, change_prec(arg2, 32), mask, 32);
  Inst *res64 =
    gen_vec_binary(gen_elem, orig, arg1, change_prec(arg2, 64), mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_mask_unary(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_end_of_line(4);

  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  for (uint32_t i = 0; i < rstate->vreg_bitsize; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, 1, i);
      Inst *inst = bb->build_inst(op, elem1);
      Inst *orig_elem = extract_vec_elem(orig, 1, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_mask_set(bool value)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_end_of_line(2);

  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  for (uint32_t i = 0; i < rstate->vreg_bitsize; i++)
    {
      Inst *inst = bb->value_inst(value, 1);
      Inst *orig_elem = extract_vec_elem(orig, 1, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_mask_binary(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5);
  get_end_of_line(6);

  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  for (uint32_t i = 0; i < rstate->vreg_bitsize; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, 1, i);
      Inst *elem2 = extract_vec_elem(arg2, 1, i);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      Inst *orig_elem = extract_vec_elem(orig, 1, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_mask_binary(Inst*(*gen_elem)(Basic_block*, Inst*,
						      Inst*))
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5);
  get_end_of_line(6);

  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  for (uint32_t i = 0; i < rstate->vreg_bitsize; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, 1, i);
      Inst *elem2 = extract_vec_elem(arg2, 1, i);
      Inst *inst = gen_elem(bb, elem1, elem2);
      Inst *orig_elem = extract_vec_elem(orig, 1, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  bb->build_inst(Op::WRITE, dest, res);
}

Inst *Parser::gen_vec_reduc(Op op, Inst *orig, Inst *arg1, Inst *arg2,
			    Inst *mask, uint32_t elem_bitsize)
{
  Inst *res = extract_vec_elem(arg2, elem_bitsize, 0);
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *inst = bb->build_inst(op, res, elem1);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      if (mask)
	cmp = bb->build_inst(Op::AND, cmp, extract_vec_elem(mask, 1, i));
      res = bb->build_inst(Op::ITE, cmp, inst, res);
    }
  for (uint32_t i = 1; i < nof_elem; i++)
    {
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      res = bb->build_inst(Op::CONCAT, orig_elem, res);
    }
  return res;
}

void Parser::process_vec_reduc(Op op)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5);
  Inst *mask = nullptr;
  if (tokens.size() > 6)
    {
      get_comma(6);
      mask = get_vreg_value(7);
      get_end_of_line(8);
    }
  else
    get_end_of_line(6);

  Inst *res8 = gen_vec_reduc(op, orig, arg1, arg2, mask, 8);
  Inst *res16 = gen_vec_reduc(op, orig, arg1, arg2, mask, 16);
  Inst *res32 = gen_vec_reduc(op, orig, arg1, arg2, mask, 32);
  Inst *res64 = gen_vec_reduc(op, orig, arg1, arg2, mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

Inst *Parser::gen_vec_reduc(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*),
			    Inst *orig, Inst *arg1, Inst *arg2, Inst *mask,
			    uint32_t elem_bitsize)
{
  Inst *res = extract_vec_elem(arg2, elem_bitsize, 0);
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *inst = gen_elem(bb, res, elem1);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      if (mask)
	cmp = bb->build_inst(Op::AND, cmp, extract_vec_elem(mask, 1, i));
      res = bb->build_inst(Op::ITE, cmp, inst, res);
    }
  for (uint32_t i = 1; i < nof_elem; i++)
    {
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      res = bb->build_inst(Op::CONCAT, orig_elem, res);
    }
  return res;
}

void Parser::process_vec_reduc(Inst*(*gen_elem)(Basic_block*, Inst*, Inst*))
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5);
  Inst *mask = nullptr;
  if (tokens.size() > 6)
    {
      get_comma(6);
      mask = get_vreg_value(7);
      get_end_of_line(8);
    }
  else
    get_end_of_line(6);

  Inst *res8 = gen_vec_reduc(gen_elem, orig, arg1, arg2, mask, 8);
  Inst *res16 = gen_vec_reduc(gen_elem, orig, arg1, arg2, mask, 16);
  Inst *res32 = gen_vec_reduc(gen_elem, orig, arg1, arg2, mask, 32);
  Inst *res64 = gen_vec_reduc(gen_elem, orig, arg1, arg2, mask, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

Inst *Parser::gen_vmerge(Inst *orig, Inst *arg1, Inst *arg2, Inst *arg3,
			 uint32_t elem_bitsize)
{
  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      Inst *elem3 = extract_vec_elem(arg3, 1, i);
      Inst *inst = bb->build_inst(Op::ITE, elem3, elem2, elem1);
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  return res;
}

void Parser::process_vmerge()
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5);
  get_comma(6);
  Inst *arg3 = get_vreg_value(7);
  get_end_of_line(8);

  Inst *res8 = gen_vmerge(orig, arg1, arg2, arg3, 8);
  Inst *res16 = gen_vmerge(orig, arg1, arg2, arg3, 16);
  Inst *res32 = gen_vmerge(orig, arg1, arg2, arg3, 32);
  Inst *res64 = gen_vmerge(orig, arg1, arg2, arg3, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vmerge_vi()
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  unsigned __int128 arg2 = get_hex_or_integer(5);
  get_comma(6);
  Inst *arg3 = get_vreg_value(7);
  get_end_of_line(8);

  Inst *imm8 = bb->value_inst(arg2, 8);
  Inst *imm16 = bb->value_inst(arg2, 16);
  Inst *imm32 = bb->value_inst(arg2, 32);
  Inst *imm64 = bb->value_inst(arg2, 64);
  Inst *res8 = gen_vmerge(orig, arg1, imm8, arg3, 8);
  Inst *res16 = gen_vmerge(orig, arg1, imm16, arg3, 16);
  Inst *res32 = gen_vmerge(orig, arg1, imm32, arg3, 32);
  Inst *res64 = gen_vmerge(orig, arg1, imm64, arg3, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vmerge_vx()
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_comma(6);
  Inst *arg3 = get_vreg_value(7);
  get_end_of_line(8);

  Inst *res8 = gen_vmerge(orig, arg1, change_prec(arg2, 8), arg3, 8);
  Inst *res16 = gen_vmerge(orig, arg1, change_prec(arg2, 16), arg3, 16);
  Inst *res32 = gen_vmerge(orig, arg1, change_prec(arg2, 32), arg3, 32);
  Inst *res64 = gen_vmerge(orig, arg1, change_prec(arg2, 64), arg3, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

Inst *Parser::gen_vec_cmp(Cond_code ccode, Inst *orig, Inst *arg1, Inst *arg2,
			  uint32_t elem_bitsize)
{
  Op op;
  bool inv = false;
  bool swap = false;
  switch (ccode)
    {
    case Cond_code::EQ:
      op = Op::EQ;
      break;
    case Cond_code::NE:
      op = Op::EQ;
      inv = true;
      break;
    case Cond_code::SLT:
      op = Op::SLT;
      break;
    case Cond_code::ULT:
      op = Op::ULT;
      break;
    case Cond_code::SLE:
      op = Op::SLT;
      swap = true;
      inv = true;
      break;
    case Cond_code::ULE:
      op = Op::ULT;
      swap = true;
      inv = true;
      break;
    case Cond_code::SGT:
      op = Op::SLT;
      swap = true;
      break;
    case Cond_code::UGT:
      op = Op::ULT;
      swap = true;
      break;
    case Cond_code::SGE:
      op = Op::SLT;
      inv = true;
      break;
    case Cond_code::UGE:
      op = Op::ULT;
      inv = true;
      break;
    case Cond_code::FEQ:
      op = Op::FEQ;
      break;
    case Cond_code::FNE:
      op = Op::FNE;
      break;
    case Cond_code::FLT:
      op = Op::FLT;
      break;
    case Cond_code::FLE:
      op = Op::FLE;
      break;
    case Cond_code::FGT:
      op = Op::FLT;
      swap = true;
      break;
    case Cond_code::FGE:
      op = Op::FLE;
      swap = true;
      break;
    default:
      throw Not_implemented("gen_vec_cmp: Invalid ccode");
    }

  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *elem1 = extract_vec_elem(arg1, elem_bitsize, i);
      Inst *elem2;
      if (arg2->bitsize == elem_bitsize)
	elem2 = arg2;
      else
	elem2 = extract_vec_elem(arg2, elem_bitsize, i);
      if (swap)
	std::swap(elem1, elem2);
      Inst *inst = bb->build_inst(op, elem1, elem2);
      if (inv)
	inst = bb->build_inst(Op::NOT, inst);
      Inst *orig_elem = extract_vec_elem(orig, 1, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  Inst *tail = bb->build_inst(Op::EXTRACT, orig, orig->bitsize - 1, nof_elem);
  return bb->build_inst(Op::CONCAT, tail, res);
}

void Parser::process_vec_cmp(Cond_code ccode)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_vreg_value(5);
  get_end_of_line(6);

  Inst *res8 = gen_vec_cmp(ccode, orig, arg1, arg2, 8);
  Inst *res16 = gen_vec_cmp(ccode, orig, arg1, arg2, 16);
  Inst *res32 = gen_vec_cmp(ccode, orig, arg1, arg2, 32);
  Inst *res64 = gen_vec_cmp(ccode, orig, arg1, arg2, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_cmp_vi(Cond_code ccode)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  unsigned __int128 arg2 = get_hex_or_integer(5);
  get_end_of_line(6);

  Inst *imm8 = bb->value_inst(arg2, 8);
  Inst *imm16 = bb->value_inst(arg2, 16);
  Inst *imm32 = bb->value_inst(arg2, 32);
  Inst *imm64 = bb->value_inst(arg2, 64);
  Inst *res8 = gen_vec_cmp(ccode, orig, arg1, imm8, 8);
  Inst *res16 = gen_vec_cmp(ccode, orig, arg1, imm16, 16);
  Inst *res32 = gen_vec_cmp(ccode, orig, arg1, imm32, 32);
  Inst *res64 = gen_vec_cmp(ccode, orig, arg1, imm64, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vec_cmp_vx(Cond_code ccode)
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_comma(4);
  Inst *arg2 = get_reg_value(5);
  get_end_of_line(6);

  Inst *res8 = gen_vec_cmp(ccode, orig, arg1, change_prec(arg2, 8), 8);
  Inst *res16 = gen_vec_cmp(ccode, orig, arg1, change_prec(arg2, 16), 16);
  Inst *res32 = gen_vec_cmp(ccode, orig, arg1, change_prec(arg2, 32), 32);
  Inst *res64 = gen_vec_cmp(ccode, orig, arg1, change_prec(arg2, 64), 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

Inst *Parser::gen_vid(Inst *orig, uint32_t elem_bitsize)
{
  Inst *res = nullptr;
  Inst *vl = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vl]);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 0; i < nof_elem; i++)
    {
      Inst *inst = bb->value_inst(i, elem_bitsize);
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      Inst *cmp = bb->build_inst(Op::ULT, bb->value_inst(i, vl->bitsize), vl);
      inst = bb->build_inst(Op::ITE, cmp, inst, orig_elem);
      if (res)
	res = bb->build_inst(Op::CONCAT, inst, res);
      else
	res = inst;
    }
  return res;
}

void Parser::process_vid()
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_end_of_line(2);

  Inst *res8 = gen_vid(orig, 8);
  Inst *res16 = gen_vid(orig, 16);
  Inst *res32 = gen_vid(orig, 32);
  Inst *res64 = gen_vid(orig, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

void Parser::process_vmv_xs()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_vreg_value(3);
  get_end_of_line(4);

  Inst *res8 = change_prec(bb->build_trunc(arg1, 8), dest->bitsize);
  Inst *res16 = change_prec(bb->build_trunc(arg1, 16), dest->bitsize);
  Inst *res32 = change_prec(bb->build_trunc(arg1, 32), dest->bitsize);
  Inst *res64 = change_prec(bb->build_trunc(arg1, 64), dest->bitsize);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
}

Inst *Parser::gen_vmv_sx(Inst *orig, Inst *arg1, uint32_t elem_bitsize)
{
  Inst *res = change_prec(arg1, elem_bitsize);
  uint32_t nof_elem = rstate->vreg_bitsize / elem_bitsize;
  for (uint32_t i = 1; i < nof_elem; i++)
    {
      Inst *orig_elem = extract_vec_elem(orig, elem_bitsize, i);
      res = bb->build_inst(Op::CONCAT, orig_elem, res);
    }
  return res;
}

void Parser::process_vmv_sx()
{
  Inst *dest = get_vreg(1);
  Inst *orig = get_vreg_value(1);
  get_comma(2);
  Inst *arg1 = get_reg_value(3);
  get_end_of_line(4);

  Inst *res8 = gen_vmv_sx(orig, arg1, 8);
  Inst *res16 = gen_vmv_sx(orig, arg1, 16);
  Inst *res32 = gen_vmv_sx(orig, arg1, 32);
  Inst *res64 = gen_vmv_sx(orig, arg1, 64);
  Inst *vsew = bb->build_inst(Op::READ, rstate->registers[RiscvRegIdx::vsew]);
  Inst *cmp8 = bb->build_inst(Op::EQ, vsew, bb->value_inst(0, 3));
  Inst *cmp16 = bb->build_inst(Op::EQ, vsew, bb->value_inst(1, 3));
  Inst *cmp32 = bb->build_inst(Op::EQ, vsew, bb->value_inst(2, 3));
  Inst *res = bb->build_inst(Op::ITE, cmp32, res32, res64);
  res = bb->build_inst(Op::ITE, cmp16, res16, res);
  res = bb->build_inst(Op::ITE, cmp8, res8, res);
  bb->build_inst(Op::WRITE, dest, res);
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
  else if (name == "add" || name == "addw" || name == "addi" || name == "addiw")
    process_ibinary(name, Op::ADD);
  else if (name == "mul" || name == "mulw")
    process_ibinary(name, Op::MUL);
  else if (name == "mulh" || name == "mulhu" || name == "mulhsu")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      Op op1 = Op::SEXT;
      Op op2 = Op::SEXT;
      if (name == "mulhsu")
	op2 = Op::ZEXT;
      else if (name == "mulhu")
	{
	  op1 = Op::ZEXT;
	  op2 = Op::ZEXT;
	}
      arg1 = bb->build_inst(op1, arg1, 2 * reg_bitsize);
      arg2 = bb->build_inst(op2, arg2, 2 * reg_bitsize);
      Inst *res = bb->build_inst(Op::MUL, arg1, arg2);
      res = bb->build_inst(Op::EXTRACT, res, 2 * reg_bitsize - 1, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "div" || name == "divw")
    process_ibinary(name, Op::SDIV);
  else if (name == "divu" || name == "divuw")
    process_ibinary(name, Op::UDIV);
  else if (name == "rem" || name == "remw")
    process_ibinary(name, Op::SREM);
  else if (name == "remu" || name == "remuw")
    process_ibinary(name, Op::UREM);
  else if (name == "slt" || name == "sltw" || name == "slti" || name == "sltiw")
    process_icmp(name, Op::SLT);
  else if (name == "sltu" || name == "sltuw"
	   || name == "sltiu" || name == "sltiuw")
    process_icmp(name, Op::ULT);
  else if (name == "sgt" || name == "sgtw")
    process_icmp(name, Op::SLT, true);
  else if (name == "sgtu" || name == "sgtuw")
    process_icmp(name, Op::ULT, true);
  else if (name == "seqz" || name == "seqzw")
    {
      // Pseudo instruction.
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      if (name == "seqzw")
	{
	  arg1 = bb->build_trunc(arg1, 32);
	}
      Inst *zero = bb->value_inst(0, arg1->bitsize);
      Inst *res = bb->build_inst(Op::EQ, arg1, zero);
      res = bb->build_inst(Op::ZEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "snez" || name == "snezw")
    {
      // Pseudo instruction.
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      if (name == "snezw")
	{
	  arg1 = bb->build_trunc(arg1, 32);
	}
      Inst *zero = bb->value_inst(0, arg1->bitsize);
      Inst *res = bb->build_inst(Op::NE, arg1, zero);
      res = bb->build_inst(Op::ZEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "and" || name == "andw"
	   || name == "andi" || name == "andiw")
    process_ibinary(name, Op::AND);
  else if (name == "or" || name == "orw" || name == "ori" || name == "oriw")
    process_ibinary(name, Op::OR);
  else if (name == "xor" || name == "xorw" || name == "xori" || name == "xoriw")
    process_ibinary(name, Op::XOR);
  else if (name == "sll" || name == "sllw" || name == "slli" || name == "slliw")
    process_ishift(name, Op::SHL);
  else if (name == "srl" || name == "srlw" || name == "srli" || name == "srliw")
    process_ishift(name, Op::LSHR);
  else if (name == "sra" || name == "sraw" || name == "srai" || name == "sraiw")
    process_ishift(name, Op::ASHR);
  else if (name == "sub" || name == "subw")
    process_ibinary(name, Op::SUB);
  else if (name == "neg" || name == "negw")
    process_iunary(name, Op::NEG);
  else if (name == "sext.w")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      Inst *res = bb->build_trunc(arg1, 32);
      res = bb->build_inst(Op::SEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "not")
    process_iunary(name, Op::NOT);
  else if (name == "mv")
    process_iunary(name, Op::MOV);
  else if (name == "li")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      // TODO: Use a correct wrapper.
      //       Sort of get_imm(3); but with correct size.
      unsigned __int128 value = get_hex_or_integer(3);
      Inst *arg1 = bb->value_inst(value, reg_bitsize);
      get_end_of_line(4);

      bb->build_inst(Op::WRITE, dest, arg1);
    }
  else if (name == "lui")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *res = get_hi(3);
      get_end_of_line(4);

      if (reg_bitsize > 32)
	res = bb->build_inst(Op::SEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "call")
    process_call();
  else if (name == "tail")
    process_tail();
  else if (name == "ld" && reg_bitsize == 64)
    process_load(8);
  else if (name == "lw")
    process_load(4);
  else if (name == "lwu")
    process_load(4, LStype::unsigned_ls);
  else if (name == "lh")
    process_load(2);
  else if (name == "lhu")
    process_load(2, LStype::unsigned_ls);
  else if (name == "lb")
    process_load(1);
  else if (name == "lbu")
    process_load(1, LStype::unsigned_ls);
  else if (name == "sd" && reg_bitsize == 64)
    process_store(8);
  else if (name == "sw")
    process_store(4);
  else if (name == "sh")
    process_store(2);
  else if (name == "sb")
    process_store(1);
  else if (name == "beq")
    process_cond_branch(Op::EQ);
  else if (name == "bne")
    process_cond_branch(Op::NE);
  else if (name == "ble")
    process_cond_branch(Op::SLE);
  else if (name == "bleu")
    process_cond_branch(Op::ULE);
  else if (name == "blt")
    process_cond_branch(Op::SLT);
  else if (name == "bltu")
    process_cond_branch(Op::ULT);
  else if (name == "bge")
    process_cond_branch(Op::SLE, true);
  else if (name == "bgeu")
    process_cond_branch(Op::ULE, true);
  else if (name == "bgt")
    process_cond_branch(Op::SLT, true);
  else if (name == "bgtu")
    process_cond_branch(Op::ULT, true);
  else if (name == "j")
    {
      Basic_block *dest_bb = get_bb(1);
      get_end_of_line(2);

      bb->build_br_inst(dest_bb);
      bb = nullptr;
    }
  else if (name == "ebreak")
    {
      bb->build_inst(Op::UB, bb->value_inst(1, 1));
      bb->build_br_inst(rstate->exit_bb);
      bb = nullptr;
    }
  else if (name == "ret" || name == "jr")
    {
      // TODO: jr and ret are pseudoinstructions. Verify that they
      // jump to the correct location.
      bb->build_br_inst(rstate->exit_bb);
      bb = nullptr;
    }
  else if (name == "nop")
    ;
  else if (name == "fld")
    process_load(8, LStype::float_ls);
  else if (name == "flw")
    process_load(4, LStype::float_ls);
  else if (name == "fsd")
    process_store(8, LStype::float_ls);
  else if (name == "fsw")
    process_store(4, LStype::float_ls);
  else if (name == "fabs.s" || name == "fabs.d")
    process_funary(name, Op::FABS);
  else if (name == "fmv.s" || name == "fmv.d")
    process_funary(name, Op::MOV);
  else if (name == "fneg.s" || name == "fneg.d")
    process_funary(name, Op::FNEG);
  else if (name == "fadd.s" || name == "fadd.d")
    process_fbinary(name, Op::FADD);
  else if (name == "fsub.s" || name == "fsub.d")
    process_fbinary(name, Op::FSUB);
  else if (name == "fmul.s" || name == "fmul.d")
    process_fbinary(name, Op::FMUL);
  else if (name == "fdiv.s" || name == "fdiv.d")
    process_fbinary(name, Op::FDIV);
  else if (name == "feq.s" || name == "feq.d")
    process_fcmp(name, Op::FEQ);
  else if (name == "flt.s" || name == "flt.d")
    process_fcmp(name, Op::FLT);
  else if (name == "fle.s" || name == "fle.d")
    process_fcmp(name, Op::FLE);
  else if (name == "fgt.s" || name == "fgt.d")
    process_fcmp(name, Op::FLT, true);
  else if (name == "fge.s" || name == "fge.d")
    process_fcmp(name, Op::FLE, true);
  else if (name == "fsgnj.s")
    {
      Inst *dest = get_freg(1);
      get_comma(2);
      Inst *arg1 = get_freg_value(3);
      get_comma(4);
      Inst *arg2 = get_freg_value(5);
      get_end_of_line(6);

      arg1 = bb->build_trunc(arg1, 32);
      Inst *signbit = bb->build_extract_bit(arg2, 31);
      Inst *res = bb->build_trunc(arg1, 31);
      res = bb->build_inst(Op::CONCAT, signbit, res);
      Inst *m1 = bb->value_m1_inst(32);
      res = bb->build_inst(Op::CONCAT, m1, res);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "fsgnj.d")
    {
      Inst *dest = get_freg(1);
      get_comma(2);
      Inst *arg1 = get_freg_value(3);
      get_comma(4);
      Inst *arg2 = get_freg_value(5);
      get_end_of_line(6);

      Inst *signbit = bb->build_extract_bit(arg2, arg2->bitsize - 1);
      Inst *res = bb->build_trunc(arg1, arg1->bitsize - 1);
      res = bb->build_inst(Op::CONCAT, signbit, res);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "fcvt.s.w")
    process_fcvt_i2f(32, 32, false);
  else if (name == "fcvt.s.wu")
    process_fcvt_i2f(32, 32, true);
  else if (name == "fcvt.d.w")
    process_fcvt_i2f(32, 64, false);
  else if (name == "fcvt.d.wu")
    process_fcvt_i2f(32, 64, true);
  else if (name == "fcvt.s.l")
    process_fcvt_i2f(64, 32, false);
  else if (name == "fcvt.s.lu")
    process_fcvt_i2f(64, 32, true);
  else if (name == "fcvt.d.l")
    process_fcvt_i2f(64, 64, false);
  else if (name == "fcvt.d.lu")
    process_fcvt_i2f(64, 64, true);
  else if (name == "fcvt.w.s")
    process_fcvt_f2i(32, 32, false);
  else if (name == "fcvt.wu.s")
    process_fcvt_f2i(32, 32, true);
  else if (name == "fcvt.l.s")
    process_fcvt_f2i(32, 64, false);
  else if (name == "fcvt.lu.s")
    process_fcvt_f2i(32, 64, true);
  else if (name == "fcvt.w.d")
    process_fcvt_f2i(64, 32, false);
  else if (name == "fcvt.wu.d")
    process_fcvt_f2i(64, 32, true);
  else if (name == "fcvt.l.d")
    process_fcvt_f2i(64, 64, false);
  else if (name == "fcvt.lu.d")
    process_fcvt_f2i(64, 64, true);
  else if (name == "fcvt.d.s")
    process_fcvt_f2f(32, 64);
  else if (name == "fcvt.s.d")
    process_fcvt_f2f(64, 32);
  else if (name == "fmin.s")
    process_fmin_fmax(32, true);
  else if (name == "fmin.d")
    process_fmin_fmax(64, true);
  else if (name == "fmax.s")
    process_fmin_fmax(32, false);
  else if (name == "fmax.d")
    process_fmin_fmax(64, false);
  else if (name == "fmv.x.s" || name == "fmv.x.w")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_freg_value(3);
      get_end_of_line(4);

      Inst *res = bb->build_trunc(arg1, 32);
      if (reg_bitsize > res->bitsize)
	res = bb->build_inst(Op::SEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "fmv.s.x" || name == "fmv.w.x")
    {
      Inst *dest = get_freg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      if (arg1->bitsize > 32)
	arg1 = bb->build_trunc(arg1, 32);
      Inst *m1 = bb->value_m1_inst(32);
      Inst *res = bb->build_inst(Op::CONCAT, m1, arg1);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "fmv.x.d")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_freg_value(3);
      get_end_of_line(4);

      bb->build_inst(Op::WRITE, dest, arg1);
    }
  else if (name == "fmv.d.x")
    {
      Inst *dest = get_freg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      bb->build_inst(Op::WRITE, dest, arg1);
    }

  // Zba
  else if (name == "add.uw")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      arg1 = bb->build_inst(Op::ZEXT, bb->build_trunc(arg1, 32), reg_bitsize);
      Inst *res = bb->build_inst(Op::ADD, arg1, arg2);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sh1add")
    process_zba_sh_add(1, false);
  else if (name == "sh1add.uw")
    process_zba_sh_add(1, true);
  else if (name == "sh2add")
    process_zba_sh_add(2, false);
  else if (name == "sh2add.uw")
    process_zba_sh_add(2, true);
  else if (name == "sh3add")
    process_zba_sh_add(3, false);
  else if (name == "sh3add.uw")
    process_zba_sh_add(3, true);
  else if (name == "slli.uw")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_imm(5);
      get_end_of_line(6);

      arg1 = bb->build_inst(Op::ZEXT, bb->build_trunc(arg1, 32), reg_bitsize);
      Inst *res = bb->build_inst(Op::SHL, arg1, arg2);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "zext.w")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      arg1 = bb->build_trunc(arg1, 32);
      Inst *res = bb->build_inst(Op::ZEXT, arg1, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }

  // Zbb
  else if (name == "andn")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *res = bb->build_inst(Op::AND, arg1, bb->build_inst(Op::NOT, arg2));
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "orn")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *res = bb->build_inst(Op::OR, arg1, bb->build_inst(Op::NOT, arg2));
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "xnor")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *res = bb->build_inst(Op::NOT, bb->build_inst(Op::XOR, arg1, arg2));
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "clz" || name == "clzw")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      bool has_w_suffix = name[name.length() - 1] == 'w';
      if (has_w_suffix)
	arg1 = bb->build_trunc(arg1, 32);
      Inst *res = gen_clz(bb, arg1);
      if (reg_bitsize == 64 && res->bitsize < reg_bitsize)
	res = bb->build_inst(Op::ZEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "ctz" || name == "ctzw")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      bool has_w_suffix = name[name.length() - 1] == 'w';
      if (has_w_suffix)
	arg1 = bb->build_trunc(arg1, 32);
      Inst *res = gen_ctz(bb, arg1);
      if (reg_bitsize == 64 && res->bitsize < reg_bitsize)
	res = bb->build_inst(Op::ZEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "cpop" || name == "cpopw")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      bool has_w_suffix = name[name.length() - 1] == 'w';
      if (has_w_suffix)
	arg1 = bb->build_trunc(arg1, 32);
      Inst *res = gen_popcount(arg1);
      if (reg_bitsize == 64 && res->bitsize < reg_bitsize)
	res = bb->build_inst(Op::ZEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "max")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *cmp = bb->build_inst(Op::SLE, arg2, arg1);
      Inst *res = bb->build_inst(Op::ITE, cmp, arg1, arg2);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "maxu")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *cmp = bb->build_inst(Op::ULE, arg2, arg1);
      Inst *res = bb->build_inst(Op::ITE, cmp, arg1, arg2);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "min")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *cmp = bb->build_inst(Op::SLT, arg1, arg2);
      Inst *res = bb->build_inst(Op::ITE, cmp, arg1, arg2);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "minu")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *cmp = bb->build_inst(Op::ULT, arg1, arg2);
      Inst *res = bb->build_inst(Op::ITE, cmp, arg1, arg2);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sext.b")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      arg1 = bb->build_trunc(arg1, 8);
      Inst *res = bb->build_inst(Op::SEXT, arg1, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sext.h")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      arg1 = bb->build_trunc(arg1, 16);
      Inst *res = bb->build_inst(Op::SEXT, arg1, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "zext.h")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      arg1 = bb->build_trunc(arg1, 16);
      Inst *res = bb->build_inst(Op::ZEXT, arg1, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "rol" || name == "rolw")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2 = get_reg_value(5);
      get_end_of_line(6);

      bool has_w_suffix = name[name.length() - 1] == 'w';
      if (has_w_suffix)
	{
	  arg1 = bb->build_trunc(arg1, 32);
	  arg2 = bb->build_trunc(arg2, 32);
	}
      if (arg1->bitsize == 32)
	arg2 = bb->build_trunc(arg2, 5);
      else
	arg2 = bb->build_trunc(arg2, 6);
      arg2 = bb->build_inst(Op::ZEXT, arg2, arg1->bitsize);
      Inst *shl = bb->build_inst(Op::SHL, arg1, arg2);
      Inst *bs = bb->value_inst(arg1->bitsize, arg1->bitsize);
      Inst *shift = bb->build_inst(Op::SUB, bs, arg2);
      Inst *lshr = bb->build_inst(Op::LSHR, arg1, shift);
      Inst *res = bb->build_inst(Op::OR, shl, lshr);
      if (has_w_suffix)
	res = bb->build_inst(Op::SEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "ror" || name == "rori" || name == "rorw" || name == "roriw")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2;
      bool is_imm =
	name[name.length() - 1] == 'i'
	|| (name[name.length() - 2] == 'i' && name[name.length() - 1] == 'w');
      if (is_imm)
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      bool has_w_suffix = name[name.length() - 1] == 'w';
      if (has_w_suffix)
	{
	  arg1 = bb->build_trunc(arg1, 32);
	  arg2 = bb->build_trunc(arg2, 32);
	}
      if (arg1->bitsize == 32)
	arg2 = bb->build_trunc(arg2, 5);
      else
	arg2 = bb->build_trunc(arg2, 6);
      arg2 = bb->build_inst(Op::ZEXT, arg2, arg1->bitsize);
      Inst *lshr = bb->build_inst(Op::LSHR, arg1, arg2);
      Inst *bs = bb->value_inst(arg1->bitsize, arg1->bitsize);
      Inst *shift = bb->build_inst(Op::SUB, bs, arg2);
      Inst *shl = bb->build_inst(Op::SHL, arg1, shift);
      Inst *res = bb->build_inst(Op::OR, lshr, shl);
      if (has_w_suffix)
	res = bb->build_inst(Op::SEXT, res, reg_bitsize);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "orc.b")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      Inst *zero = bb->value_inst(0, 8);
      Inst *res = nullptr;
      for (unsigned i = 0; i < reg_bitsize / 8; i++)
	{
	  Inst *byte = bb->build_inst(Op::EXTRACT, arg1, i * 8 + 7, i * 8);
	  Inst *cmp = bb->build_inst(Op::NE, byte, zero);
	  byte = bb->build_inst(Op::SEXT, cmp, 8);
	  if (res)
	    res = bb->build_inst(Op::CONCAT, byte, res);
	  else
	    res = byte;
	}
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "rev8")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_end_of_line(4);

      Inst *res = gen_bswap(bb, arg1);
      bb->build_inst(Op::WRITE, dest, res);
    }

  // Zbs
  else if (name == "bclr" || name == "bclri")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2;
      bool is_imm = name[name.length() - 1] == 'i';
      if (is_imm)
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *mask = bb->value_inst(reg_bitsize - 1, reg_bitsize);
      arg2 = bb->build_inst(Op::AND, arg2, mask);
      Inst *res = bb->build_inst(Op::SHL, bb->value_inst(1, reg_bitsize), arg2);
      res = bb->build_inst(Op::AND, arg1, bb->build_inst(Op::NOT, res));
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "bext" || name == "bexti")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2;
      bool is_imm = name[name.length() - 1] == 'i';
      if (is_imm)
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *mask = bb->value_inst(reg_bitsize - 1, reg_bitsize);
      arg2 = bb->build_inst(Op::AND, arg2, mask);
      Inst *res = bb->build_inst(Op::LSHR, arg1, arg2);
      res = bb->build_inst(Op::AND, res, bb->value_inst(1, reg_bitsize));
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "binv" || name == "binvi")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2;
      bool is_imm = name[name.length() - 1] == 'i';
      if (is_imm)
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *mask = bb->value_inst(reg_bitsize - 1, reg_bitsize);
      arg2 = bb->build_inst(Op::AND, arg2, mask);
      Inst *res = bb->build_inst(Op::SHL, bb->value_inst(1, reg_bitsize), arg2);
      res = bb->build_inst(Op::XOR, arg1, res);
      bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "bset" || name == "bseti")
    {
      Inst *dest = get_reg(1);
      get_comma(2);
      Inst *arg1 = get_reg_value(3);
      get_comma(4);
      Inst *arg2;
      bool is_imm = name[name.length() - 1] == 'i';
      if (is_imm)
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      Inst *mask = bb->value_inst(reg_bitsize - 1, reg_bitsize);
      arg2 = bb->build_inst(Op::AND, arg2, mask);
      Inst *res = bb->build_inst(Op::SHL, bb->value_inst(1, reg_bitsize), arg2);
      res = bb->build_inst(Op::OR, arg1, res);
      bb->build_inst(Op::WRITE, dest, res);
    }

  //
  // Vector instructions
  //

  // Configuration-setting instructions
  else if (name == "vsetivli")
    process_vsetvli(true);
  else if (name == "vsetvli")
    process_vsetvli(false);

  // Loads and stores
  else if (name == "vle8.v")
    process_vle(8);
  else if (name == "vle16.v")
    process_vle(16);
  else if (name == "vle32.v")
    process_vle(32);
  else if (name == "vle64.v")
    process_vle(64);
  else if (name == "vse8.v")
    process_vse(8);
  else if (name == "vse16.v")
    process_vse(16);
  else if (name == "vse32.v")
    process_vse(32);
  else if (name == "vse64.v")
    process_vse(64);

  // Integer arithmetic - add and subtract
  else if (name == "vadd.vv")
    process_vec_binary(Op::ADD);
  else if (name == "vadd.vx")
    process_vec_binary_vx(Op::ADD);
  else if (name == "vadd.vi")
    process_vec_binary_vi(Op::ADD);
  else if (name == "vsub.vv")
    process_vec_binary(Op::SUB);
  else if (name == "vsub.vx")
    process_vec_binary_vx(Op::SUB);
  else if (name == "vrsub.vx")
    process_vec_binary_vx(gen_rsub);
  else if (name == "vrsub.vi")
    process_vec_binary_vi(gen_rsub);
  else if (name == "vneg.v")
    process_vec_unary(Op::NEG);

  // Integer arithmetic - widening add and subtract

  // Integer arithmetic - extension

  // Integer arithmetic - bitwise logical
  else if (name == "vand.vv")
    process_vec_binary(Op::AND);
  else if (name == "vand.vx")
    process_vec_binary_vx(Op::AND);
  else if (name == "vand.vi")
    process_vec_binary_vi(Op::AND);
  else if (name == "vor.vv")
    process_vec_binary(Op::OR);
  else if (name == "vor.vx")
    process_vec_binary_vx(Op::OR);
  else if (name == "vor.vi")
    process_vec_binary_vi(Op::OR);
  else if (name == "vxor.vv")
    process_vec_binary(Op::XOR);
  else if (name == "vxor.vx")
    process_vec_binary_vx(Op::XOR);
  else if (name == "vxor.vi")
    process_vec_binary_vi(Op::XOR);
  else if (name == "vnot.v")
    process_vec_unary(Op::NOT);

  // Integer arithmetic - shift instructions
  else if (name == "vsll.vv")
    process_vec_binary(Op::SHL);
  else if (name == "vsll.vx")
    process_vec_binary_vx(Op::SHL);
  else if (name == "vsll.vi")
    process_vec_binary_vi(Op::SHL);
  else if (name == "vsrl.vv")
    process_vec_binary(Op::LSHR);
  else if (name == "vsrl.vx")
    process_vec_binary_vx(Op::LSHR);
  else if (name == "vsrl.vi")
    process_vec_binary_vi(Op::LSHR);
  else if (name == "vsra.vv")
    process_vec_binary(Op::ASHR);
  else if (name == "vsra.vx")
    process_vec_binary_vx(Op::ASHR);
  else if (name == "vsra.vi")
    process_vec_binary_vi(Op::ASHR);

  // Integer arithmetic - narrowing shift instructions

  // Integer arithmetic - compare
  else if (name == "vmseq.vv")
    process_vec_cmp(Cond_code::EQ);
  else if (name == "vmseq.vx")
    process_vec_cmp_vx(Cond_code::EQ);
  else if (name == "vmseq.vi")
    process_vec_cmp_vi(Cond_code::EQ);
  else if (name == "vmsne.vv")
    process_vec_cmp(Cond_code::NE);
  else if (name == "vmsne.vx")
    process_vec_cmp_vx(Cond_code::NE);
  else if (name == "vmsne.vi")
    process_vec_cmp_vi(Cond_code::NE);
  else if (name == "vmsltu.vv")
    process_vec_cmp(Cond_code::ULT);
  else if (name == "vmsltu.vx")
    process_vec_cmp_vx(Cond_code::ULT);
  else if (name == "vmsltu.vi")
    process_vec_cmp_vi(Cond_code::ULT);
  else if (name == "vmslt.vv")
    process_vec_cmp(Cond_code::SLT);
  else if (name == "vmslt.vx")
    process_vec_cmp_vx(Cond_code::SLT);
  else if (name == "vmslt.vi")
    process_vec_cmp_vi(Cond_code::SLT);
  else if (name == "vmsleu.vv")
    process_vec_cmp(Cond_code::ULE);
  else if (name == "vmsleu.vx")
    process_vec_cmp_vx(Cond_code::ULE);
  else if (name == "vmsleu.vi")
    process_vec_cmp_vi(Cond_code::ULE);
  else if (name == "vmsle.vv")
    process_vec_cmp(Cond_code::SLE);
  else if (name == "vmsle.vx")
    process_vec_cmp_vx(Cond_code::SLE);
  else if (name == "vmsle.vi")
    process_vec_cmp_vi(Cond_code::SLE);
  else if (name == "vmsgtu.vv")
    process_vec_cmp(Cond_code::UGT);
  else if (name == "vmsgtu.vx")
    process_vec_cmp_vx(Cond_code::UGT);
  else if (name == "vmsgtu.vi")
    process_vec_cmp_vi(Cond_code::UGT);
  else if (name == "vmsgt.vv")
    process_vec_cmp(Cond_code::SGT);
  else if (name == "vmsgt.vx")
    process_vec_cmp_vx(Cond_code::SGT);
  else if (name == "vmsgt.vi")
    process_vec_cmp_vi(Cond_code::SGT);
  else if (name == "vmsgeu.vv")
    process_vec_cmp(Cond_code::UGE);
  else if (name == "vmsgeu.vx")
    process_vec_cmp_vx(Cond_code::UGE);
  else if (name == "vmsgeu.vi")
    process_vec_cmp_vi(Cond_code::UGE);
  else if (name == "vmsge.vv")
    process_vec_cmp(Cond_code::SGE);
  else if (name == "vmsge.vx")
    process_vec_cmp_vx(Cond_code::SGE);
  else if (name == "vmsge.vi")
    process_vec_cmp_vi(Cond_code::SGE);

  // Integer arithmetic - min/max
  else if (name == "vminu.vv")
    process_vec_binary(gen_umin);
  else if (name == "vminu.vx")
    process_vec_binary_vx(gen_umin);
  else if (name == "vmin.vv")
    process_vec_binary(gen_smin);
  else if (name == "vmin.vx")
    process_vec_binary_vx(gen_smin);
  else if (name == "vmaxu.vv")
    process_vec_binary(gen_umax);
  else if (name == "vmaxu.vx")
    process_vec_binary_vx(gen_umax);
  else if (name == "vmax.vv")
    process_vec_binary(gen_smax);
  else if (name == "vmax.vx")
    process_vec_binary_vx(gen_smax);

  // Integer arithmetic - multiply
  else if (name == "vmul.vv")
    process_vec_binary(Op::MUL);
  else if (name == "vmul.vx")
    process_vec_binary_vx(Op::MUL);
  else if (name == "vmulh.vv")
    process_vec_binary(gen_mulh);
  else if (name == "vmulh.vx")
    process_vec_binary_vx(gen_mulh);
  else if (name == "vmulhu.vv")
    process_vec_binary(gen_mulhu);
  else if (name == "vmulhu.vx")
    process_vec_binary_vx(gen_mulhu);
  else if (name == "vmulhsu.vv")
    process_vec_binary(gen_mulhsu);
  else if (name == "vmulhsu.vx")
    process_vec_binary_vx(gen_mulhsu);

  // Integer arithmetic - divide
  else if (name == "vdiv.vv")
    process_vec_binary(Op::SDIV);
  else if (name == "vdiv.vx")
    process_vec_binary_vx(Op::SDIV);
  else if (name == "vdivu.vv")
    process_vec_binary(Op::UDIV);
  else if (name == "vdivu.vx")
    process_vec_binary_vx(Op::UDIV);
  else if (name == "vremu.vv")
    process_vec_binary(Op::UREM);
  else if (name == "vremu.vx")
    process_vec_binary_vx(Op::UREM);
  else if (name == "vrem.vv")
    process_vec_binary(Op::SREM);
  else if (name == "vrem.vx")
    process_vec_binary_vx(Op::SREM);

  // Integer arithmetic - widening multiply

  // Integer arithmetic - multiply add

  // Integer arithmetic - widening multiply add

  // Integer arithmetic - merge
  else if (name == "vmerge.vvm")
    process_vmerge();
  else if (name == "vmerge.vxm")
    process_vmerge_vx();
  else if (name == "vmerge.vim")
    process_vmerge_vi();

  // Integer arithmetic - move
  else if (name == "vmv.v.v")
    process_vec_unary(Op::MOV);
  else if (name == "vmv.v.x")
    process_vec_unary_vx(Op::MOV);
  else if (name == "vmv.v.i")
    process_vec_unary_vi(Op::MOV);

  // Floating-point - add/subtract
  else if (name == "vfadd.vv")
    process_vec_binary(Op::FADD);
  else if (name == "vfsub.vv")
    process_vec_binary(Op::FSUB);

  // Floating-point - multiply/divide
  else if (name == "vfmul.vv")
    process_vec_binary(Op::FMUL);
  else if (name == "vfdiv.vv")
    process_vec_binary(Op::FDIV);

  // Floating-point - compare
  else if (name == "vmfeq.vv")
    process_vec_cmp(Cond_code::FEQ);
  else if (name == "vmfeq.vf")
    process_vec_cmp_vx(Cond_code::FEQ);
  else if (name == "vmfne.vv")
    process_vec_cmp(Cond_code::FNE);
  else if (name == "vmfne.vf")
    process_vec_cmp_vx(Cond_code::FNE);
  else if (name == "vmflt.vv")
    process_vec_cmp(Cond_code::FLT);
  else if (name == "vmflt.vf")
    process_vec_cmp_vx(Cond_code::FLT);
  else if (name == "vmfle.vv")
    process_vec_cmp(Cond_code::FLE);
  else if (name == "vmfle.vf")
    process_vec_cmp_vx(Cond_code::FLE);
  else if (name == "vmfgt.vv")
    process_vec_cmp(Cond_code::FGT);
  else if (name == "vmfgt.vf")
    process_vec_cmp_vx(Cond_code::FGT);
  else if (name == "vmfge.vv")
    process_vec_cmp(Cond_code::FGE);
  else if (name == "vmfge.vf")
    process_vec_cmp_vx(Cond_code::FGE);

  // Vector reduction operations
  else if (name == "vredsum.vs")
    process_vec_reduc(Op::ADD);
  else if (name == "vredmaxu.vs")
    process_vec_reduc(gen_umax);
  else if (name == "vredmax.vs")
    process_vec_reduc(gen_smax);
  else if (name == "vredminu.vs")
    process_vec_reduc(gen_umin);
  else if (name == "vredmin.vs")
    process_vec_reduc(gen_smin);
  else if (name == "vredand.vs")
    process_vec_reduc(Op::AND);
  else if (name == "vredor.vs")
    process_vec_reduc(Op::OR);
  else if (name == "vredxor.vs")
    process_vec_reduc(Op::XOR);

  // Vector mask instructions
  else if (name == "vmand.mm")
    process_vec_mask_binary(Op::AND);
  else if (name == "vmnand.mm")
    process_vec_mask_binary(gen_nand);
  else if (name == "vmandn.mm")
    process_vec_mask_binary(gen_andn);
  else if (name == "vmxor.mm")
    process_vec_mask_binary(Op::XOR);
  else if (name == "vmor.mm")
    process_vec_mask_binary(Op::OR);
  else if (name == "vmnor.mm")
    process_vec_mask_binary(gen_nor);
  else if (name == "vmorn.mm")
    process_vec_mask_binary(gen_orn);
  else if (name == "vmxnor.mm")
    process_vec_mask_binary(gen_xnor);
  else if (name == "vmclr.m")
    process_vec_mask_set(false);
  else if (name == "vmset.m")
    process_vec_mask_set(true);
  else if (name == "vmnot.m")
    process_vec_mask_unary(Op::NOT);
  else if (name == "vid.v")
    process_vid();

  // Vector permutation instructions
  else if (name == "vmv.x.s")
    process_vmv_xs();
  else if (name == "vmv.s.x")
    process_vmv_sx();

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
      else if (buf[pos] == '.' && buf[pos + 1] == 'L' )
	lex_label_or_label_def();
      else if (isalpha(buf[pos]) || buf[pos] == '_' || buf[pos] == '.')
	lex_name();
      else if (buf[pos] == '%' && buf[pos + 1] == 'l' && buf[pos + 2] == 'o')
	lex_hilo();
      else if (buf[pos] == '%' && buf[pos + 1] == 'h' && buf[pos + 2] == 'i')
	lex_hilo();
      else if (buf[pos] == ',')
	{
	  tokens.emplace_back(Lexeme::comma, pos, 1);
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
      if (cmd == ".dword"
	  || cmd == ".word"
	  || cmd == ".half"
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
	    throw Parse_error(".word value is not a number", line_number);
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
	  else if (cmd == ".half")
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
	  && buf[pos + 5] == 'l'
	  && (buf[pos + 6] == ' ' || buf[pos + 6] == '\t'))
	{
	  pos += 7;
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
  Inst *bs = rstate->entry_bb->value_inst(reg_bitsize, 32);
  zero_reg = rstate->entry_bb->build_inst(Op::REGISTER, bs);

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

	    // TODO: Do not hard code ID values.
	    int next_id = -126;
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

Function *parse_riscv(std::string const& file_name, riscv_state *state)
{
  Parser p(state);
  Function *func = p.parse(file_name);
  reverse_post_order(func);
  return func;
}

} // end namespace smtgcc
