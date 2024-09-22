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

// TODO: Check that all instructions are supported by asm. For example,
// I am not sure that the "w" version of sgt is supported...

struct parser {
  parser(riscv_state *rstate) : rstate{rstate} {}

  enum class lexeme {
    label,
    label_def,
    variable,
    name,
    integer,
    hex,
    comma,
    assign,
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

  struct token {
    lexeme kind;
    int pos;
    int size;
  };
  std::vector<token> tokens;
  std::map<std::string, std::vector<unsigned char>> sym_name2data;

  int line_number = 0;
  size_t pos;

  std::vector<char> buf;

  std::optional<std::string> parse_label_def();
  std::string parse_cmd();
  void parse_data(std::vector<unsigned char>& data);
  void skip_line();
  void skip_whitespace();
  void parse_rodata();
  Function *parse(std::string const& file_name);

  riscv_state *rstate;
  Module *module;
  uint32_t reg_bitsize;
  Function *src_func;

private:
  Function *func = nullptr;
  Basic_block *bb = nullptr;
  std::map<std::string, Basic_block *> label2bb;
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

  std::string token_string(const token& tok);

  std::string get_name(const char *p);
  uint32_t get_u32(const char *p);
  unsigned __int128 get_hex(const char *p);

  unsigned __int128 get_hex_or_integer(unsigned idx);
  Inst *get_reg(unsigned idx);
  Inst *get_freg(unsigned idx);
  Inst *get_hilo_addr(const token& tok);
  Inst *get_hi(unsigned idx);
  Inst *get_lo(unsigned idx);
  Inst *get_imm(unsigned idx);
  Inst *get_reg_value(unsigned idx);
  Inst *get_freg_value(unsigned idx);
  Basic_block *get_bb(unsigned idx);
  Basic_block *get_bb_def(unsigned idx);
  std::string get_name(unsigned idx);
  void get_comma(unsigned idx);
  void get_left_paren(unsigned idx);
  void get_right_paren(unsigned idx);
  void get_end_of_line(unsigned idx);
  void process_cond_branch(Op op);
  Inst *gen_bswap(Inst *arg);
  Inst *gen_clz(Inst *arg);
  Inst *gen_ctz(Inst *arg);
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
  void process_funary(std::string name, Op op);
  void process_fbinary(std::string name, Op op);
  void process_fcmp(std::string name, Op op);
  void process_iunary(std::string name, Op op);
  void process_ibinary(std::string name, Op op);
  void process_icmp(std::string name, Op op);
  void process_ishift(std::string name, Op op);

  void parse_function();

  void skip_space_and_comments();
};

void parser::skip_space_and_comments()
{
  while (isspace(buf[pos]))
    pos++;
  if (buf[pos] == ';')
    {
      while (buf[pos])
	pos++;
    }
}

void parser::lex_label_or_label_def(void)
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
      tokens.emplace_back(lexeme::label_def, start_pos, pos - start_pos);
    }
  else
    tokens.emplace_back(lexeme::label, start_pos, pos - start_pos);
}

void parser::lex_hex(void)
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
  tokens.emplace_back(lexeme::hex, start_pos, pos - start_pos);
}

void parser::lex_integer(void)
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
  tokens.emplace_back(lexeme::integer, start_pos, pos - start_pos);
}

void parser::lex_hex_or_integer(void)
{
  assert(isdigit(buf[pos]) || buf[pos] == '-');
  if (buf[pos] == '0' && (buf[pos + 1] == 'x' || buf[pos + 1] == 'X'))
    lex_hex();
  else
    lex_integer();
}

void parser::lex_name(void)
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
  tokens.emplace_back(lexeme::name, start_pos, pos - start_pos);
}

void parser::lex_hilo(void)
{
  int start_pos = pos;
  bool is_lo = (buf[pos] == '%' && buf[pos + 1] == 'l' && buf[pos + 2] == 'o');
  lexeme op = is_lo ? lexeme::lo : lexeme::hi;
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

std::string parser::token_string(const token& tok)
{
  return std::string(&buf[tok.pos], tok.size);
}

std::string parser::get_name(const char *p)
{
  std::string name;
  while (isalnum(*p) || *p == '_' || *p == '-' || *p == '.')
    name.push_back(*p++);
  return name;
}

std::string parser::get_name(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != lexeme::name)
    throw Parse_error("expected a name after " + token_string(tokens[idx - 1]),
		      line_number);
  return get_name(&buf[tokens[idx].pos]);
}

uint32_t parser::get_u32(const char *p)
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

unsigned __int128 parser::get_hex(const char *p)
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

unsigned __int128 parser::get_hex_or_integer(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::hex && tokens[idx].kind != lexeme::integer)
    throw Parse_error("expected a hexadecimal or decimal integer instead of "
		      + token_string(tokens[idx]), line_number);

  int pos = tokens[idx].pos;
  if (buf[pos] == '-')
    pos++;
  unsigned __int128 val;
  if (tokens[idx].kind == lexeme::integer)
    val = get_u32(&buf[pos]);
  else
    val = get_hex(&buf[pos]);
  if (buf[tokens[idx].pos] == '-')
    val = -val;
  return val;
}

Inst *parser::get_reg(unsigned idx)
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
      && buf[tokens[idx].pos + 0] == 's'
      && buf[tokens[idx].pos + 1] == 'p')
    return rstate->registers[2];
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 'r'
      && buf[tokens[idx].pos + 1] == 'a')
    return rstate->registers[1];
  if (tokens[idx].kind != lexeme::name
      || (buf[tokens[idx].pos] != 'a'
	  && buf[tokens[idx].pos] != 's'
	  && buf[tokens[idx].pos] != 't'))
    throw Parse_error("expected a register instead of "
		      + token_string(tokens[idx]), line_number);
  // TODO: Check length.
  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  if (tokens[idx].size == 3)
    value = value * 10 + (buf[tokens[idx].pos + 2] - '0');
  if (buf[tokens[idx].pos] == 'a')
    return rstate->registers[10 + value];
  else if (buf[tokens[idx].pos] == 's')
    {
      if (value < 2)
	return rstate->registers[8 + value];
      else
	return rstate->registers[18 - 2 + value];
    }
  else if (buf[tokens[idx].pos] == 't')
    {
      if (value < 3)
	return rstate->registers[5 + value];
      else
	return rstate->registers[28 - 3 + value];
    }
  else
    throw Parse_error("expected a register instead of "
		      + token_string(tokens[idx]), line_number);
}

Inst *parser::get_freg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  int pos = tokens[idx].pos;
  if (buf[pos] != 'f')
    throw Parse_error("expected a floating point register", line_number);
  pos++;
  bool is_pseudo_reg = false;
  if (!isdigit(buf[pos]))
    {
      if (buf[pos] != 'a' && buf[pos] != 's' && buf[pos] != 't')
	throw Parse_error("invalid floating point register "
			  + token_string(tokens[idx]), line_number);
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
	    return rstate->fregisters[8];
	  else if (value == 1)
	    return rstate->fregisters[9];
	  else
	    return rstate->fregisters[16 + value];
	}
      else if (c == 't')
	{
	  if (value <= 7)
	    return rstate->fregisters[value];
	  else
	    return rstate->fregisters[value + 20];
	}
      else
	return rstate->fregisters[10 + value];
    }
  else
    return rstate->fregisters[value];
}

Inst *parser::get_hilo_addr(const token& tok)
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
  if (!rstate->sym_name2mem.contains(sym_name))
    throw Parse_error("unknown symbol " + sym_name, line_number);
  Inst *addr = rstate->sym_name2mem.at(sym_name);
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

Inst *parser::get_hi(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::hi)
    throw Parse_error("expected %lo instead of "
		      + token_string(tokens[idx]), line_number);
  Inst *high = bb->value_inst(31, 32);
  Inst *low = bb->value_inst(12, 32);
  Inst *addr = get_hilo_addr(tokens[idx]);
  Inst *res = bb->build_inst(Op::EXTRACT, addr, high, low);
  Inst *zero = bb->value_inst(0, 12);
  return bb->build_inst(Op::CONCAT, res, zero);
}

Inst *parser::get_lo(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::lo)
    throw Parse_error("expected %lo instead of "
		      + token_string(tokens[idx]), line_number);
  return bb->build_trunc(get_hilo_addr(tokens[idx]), 12);
}

Inst *parser::get_imm(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  Inst *inst;
  if (tokens[idx].kind == lexeme::lo)
    inst = get_lo(idx);
  else
    inst = bb->value_inst(get_hex_or_integer(idx), 12);
  Inst *bitsize = bb->value_inst(reg_bitsize, 32);
  return bb->build_inst(Op::SEXT, inst, bitsize);
}

Inst *parser::get_reg_value(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].size == 4
      && buf[tokens[idx].pos + 0] == 'z'
      && buf[tokens[idx].pos + 1] == 'e'
      && buf[tokens[idx].pos + 2] == 'r'
      && buf[tokens[idx].pos + 3] == 'o')
    return bb->value_inst(0, reg_bitsize);
  return bb->build_inst(Op::READ, get_reg(idx));
}

Inst *parser::get_freg_value(unsigned idx)
{
  return bb->build_inst(Op::READ, get_freg(idx));
}

Basic_block *parser::get_bb(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::label)
    throw Parse_error("expected a label instead of "
		      + token_string(tokens[idx]), line_number);
  std::string label(&buf[tokens[idx].pos], tokens[idx].size);
  auto I = label2bb.find(label);
  if (I != label2bb.end())
    return I->second;
  Basic_block *new_bb = func->build_bb();
  label2bb.insert({label, new_bb});
  return new_bb;
}

Basic_block *parser::get_bb_def(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::label_def)
    throw Parse_error("expected a label instead of "
		      + token_string(tokens[idx]), line_number);
  assert(tokens[idx].size > 0
	 && buf[tokens[idx].pos + tokens[idx].size - 1] == ':');
  std::string label(&buf[tokens[idx].pos], tokens[idx].size - 1);
  auto I = label2bb.find(label);
  if (I != label2bb.end())
    return I->second;
  Basic_block *new_bb = func->build_bb();
  label2bb.insert({label, new_bb});
  return new_bb;
}

void parser::get_comma(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != lexeme::comma)
    throw Parse_error("expected a ',' after " + token_string(tokens[idx - 1]),
		      line_number);
}

void parser::get_left_paren(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != lexeme::left_paren)
    throw Parse_error("expected a '(' after " + token_string(tokens[idx - 1]),
		      line_number);
}

void parser::get_right_paren(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != lexeme::right_paren)
    throw Parse_error("expected a ')' after " + token_string(tokens[idx - 1]),
		      line_number);
}

void parser::get_end_of_line(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() > idx)
    throw Parse_error("expected end of line after " +
		      token_string(tokens[idx - 1]), line_number);
}

void parser::process_cond_branch(Op op)
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = get_reg_value(3);
  get_comma(4);
  Basic_block *true_bb = get_bb(5);
  get_end_of_line(6);

  Basic_block *false_bb = func->build_bb();
  Inst *cond = bb->build_inst(op, arg1, arg2);
  bb->build_br_inst(cond, true_bb, false_bb);
  bb = false_bb;
}

Inst *parser::gen_bswap(Inst *arg)
{
  Inst *inst = bb->build_trunc(arg, 8);
  for (uint32_t i = 8; i < arg->bitsize; i += 8)
    {
      Inst *high = bb->value_inst(i + 7, 32);
      Inst *low = bb->value_inst(i, 32);
      Inst *byte = bb->build_inst(Op::EXTRACT, arg, high, low);
      inst = bb->build_inst(Op::CONCAT, inst, byte);
    }
  return inst;
}

Inst *parser::gen_clz(Inst *arg)
{
  Inst *zero = bb->value_inst(0, arg->bitsize);
  Inst *ub = bb->build_inst(Op::EQ, arg, zero);
  bb->build_inst(Op::UB, ub);
  Inst *inst = bb->value_inst(arg->bitsize, 32);
  for (unsigned i = 0; i < arg->bitsize; i++)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *val = bb->value_inst(arg->bitsize - i - 1, 32);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  return inst;
}

Inst *parser::gen_ctz(Inst *arg)
{
  Inst *zero = bb->value_inst(0, arg->bitsize);
  Inst *ub = bb->build_inst(Op::EQ, arg, zero);
  bb->build_inst(Op::UB, ub);
  Inst *inst = bb->value_inst(arg->bitsize, 32);
  for (int i = arg->bitsize - 1; i >= 0; i--)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      Inst *val = bb->value_inst(i, 32);
      inst = bb->build_inst(Op::ITE, bit, val, inst);
    }
  return inst;
}

Inst *parser::gen_parity(Inst *arg)
{
  Inst *inst = bb->build_extract_bit(arg, 0);
  for (uint32_t i = 1; i < arg->bitsize; i++)
    {
      Inst *bit = bb->build_extract_bit(arg, i);
      inst = bb->build_inst(Op::XOR, inst, bit);
    }
  inst = bb->build_inst(Op::ZEXT, inst, bb->value_inst(reg_bitsize, 32));

  return inst;
}

Inst *parser::gen_popcount(Inst *arg)
{
  Inst *bs = bb->value_inst(reg_bitsize, 32);
  Inst *bit = bb->build_extract_bit(arg, 0);
  Inst *inst = bb->build_inst(Op::ZEXT, bit, bs);
  for (uint32_t i = 1; i < arg->bitsize; i++)
    {
      bit = bb->build_extract_bit(arg, i);
      Inst *ext = bb->build_inst(Op::ZEXT, bit, bs);
      inst = bb->build_inst(Op::ADD, inst, ext);
    }
  return inst;
}

Inst *parser::gen_sdiv(Inst *arg1, Inst *arg2)
{
  Inst *zero = bb->value_inst(0, arg2->bitsize);
  bb->build_inst(Op::UB, bb->build_inst(Op::EQ, arg2, zero));
  return bb->build_inst(Op::SDIV, arg1, arg2);
}

Inst *parser::gen_udiv(Inst *arg1, Inst *arg2)
{
  Inst *zero = bb->value_inst(0, arg2->bitsize);
  bb->build_inst(Op::UB, bb->build_inst(Op::EQ, arg2, zero));
  return bb->build_inst(Op::UDIV, arg1, arg2);
}

Inst *parser::read_arg(uint32_t reg, uint32_t bitsize)
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

void parser::write_retval(Inst *retval)
{
  assert(reg_bitsize == 32);
  if (retval->bitsize == 32)
    bb->build_inst(Op::WRITE, rstate->registers[10 + 0], retval);
  else
    {
      assert(retval->bitsize == 64);
      Inst *a0 = bb->build_trunc(retval, reg_bitsize);
      Inst *high = bb->value_inst(2 * reg_bitsize - 1, 32);
      Inst *low = bb->value_inst(reg_bitsize, 32);
      Inst *a1 = bb->build_inst(Op::EXTRACT, retval, high, low);
      bb->build_inst(Op::WRITE, rstate->registers[10 + 0], a0);
      bb->build_inst(Op::WRITE, rstate->registers[10 + 1], a1);
    }
}

void parser::process_call()
{
  std::string name = get_name(1);
  get_end_of_line(2);

  if (name == "__bswapdi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 64);
      Inst *res = gen_bswap(arg);
      write_retval(res);
      return;
    }
  if (name == "__bswapsi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 32);
      Inst *res = gen_bswap(arg);
      write_retval(res);
      return;
    }
  if (name == "__clzdi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 64);
      Inst *res = gen_clz(arg);
      res = bb->build_inst(Op::SEXT, res, bb->value_inst(64, 32));
      write_retval(res);
      return;
    }
  if (name == "__clzsi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 32);
      Inst *res = gen_clz(arg);
      write_retval(res);
      return;
    }
  if (name == "__ctzdi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 64);
      Inst *res = gen_ctz(arg);
      res = bb->build_inst(Op::SEXT, res, bb->value_inst(64, 32));
      write_retval(res);
      return;
    }
  if (name == "__ctzsi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 32);
      Inst *res = gen_ctz(arg);
      write_retval(res);
      return;
    }
  if (name == "__divdi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(10 + 0, 64);
      Inst *arg2 = read_arg(10 + 2, 64);
      Inst *res = gen_sdiv(arg1, arg2);
      write_retval(res);
      return;
    }
  if (name == "__udivdi3" && reg_bitsize == 32)
    {
      Inst *arg1 = read_arg(10 + 0, 64);
      Inst *arg2 = read_arg(10 + 2, 64);
      Inst *res = gen_udiv(arg1, arg2);
      write_retval(res);
      return;
    }
  if (name == "__popcountdi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 64);
      Inst *res = gen_popcount(arg);
      write_retval(res);
      return;
    }
  if (name == "__popcountsi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 32);
      Inst *res = gen_popcount(arg);
      write_retval(res);
      return;
    }
  if (name == "__paritydi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 64);
      Inst *res = gen_parity(arg);
      write_retval(res);
      return;
    }
  if (name == "__paritysi2" && reg_bitsize == 32)
    {
      Inst *arg = read_arg(10 + 0, 32);
      Inst *res = gen_parity(arg);
      write_retval(res);
      return;
    }

  throw Not_implemented("call " + name);
}

void parser::process_tail()
{
  std::string name = get_name(1);
  get_end_of_line(2);

  throw Not_implemented("tail " + name);
}

void parser::store_ub_check(Inst *ptr, uint64_t size)
{
  Inst *ptr_mem_id = bb->build_extract_id(ptr);

  // It is UB to write to constant memory.
  Inst *is_const = bb->build_inst(Op::IS_CONST_MEM, ptr_mem_id);
  bb->build_inst(Op::UB, is_const);

  // It is UB if the store overflows into a different memory object.
  Inst *size_inst = bb->value_inst(size - 1, 32);
  Inst *last_addr = bb->build_inst(Op::ADD, ptr, size_inst);
  Inst *last_mem_id = bb->build_extract_id(last_addr);
  Inst *is_ub = bb->build_inst(Op::NE, ptr_mem_id, last_mem_id);
  bb->build_inst(Op::UB, is_ub);

  // It is UB if the end is outside the memory object -- the start is
  // obviously in the memory object if the end is within the object.
  // Otherwise, the  previous overflow check would have failed.
  Inst *mem_size = bb->build_inst(Op::GET_MEM_SIZE, ptr_mem_id);
  Inst *offset = bb->build_extract_offset(last_addr);
  Inst *out_of_bound = bb->build_inst(Op::UGE, offset, mem_size);
  bb->build_inst(Op::UB, out_of_bound);
}

void parser::load_ub_check(Inst *ptr, uint64_t size)
{
  Inst *ptr_mem_id = bb->build_extract_id(ptr);

  // It is UB if the store overflows into a different memory object.
  Inst *size_inst = bb->value_inst(size - 1, 32);
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
  Inst *out_of_bound = bb->build_inst(Op::UGE, offset, mem_size);
  bb->build_inst(Op::UB, out_of_bound);
}

void parser::process_load(int size, LStype lstype)
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
      Inst *bitsize = bb->value_inst(reg_bitsize, 32);
      Op op = lstype == LStype::unsigned_ls ? Op::ZEXT : Op::SEXT;
      value = bb->build_inst(op, value, bitsize);
    }
  bb->build_inst(Op::WRITE, dest, value);
}

void parser::process_store(int size, LStype lstype)
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
      Inst *high = bb->value_inst(i * 8 + 7, 32);
      Inst *low = bb->value_inst(i * 8, 32);
      Inst *byte = bb->build_inst(Op::EXTRACT, value, high, low);
      bb->build_inst(Op::STORE, addr, byte);
    }
}

void parser::process_funary(std::string name, Op op)
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

void parser::process_fbinary(std::string name, Op op)
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

void parser::process_fcmp(std::string name, Op op)
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
  Inst *res = bb->build_inst(op, arg1, arg2);
  Inst *bitsize = bb->value_inst(reg_bitsize, 32);
  res = bb->build_inst(Op::ZEXT, res, bitsize);
  bb->build_inst(Op::WRITE, dest, res);
}

void parser::process_iunary(std::string name, Op op)
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
    {
      Inst *bitsize = bb->value_inst(reg_bitsize, 32);
      res = bb->build_inst(Op::SEXT, res, bitsize);
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void parser::process_ibinary(std::string name, Op op)
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
    {
      Inst *bitsize = bb->value_inst(reg_bitsize, 32);
      res = bb->build_inst(Op::SEXT, res, bitsize);
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void parser::process_ishift(std::string name, Op op)
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
      arg2 = bb->build_trunc(arg2, 5);
      Inst *bitsize = bb->value_inst(32, 32);
      arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize);
    }
  else
    {
      arg2 = bb->build_trunc(arg2, 6);
      Inst *bitsize = bb->value_inst(reg_bitsize, 32);
      arg2 = bb->build_inst(Op::ZEXT, arg2, bitsize);
    }
  Inst *res = bb->build_inst(op, arg1, arg2);
  if (has_w_suffix)
    {
      Inst *bitsize = bb->value_inst(reg_bitsize, 32);
      res = bb->build_inst(Op::SEXT, res, bitsize);
    }
  bb->build_inst(Op::WRITE, dest, res);
}

void parser::process_icmp(std::string name, Op op)
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
  Inst *res = bb->build_inst(op, arg1, arg2);
  Inst *bitsize = bb->value_inst(reg_bitsize, 32);
  res = bb->build_inst(Op::ZEXT, res, bitsize);
  bb->build_inst(Op::WRITE, dest, res);
}

void parser::parse_function()
{
  if (tokens[0].kind == lexeme::label_def)
  {
    Basic_block *dest_bb = get_bb_def(0);
    get_end_of_line(1);

    if (bb)
      bb->build_br_inst(dest_bb);
    bb = dest_bb;
    return;
  }

  std::string name = get_name(&buf[tokens[0].pos]);
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
      Inst *bitsize = bb->value_inst(2 * reg_bitsize, 32);
      arg1 = bb->build_inst(op1, arg1, bitsize);
      arg2 = bb->build_inst(op2, arg2, bitsize);
      Inst *res = bb->build_inst(Op::MUL, arg1, arg2);
      Inst *high = bb->value_inst(2 * reg_bitsize - 1, 32);
      Inst *low = bb->value_inst(reg_bitsize, 32);
      res = bb->build_inst(Op::EXTRACT, res, high, low);
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
    process_icmp(name, Op::SGT);
  else if (name == "sgtu" || name == "sgtuw")
    process_icmp(name, Op::UGT);
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
      Inst *bitsize = bb->value_inst(reg_bitsize, 32);
      res = bb->build_inst(Op::ZEXT, res, bitsize);
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
      Inst *bitsize = bb->value_inst(reg_bitsize, 32);
      res = bb->build_inst(Op::ZEXT, res, bitsize);
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
      Inst *bitsize = bb->value_inst(reg_bitsize, 32);
      res = bb->build_inst(Op::SEXT, res, bitsize);
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
	{
	  Inst *bitsize = bb->value_inst(reg_bitsize, 32);
	  res = bb->build_inst(Op::SEXT, res, bitsize);
	}
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
    process_cond_branch(Op::SGE);
  else if (name == "bgeu")
    process_cond_branch(Op::UGE);
  else if (name == "bgt")
    process_cond_branch(Op::SGT);
  else if (name == "bgtu")
    process_cond_branch(Op::UGT);
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
    process_fcmp(name, Op::FGT);
  else if (name == "fge.s" || name == "fge.d")
    process_fcmp(name, Op::FGE);
  else
    throw Parse_error("unhandled instruction: "s + name, line_number);
}

void parser::lex_line(void)
{
  tokens.clear();
  while (buf[pos] != '\n')
    {
      skip_space_and_comments();
      if (!buf[pos])
	break;
      if (isdigit(buf[pos]) || buf[pos] == '-')
	lex_hex_or_integer();
      else if (buf[pos] == '.' && buf[pos + 1] == 'L' ) // TODO: pos+1 check.
	lex_label_or_label_def();
      else if (isalpha(buf[pos]) || buf[pos] == '_' || buf[pos] == '.')
	lex_name();
      else if (buf[pos] == '%' && buf[pos + 1] == 'l' && buf[pos + 2] == 'o')
	lex_hilo();
      else if (buf[pos] == '%' && buf[pos + 1] == 'h' && buf[pos + 2] == 'i')
	lex_hilo();
      else if (buf[pos] == ',')
	{
	  tokens.emplace_back(lexeme::comma, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '=')
	{
	  tokens.emplace_back(lexeme::assign, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '[')
	{
	  tokens.emplace_back(lexeme::left_bracket, pos, 1);
	  pos++;
	}
      else if (buf[pos] == ']')
	{
	  tokens.emplace_back(lexeme::right_bracket, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '(')
	{
	  tokens.emplace_back(lexeme::left_paren, pos, 1);
	  pos++;
	}
      else if (buf[pos] == ')')
	{
	  tokens.emplace_back(lexeme::right_paren, pos, 1);
	  pos++;
	}
      else
	throw Parse_error("syntax error", line_number);
    }
  pos++;
}

std::optional<std::string> parser::parse_label_def()
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
  std::string label(&buf[start_pos], pos - start_pos);
  pos++;
  if (buf[pos] == '\n')
    {
      pos++;
      return label;
    }
  return {};
}

std::string parser::parse_cmd()
{
  size_t start_pos = pos;
  while (buf[pos] == '.' || isalnum(buf[pos]))
    pos++;
  assert(buf[pos] == ' ' || buf[pos] == '\t' || buf[pos] == '\n');
  return std::string(&buf[start_pos], pos - start_pos);
}

void parser::parse_data(std::vector<unsigned char>& data)
{
  for (;;)
    {
      size_t start_pos = pos;

      if (pos == buf.size() - 1)
	break;
      assert(pos < buf.size());

      skip_whitespace();
      std::string cmd = parse_cmd();
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
	  // TODO: Implement. Note: we must handle escape sequences such as \n.
	  throw Parse_error(".string/.ascii not implemented", line_number);
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

void parser::skip_line()
{
  while (buf[pos] != '\n')
    {
      pos++;
    }
  pos++;
}

void parser::skip_whitespace()
{
  while (buf[pos] == ' ' || buf[pos] == '\t')
    {
      pos++;
    }
}

void parser::parse_rodata()
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
	  std::string name(&buf[first_pos], pos - first_pos);
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

      pos = start_pos;

      if (parser_state == state::memory_section)
	{
	  std::optional<std::string> label = parse_label_def();
	  if (label)
	    {
	      // TODO: Change to check for duplicated labels.
	      assert(!sym_name2data.contains(*label));

	      parse_data(sym_name2data[*label]);

	      // TODO: Change to check.
	      assert(!sym_name2data[*label].empty());

	      continue;
	    }
	}

      skip_line();
      parser_state = state::global;
    }
}

Function *parser::parse(std::string const& file_name)
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

    line_number++;

    if (parser_state == state::global)
      {
	std::optional<std::string> label = parse_label_def();
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
	std::string name = get_name(&buf[tokens[0].pos]);
	if (name == ".size")
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
  parser p(state);
  Function *func = p.parse(file_name);
  reverse_post_order(func);
  return func;
}

} // end namespace smtgcc
