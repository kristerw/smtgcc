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
  Parser(bpf_state *rstate) : rstate{rstate} {}

  enum class Cond_code {
    EQ, NE, GT, GE, LT, LE, SGT, SGE, SLT, SLE
  };

  enum class Lexeme {
    label,
    label_def,
    name,
    reg,
    integer,
    hex,
    comma,
    plus,
    minus,
    left_bracket,
    right_bracket
  };

  struct Token {
    Lexeme kind;
    int pos;
    int size;
  };
  std::vector<Token> tokens;

  Function *parse(std::string const& file_name);

  bpf_state *rstate;
  Module *module;
  Function *src_func;

private:
  Function *func = nullptr;
  Basic_block *bb = nullptr;
  std::map<std::string_view, Basic_block *> label2bb;
  std::map<uint32_t, Inst *> id2inst;

  void skip_space_and_comments();
  void lex_label_or_label_def();
  void lex_hex();
  void lex_integer();
  void lex_imm();
  void lex_name();
  void lex_reg();

  uint64_t get_u64(const char *p);
  unsigned __int128 get_hex(const char *p);
  unsigned __int128 get_hex_or_integer(unsigned idx);
  Inst *get_imm(unsigned idx);
  Inst *get_reg(unsigned idx);
  Inst *get_reg_value(unsigned idx);
  Inst *get_reg_or_imm_value(unsigned idx);
  void get_left_bracket(unsigned idx);
  void get_right_bracket(unsigned idx);
  void get_plus(unsigned idx);

  void write_reg(Inst *reg, Inst *value);
  std::string_view token_string(const Token& tok);
  std::string_view get_name(unsigned idx);
  void build_mem(const std::string& sym_name);
  Inst *get_sym_addr(unsigned idx);
  Basic_block *get_bb(unsigned idx);
  Basic_block *get_bb_def(unsigned idx);
  void get_comma(unsigned idx);
  void get_end_of_line(unsigned idx);
  Inst *extract_vec_elem(Inst *inst, uint32_t elem_bitsize, uint32_t idx);
  Inst *load_value(Inst *ptr, uint64_t size);
  void store_value(Inst *ptr, Inst *value);
  Inst *build_cond(Cond_code cc, Inst *arg1, Inst *arg2);
  void process_unary(Op op);
  void process_unary32(Op op);
  void process_binary(Op op);
  void process_binary32(Op op);
  void process_div(Op op);
  void process_div32(Op op);
  void process_shift(Op op);
  void process_shift32(Op op);
  void process_mov();
  void process_mov32();
  void process_movs();
  void process_movs32();
  void process_bswap();
  void process_ja();
  void process_cond_branch(Cond_code cc);
  void process_cond_branch32(Cond_code cc);
  void process_call();
  void process_exit();
  void process_store(uint32_t size);
  void process_load(uint32_t size, bool sign_extend = false);

  void parse_function();
  void lex_line();
};

void Parser::skip_space_and_comments()
{
  while (isspace(buf[pos]))
    pos++;
  if (buf[pos] == '#'
      && (!isdigit(buf[pos + 1])
	  && buf[pos + 1] != '-'
	  && buf[pos + 1] != ':'))
    {
      while (buf[pos] != '\n')
	pos++;
    }
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
  assert(isdigit(buf[pos]) || buf[pos] == '-');
  if (buf[pos] == '0' && (buf[pos + 1] == 'x' || buf[pos + 1] == 'X'))
    lex_hex();
  else
    lex_integer();
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

void Parser::lex_reg()
{
  assert(buf[pos] == '%');
  int start_pos = pos++;
  if (buf[pos] == 'r')
    {
      pos++;
      while (isdigit(buf[pos]))
	pos++;
    }
  else if (buf[pos] == 'f' && buf[pos + 1] == 'p')
    pos += 2;
  tokens.emplace_back(Lexeme::reg, start_pos, pos - start_pos);
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
  return bb->value_inst(get_hex_or_integer(idx), 64);
}

Inst *Parser::get_reg(unsigned idx)
{
  if (tokens[idx].kind == Lexeme::reg
      && tokens[idx].size == 3
      && buf[tokens[idx].pos] == '%'
      && buf[tokens[idx].pos + 1] == 'f'
      && buf[tokens[idx].pos + 2] == 'p')
    return rstate->registers[BpfRegIdx::fp];
  if (tokens[idx].kind == Lexeme::reg
      && tokens[idx].size == 4
      && buf[tokens[idx].pos] == '%'
      && buf[tokens[idx].pos + 1] == 'r'
      && buf[tokens[idx].pos + 2] == '1'
      && buf[tokens[idx].pos + 3] == '0')
    return rstate->registers[BpfRegIdx::fp];

  if (tokens[idx].kind == Lexeme::reg
      && tokens[idx].size == 3
      && buf[tokens[idx].pos] == '%'
      && buf[tokens[idx].pos + 1] == 'r'
      && isdigit(buf[tokens[idx].pos + 2]))
    {
      uint32_t value = buf[tokens[idx].pos + 2] - '0';
      return rstate->registers[BpfRegIdx::r0 + value];
    }

  throw Parse_error("expected a register instead of "
		    + std::string(token_string(tokens[idx])), line_number);
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
  if (tokens[idx].kind == Lexeme::reg)
    value = get_reg_value(idx);
  else
    value = get_imm(idx);
  return value;
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

void Parser::get_plus(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::plus)
    throw Parse_error("expected a '+' after "
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

void Parser::build_mem(const std::string& sym_name)
{
  assert(!rstate->sym_name2mem.contains(sym_name));
  if (!sym_name2data.contains(sym_name))
    throw Parse_error("unknown symbol " + sym_name, line_number);
  std::vector<unsigned char>& data = sym_name2data.at(sym_name);

  if (rstate->next_local_id == 0)
    throw Not_implemented("too many local variables");

  Basic_block *entry_bb = rstate->entry_bb;
  Inst *id = entry_bb->value_inst(rstate->next_local_id++, module->ptr_id_bits);
  Inst *mem_size = entry_bb->value_inst(data.size(), module->ptr_offset_bits);
  Inst *flags = entry_bb->value_inst(MEM_CONST, 32);
  Inst *mem = entry_bb->build_inst(Op::MEMORY, id, mem_size, flags);

  assert(!rstate->sym_name2mem.contains(sym_name));
  rstate->sym_name2mem.insert({sym_name, mem});

  for (size_t i = 0; i < data.size(); i++)
    {
      Inst *off = entry_bb->value_inst(i, mem->bitsize);
      Inst *ptr = entry_bb->build_inst(Op::ADD, mem, off);
      Inst *byte = entry_bb->value_inst(data[i], 8);
      entry_bb->build_inst(Op::STORE, ptr, byte);
    }
}

Inst *Parser::get_sym_addr(unsigned idx)
{
  std::string name = std::string(get_name(idx));
  uint64_t offset = 0;
  if (sym_alias.contains(name))
    std::tie(name, offset) = sym_alias[name];
  if (!rstate->sym_name2mem.contains(name))
    build_mem(name);
  Inst *inst = rstate->sym_name2mem[name];
  if (offset)
    inst = bb->build_inst(Op::ADD, inst, bb->value_inst(offset, inst->bitsize));
  return inst;
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

Inst *Parser::build_cond(Cond_code cc, Inst *arg1, Inst *arg2)
{
  switch (cc)
    {
    case Cond_code::EQ:
      return bb->build_inst(Op::EQ, arg1, arg2);
    case Cond_code::NE:
      return bb->build_inst(Op::NOT, bb->build_inst(Op::EQ, arg1, arg2));
    case Cond_code::GE:
      return bb->build_inst(Op::NOT, bb->build_inst(Op::ULT, arg1, arg2));
    case Cond_code::GT:
      return bb->build_inst(Op::ULT, arg2, arg1);
    case Cond_code::SGE:
      return bb->build_inst(Op::NOT, bb->build_inst(Op::SLT, arg1, arg2));
    case Cond_code::SGT:
      return bb->build_inst(Op::SLT, arg2, arg1);
    case Cond_code::LE:
      return bb->build_inst(Op::NOT, bb->build_inst(Op::ULT, arg2, arg1));
    case Cond_code::LT:
      return bb->build_inst(Op::ULT, arg1, arg2);
    case Cond_code::SLE:
      return bb->build_inst(Op::NOT, bb->build_inst(Op::SLT, arg2, arg1));
    case Cond_code::SLT:
      return bb->build_inst(Op::SLT, arg1, arg2);
    }

  throw Parse_error("unhandled condition code", line_number);
}

void Parser::process_unary(Op op)
{
  Inst *dest = get_reg(1);
  Inst *arg1 = get_reg_value(1);
  get_end_of_line(2);

  Inst *res = bb->build_inst(op, arg1);
  write_reg(dest, res);
}

void Parser::process_unary32(Op op)
{
  Inst *dest = get_reg(1);
  Inst *arg1 = bb->build_trunc(get_reg_value(1), 32);
  get_end_of_line(2);

  Inst *res = bb->build_inst(op, arg1);
  write_reg(dest, res);
}

void Parser::process_binary(Op op)
{
  Inst *dest = get_reg(1);
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = get_reg_or_imm_value(3);
  get_end_of_line(4);

  Inst *res = bb->build_inst(op, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_binary32(Op op)
{
  Inst *dest = get_reg(1);
  Inst *arg1 = bb->build_trunc(get_reg_value(1), 32);
  get_comma(2);
  Inst *arg2 = bb->build_trunc(get_reg_or_imm_value(3), 32);
  get_end_of_line(4);

  Inst *res = bb->build_inst(op, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_div(Op op)
{
  Inst *dest = get_reg(1);
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = get_reg_or_imm_value(3);
  get_end_of_line(4);

  Inst *res = bb->build_inst(op, arg1, arg2);
  Inst *zero = bb->value_inst(0, arg2->bitsize);
  Inst *is_zero = bb->build_inst(Op::EQ, arg2, zero);
  res = bb->build_inst(Op::ITE, is_zero, zero, res);
  write_reg(dest, res);
}

void Parser::process_div32(Op op)
{
  Inst *dest = get_reg(1);
  Inst *arg1 = bb->build_trunc(get_reg_value(1), 32);
  get_comma(2);
  Inst *arg2 = bb->build_trunc(get_reg_or_imm_value(3), 32);
  get_end_of_line(4);

  Inst *res = bb->build_inst(op, arg1, arg2);
  Inst *zero = bb->value_inst(0, arg2->bitsize);
  Inst *is_zero = bb->build_inst(Op::EQ, arg2, zero);
  res = bb->build_inst(Op::ITE, is_zero, zero, res);
  write_reg(dest, res);
}

void Parser::process_shift(Op op)
{
  Inst *dest = get_reg(1);
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = get_reg_or_imm_value(3);
  get_end_of_line(4);

  Inst *mask = bb->value_inst(63, 64);
  arg2 = bb->build_inst(Op::AND, arg2, mask);
  Inst *res = bb->build_inst(op, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_shift32(Op op)
{
  Inst *dest = get_reg(1);
  Inst *arg1 = bb->build_trunc(get_reg_value(1), 32);
  get_comma(2);
  Inst *arg2 = bb->build_trunc(get_reg_or_imm_value(3), 32);
  get_end_of_line(4);

  Inst *mask = bb->value_inst(31, 32);
  arg2 = bb->build_inst(Op::AND, arg2, mask);
  Inst *res = bb->build_inst(op, arg1, arg2);
  write_reg(dest, res);
}

void Parser::process_mov()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1;
  if (tokens[3].kind == Lexeme::name)
    {
      unsigned idx = 3;
      arg1 = get_sym_addr(idx++);
      if (tokens.size() > idx && tokens[idx].kind == Lexeme::plus)
	{
	  idx++;
	  Inst *offset = get_imm(idx++);
	  arg1 = bb->build_inst(Op::ADD, arg1, offset);
	}
      else if (tokens.size() > idx && tokens[idx].kind == Lexeme::minus)
	{
	  idx++;
	  Inst *offset = get_imm(idx++);
	  arg1 = bb->build_inst(Op::SUB, arg1, offset);
	}
      get_end_of_line(idx);
    }
  else
    {
      arg1 = get_reg_or_imm_value(3);
      get_end_of_line(4);
    }

  write_reg(dest, arg1);
}

void Parser::process_mov32()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = bb->build_trunc(get_reg_or_imm_value(3), 32);
  get_end_of_line(4);

  write_reg(dest, arg1);
}

void Parser::process_movs()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_or_imm_value(3);
  get_comma(4);
  uint64_t arg2 = get_hex_or_integer(5);
  get_end_of_line(6);

  arg1 = bb->build_trunc(arg1, arg2);
  Inst *res = bb->build_inst(Op::SEXT, arg1, 64);
  write_reg(dest, res);
}

void Parser::process_movs32()
{
  Inst *dest = get_reg(1);
  get_comma(2);
  Inst *arg1 = get_reg_or_imm_value(3);
  get_comma(4);
  uint64_t arg2 = get_hex_or_integer(5);
  get_end_of_line(6);

  arg1 = bb->build_trunc(arg1, arg2);
  Inst *res = bb->build_inst(Op::SEXT, arg1, 32);
  write_reg(dest, res);
}

void Parser::process_bswap()
{
  Inst *dest = get_reg(1);
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  uint64_t arg2 = get_hex_or_integer(3);
  get_end_of_line(4);

  Inst *res = gen_bswap(bb, bb->build_trunc(arg1, arg2));
  write_reg(dest, res);
}

void Parser::process_ja()
{
  Basic_block *dest_bb = get_bb(1);
  get_end_of_line(2);

  bb->build_br_inst(dest_bb);
  bb = nullptr;
}

void Parser::process_cond_branch(Cond_code cc)
{
  Inst *arg1 = get_reg_value(1);
  get_comma(2);
  Inst *arg2 = get_reg_or_imm_value(3);
  get_comma(4);
  Basic_block *true_bb = get_bb(5);
  get_end_of_line(6);

  Basic_block *false_bb = func->build_bb();
  bb->build_br_inst(build_cond(cc, arg1, arg2), true_bb, false_bb);
  bb = false_bb;
}

void Parser::process_cond_branch32(Cond_code cc)
{
  Inst *arg1 = bb->build_trunc(get_reg_value(1), 32);
  get_comma(2);
  Inst *arg2 = bb->build_trunc(get_reg_or_imm_value(3), 32);
  get_comma(4);
  Basic_block *true_bb = get_bb(5);
  get_end_of_line(6);

  Basic_block *false_bb = func->build_bb();
  bb->build_br_inst(build_cond(cc, arg1, arg2), true_bb, false_bb);
  bb = false_bb;
}

void Parser::process_call()
{
  std::string_view name = get_name(1);
  get_end_of_line(2);

  if (name == "abort")
    {
      Inst *b1 = bb->value_inst(1, 1);
      bb->build_inst(Op::WRITE, rstate->registers[BpfRegIdx::abort], b1);
      bb->build_br_inst(rstate->exit_bb);
      bb = func->build_bb();
      return;
    }
  if (name == "exit")
    {
      Inst *b1 = bb->value_inst(1, 1);
      Inst *exit_val =
	bb->build_inst(Op::READ, rstate->registers[BpfRegIdx::r1]);
      exit_val = bb->build_trunc(exit_val, 32);
      bb->build_inst(Op::WRITE, rstate->registers[BpfRegIdx::exit], b1);
      bb->build_inst(Op::WRITE, rstate->registers[BpfRegIdx::exit_val],
		     exit_val);
      bb->build_br_inst(rstate->exit_bb);
      bb = func->build_bb();
      return;
    }

  throw Not_implemented("call " + std::string(name));
}

void Parser::process_exit()
{
  get_end_of_line(1);

  bb->build_br_inst(rstate->exit_bb);
  bb = func->build_bb();
}

void Parser::process_store(uint32_t size)
{
  get_left_bracket(1);
  Inst *ptr = get_reg_value(2);
  get_plus(3);
  Inst *offset = get_imm(4);
  get_right_bracket(5);
  get_comma(6);
  Inst *value = get_reg_or_imm_value(7);
  get_end_of_line(8);

  ptr = bb->build_inst(Op::ADD, ptr, offset);
  if (size * 8 < value->bitsize)
    value = bb->build_trunc(value, size * 8);
  store_value(ptr, value);
}

void Parser::process_load(uint32_t size, bool sign_extend)
{
  Inst *dest = get_reg(1);
  get_comma(2);
  get_left_bracket(3);
  Inst *ptr = get_reg_value(4);
  get_plus(5);
  Inst *offset = get_imm(6);
  get_right_bracket(7);
  get_end_of_line(8);

  ptr = bb->build_inst(Op::ADD, ptr, offset);
  Inst *value = load_value(ptr, size);
  if (sign_extend)
    value = bb->build_inst(Op::SEXT, value, 64);
  write_reg(dest, value);
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
  else if (name == "add")
    process_binary(Op::ADD);
  else if (name == "add32")
    process_binary32(Op::ADD);
  else if (name == "and")
    process_binary(Op::AND);
  else if (name == "and32")
    process_binary32(Op::AND);
  else if (name == "arsh")
    process_shift(Op::ASHR);
  else if (name == "arsh32")
    process_shift32(Op::ASHR);
  else if (name == "bswap")
    process_bswap();
  else if (name == "call")
    process_call();
  else if (name == "div")
    process_div(Op::UDIV);
  else if (name == "div32")
    process_div32(Op::UDIV);
  else if (name == "endbe")
    process_bswap();
  else if (name == "exit")
    process_exit();
  else if (name == "ja")
    process_ja();
  else if (name == "jeq")
    process_cond_branch(Cond_code::EQ);
  else if (name == "jeq32")
    process_cond_branch32(Cond_code::EQ);
  else if (name == "jne")
    process_cond_branch(Cond_code::NE);
  else if (name == "jne32")
    process_cond_branch32(Cond_code::NE);
  else if (name == "jge")
    process_cond_branch(Cond_code::GE);
  else if (name == "jge32")
    process_cond_branch32(Cond_code::GE);
  else if (name == "jgt")
    process_cond_branch(Cond_code::GT);
  else if (name == "jgt32")
    process_cond_branch32(Cond_code::GT);
  else if (name == "jsge")
    process_cond_branch(Cond_code::SGE);
  else if (name == "jsge32")
    process_cond_branch32(Cond_code::SGE);
  else if (name == "jsgt")
    process_cond_branch(Cond_code::SGT);
  else if (name == "jsgt32")
    process_cond_branch32(Cond_code::SGT);
  else if (name == "jle")
    process_cond_branch(Cond_code::LE);
  else if (name == "jle32")
    process_cond_branch32(Cond_code::LE);
  else if (name == "jlt")
    process_cond_branch(Cond_code::LT);
  else if (name == "jlt32")
    process_cond_branch32(Cond_code::LT);
  else if (name == "jsle")
    process_cond_branch(Cond_code::SLE);
  else if (name == "jsle32")
    process_cond_branch32(Cond_code::SLE);
  else if (name == "jslt")
    process_cond_branch(Cond_code::SLT);
  else if (name == "jslt32")
    process_cond_branch32(Cond_code::SLT);
  else if (name == "lddw")
    process_mov();
  else if (name == "ldxsb")
    process_load(1, true);
  else if (name == "ldxsh")
    process_load(2, true);
  else if (name == "ldxsw")
    process_load(4, true);
  else if (name == "ldxb")
    process_load(1);
  else if (name == "ldxh")
    process_load(2);
  else if (name == "ldxw")
    process_load(4);
  else if (name == "ldxdw")
    process_load(8);
  else if (name == "lsh")
    process_shift(Op::SHL);
  else if (name == "lsh32")
    process_shift32(Op::SHL);
  else if (name == "mod")
    process_binary(Op::UREM);
  else if (name == "mod32")
    process_binary32(Op::UREM);
  else if (name == "mov")
    process_mov();
  else if (name == "mov32")
    process_mov32();
  else if (name == "movs")
    process_movs();
  else if (name == "movs32")
    process_movs32();
  else if (name == "mul")
    process_binary(Op::MUL);
  else if (name == "mul32")
    process_binary32(Op::MUL);
  else if (name == "neg")
    process_unary(Op::NEG);
  else if (name == "neg32")
    process_unary32(Op::NEG);
  else if (name == "or")
    process_binary(Op::OR);
  else if (name == "or32")
    process_binary32(Op::OR);
  else if (name == "rsh")
    process_shift(Op::LSHR);
  else if (name == "rsh32")
    process_shift32(Op::LSHR);
  else if (name == "sdiv")
    process_div(Op::SDIV);
  else if (name == "sdiv32")
    process_div32(Op::SDIV);
  else if (name == "smod")
    process_binary(Op::SREM);
  else if (name == "smod32")
    process_binary32(Op::SREM);
  else if (name == "stb" || name == "stxb")
    process_store(1);
  else if (name == "sth" || name == "stxh")
    process_store(2);
  else if (name == "stw" || name == "stxw")
    process_store(4);
  else if (name == "stdw" || name == "stxdw")
    process_store(8);
  else if (name == "sub")
    process_binary(Op::SUB);
  else if (name == "sub32")
    process_binary32(Op::SUB);
  else if (name == "xor")
    process_binary(Op::XOR);
  else if (name == "xor32")
    process_binary32(Op::XOR);
  else
    throw Parse_error("unhandled instruction: "s + std::string(name),
		      line_number);
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
      else if (buf[pos] == '%')
	lex_reg();
      else if (isdigit(buf[pos]) || buf[pos] == '-')
	lex_imm();
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
      else
	throw Parse_error("syntax error", line_number);
    }
  pos++;
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

	    for (const auto& [name, mem] : rstate->sym_name2mem)
	      {
		if (!sym_name2data.contains(name))
		  continue;
		std::vector<unsigned char>& data = sym_name2data.at(name);
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

Function *parse_bpf(std::string const& file_name, bpf_state *state)
{
  Parser p(state);
  Function *func = p.parse(file_name);
  reverse_post_order(func);
  return func;
}

} // end namespace smtgcc
