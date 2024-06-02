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

const int stack_size = 1024;

// TODO: Check that all instructions are supported by asm. For example,
// I am not sure that the "w" version of sgt is supported...

struct parser {
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
  std::vector<Instruction *> registers;
  std::vector<Instruction *> fregisters;
  std::vector<Basic_block *> ret_bbs;
  std::map<std::string, Instruction *> sym_name2mem;

  int line_number = 0;
  int pos;

  static const int max_line_len = 1000;
  char buf[max_line_len];

  Function *parse(std::string const& file_name, riscv_state *state);

  Module *module;
  uint32_t reg_bitsize;
  Function *src_func;

private:
  Function *current_func = nullptr;
  Basic_block *current_bb = nullptr;
  std::map<uint32_t, Basic_block *> id2bb;
  std::map<Basic_block *, uint32_t> bb2id;
  std::map<uint32_t, Instruction *> id2inst;

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
  Instruction *get_reg(unsigned idx);
  Instruction *get_freg(unsigned idx);
  Instruction *get_hilo_addr(const token& tok);
  Instruction *get_hi(unsigned idx);
  Instruction *get_lo(unsigned idx);
  Instruction *get_imm(unsigned idx);
  Instruction *get_reg_value(unsigned idx);
  Instruction *get_freg_value(unsigned idx);
  Basic_block *get_bb(unsigned idx);
  Basic_block *get_bb_def(unsigned idx);
  std::string get_name(unsigned idx);
  void get_comma(unsigned idx);
  void get_left_paren(unsigned idx);
  void get_right_paren(unsigned idx);
  void get_end_of_line(unsigned idx);
  void gen_cond_branch(Op opcode);
  void gen_call();
  void gen_tail();
  void store_ub_check(Instruction *ptr, uint64_t size);
  void load_ub_check(Instruction *ptr, uint64_t size);
  void gen_load(int size, LStype lstype = LStype::signed_ls);
  void gen_store(int size, LStype lstype = LStype::signed_ls);
  void gen_funary(std::string name, Op op);
  void gen_fbinary(std::string name, Op op);
  void gen_fcmp(std::string name, Op op);

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
  if (!isdigit(buf[pos]))
    throw Parse_error("expected a digit after \".L\"", line_number);
  pos++;
  if (isdigit(buf[pos]) && buf[pos - 1] == '0')
    throw Parse_error("octal numbers are not supported in labels", line_number);
  while (isdigit(buf[pos]))
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
  assert(isalpha(buf[pos]) || buf[pos] == '_' || buf[pos] == '.');
  int start_pos = pos;
  pos++;
  while (isalnum(buf[pos]) || buf[pos] == '_' || buf[pos] == '-' || buf[pos] == '.')
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

Instruction *parser::get_reg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 's'
      && buf[tokens[idx].pos + 1] == 'p')
    return registers[2];
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 'r'
      && buf[tokens[idx].pos + 1] == 'a')
    return registers[1];
  if (tokens[idx].kind != lexeme::name
      || (buf[tokens[idx].pos] != 'a'
	  && buf[tokens[idx].pos] != 's'
	  && buf[tokens[idx].pos] != 't'))
    throw Parse_error("expected a register instead of "
		      + token_string(tokens[idx]), line_number);
  // TODO: Check length.
  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  if (tokens[idx].size == 3)
    value = value * 10 + (buf[tokens[idx].pos + 1] - '0');
  if (buf[tokens[idx].pos] == 'a')
    return registers[10 + value];
  else if (buf[tokens[idx].pos] == 's')
    {
      if (value < 2)
	return registers[8 + value];
      else
	return registers[18 - 2 + value];
    }
  else if (buf[tokens[idx].pos] == 't')
    {
      if (value < 3)
	return registers[5 + value];
      else
	return registers[28 - 3 + value];
    }
  else
    throw Parse_error("expected a register instead of "
		      + token_string(tokens[idx]), line_number);
}

Instruction *parser::get_freg(unsigned idx)
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
  if (tokens[idx].size == pos)
    value = value * 10 + (buf[pos + 1] - '0');
  if (is_pseudo_reg)
    {
      char c = buf[tokens[idx].pos + 1];
      assert(c == 'a' || c =='s' || c =='t');
      if (c == 's')
	{
	  if (value == 0)
	    return fregisters[8];
	  else if (value == 1)
	    return fregisters[9];
	  else
	    return fregisters[16 + value];
	}
      else if (c == 't')
	{
	  if (value <= 7)
	    return fregisters[value];
	  else
	    return fregisters[value + 21];
	}
      else
	return fregisters[10 + value];
    }
  else
    return fregisters[value];
}

Instruction *parser::get_hilo_addr(const token& tok)
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
  if (!sym_name2mem.contains(sym_name))
    throw Parse_error("unknown symbol " + sym_name, line_number);
  Instruction *addr = sym_name2mem.at(sym_name);
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
      Instruction *value_inst = current_bb->value_inst(value, addr->bitsize);
      addr = current_bb->build_inst(Op::ADD, addr, value_inst);
    }
  assert(buf[pos] == ')');
  return addr;
}

Instruction *parser::get_hi(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::hi)
    throw Parse_error("expected %lo instead of "
		      + token_string(tokens[idx]), line_number);
  Instruction *high = current_bb->value_inst(31, 32);
  Instruction *low = current_bb->value_inst(12, 32);
  Instruction *addr = get_hilo_addr(tokens[idx]);
  Instruction *res = current_bb->build_inst(Op::EXTRACT, addr, high, low);
  Instruction *zero = current_bb->value_inst(0, 12);
  return current_bb->build_inst(Op::CONCAT, res, zero);
}

Instruction *parser::get_lo(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::lo)
    throw Parse_error("expected %lo instead of "
		      + token_string(tokens[idx]), line_number);
  return current_bb->build_trunc(get_hilo_addr(tokens[idx]), 12);
}

Instruction *parser::get_imm(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  Instruction *inst;
  if (tokens[idx].kind == lexeme::lo)
    inst = get_lo(idx);
  else
    inst = current_bb->value_inst(get_hex_or_integer(idx), 12);
  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
  return current_bb->build_inst(Op::SEXT, inst, bitsize);
}

Instruction *parser::get_reg_value(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].size == 4
      && buf[tokens[idx].pos + 0] == 'z'
      && buf[tokens[idx].pos + 1] == 'e'
      && buf[tokens[idx].pos + 2] == 'r'
      && buf[tokens[idx].pos + 3] == 'o')
    return current_bb->value_inst(0, reg_bitsize);
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 'r'
      && buf[tokens[idx].pos + 1] == 'a')
    return current_bb->build_inst(Op::READ, registers[1]);
  if (tokens[idx].size == 2
      && buf[tokens[idx].pos + 0] == 's'
      && buf[tokens[idx].pos + 1] == 'p')
    return current_bb->build_inst(Op::READ, registers[2]);
  if (tokens[idx].kind != lexeme::name
      || (buf[tokens[idx].pos] != 'a'
	  && buf[tokens[idx].pos] != 's'
	  && buf[tokens[idx].pos] != 't'))
    throw Parse_error("expected a register instead of "
		      + token_string(tokens[idx]), line_number);
  // TODO: Check length.
  uint32_t value = buf[tokens[idx].pos + 1] - '0';
  if (tokens[idx].size == 3)
    value = value * 10 + (buf[tokens[idx].pos + 1] - '0');
  if (buf[tokens[idx].pos] == 'a')
    return current_bb->build_inst(Op::READ, registers[10 + value]);
  else if (buf[tokens[idx].pos] == 's')
    {
      if (value < 2)
	return current_bb->build_inst(Op::READ, registers[8 + value]);
      else
	return current_bb->build_inst(Op::READ, registers[18 - 2 + value]);
    }
  else if (buf[tokens[idx].pos] == 't')
    {
      if (value < 3)
	return current_bb->build_inst(Op::READ, registers[5 + value]);
      else
	return current_bb->build_inst(Op::READ, registers[28 - 3 + value]);
    }
  else
    throw Parse_error("expected a register instead of "
		      + token_string(tokens[idx]), line_number);
}

Instruction *parser::get_freg_value(unsigned idx)
{
  return current_bb->build_inst(Op::READ, get_freg(idx));
}

Basic_block *parser::get_bb(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
 if (tokens[idx].kind != lexeme::label)
   throw Parse_error("expected a label instead of "
		     + token_string(tokens[idx]), line_number);
  uint32_t id = get_u32(&buf[tokens[idx].pos + 2]);
  auto I = id2bb.find(id);
  if (I != id2bb.end())
    return I->second;
  Basic_block *bb = current_func->build_bb();
  id2bb[id] = bb;
  bb2id[bb] = id;
  return bb;
}

Basic_block *parser::get_bb_def(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
 if (tokens[idx].kind != lexeme::label_def)
   throw Parse_error("expected a label instead of "
		     + token_string(tokens[idx]), line_number);
  uint32_t id = get_u32(&buf[tokens[idx].pos + 2]);
  auto I = id2bb.find(id);
  if (I != id2bb.end())
    return I->second;
  Basic_block *bb = current_func->build_bb();
  id2bb[id] = bb;
  bb2id[bb] = id;
  return bb;
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

void parser::gen_cond_branch(Op opcode)
{
  Instruction *arg1 = get_reg_value(1);
  get_comma(2);
  Instruction *arg2 = get_reg_value(3);
  get_comma(4);
  Basic_block *true_bb = get_bb(5);
  get_end_of_line(6);

  Basic_block *false_bb = current_func->build_bb();
  Instruction *cond = current_bb->build_inst(opcode, arg1, arg2);
  current_bb->build_br_inst(cond, true_bb, false_bb);
  current_bb = false_bb;
}

void parser::gen_call()
{
  std::string name = get_name(1);
  get_end_of_line(2);

  throw Not_implemented("call " + name);
}

void parser::gen_tail()
{
  std::string name = get_name(1);
  get_end_of_line(2);

  throw Not_implemented("tail " + name);
}

void parser::store_ub_check(Instruction *ptr, uint64_t size)
{
  Instruction *ptr_mem_id = current_bb->build_extract_id(ptr);

  // It is UB to write to constant memory.
  Instruction *is_const = current_bb->build_inst(Op::IS_CONST_MEM, ptr_mem_id);
  current_bb->build_inst(Op::UB, is_const);

  // It is UB if the store overflow a memory object.
  Instruction *size_inst = current_bb->value_inst(size - 1, 32);
  Instruction *last_addr = current_bb->build_inst(Op::ADD, ptr, size_inst);
  Instruction *last_mem_id = current_bb->build_extract_id(last_addr);
  Instruction *is_ub = current_bb->build_inst(Op::NE, ptr_mem_id, last_mem_id);
  current_bb->build_inst(Op::UB, is_ub);

  // It is UB if the end is outside the memory object.
  Instruction *mem_size = current_bb->build_inst(Op::GET_MEM_SIZE, ptr_mem_id);
  Instruction *offset = current_bb->build_extract_offset(last_addr);
  Instruction *out_of_bound = current_bb->build_inst(Op::UGE, offset, mem_size);
  current_bb->build_inst(Op::UB, out_of_bound);
}

void parser::load_ub_check(Instruction *ptr, uint64_t size)
{
  Instruction *ptr_mem_id = current_bb->build_extract_id(ptr);

  // It is UB if the load overflow a memory object.
  Instruction *size_inst = current_bb->value_inst(size - 1, 32);
  Instruction *last_addr = current_bb->build_inst(Op::ADD, ptr, size_inst);
  Instruction *last_mem_id = current_bb->build_extract_id(last_addr);
  Instruction *is_ub = current_bb->build_inst(Op::NE, ptr_mem_id, last_mem_id);
  current_bb->build_inst(Op::UB, is_ub);

  // It is UB if the end is outside the memory object.
  Instruction *mem_size = current_bb->build_inst(Op::GET_MEM_SIZE, ptr_mem_id);
  Instruction *offset = current_bb->build_extract_offset(last_addr);
  Instruction *out_of_bound = current_bb->build_inst(Op::UGE, offset, mem_size);
  current_bb->build_inst(Op::UB, out_of_bound);
}

void parser::gen_load(int size, LStype lstype)
{
  Instruction *ptr;
  Instruction *dest;
  if (lstype == LStype::float_ls)
    dest = get_freg(1);
  else
    dest = get_reg(1);
  get_comma(2);
  Instruction *offset = get_imm(3);
  get_left_paren(4);
  Instruction *base = get_reg_value(5);
  get_right_paren(6);
  get_end_of_line(7);

  ptr = current_bb->build_inst(Op::ADD, base, offset);
  load_ub_check(ptr, size);
  Instruction *value = nullptr;
  for (int i = 0; i < size; i++)
    {
      Instruction *size_inst = current_bb->value_inst(i, ptr->bitsize);
      Instruction *addr = current_bb->build_inst(Op::ADD, ptr, size_inst);
      Instruction *byte = current_bb->build_inst(Op::LOAD, addr);
      if (value)
	value = current_bb->build_inst(Op::CONCAT, byte, value);
      else
	value = byte;
    }
  if (lstype == LStype::float_ls && size == 4)
    {
      Instruction *m1 = current_bb->value_m1_inst(32);
      value = current_bb->build_inst(Op::CONCAT, m1, value);
    }
  else if (value->bitsize < reg_bitsize)
    {
      Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
      Op op = lstype == LStype::unsigned_ls ? Op::ZEXT : Op::SEXT;
      value = current_bb->build_inst(op, value, bitsize);
    }
  current_bb->build_inst(Op::WRITE, dest, value);
}

void parser::gen_store(int size, LStype lstype)
{
  Instruction *ptr;
  Instruction *value;
  if (lstype == LStype::float_ls)
    value = get_freg_value(1);
  else
    value = get_reg_value(1);
  get_comma(2);
  Instruction *offset = get_imm(3);
  get_left_paren(4);
  Instruction *base = get_reg_value(5);
  get_right_paren(6);
  get_end_of_line(7);

  ptr = current_bb->build_inst(Op::ADD, base, offset);
  store_ub_check(ptr, size);
  for (int i = 0; i < size; i++)
    {
      Instruction *size_inst = current_bb->value_inst(i, ptr->bitsize);
      Instruction *addr = current_bb->build_inst(Op::ADD, ptr, size_inst);
      Instruction *high = current_bb->value_inst(i * 8 + 7, 32);
      Instruction *low = current_bb->value_inst(i * 8, 32);
      Instruction *byte = current_bb->build_inst(Op::EXTRACT, value, high, low);
      current_bb->build_inst(Op::STORE, addr, byte);
    }
}

void parser::gen_funary(std::string name, Op op)
{
  Instruction *dest = get_freg(1);
  get_comma(2);
  Instruction *arg1 = get_freg_value(3);
  get_end_of_line(4);

  bool is_single_prec =
    name[name.length() - 2] == '.' && name[name.length() - 1] == 's';
  if (is_single_prec)
    {
      arg1 = current_bb->build_trunc(arg1, 32);
    }
  Instruction *res = current_bb->build_inst(op, arg1);
  if (is_single_prec)
    {
      Instruction *m1 = current_bb->value_m1_inst(32);
      res = current_bb->build_inst(Op::CONCAT, m1, res);
    }
  current_bb->build_inst(Op::WRITE, dest, res);
}

void parser::gen_fbinary(std::string name, Op op)
{
  Instruction *dest = get_freg(1);
  get_comma(2);
  Instruction *arg1 = get_freg_value(3);
  get_comma(4);
  Instruction *arg2 = get_freg_value(5);
  get_end_of_line(6);

  bool is_single_prec =
    name[name.length() - 2] == '.' && name[name.length() - 1] == 's';
  if (is_single_prec)
    {
      arg1 = current_bb->build_trunc(arg1, 32);
      arg2 = current_bb->build_trunc(arg2, 32);
    }
  Instruction *res = current_bb->build_inst(op, arg1, arg2);
  if (is_single_prec)
    {
      Instruction *m1 = current_bb->value_m1_inst(32);
      res = current_bb->build_inst(Op::CONCAT, m1, res);
    }
  current_bb->build_inst(Op::WRITE, dest, res);
}

void parser::gen_fcmp(std::string name, Op op)
{
  Instruction *dest = get_reg(1);
  get_comma(2);
  Instruction *arg1 = get_freg_value(3);
  get_comma(4);
  Instruction *arg2 = get_freg_value(5);
  get_end_of_line(6);

  bool is_single_prec =
    name[name.length() - 2] == '.' && name[name.length() - 1] == 's';
  if (is_single_prec)
    {
      arg1 = current_bb->build_trunc(arg1, 32);
      arg2 = current_bb->build_trunc(arg2, 32);
    }
  Instruction *res = current_bb->build_inst(op, arg1, arg2);
  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
  res = current_bb->build_inst(Op::ZEXT, res, bitsize);
  current_bb->build_inst(Op::WRITE, dest, res);
}

void parser::parse_function()
{
  if (tokens[0].kind == lexeme::label_def)
  {
    Basic_block *bb = get_bb_def(0);
    get_end_of_line(1);

    if (current_bb)
      current_bb->build_br_inst(bb);
    current_bb = bb;
    return;
  }

  std::string name = get_name(&buf[tokens[0].pos]);
  if (name == "add" || name == "addw" || name == "addi" || name == "addiw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "addi" || name == "addiw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "addw" || name == "addiw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::ADD, arg1, arg2);
      if (name == "addw" || name == "addiw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "mul" || name == "mulw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "mulw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::MUL, arg1, arg2);
      if (name == "mulw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "mulh" || name == "mulhu" || name == "mulhsu")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
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
      Instruction *bitsize = current_bb->value_inst(2 * reg_bitsize, 32);
      arg1 = current_bb->build_inst(op1, arg1, bitsize);
      arg2 = current_bb->build_inst(op2, arg2, bitsize);
      Instruction *res = current_bb->build_inst(Op::MUL, arg1, arg2);
      Instruction *high = current_bb->value_inst(2 * reg_bitsize - 1, 32);
      Instruction *low = current_bb->value_inst(reg_bitsize, 32);
      res = current_bb->build_inst(Op::EXTRACT, res, high, low);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "div" || name == "divw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "divw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::SDIV, arg1, arg2);
      if (name == "divw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "divu" || name == "divuw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "divuw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::UDIV, arg1, arg2);
      if (name == "divuw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "rem" || name == "remw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "remw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::SREM, arg1, arg2);
      if (name == "remw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "remu" || name == "remuw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "remuw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::UREM, arg1, arg2);
      if (name == "remuw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "slt" || name == "sltw" || name == "slti" || name == "sltiw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "slti" || name == "sltiw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "sltw" || name == "sltiw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::SLT, arg1, arg2);
      Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
      res = current_bb->build_inst(Op::ZEXT, res, bitsize);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sltu" || name == "sltuw"
	   || name == "sltiu" || name == "sltiuw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "sltiu" || name == "sltiuw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "sltuw" || name == "sltiuw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::ULT, arg1, arg2);
      Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
      res = current_bb->build_inst(Op::ZEXT, res, bitsize);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sgt" || name == "sgtw")
    {
      // Pseudo instruction.
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "sgtuw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::SGT, arg1, arg2);
      Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
      res = current_bb->build_inst(Op::ZEXT, res, bitsize);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sgtu" || name == "sgtuw")
    {
      // Pseudo instruction.
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "sgtuw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::UGT, arg1, arg2);
      Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
      res = current_bb->build_inst(Op::ZEXT, res, bitsize);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "seqz" || name == "seqzw")
    {
      // Pseudo instruction.
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_end_of_line(4);

      if (name == "seqzw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	}
      Instruction *zero = current_bb->value_inst(0, arg1->bitsize);
      Instruction *res = current_bb->build_inst(Op::EQ, arg1, zero);
      Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
      res = current_bb->build_inst(Op::ZEXT, res, bitsize);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "snez" || name == "snezw")
    {
      // Pseudo instruction.
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_end_of_line(4);

      if (name == "snezw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	}
      Instruction *zero = current_bb->value_inst(0, arg1->bitsize);
      Instruction *res = current_bb->build_inst(Op::NE, arg1, zero);
      Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
      res = current_bb->build_inst(Op::ZEXT, res, bitsize);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "and" || name == "andw" || name == "andi" || name == "andiw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "andi" || name == "andiw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "andw" || name == "andiw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::AND, arg1, arg2);
      if (name == "andw" || name == "andiw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "or" || name == "orw" || name == "ori" || name == "oriw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "ori" || name == "oriw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "orw" || name == "oriw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::OR, arg1, arg2);
      if (name == "orw" || name == "oriw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "xor" || name == "xorw" || name == "xori" || name == "xoriw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "xori" || name == "xoriw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "xorw" || name == "xoriw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::XOR, arg1, arg2);
      if (name == "xorw" || name == "xoriw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sll" || name == "sllw" || name == "slli" || name == "slliw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "slli" || name == "slliw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (reg_bitsize == 32 || name == "sllw" || name == "slliw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 5);
	  Instruction *bitsize = current_bb->value_inst(32, 32);
	  arg2 = current_bb->build_inst(Op::ZEXT, arg2, bitsize);
	}
      else
	{
	  arg2 = current_bb->build_trunc(arg2, 6);
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  arg2 = current_bb->build_inst(Op::ZEXT, arg2, bitsize);
	}
      Instruction *res = current_bb->build_inst(Op::SHL, arg1, arg2);
      if (name == "sllw" || name == "slliw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "srl" || name == "srlw" || name == "srli" || name == "srliw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "srli" || name == "srliw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (reg_bitsize == 32 || name == "srlw" || name == "srliw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 5);
	  Instruction *bitsize = current_bb->value_inst(32, 32);
	  arg2 = current_bb->build_inst(Op::ZEXT, arg2, bitsize);
	}
      else
	{
	  arg2 = current_bb->build_trunc(arg2, 6);
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  arg2 = current_bb->build_inst(Op::ZEXT, arg2, bitsize);
	}
      Instruction *res = current_bb->build_inst(Op::LSHR, arg1, arg2);
      if (name == "srlw" || name == "srliw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sra" || name == "sraw" || name == "srai" || name == "sraiw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2;
      if (name == "srai" || name == "sraiw")
	arg2 = get_imm(5);
      else
	arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (reg_bitsize == 32 || name == "sraw" || name == "sraiw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 5);
	  Instruction *bitsize = current_bb->value_inst(32, 32);
	  arg2 = current_bb->build_inst(Op::ZEXT, arg2, bitsize);
	}
      else
	{
	  arg2 = current_bb->build_trunc(arg2, 6);
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  arg2 = current_bb->build_inst(Op::ZEXT, arg2, bitsize);
	}
      Instruction *res = current_bb->build_inst(Op::ASHR, arg1, arg2);
      if (name == "sraw" || name == "sraiw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sub" || name == "subw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_comma(4);
      Instruction *arg2 = get_reg_value(5);
      get_end_of_line(6);

      if (name == "subw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	  arg2 = current_bb->build_trunc(arg2, 32);
	}
      Instruction *res = current_bb->build_inst(Op::SUB, arg1, arg2);
      if (name == "subw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "neg" || name == "negw")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_end_of_line(4);

      if (name == "negw")
	{
	  arg1 = current_bb->build_trunc(arg1, 32);
	}
      Instruction *res = current_bb->build_inst(Op::NEG, arg1);
      if (name == "negw")
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "sext.w")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_end_of_line(4);

      Instruction *res = current_bb->build_trunc(arg1, 32);
      Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
      res = current_bb->build_inst(Op::SEXT, res, bitsize);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "not")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_end_of_line(4);

      Instruction *res = current_bb->build_inst(Op::NOT, arg1);
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "mv")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *arg1 = get_reg_value(3);
      get_end_of_line(4);

      current_bb->build_inst(Op::WRITE, dest, arg1);
    }
  else if (name == "li")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      // TODO: Use a correct wrapper.
      //       Sort of get_imm(3); but with correct size.
      unsigned __int128 value = get_hex_or_integer(3);
      Instruction *arg1 = current_bb->value_inst(value, reg_bitsize);
      get_end_of_line(4);

      current_bb->build_inst(Op::WRITE, dest, arg1);
    }
  else if (name == "lui")
    {
      Instruction *dest = get_reg(1);
      get_comma(2);
      Instruction *res = get_hi(3);
      get_end_of_line(4);

      if (reg_bitsize > 32)
	{
	  Instruction *bitsize = current_bb->value_inst(reg_bitsize, 32);
	  res = current_bb->build_inst(Op::SEXT, res, bitsize);
	}
      current_bb->build_inst(Op::WRITE, dest, res);
    }
  else if (name == "call")
    gen_call();
  else if (name == "tail")
    gen_tail();
  else if (name == "ld" && reg_bitsize == 64)
    gen_load(8);
  else if (name == "lw")
    gen_load(4);
  else if (name == "lh")
    gen_load(2);
  else if (name == "lhu")
    gen_load(2, LStype::unsigned_ls);
  else if (name == "lb")
    gen_load(1);
  else if (name == "lbu")
    gen_load(1, LStype::unsigned_ls);
  else if (name == "sd" && reg_bitsize == 64)
    gen_store(8);
  else if (name == "sw")
    gen_store(4);
  else if (name == "sh")
    gen_store(2);
  else if (name == "sb")
    gen_store(1);
  else if (name == "beq")
    gen_cond_branch(Op::EQ);
  else if (name == "bne")
    gen_cond_branch(Op::NE);
  else if (name == "ble")
    gen_cond_branch(Op::SLE);
  else if (name == "bleu")
    gen_cond_branch(Op::ULE);
  else if (name == "blt")
    gen_cond_branch(Op::SLT);
  else if (name == "bltu")
    gen_cond_branch(Op::ULT);
  else if (name == "bge")
    gen_cond_branch(Op::SGE);
  else if (name == "bgeu")
    gen_cond_branch(Op::UGE);
  else if (name == "bgt")
    gen_cond_branch(Op::SGT);
  else if (name == "bgtu")
    gen_cond_branch(Op::UGT);
  else if (name == "j")
    {
      Basic_block *dest_bb = get_bb(1);
      get_end_of_line(2);

      current_bb->build_br_inst(dest_bb);
      current_bb = nullptr;
    }
  else if (name == "ebreak")
    {
      current_bb->build_inst(Op::UB, current_bb->value_inst(1, 1));
      ret_bbs.push_back(current_bb);
      current_bb = nullptr;
    }
  else if (name == "ret" || name == "jr")
    {
      // TODO: jr and ret are pseudoinstructions. Verify that they
      // jump to the correct location.
      ret_bbs.push_back(current_bb);
      current_bb = nullptr;
    }
  else if (name == "fld")
    gen_load(8, LStype::float_ls);
  else if (name == "flw")
    gen_load(4, LStype::float_ls);
  else if (name == "fsd")
    gen_store(8, LStype::float_ls);
  else if (name == "fsw")
    gen_store(4, LStype::float_ls);
  else if (name == "fabs.s" || name == "fabs.d")
    gen_funary(name, Op::FABS);
  else if (name == "fneg.s" || name == "fneg.d")
    gen_funary(name, Op::FNEG);
  else if (name == "fadd.s" || name == "fadd.d")
    gen_fbinary(name, Op::FADD);
  else if (name == "fsub.s" || name == "fsub.d")
    gen_fbinary(name, Op::FSUB);
  else if (name == "fmul.s" || name == "fmul.d")
    gen_fbinary(name, Op::FMUL);
  else if (name == "fdiv.s" || name == "fdiv.d")
    gen_fbinary(name, Op::FDIV);
  else if (name == "feq.s" || name == "feq.d")
    gen_fcmp(name, Op::FEQ);
  else if (name == "flt.s" || name == "flt.d")
    gen_fcmp(name, Op::FLT);
  else if (name == "fle.s" || name == "fle.d")
    gen_fcmp(name, Op::FLE);
  else if (name == "fgt.s" || name == "fgt.d")
    gen_fcmp(name, Op::FGT);
  else if (name == "fge.s" || name == "fge.d")
    gen_fcmp(name, Op::FGE);
  else
    throw Parse_error("unhandled instruction: "s + name, line_number);
}

void parser::lex_line(void)
{
  pos = 0;
  tokens.clear();
  while (buf[pos])
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
}

Function *parser::parse(std::string const& file_name, riscv_state *rstate)
{
  enum class state {
    global,
    function,
    basic_block,
    instruction,
    done
  };

  std::ifstream in(file_name);
  if (!in)
    throw Parse_error("Could not open file.", 0);

  module = rstate->module;
  reg_bitsize = rstate->reg_bitsize;
  assert(module->functions.size() == 1);
  src_func = module->functions[0];

  state parser_state = state::global;
  while (parser_state != state::done && in.getline(buf, max_line_len)) {
    line_number++;

    // The parsing code has problems with Unicode characters in labels.
    // Report "not implemented" for all uses of non-ASCII characters for
    // now.
    for (std::streamsize i = 0; i < in.gcount(); i++)
      {
	if ((unsigned char)buf[i] > 127)
	  throw Not_implemented("non-ASCII character in assembly file");
      }

    if (parser_state == state::global)
      {
	// TODO: Implement real parsing.
	// Just eat lines until we find "foo:" for now.
	size_t len = strlen(buf);
	if (len < 2 || buf[len - 1] != ':')
	  continue;
	buf[len - 1] = 0;
	if (buf == rstate->func_name)
	  {
	    current_func = module->build_function("tgt");
	    Basic_block *entry_bb = current_func->build_bb();
	    for (int i = 0; i < 32; i++)
	      {
		Instruction *bitsize = entry_bb->value_inst(reg_bitsize, 32);
		Instruction *reg = entry_bb->build_inst(Op::REGISTER, bitsize);
		registers.push_back(reg);
	      }
	    for (int i = 0; i < 32; i++)
	      {
		Instruction *bitsize = entry_bb->value_inst(64, 32);
		Instruction *reg = entry_bb->build_inst(Op::REGISTER, bitsize);
		fregisters.push_back(reg);
	      }

	    Basic_block *bb = current_func->build_bb();
	    entry_bb->build_br_inst(bb);

	    current_bb = bb;

	    for (const auto& mem_obj : rstate->memory_objects)
	      {
		Instruction *id =
		  current_bb->value_inst(mem_obj.id, module->ptr_id_bits);
		Instruction *size =
		  current_bb->value_inst(mem_obj.size, module->ptr_offset_bits);
		Instruction *flags = current_bb->value_inst(mem_obj.flags, 32);
		Instruction *inst =
		  entry_bb->build_inst(Op::MEMORY, id, size, flags);
		sym_name2mem.insert({mem_obj.sym_name, inst});
	      }

	    // Set up the stack.
	    // TODO: Set up memory consistent with the src function.
	    Instruction *id = bb->value_inst(-128, module->ptr_id_bits);
	    Instruction *mem_size =
	      bb->value_inst(stack_size, module->ptr_offset_bits);
	    Instruction *flags = bb->value_inst(0, 32);
	    Instruction *stack =
	      entry_bb->build_inst(Op::MEMORY, id, mem_size, flags);
	    Instruction *size = bb->value_inst(stack_size, stack->bitsize);
	    stack = bb->build_inst(Op::ADD, stack, size);
	    current_bb->build_inst(Op::WRITE, registers[2], stack);

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
	    // Make this valid by branching back to current_bb (this is then
	    // removed when building RPO as the BB is unreachable).
	    if (current_bb)
	      {
		current_bb->build_br_inst(current_bb);
		current_bb = nullptr;
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

  if (in.gcount() >= max_line_len - 1)
    throw Parse_error("line too long", line_number);
  if (parser_state != state::done)
    throw Parse_error("EOF in the middle of a function", line_number);

  // TODO: This should be a state check to ensure we are not within a function.
  // Hmm. But we probably want to check size too, but with a throw if 0.
  // Note: ret_bbs.size()); may be 0 for e.g. a function such as
  // int foo(void) {
  //     __builtin_trap();
  // }
  // Hmm. But we should treat the ebreak as a return in that case.
  Basic_block *exit_bb = current_func->build_bb();
  for (auto bb : ret_bbs)
    bb->build_br_inst(exit_bb);
  exit_bb->build_ret_inst();

  rstate->registers = registers;

  return current_func;
}

} // end anonymous namespace

Function *parse_riscv(std::string const& file_name, riscv_state *state)
{
  parser p;
  Function *func = p.parse(file_name, state);
  reverse_post_order(func);
  validate(func);
  return func;
}

} // end namespace smtgcc
