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
  Parser(m68k_state *rstate)
    : ParserBase(rstate->sym_name2mem)
    , rstate{rstate} {}

  enum class Cond_code {
    CC, CS, EQ, GE, GT, HI, LE, LS, LT, MI, NE, PL, VC, VS, F, T
  };

  enum class Lexeme {
    label,
    label_def,
    name,
    areg,
    dreg,
    freg,
    integer,
    hex,
    comma,
    at,
    colon,
    hash,
    plus,
    minus,
    asterisk,
    left_paren,
    right_paren,
    left_brace,
    right_brace,
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

  m68k_state *rstate;
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
  void lex_string();
  void lex_name();
  void lex_reg();

  uint32_t get_u32(const char *p);
  unsigned __int128 get_hex(const char *p);
  Inst *get_integer(unsigned idx, uint32_t bitsize = 32);
  Inst *get_hex(unsigned idx, uint32_t bitsize = 32);
  Inst *get_hex_or_integer(unsigned idx, uint32_t bitsize = 32);
  bool is_kind(unsigned idx, Lexeme kind);
  Inst *get_areg(unsigned idx);
  Inst *get_areg_value(unsigned idx, uint32_t bitsize = 32);
  Inst *get_dreg(unsigned idx);
  Inst *get_dreg_value(unsigned idx, uint32_t bitsize = 32);
  std::pair<Inst *, Inst *> get_freg(unsigned idx);
  Inst *get_freg_value(unsigned idx, uint32_t bitsize);
  std::pair<Inst*, unsigned> load_arg(unsigned idx, uint32_t bitsize);
  unsigned store_arg(unsigned idx, Inst *value);
  void write_areg(Inst *reg, Inst *value);
  void write_dreg(Inst *reg, Inst *value);
  void write_freg(std::pair<Inst *, Inst *> reg, Inst *value);
  void set_xnzvc(Inst *x, Inst *n, Inst *z, Inst *v, Inst *c);
  void set_nz00(Inst *inst);

  std::string_view token_string(const Token& tok);
  std::string_view get_name(unsigned idx);
  void build_mem(const std::string& sym_name);
  std::pair<Inst *, unsigned> get_addr_name(unsigned idx);
  std::pair<Inst *, unsigned> get_addr(unsigned idx, uint32_t access_bitsize);
  Basic_block *get_bb(unsigned idx);
  Basic_block *get_bb_def(unsigned idx);
  void get_comma(unsigned idx);
  void get_hash(unsigned idx);
  void get_left_paren(unsigned idx);
  void get_right_paren(unsigned idx);
  void get_end_of_line(unsigned idx);
  Inst *build_cond(Cond_code cc);
  void process_binary_bitwise(Op op, uint32_t bitsize);
  void process_add(uint32_t bitsize);
  void process_sub(uint32_t bitsize);
  void process_fbinary(Op op, uint32_t bitsize);
  void process_shift(Op op, uint32_t bitsize);
  void process_clr(uint32_t bitsize);
  void process_cmp(uint32_t bitsize);
  void process_ext(uint32_t src_bitsize, uint32_t dest_bitsize);
  void process_fmove(uint32_t bitsize);
  void process_jcc(Cond_code cc);
  void process_lea();
  void process_move(uint32_t bitsize);
  void process_moveq();
  void process_neg(uint32_t bitsize);
  void process_scc(Cond_code cc);
  void process_tst(uint32_t bitsize);
  void process_call();

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
  if (!isdigit(buf[pos]))
    throw Parse_error("expected a digit after \".L\"", line_number);
  pos++;
  while (isdigit(buf[pos]))
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

void Parser::lex_reg()
{
  assert(buf[pos] == '%');
  int start_pos = pos++;
  if (buf[pos] == 'd' && ('0' <= buf[pos + 1] && buf[pos + 1] <= '7'))
    {
      pos += 2;
      tokens.emplace_back(Lexeme::dreg, start_pos, pos - start_pos);
    }
  else if (buf[pos] == 'a' && ('0' <= buf[pos + 1] && buf[pos + 1] <= '7'))
    {
      pos += 2;
      tokens.emplace_back(Lexeme::areg, start_pos, pos - start_pos);
    }
  else if (buf[pos] == 'f'
	   && buf[pos + 1] == 'p'
	   && ('0' <= buf[pos + 2] && buf[pos + 2] <= '7'))
    {
      pos += 3;
      tokens.emplace_back(Lexeme::freg, start_pos, pos - start_pos);
    }
  else if (buf[pos] == 's' && buf[pos + 1] == 'p')
    {
      pos += 2;
      tokens.emplace_back(Lexeme::areg, start_pos, pos - start_pos);
    }
  else
    throw Parse_error("invalid register", line_number);
}

uint32_t Parser::get_u32(const char *p)
{
  assert(isdigit(*p));
  unsigned __int128 value = 0;
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

Inst *Parser::get_integer(unsigned idx, uint32_t bitsize)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::integer)
    throw Parse_error("expected a decimal integer instead of "
		      + std::string(token_string(tokens[idx])), line_number);

  int pos = tokens[idx].pos;
  if (buf[pos] == '-')
    pos++;
  unsigned __int128 val = get_u32(&buf[pos]);
  if (buf[tokens[idx].pos] == '-')
    val = -val;
  return bb->value_inst(val, bitsize);
}

Inst *Parser::get_hex(unsigned idx, uint32_t bitsize)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::hex)
    throw Parse_error("expected a hexadecimal integer instead of "
		      + std::string(token_string(tokens[idx])), line_number);

  int pos = tokens[idx].pos;
  unsigned __int128 val = get_hex(&buf[pos]);
  return bb->value_inst(val, bitsize);
}

Inst *Parser::get_hex_or_integer(unsigned idx, uint32_t bitsize)
{
  if (is_kind(idx, Lexeme::integer))
    return get_integer(idx, bitsize);
  else
    return get_hex(idx, bitsize);
}

bool Parser::is_kind(unsigned idx, Lexeme kind)
{
  if (tokens.size() <= idx)
    return false;
  return tokens[idx].kind == kind;
}

Inst *Parser::get_areg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::areg)
    throw Parse_error("expected an a-register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  uint32_t value;
  if (buf[tokens[idx].pos + 1] == 's')
    value = 7;
  else
    value = buf[tokens[idx].pos + 2] - '0';
  return rstate->registers[M68kRegIdx::a0 + value];
}

Inst *Parser::get_areg_value(unsigned idx, uint32_t bitsize)
{
  return bb->build_trunc(bb->build_inst(Op::READ, get_areg(idx)), bitsize);
}

Inst *Parser::get_dreg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::dreg)
    throw Parse_error("expected a d-register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  uint32_t value = buf[tokens[idx].pos + 2] - '0';
  return rstate->registers[M68kRegIdx::d0 + value];
}

Inst *Parser::get_dreg_value(unsigned idx, uint32_t bitsize)
{
  return bb->build_trunc(bb->build_inst(Op::READ, get_dreg(idx)), bitsize);
}

std::pair<Inst *, Inst *> Parser::get_freg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != Lexeme::freg)
    throw Parse_error("expected a fp-register instead of "
		      + std::string(token_string(tokens[idx])), line_number);
  uint32_t value = buf[tokens[idx].pos + 3] - '0';
  Inst *reg64 = rstate->registers[M68kRegIdx::fp0_64 + value];
  Inst * reg32 = rstate->registers[M68kRegIdx::fp0_32 + value];
  return {reg64, reg32};
}

Inst *Parser::get_freg_value(unsigned idx, uint32_t bitsize)
{
  assert(bitsize == 64 || bitsize == 32);
  auto [reg64, reg32] = get_freg(idx);
  return bb->build_inst(Op::READ, bitsize == 64 ? reg64 : reg32);
}

std::pair<Inst*, unsigned> Parser::load_arg(unsigned idx, uint32_t bitsize)
{
  auto [ptr, new_idx] = get_addr(idx, bitsize);
  Inst *value = bb->build_inst(Op::LOAD_BE, ptr, bitsize / 8);
  return {value, new_idx};
}

unsigned Parser::store_arg(unsigned idx, Inst *value)
{
  auto [ptr, new_idx] = get_addr(idx, value->bitsize);
  bb->build_inst(Op::STORE_BE, ptr, value);
  return new_idx;
}

void Parser::write_areg(Inst *reg, Inst *value)
{
  assert(reg->op == Op::REGISTER);
  assert(value->bitsize == 32);
  bb->build_inst(Op::WRITE, reg, value);
}

void Parser::write_dreg(Inst *reg, Inst *value)
{
  assert(reg->op == Op::REGISTER);
  assert(value->bitsize == 8 || value->bitsize == 16 || value->bitsize == 32);
  if (value->bitsize < 32)
    {
      Inst *orig = bb->build_inst(Op::READ, reg);
      Inst *high_part = bb->build_inst(Op::EXTRACT, orig, 31, value->bitsize);
      value = bb->build_inst(Op::CONCAT, high_part, value);
    }
  bb->build_inst(Op::WRITE, reg, value);
}

void Parser::write_freg(std::pair<Inst *, Inst *> reg, Inst *value)
{
  assert(value->bitsize == 64 || value->bitsize == 32);
  if (value->bitsize == 64)
    {
      bb->build_inst(Op::WRITE, reg.first, value);
      Inst *value32 = bb->build_inst(Op::FCHPREC, value, 32);
      bb->build_inst(Op::WRITE, reg.second, value32);
    }
  else
    {
      Inst *value64 = bb->build_inst(Op::FCHPREC, value, 64);
      bb->build_inst(Op::WRITE, reg.first, value64);
      bb->build_inst(Op::WRITE, reg.second, value);
    }
}

void Parser::set_xnzvc(Inst *x, Inst *n, Inst *z, Inst *v, Inst *c)
{
  if (x)
    bb->build_inst(Op::WRITE, rstate->registers[M68kRegIdx::x], x);
  if (n)
    bb->build_inst(Op::WRITE, rstate->registers[M68kRegIdx::n], n);
  if (z)
    bb->build_inst(Op::WRITE, rstate->registers[M68kRegIdx::z], z);
  if (v)
    bb->build_inst(Op::WRITE, rstate->registers[M68kRegIdx::v], v);
  if (c)
    bb->build_inst(Op::WRITE, rstate->registers[M68kRegIdx::c], c);
}

// Set the condition flags so that:
//  X is unchanged.
//  N and Z are set according to inst.
//  V and C are set to 0.
void Parser::set_nz00(Inst *inst)
{
  Inst *zero = bb->value_inst(0, inst->bitsize);
  Inst *n = bb->build_inst(Op::SLT, inst, zero);
  Inst *z = bb->build_inst(Op::EQ, inst, zero);
  Inst *v = bb->value_inst(0, 1);
  Inst *c = bb->value_inst(0, 1);
  set_xnzvc(nullptr, n, z, v, c);
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
  std::vector<Inst *>& data = sym_name2data.at(sym_name);

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
      entry_bb->build_inst(Op::STORE, ptr, data[i]);
    }
}

// Handle addresses of the form:
//  * name
//  * name + 16
//  * name - 16
//  * (name)
//  * (name + 16)
//  * (name - 16)
std::pair<Inst *, unsigned> Parser::get_addr_name(unsigned idx)
{
  bool has_paren = is_kind(idx, Lexeme::left_paren);
  if (has_paren)
    idx++;
  std::string name = std::string(get_name(idx++));
  uint64_t offset = 0;
  if (sym_alias.contains(name))
    std::tie(name, offset) = sym_alias[name];
  if (!rstate->sym_name2mem.contains(name))
    build_mem(name);
  Inst *inst = rstate->sym_name2mem[name];
  if (offset)
    inst = bb->build_inst(Op::ADD, inst, bb->value_inst(offset, inst->bitsize));
  if (is_kind(idx, Lexeme::plus))
    {
      idx++;
      inst = bb->build_inst(Op::ADD, inst, get_hex_or_integer(idx++, 32));
    }
  else if (is_kind(idx, Lexeme::minus))
    {
      idx++;
      inst = bb->build_inst(Op::SUB, inst, get_hex_or_integer(idx++, 32));
    }
  if (has_paren)
    get_right_paren(idx++);
  return {inst, idx};
}


std::pair<Inst *, unsigned> Parser::get_addr(unsigned idx, uint32_t access_bitsize)
{
  if (is_kind(idx, Lexeme::name)
      || (is_kind(idx, Lexeme::left_paren) && is_kind(idx + 1, Lexeme::name)))
    return get_addr_name(idx);

  bool minus = false;
  if (is_kind(idx, Lexeme::minus))
    {
      minus = true;
      idx++;
    }
  Inst *value = nullptr;
  if (is_kind(idx, Lexeme::integer))
    {
      value = get_hex_or_integer(idx, 32);
      if (minus)
	{
	  value = bb->build_inst(Op::NEG, value);
	  minus = false;
	}
      idx++;
      if (is_kind(idx, Lexeme::name))
	{
	  // Integer addresses may have a ".w" prefix if it fits in 16
	  // bits. Skip it, if present.
	  std::string_view name = get_name(idx);
	  if (name == ".w")
	    idx++;
	}
    }
  Inst *inst = nullptr;
  if (is_kind(idx, Lexeme::left_paren))
    {
      get_left_paren(idx++);
      Inst *reg = get_areg(idx);
      inst = get_areg_value(idx++);
      if (minus)
	{
	  Inst *offset = bb->value_inst(access_bitsize / 8, 32);
	  inst = bb->build_inst(Op::SUB, inst, offset);
	  write_areg(reg, inst);
	}
      if (is_kind(idx, Lexeme::comma))
	throw Parse_error("Register indirect with index not implemented",
			  line_number);
      get_right_paren(idx++);
      if (!value && is_kind(idx, Lexeme::plus))
	{
	  assert(!minus);
	  Inst *offset = bb->value_inst(access_bitsize / 8, 32);
	  write_areg(reg, bb->build_inst(Op::ADD, inst, offset));
	  idx++;
	}
    }
  if (!inst && !value)
    throw Parse_error("expected an address", line_number);
  if (!inst)
    return {value, idx};
  if (value)
    inst = bb->build_inst(Op::ADD, inst, value);
  return {inst, idx};
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

void Parser::get_hash(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != Lexeme::hash)
    throw Parse_error("expected a '#' after "
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

Inst *Parser::build_cond(Cond_code cc)
{
  switch (cc)
    {
    case Cond_code::CC:
      {
	Inst *c = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::c]);
	return bb->build_inst(Op::NOT, c);
      }
    case Cond_code::CS:
      return bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::c]);
    case Cond_code::EQ:
      return bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::z]);
    case Cond_code::GE:
      {
	Inst *n = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::n]);
	Inst *not_n = bb->build_inst(Op::NOT, n);
	Inst *v = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::v]);
	Inst *not_v = bb->build_inst(Op::NOT, v);
	Inst *cond1 = bb->build_inst(Op::AND, n, v);
	Inst *cond2 = bb->build_inst(Op::AND, not_n, not_v);
	return bb->build_inst(Op::OR, cond1, cond2);
      }
    case Cond_code::GT:
      {
	Inst *ge = build_cond(Cond_code::GE);
	Inst *ne = build_cond(Cond_code::NE);
	return bb->build_inst(Op::AND, ge, ne);
      }
    case Cond_code::HI:
      {
	Inst *cc = build_cond(Cond_code::CC);
	Inst *ne = build_cond(Cond_code::NE);
	return bb->build_inst(Op::AND, cc, ne);
      }
    case Cond_code::LE:
      {
	Inst *lt = build_cond(Cond_code::LT);
	Inst *eq = build_cond(Cond_code::EQ);
	return bb->build_inst(Op::OR, lt, eq);
      }
    case Cond_code::LS:
      {
	Inst *cs = build_cond(Cond_code::CS);
	Inst *eq = build_cond(Cond_code::EQ);
	return bb->build_inst(Op::OR, cs, eq);
      }
    case Cond_code::LT:
      {
	Inst *n = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::n]);
	Inst *not_n = bb->build_inst(Op::NOT, n);
	Inst *v = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::v]);
	Inst *not_v = bb->build_inst(Op::NOT, v);
	Inst *cond1 = bb->build_inst(Op::AND, n, not_v);
	Inst *cond2 = bb->build_inst(Op::AND, not_n, v);
	return bb->build_inst(Op::OR, cond1, cond2);
      }
    case Cond_code::MI:
      return bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::n]);
    case Cond_code::NE:
      {
	Inst *z = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::z]);
	return bb->build_inst(Op::NOT, z);
      }
    case Cond_code::PL:
      {
	Inst *n = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::n]);
	return bb->build_inst(Op::NOT, n);
      }
    case Cond_code::VC:
      {
	Inst *v = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::v]);
	return bb->build_inst(Op::NOT, v);
      }
    case Cond_code::VS:
      return bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::v]);
    case Cond_code::F:
      return bb->value_inst(0, 1);
    case Cond_code::T:
      return bb->value_inst(1, 1);
    }

  throw Parse_error("unhandled condition code", line_number);
}

void Parser::process_binary_bitwise(Op op, uint32_t bitsize)
{
  Inst *arg1;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::dreg))
    arg1 = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::hash))
    {
      idx++;
      if (is_kind(idx, Lexeme::name))
	std::tie(arg1, idx) = get_addr(idx, bitsize);
      else
	arg1 = get_hex_or_integer(idx++, bitsize);
    }
  else
    std::tie(arg1, idx) = load_arg(idx, bitsize);
  get_comma(idx++);

  Inst *arg2;
  unsigned dest_idx = idx;
  if (is_kind(idx, Lexeme::dreg))
    arg2 = get_dreg_value(idx++, bitsize);
  else
    std::tie(arg2, idx) = load_arg(idx, bitsize);
  get_end_of_line(idx);

  Inst *res = bb->build_inst(op, arg1, arg2);

  set_nz00(res);

  idx = dest_idx;
  if (is_kind(idx, Lexeme::dreg))
    write_dreg(get_dreg(idx++), res);
  else
    store_arg(idx, res);
}

void Parser::process_add(uint32_t bitsize)
{
  Inst *arg1;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::dreg))
    arg1 = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::areg))
    arg1 = get_areg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::hash))
    {
      idx++;
      if (is_kind(idx, Lexeme::name))
	std::tie(arg1, idx) = get_addr(idx, bitsize);
      else
	arg1 = get_hex_or_integer(idx++, bitsize);
    }
  else
    std::tie(arg1, idx) = load_arg(idx, bitsize);
  get_comma(idx++);

  Inst *arg2;
  unsigned dest_idx = idx;
  bool update_cc = true;
  if (is_kind(idx, Lexeme::dreg))
    arg2 = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::areg))
    {
      update_cc = false;
      arg2 = get_areg_value(idx++, 32);
      if (bitsize < 32)
	{
	  arg1 = bb->build_inst(Op::SEXT, arg1, 32);
	  bitsize = 32;
	}
    }
  else
    std::tie(arg2, idx) = load_arg(idx, bitsize);
  get_end_of_line(idx);

  Inst *res = bb->build_inst(Op::ADD, arg2, arg1);

  if (update_cc)
    {
      // Extract the sign bits for the values.
      Inst *sm = bb->build_extract_bit(arg1, bitsize - 1);
      Inst *not_sm = bb->build_inst(Op::NOT, sm);
      Inst *dm = bb->build_extract_bit(arg2, bitsize - 1);
      Inst *not_dm = bb->build_inst(Op::NOT, dm);

      Inst *rm = bb->build_extract_bit(res, bitsize - 1);
      Inst *not_rm = bb->build_inst(Op::NOT, rm);

      Inst *zero = bb->value_inst(0, bitsize);
      Inst *n = rm;
      Inst *z = bb->build_inst(Op::EQ, res, zero);

      // V = (Sm & Dm & !Rm) | (!Sm & !Dm & Rm)
      Inst *v1 =
	bb->build_inst(Op::AND, bb->build_inst(Op::AND, sm, dm), not_rm);
      Inst *v2 =
	bb->build_inst(Op::AND, bb->build_inst(Op::AND, not_sm, not_dm), rm);
      Inst *v = bb->build_inst(Op::OR, v1, v2);

      // C = (Sm & Dm) | (!Rm & Dm) | (Sm & !Rm)
      Inst *c1 = bb->build_inst(Op::AND, sm, dm);
      Inst *c2 = bb->build_inst(Op::AND, not_rm, dm);
      Inst *c3 = bb->build_inst(Op::AND, sm, not_rm);
      Inst *c = bb->build_inst(Op::OR, bb->build_inst(Op::OR, c1, c2), c3);

      Inst *x = c;
      set_xnzvc(x, n, z, v, c);
    }

  idx = dest_idx;
  if (is_kind(idx, Lexeme::dreg))
    write_dreg(get_dreg(idx++), res);
  else if (is_kind(idx, Lexeme::areg))
    write_dreg(get_areg(idx++), res);
  else
    store_arg(idx, res);
}

void Parser::process_sub(uint32_t bitsize)
{
  Inst *arg1;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::dreg))
    arg1 = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::areg))
    arg1 = get_areg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::hash))
    {
      idx++;
      if (is_kind(idx, Lexeme::name))
	std::tie(arg1, idx) = get_addr(idx, bitsize);
      else
	arg1 = get_hex_or_integer(idx++, bitsize);
    }
  else
    std::tie(arg1, idx) = load_arg(idx, bitsize);
  get_comma(idx++);

  Inst *arg2;
  unsigned dest_idx = idx;
  bool update_cc = true;
  if (is_kind(idx, Lexeme::dreg))
    arg2 = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::areg))
    {
      update_cc = false;
      arg2 = get_areg_value(idx++, 32);
      if (bitsize < 32)
	{
	  arg1 = bb->build_inst(Op::SEXT, arg1, 32);
	  bitsize = 32;
	}
    }
  else
    std::tie(arg2, idx) = load_arg(idx, bitsize);
  get_end_of_line(idx);

  Inst *res = bb->build_inst(Op::SUB, arg2, arg1);

  if (update_cc)
    {
      // Extract the sign bits for the values.
      Inst *sm = bb->build_extract_bit(arg1, bitsize - 1);
      Inst *not_sm = bb->build_inst(Op::NOT, sm);
      Inst *dm = bb->build_extract_bit(arg2, bitsize - 1);
      Inst *not_dm = bb->build_inst(Op::NOT, dm);

      Inst *rm = bb->build_extract_bit(res, bitsize - 1);
      Inst *not_rm = bb->build_inst(Op::NOT, rm);

      Inst *zero = bb->value_inst(0, bitsize);
      Inst *n = rm;
      Inst *z = bb->build_inst(Op::EQ, res, zero);

      // V = (!Sm & Dm & !Rm) | (Sm & !Dm & Rm)
      Inst *v1 =
	bb->build_inst(Op::AND, bb->build_inst(Op::AND, not_sm, dm), not_rm);
      Inst *v2 =
	bb->build_inst(Op::AND, bb->build_inst(Op::AND, sm, not_dm), rm);
      Inst *v = bb->build_inst(Op::OR, v1, v2);

      // C = (Sm & !Dm) | (Rm & !Dm) | (Sm & Rm)
      Inst *c1 = bb->build_inst(Op::AND, sm, not_dm);
      Inst *c2 = bb->build_inst(Op::AND, rm, not_dm);
      Inst *c3 = bb->build_inst(Op::AND, sm, rm);
      Inst *c = bb->build_inst(Op::OR, bb->build_inst(Op::OR, c1, c2), c3);

      Inst *x = c;
      set_xnzvc(x, n, z, v, c);
    }

  idx = dest_idx;
  if (is_kind(idx, Lexeme::dreg))
    write_dreg(get_dreg(idx++), res);
  else if (is_kind(idx, Lexeme::areg))
    write_dreg(get_areg(idx++), res);
  else
    store_arg(idx, res);
}

void Parser::process_fbinary(Op op, uint32_t bitsize)
{
  Inst *arg1;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::freg))
    arg1 = get_freg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::hash))
    {
      idx++;
      if (is_kind(idx, Lexeme::name))
	std::tie(arg1, idx) = get_addr(idx, bitsize);
      else
	arg1 = get_hex_or_integer(idx++, bitsize);
    }
  else
    std::tie(arg1, idx) = load_arg(idx, bitsize);
  get_comma(idx++);

  Inst *arg2;
  unsigned dest_idx = idx;
  if (is_kind(idx, Lexeme::freg))
    arg2 = get_freg_value(idx++, bitsize);
  else
    std::tie(arg2, idx) = load_arg(idx, bitsize);
  get_end_of_line(idx);

  Inst *res = bb->build_inst(op, arg2, arg1);

  idx = dest_idx;
  if (is_kind(idx, Lexeme::freg))
    write_freg(get_freg(idx++), res);
  else
    store_arg(idx, res);
}

void Parser::process_shift(Op op, uint32_t bitsize)
{
  Inst *arg1;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::dreg))
    {
      arg1 = get_dreg_value(idx++, bitsize);
      arg1 = bb->build_inst(Op::ZEXT, bb->build_trunc(arg1, 6), bitsize);
    }
  else
    {
      get_hash(idx++);
      if (is_kind(idx, Lexeme::name))
	std::tie(arg1, idx) = get_addr(idx, bitsize);
      else
	arg1 = get_hex_or_integer(idx++, bitsize);
    }
  get_comma(idx++);

  Inst *arg2;
  unsigned dest_idx = idx;
  arg2 = get_dreg_value(idx++, bitsize);
  get_end_of_line(idx);

  Inst *res = bb->build_inst(op, arg2, arg1);

  Inst *b0 = bb->value_inst(0, 1);
  Inst *x = bb->build_inst(Op::READ, rstate->registers[M68kRegIdx::x]);
  Inst *c;
  if (op == Op::SHL)
    {
      Inst *shift = bb->build_inst(Op::ZEXT, arg1, arg1->bitsize + 1);
      x = bb->build_inst(Op::CONCAT, x, res);
      x = bb->build_inst(op, x, shift);
      x = bb->build_extract_bit(x, x->bitsize - 1);
      c = bb->build_inst(Op::CONCAT, b0, res);
      c = bb->build_inst(op, c, shift);
      c = bb->build_extract_bit(c, c->bitsize - 1);
    }
  else
    {
      Inst *shift = bb->build_inst(Op::ZEXT, arg1, arg1->bitsize + 1);
      x = bb->build_inst(Op::CONCAT, x, res);
      x = bb->build_inst(op, shift, x);
      x = bb->build_extract_bit(x, 0);
      c = bb->build_inst(Op::CONCAT, res, b0);
      c = bb->build_inst(op, c, shift);
      c = bb->build_extract_bit(c, 0);
    }
  Inst *zero = bb->value_inst(0, bitsize);
  Inst *n = bb->build_extract_bit(res, bitsize - 1);
  Inst *z = bb->build_inst(Op::EQ, res, zero);
  Inst *v = bb->value_inst(0, 1);
  set_xnzvc(x, n, z, v, c);

  idx = dest_idx;
  write_dreg(get_dreg(idx++), res);
}

void Parser::process_clr(uint32_t bitsize)
{
  Inst *zero = bb->value_inst(0, bitsize);
  if (is_kind(1, Lexeme::dreg))
    {
      Inst *dest = get_dreg(1);
      get_end_of_line(2);

      write_dreg(dest, zero);
    }
  else
    {
      unsigned idx = store_arg(1, zero);
      get_end_of_line(idx);
    }

  Inst *n = bb->value_inst(0, 1);
  Inst *z = bb->value_inst(1, 1);
  Inst *v = bb->value_inst(0, 1);
  Inst *c = bb->value_inst(0, 1);
  set_xnzvc(nullptr, n, z, v, c);
}

void Parser::process_cmp(uint32_t bitsize)
{
  Inst *arg1;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::dreg))
    arg1 = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::areg))
    arg1 = get_areg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::hash))
    {
      idx++;
      if (is_kind(idx, Lexeme::name))
	std::tie(arg1, idx) = get_addr(idx, bitsize);
      else
	arg1 = get_hex_or_integer(idx++, bitsize);
    }
  else
    std::tie(arg1, idx) = load_arg(idx, bitsize);
  get_comma(idx++);

  Inst *arg2;
  if (is_kind(idx, Lexeme::dreg))
    arg2 = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::areg))
    arg2 = get_areg_value(idx++, bitsize);
  else
    std::tie(arg2, idx) = load_arg(idx, bitsize);
  get_end_of_line(idx);

  Inst *res = bb->build_inst(Op::SUB, arg2, arg1);

  // Extract the sign bits for the values.
  Inst *sm = bb->build_extract_bit(arg1, bitsize - 1);
  Inst *not_sm = bb->build_inst(Op::NOT, sm);
  Inst *dm = bb->build_extract_bit(arg2, bitsize - 1);
  Inst *not_dm = bb->build_inst(Op::NOT, dm);
  Inst *rm = bb->build_extract_bit(res, bitsize - 1);
  Inst *not_rm = bb->build_inst(Op::NOT, rm);

  Inst *n = rm;
  Inst *z = bb->build_inst(Op::EQ, arg1, arg2);
  Inst *v1 =
    bb->build_inst(Op::AND, bb->build_inst(Op::AND, not_sm, dm), not_rm);
  Inst *v2 =
    bb->build_inst(Op::AND, bb->build_inst(Op::AND, sm, not_dm), rm);
  Inst *v = bb->build_inst(Op::OR, v1, v2);
  Inst *c1 = bb->build_inst(Op::AND, sm, not_dm);
  Inst *c2 = bb->build_inst(Op::AND, rm, not_dm);
  Inst *c3 = bb->build_inst(Op::AND, sm, rm);
  Inst *c = bb->build_inst(Op::OR, bb->build_inst(Op::OR, c1, c2), c3);

  set_xnzvc(nullptr, n, z, v, c);
}

void Parser::process_ext(uint32_t src_bitsize, uint32_t dest_bitsize)
{
  Inst *dest = get_dreg(1);
  Inst *arg = get_dreg_value(1, src_bitsize);
  get_end_of_line(2);

  set_nz00(arg);

  Inst *res = bb->build_inst(Op::SEXT, arg, dest_bitsize);
  write_dreg(dest, res);
}

void Parser::process_fmove(uint32_t bitsize)
{
  Inst *value;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::freg))
    value = get_freg_value(idx++, bitsize);
  else
    std::tie(value, idx) = load_arg(idx, bitsize);
  get_comma(idx++);

  if (is_kind(idx, Lexeme::freg))
    {
      std::pair<Inst *, Inst *> dest = get_freg(idx++);
      get_end_of_line(idx);

      write_freg(dest, value);
    }
  else
    {
      idx = store_arg(idx, value);
      get_end_of_line(idx);
    }
}

void Parser::process_jcc(Cond_code cc)
{
  Basic_block *true_bb = get_bb(1);
  get_end_of_line(2);

  Inst *cond = build_cond(cc);
  Basic_block *false_bb = func->build_bb();
  bb->build_br_inst(cond, true_bb, false_bb);
  bb = false_bb;
}

void Parser::process_lea()
{
  auto [ptr, idx] = get_addr(1, 32);
  get_comma(idx++);
  Inst *dest = get_areg(idx++);
  get_end_of_line(idx);

  write_dreg(dest, ptr);
}

void Parser::process_move(uint32_t bitsize)
{
  Inst *value;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::dreg))
    value = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::areg))
    value = get_areg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::hash))
    {
      idx++;
      if (is_kind(idx, Lexeme::name))
	std::tie(value, idx) = get_addr(idx, bitsize);
      else
	value = get_hex_or_integer(idx++, bitsize);
    }
  else
    std::tie(value, idx) = load_arg(idx, bitsize);
  get_comma(idx++);

  if (!is_kind(idx, Lexeme::areg))
    set_nz00(value);

  if (is_kind(idx, Lexeme::dreg))
    {
      Inst *dest = get_dreg(idx++);
      get_end_of_line(idx);

      write_dreg(dest, value);
    }
  else if (is_kind(idx, Lexeme::areg))
    {
      Inst *dest = get_areg(idx++);
      get_end_of_line(idx);

      if (bitsize < 32)
	value = bb->build_inst(Op::SEXT, value, 32);
      write_areg(dest, value);
    }
  else
    {
      idx = store_arg(idx, value);
      get_end_of_line(idx);
    }
}

void Parser::process_moveq()
{
  get_hash(1);
  Inst *value = get_hex_or_integer(2, 8);
  get_comma(3);
  Inst *dest = get_dreg(4);
  get_end_of_line(5);

  set_nz00(value);

  write_dreg(dest, bb->build_inst(Op::SEXT, value, 32));
}

void Parser::process_neg(uint32_t bitsize)
{
  Inst *arg;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::dreg))
    arg = get_dreg_value(idx++, bitsize);
  else
    std::tie(arg, idx) = load_arg(idx, bitsize);
  get_end_of_line(idx);

  Inst *res = bb->build_inst(Op::NEG, arg);

  // Extract the sign bits for the values.
  Inst *dm = bb->build_extract_bit(arg, bitsize - 1);
  Inst *rm = bb->build_extract_bit(res, bitsize - 1);

  Inst *zero = bb->value_inst(0, bitsize);
  Inst *n = rm;
  Inst *z = bb->build_inst(Op::EQ, arg, zero);
  Inst *v = bb->build_inst(Op::AND, dm, rm);
  Inst *c = bb->build_inst(Op::OR, dm, rm);
  Inst *x = c;
  set_xnzvc(x, n, z, v, c);

  if (is_kind(1, Lexeme::dreg))
    write_dreg(get_dreg(1), res);
  else
    store_arg(1, res);
}

void Parser::process_scc(Cond_code cc)
{
  Inst *res = bb->build_inst(Op::SEXT, build_cond(cc), 8);

  if (is_kind(1, Lexeme::dreg))
    {
      Inst *dest = get_dreg(1);
      get_end_of_line(2);

      write_dreg(dest, res);
    }
  else
    {
      unsigned idx = store_arg(1, res);
      get_end_of_line(idx);
    }
}

void Parser::process_tst(uint32_t bitsize)
{
  Inst *arg;
  unsigned idx = 1;
  if (is_kind(idx, Lexeme::dreg))
    arg = get_dreg_value(idx++, bitsize);
  else if (is_kind(idx, Lexeme::areg))
    arg = get_areg_value(idx++, bitsize);
  else
    std::tie(arg, idx) = load_arg(idx, bitsize);
  get_end_of_line(idx);

  set_nz00(arg);
}

void Parser::process_call()
{
  std::string_view name = get_name(1);
  get_end_of_line(2);

  if (name == "abort" || name == "__assert_fail")
    {
      Inst *b1 = bb->value_inst(1, 1);
      bb->build_inst(Op::WRITE, rstate->registers[M68kRegIdx::abort], b1);
      bb->build_br_inst(rstate->exit_bb);
      bb = func->build_bb();
      return;
    }

  throw Not_implemented("call " + std::string(name));
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
	    throw Not_implemented("attribute patchable_function_entry");
	}
      throw Parse_error(".section in the middle of a function", line_number);
    }
  else if (name == "add.l" || name == "addq.l")
    process_add(32);
  else if (name == "add.w" || name == "addq.w")
    process_add(16);
  else if (name == "add.b" || name == "addq.b")
    process_add(8);
  else if (name == "and.l")
    process_binary_bitwise(Op::AND, 32);
  else if (name == "and.w")
    process_binary_bitwise(Op::AND, 16);
  else if (name == "and.b")
    process_binary_bitwise(Op::AND, 8);
  else if (name == "asr.l")
    process_shift(Op::ASHR, 32);
  else if (name == "asr.w")
    process_shift(Op::ASHR, 16);
  else if (name == "asr.b")
    process_shift(Op::ASHR, 8);
  else if (name == "clr.l")
    process_clr(32);
  else if (name == "clr.w")
    process_clr(16);
  else if (name == "clr.b")
    process_clr(8);
  else if (name == "cmp.l")
    process_cmp(32);
  else if (name == "cmp.w")
    process_cmp(16);
  else if (name == "cmp.b")
    process_cmp(8);
  else if (name == "eor.l")
    process_binary_bitwise(Op::XOR, 32);
  else if (name == "eor.w")
    process_binary_bitwise(Op::XOR, 16);
  else if (name == "eor.b")
    process_binary_bitwise(Op::XOR, 8);
  else if (name == "ext.l")
    process_ext(16, 32);
  else if (name == "ext.w")
    process_ext(8, 16);
  else if (name == "extb.l")
    process_ext(8, 32);
  else if (name == "fadd.s")
    process_fbinary(Op::FADD, 32);
  else if (name == "fadd.d")
    process_fbinary(Op::FADD, 64);
  else if (name == "fdiv.s")
    process_fbinary(Op::FDIV, 32);
  else if (name == "fdiv.d")
    process_fbinary(Op::FDIV, 64);
  else if (name == "fmove.s")
    process_fmove(32);
  else if (name == "fmove.d")
    process_fmove(64);
  else if (name == "fmul.s")
    process_fbinary(Op::FMUL, 32);
  else if (name == "fmul.d")
    process_fbinary(Op::FMUL, 64);
  else if (name == "fsub.s")
    process_fbinary(Op::FSUB, 32);
  else if (name == "fsub.d")
    process_fbinary(Op::FSUB, 64);
  else if (name == "jcc")
    process_jcc(Cond_code::CC);
  else if (name == "jcs")
    process_jcc(Cond_code::CS);
  else if (name == "jeq")
    process_jcc(Cond_code::EQ);
  else if (name == "jge")
    process_jcc(Cond_code::GE);
  else if (name == "jgt")
    process_jcc(Cond_code::GT);
  else if (name == "jhi")
    process_jcc(Cond_code::HI);
  else if (name == "jle")
    process_jcc(Cond_code::LE);
  else if (name == "jls")
    process_jcc(Cond_code::LS);
  else if (name == "jlt")
    process_jcc(Cond_code::LT);
  else if (name == "jmi")
    process_jcc(Cond_code::MI);
  else if (name == "jne")
    process_jcc(Cond_code::NE);
  else if (name == "jpl")
    process_jcc(Cond_code::PL);
  else if (name == "jvc")
    process_jcc(Cond_code::VC);
  else if (name == "jvs")
    process_jcc(Cond_code::VS);
  else if (name == "jsr")
    process_call();
  else if (name == "lea")
    process_lea();
  else if (name == "lsl.l")
    process_shift(Op::SHL, 32);
  else if (name == "lsl.w")
    process_shift(Op::SHL, 16);
  else if (name == "lsl.b")
    process_shift(Op::SHL, 8);
  else if (name == "lsr.l")
    process_shift(Op::LSHR, 32);
  else if (name == "lsr.w")
    process_shift(Op::LSHR, 16);
  else if (name == "lsr.b")
    process_shift(Op::LSHR, 8);
  else if (name == "move.l")
    process_move(32);
  else if (name == "move.w")
    process_move(16);
  else if (name == "move.b")
    process_move(8);
  else if (name == "moveq")
    process_moveq();
  else if (name == "neg.l")
    process_neg(32);
  else if (name == "neg.w")
    process_neg(16);
  else if (name == "neg.b")
    process_neg(8);
  else if (name == "or.l")
    process_binary_bitwise(Op::OR, 32);
  else if (name == "or.w")
    process_binary_bitwise(Op::OR, 16);
  else if (name == "or.b")
    process_binary_bitwise(Op::OR, 8);
  else if (name == "rts")
    {
      get_end_of_line(1);

      bb->build_br_inst(rstate->exit_bb);
      bb = nullptr;
    }
  else if (name == "scc")
    process_scc(Cond_code::CC);
  else if (name == "scs")
    process_scc(Cond_code::CS);
  else if (name == "seq")
    process_scc(Cond_code::EQ);
  else if (name == "sge")
    process_scc(Cond_code::GE);
  else if (name == "sgt")
    process_scc(Cond_code::GT);
  else if (name == "shi")
    process_scc(Cond_code::HI);
  else if (name == "sle")
    process_scc(Cond_code::LE);
  else if (name == "sls")
    process_scc(Cond_code::LS);
  else if (name == "slt")
    process_scc(Cond_code::LT);
  else if (name == "smi")
    process_scc(Cond_code::MI);
  else if (name == "sne")
    process_scc(Cond_code::NE);
  else if (name == "spl")
    process_scc(Cond_code::PL);
  else if (name == "svc")
    process_scc(Cond_code::VC);
  else if (name == "svs")
    process_scc(Cond_code::VS);
  else if (name == "sf")
    process_scc(Cond_code::F);
  else if (name == "st")
    process_scc(Cond_code::T);
  else if (name == "sub.l" || name == "subq.l")
    process_sub(32);
  else if (name == "sub.w" || name == "subq.w")
    process_sub(16);
  else if (name == "sub.b" || name == "subq.b")
    process_sub(8);
  else if (name == "trap")
    {
      get_hash(1);
      get_hex_or_integer(2, 32);
      get_end_of_line(3);

      bb->build_inst(Op::UB, bb->value_inst(1, 1));
      bb->build_br_inst(rstate->exit_bb);
      bb = nullptr;
    }
  else if (name == "tst.l")
    process_tst(32);
  else if (name == "tst.w")
    process_tst(16);
  else if (name == "tst.b")
    process_tst(8);
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
	  && tokens.back().kind != Lexeme::hash)
        {
	  tokens.emplace_back(Lexeme::minus, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '+')
        {
	  tokens.emplace_back(Lexeme::plus, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '*')
        {
	  tokens.emplace_back(Lexeme::asterisk, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '0' && buf[pos + 1] == 'x')
	lex_hex();
      else if (isdigit(buf[pos])
	       || (buf[pos] == '-' && isdigit(buf[pos + 1])))
	lex_integer();
      else if (buf[pos] == '%')
        lex_reg();
      else if (buf[pos] == '#')
	{
	  tokens.emplace_back(Lexeme::hash, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '"')
	lex_string();
      else if (buf[pos] == '.'
	       && buf[pos + 1] == 'L'
	       && isdigit(buf[pos + 2]))
	lex_label_or_label_def();
      else if (isalpha(buf[pos]) || buf[pos] == '_' || buf[pos] == '.')
	lex_name();
      else if (buf[pos] == ',')
	{
	  tokens.emplace_back(Lexeme::comma, pos, 1);
	  pos++;
	}
      else if (buf[pos] == ':')
	{
	  tokens.emplace_back(Lexeme::colon, pos, 1);
	  pos++;
	}
      else if (buf[pos] == '@')
	{
	  tokens.emplace_back(Lexeme::at, pos, 1);
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

Function *parse_m68k(std::string const& file_name, m68k_state *state)
{
  Parser p(state);
  Function *func = p.parse(file_name);
  reverse_post_order(func);
  return func;
}

} // end namespace smtgcc
