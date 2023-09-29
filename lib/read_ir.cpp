#include <fstream>
#include <cassert>
#include <limits>

#include "smtgcc.h"

namespace smtgcc {
namespace {

struct parser {
  enum class lexeme {
    label,
    label_def,
    reg,
    name,
    integer,
    hex,
    comma,
    assign,
    left_bracket,
    right_bracket
  };

  struct token {
    lexeme kind;
    int pos;
    int size;
  };
  std::vector<token> tokens;

  struct br_inst {
    std::vector<Instruction *> args;
    std::vector<std::string> labels;
  };

  std::map<std::string, Instruction_info> name2info;

  parser()
  {
    for (auto& info : inst_info)
      {
	name2info[info.name] = info;
      }
  }

  ~parser()
  {
    if (module)
      destroy_module(module);
  }

  int line_number = 0;
  int pos;

  // Lines may be very long for the phi nodes in tests such as
  // gcc.c-torture/compile/20001226-1.c
  static const int max_line_len = 100000;
  char buf[max_line_len];

  void parse(std::string const& file_name);

  Module *module = nullptr;

private:
  Function *current_func = nullptr;
  Basic_block *current_bb = nullptr;
  std::map<uint32_t, Basic_block *> id2bb;
  std::map<Basic_block *, uint32_t> bb2id;
  std::map<uint32_t, Instruction *> id2inst;
  std::map<Basic_block *, br_inst> bb2br_args;

  void lex_line(void);
  void lex_label_or_label_def(void);
  void lex_reg(void);
  void lex_hex(void);
  void lex_integer(void);
  void lex_hex_or_integer(void);
  void lex_name(void);

  std::string token_string(const token& tok);

  std::string get_name(const char *p);
  uint32_t get_u32(const char *p);
  unsigned __int128 get_hex(const char *p);

  uint32_t get_uint32(unsigned idx);
  unsigned __int128 get_hex_or_integer(unsigned idx);
  Instruction *get_arg(unsigned idx);
  Basic_block *get_bb(unsigned idx);
  std::string get_bb_string(unsigned idx);
  Basic_block *get_bb_from_string(std::string str);
  void get_comma(unsigned idx);
  void get_left_bracket(unsigned idx);
  void get_right_bracket(unsigned idx);
  void get_end_of_line(unsigned idx);

  void parse_config();
  void parse_function();
  void parse_basic_block();
  Op parse_instruction();

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
  if (!isdigit(buf[pos]))
    throw Parse_error("expected a digit after '.'", line_number);
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

void parser::lex_reg(void)
{
  assert(buf[pos] == '%');
  int start_pos = pos;
  pos++;
  if (!isdigit(buf[pos]))
    throw Parse_error("expected a digit after '%'", line_number);
  pos++;
  if (isdigit(buf[pos]) && buf[pos - 1] == '0')
    throw Parse_error("octal numbers are not supported after '%'", line_number);
  while (isdigit(buf[pos]))
    pos++;
  tokens.emplace_back(lexeme::reg, start_pos, pos - start_pos);
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
  assert(isdigit(buf[pos]));
  int start_pos = pos;
  pos++;
  if (isdigit(buf[pos]) && buf[pos - 1] == '0')
    throw Parse_error("octal numbers are not supported", line_number);
  while (isdigit(buf[pos]))
    pos++;
  tokens.emplace_back(lexeme::integer, start_pos, pos - start_pos);
}

void parser::lex_hex_or_integer(void)
{
  assert(isdigit(buf[pos]));
  if (buf[pos] == '0' && (buf[pos + 1] == 'x' || buf[pos + 1] == 'X'))
    lex_hex();
  else
    lex_integer();
}

void parser::lex_name(void)
{
  assert(isalpha(buf[pos]) || buf[pos] == '_');
  int start_pos = pos;
  pos++;
  while (isalnum(buf[pos])
	 || buf[pos] == '_'
	 || buf[pos] == '-'
	 || buf[pos] == '.')
    pos++;
  tokens.emplace_back(lexeme::name, start_pos, pos - start_pos);
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

uint32_t parser::get_u32(const char *p)
{
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

uint32_t parser::get_uint32(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::integer)
    throw Parse_error("expected a decimal integer instead of "
		      + token_string(tokens[idx]), line_number);
  return get_u32(&buf[tokens[idx].pos]);
}

unsigned __int128 parser::get_hex_or_integer(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::hex && tokens[idx].kind != lexeme::integer)
    throw Parse_error("expected a hexadecimal or decimal integer instead of "
		      + token_string(tokens[idx]), line_number);
  if (tokens[idx].kind == lexeme::integer)
    return get_u32(&buf[tokens[idx].pos]);
  else
    return get_hex(&buf[tokens[idx].pos]);
}

Instruction *parser::get_arg(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
  if (tokens[idx].kind != lexeme::reg)
    throw Parse_error("expected a reg instead of "
		      + token_string(tokens[idx]), line_number);
  uint32_t arg_id = get_u32(&buf[tokens[idx].pos + 1]);
  if (!id2inst.contains(arg_id))
    throw Parse_error(token_string(tokens[idx])
		      + " is not defined before use", line_number);
  return id2inst.at(arg_id);
}

Basic_block *parser::get_bb(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
 if (tokens[idx].kind != lexeme::label)
   throw Parse_error("expected a label instead of "
		     + token_string(tokens[idx]), line_number);
  uint32_t id = get_u32(&buf[tokens[idx].pos + 1]);
  auto I = id2bb.find(id);
  if (I != id2bb.end())
    return I->second;
  Basic_block *bb = current_func->build_bb();
  id2bb[id] = bb;
  bb2id[bb] = id;
  return bb;
}

std::string parser::get_bb_string(unsigned idx)
{
  if (tokens.size() <= idx)
    throw Parse_error("expected more arguments", line_number);
 if (tokens[idx].kind != lexeme::label)
   throw Parse_error("expected a label instead of "
		     + token_string(tokens[idx]), line_number);
  return token_string(tokens[idx]);
}

Basic_block *parser::get_bb_from_string(std::string str)
{
  uint32_t id = get_u32(str.c_str() + 1);
  if (!id2bb.contains(id))
    throw Parse_error("basic block " + str + " is not defined", 0); // XXX line no
  return id2bb.at(id);
}

void parser::get_comma(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != lexeme::comma)
    throw Parse_error("expected a ',' after " + token_string(tokens[idx - 1]),
		      line_number);
}

void parser::get_left_bracket(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != lexeme::left_bracket)
    throw Parse_error("expected a '[' after " + token_string(tokens[idx - 1]),
		      line_number);
}

void parser::get_right_bracket(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() <= idx || tokens[idx].kind != lexeme::right_bracket)
    throw Parse_error("expected a ']' after " + token_string(tokens[idx - 1]),
		      line_number);
}

void parser::get_end_of_line(unsigned idx)
{
  assert(idx > 0);
  if (tokens.size() > idx)
    throw Parse_error("expected end of line after " +
		      token_string(tokens[idx - 1]), line_number);
}

void parser::parse_config()
{
  if (tokens[0].kind != lexeme::name)
    throw Parse_error("expected \"config\" as first statement", line_number);
  std::string config = get_name(&buf[tokens[0].pos]);
  if (config != "config")
    throw Parse_error("expected \"config\" as first statement", line_number);

  uint32_t ptr_bits = get_uint32(1);
  get_comma(2);
  uint32_t ptr_id_bits = get_uint32(3);
  get_comma(4);
  uint32_t ptr_offset_bits = get_uint32(5);
  get_end_of_line(6);

  module = create_module(ptr_bits, ptr_id_bits, ptr_offset_bits);
}

void parser::parse_function()
{
  if (tokens[0].kind != lexeme::name)
    throw Parse_error("expected \"function\"", line_number);
  std::string function = get_name(&buf[tokens[0].pos]);
  if (function != "function")
    throw Parse_error("expected \"function\"", line_number);

  if (tokens.size() < 2 || tokens[1].kind != lexeme::name)
    throw Parse_error("expected a function name", line_number);
  if (tokens.size() > 2)
    throw Parse_error("extra tokens after function name", line_number);
  std::string name = get_name(&buf[tokens[1].pos]);

  current_func = module->build_function(name);
}

void parser::parse_basic_block()
{
  if (tokens[0].kind != lexeme::label_def)
    throw Parse_error("expected a label definition", line_number);
  if (tokens.size() > 1)
    throw Parse_error("expected end of line after " + token_string(tokens[0]),
		      line_number);

  uint32_t id = get_u32(&buf[tokens[0].pos + 1]);
  if (id2bb.contains(id))
    {
      // We have already created a BB for this ID. This could be because
      // of a forward declaration when a branch instruction branches to
      // a BB that has not been defined yet (in which case the BB is empty).
      // But it may be because the function incorrectly has two BBs with
      // the same ID (in which case the BB has instructions).
      if (id2bb.at(id)->last_inst)
	throw Parse_error(token_string(tokens[0]) + " is redefined",
			  line_number);
      current_bb = id2bb.at(id);
    }
  else
    {
      current_bb = current_func->build_bb();
      id2bb[id] = current_bb;
      bb2id[current_bb] = id;
    }
}

Op parser::parse_instruction()
{
  if (tokens[0].kind != lexeme::reg && tokens[0].kind != lexeme::name)
    throw Parse_error("expected an instruction or assignment statement",
		      line_number);

  unsigned name_idx = 0;
  if (tokens[0].kind == lexeme::reg)
    {
      if (tokens.size() < 2 || tokens[1].kind != lexeme::assign)
	throw Parse_error("expected `=` after " + token_string(tokens[0]),
			  line_number);
      if (tokens.size() < 3 || tokens[2].kind != lexeme::name)
	throw Parse_error("expected an instruction name after '='",
			  line_number);
      name_idx = 2;
    }
  const std::string name = get_name(&buf[tokens[name_idx].pos]);

  auto I = name2info.find(name);
  if (I == name2info.end())
    throw Parse_error("invalid instruction name", line_number);
  const Instruction_info& info = I->second;
  if (info.has_lhs && name_idx == 0)
    throw Parse_error("instruction return value is ignored", line_number);
  if (!info.has_lhs && name_idx != 0)
    throw Parse_error(name + " does not return a value", line_number);

  if (info.has_lhs)
    {
      uint32_t lhs_id = get_u32(&buf[tokens[0].pos + 1]);
      if (id2inst.contains(lhs_id))
	throw Parse_error("redefinition of " + token_string(tokens[0]),
			  line_number);

      if (info.iclass == Inst_class::iunary
	  || info.iclass == Inst_class::funary)
	{
	  Instruction *arg1 = get_arg(3);
	  get_end_of_line(4);

	  id2inst[lhs_id] = current_bb->build_inst(info.opcode, arg1);
	}
      else if (info.iclass == Inst_class::ibinary
	       || info.iclass == Inst_class::fbinary
	       || info.iclass == Inst_class::icomparison
	       || info.iclass == Inst_class::fcomparison
	       || info.iclass == Inst_class::conv)
	{
	  Instruction *arg1 = get_arg(3);
	  get_comma(4);
	  Instruction *arg2 = get_arg(5);
	  get_end_of_line(6);

	  id2inst[lhs_id] = current_bb->build_inst(info.opcode, arg1, arg2);
	}
      else if (info.iclass == Inst_class::ternary)
	{
	  Instruction *arg1 = get_arg(3);
	  get_comma(4);
	  Instruction *arg2 = get_arg(5);
	  get_comma(6);
	  Instruction *arg3 = get_arg(7);
	  get_end_of_line(8);

	  id2inst[lhs_id] =
	    current_bb->build_inst(info.opcode, arg1, arg2, arg3);
	}
      else if (info.opcode == Op::PHI)
	{
	  // TODO: This assumes there are no loops.
	  // TODO: Need to test (and improve) error handling.
	  unsigned idx = 3;

	  // Read the first phi arg to get the bitsize needed to create the
	  // phi instruction.
	  get_left_bracket(idx++);
	  Instruction *arg_inst = get_arg(idx++);
	  get_comma(idx++);
	  Basic_block *arg_bb = get_bb(idx++);
	  get_right_bracket(idx++);

	  Instruction *phi = current_bb->build_phi_inst(arg_inst->bitsize);
	  phi->add_phi_arg(arg_inst, arg_bb);
	  id2inst[lhs_id] = phi;

	  // Add the remaining phi args.
	  while (idx < tokens.size())
	    {
	      get_comma(idx++);
	      get_left_bracket(idx++);
	      Instruction *arg_inst = get_arg(idx++);
	      get_comma(idx++);
	      Basic_block *arg_bb = get_bb(idx++);
	      get_right_bracket(idx++);
	      phi->add_phi_arg(arg_inst, arg_bb);
	    }
	}
      else if (info.opcode == Op::VALUE)
	{
	  unsigned __int128 value = get_hex_or_integer(3);
	  get_comma(4);
	  uint32_t bitsize = get_uint32(5);
	  get_end_of_line(6);

	  id2inst[lhs_id] = current_bb->value_inst(value, bitsize);
	}
      else
	throw Not_implemented("parse_instruction: " + name);
    }
  else
    {
      if (info.iclass == Inst_class::iunary ||
	  info.iclass == Inst_class::funary)
	{
	  Instruction *arg1 = get_arg(1);
	  get_end_of_line(2);

	  current_bb->build_inst(info.opcode, arg1);
	}
      else if (info.iclass == Inst_class::ibinary
	       || info.iclass == Inst_class::fbinary
	       || info.iclass == Inst_class::icomparison
	       || info.iclass == Inst_class::fcomparison)
	{
	  Instruction *arg1 = get_arg(1);
	  get_comma(2);
	  Instruction *arg2 = get_arg(3);
	  get_end_of_line(4);

	  current_bb->build_inst(info.opcode, arg1, arg2);
	}
      else if (info.opcode == Op::BR)
	{
	  br_inst br_args;
	  if (tokens.size() > 2)
	    {
	      br_args.args.push_back(get_arg(1));
	      get_comma(2);
	      br_args.labels.push_back(get_bb_string(3));
	      get_comma(4);
	      br_args.labels.push_back(get_bb_string(5));
	      get_end_of_line(6);
	    }
	  else
	    {
	      br_args.labels.push_back(get_bb_string(1));
	      get_end_of_line(2);
	    }
	  bb2br_args[current_bb] = br_args;
	}
      else if (info.opcode == Op::RET)
	{
	  if (tokens.size() == 1)
	    {
	      get_end_of_line(1);
	      current_bb->build_ret_inst();
	    }
	  else if (tokens.size() == 2)
	    {
	      Instruction *arg = get_arg(1);
	      get_end_of_line(2);
	      current_bb->build_ret_inst(arg);
	    }
	  else
	    {
	      Instruction *arg1 = get_arg(1);
	      get_comma(2);
	      Instruction *arg2 = get_arg(3);
	      get_end_of_line(4);
	      current_bb->build_ret_inst(arg1, arg2);
	    }
	}
      else
	throw Not_implemented("parse_instruction: " + name);
    }

  return info.opcode;
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
      if (buf[pos] == '.')
	lex_label_or_label_def();
      else if (buf[pos] == '%')
	lex_reg();
      else if (isdigit(buf[pos]))
	lex_hex_or_integer();
      else if (isalpha(buf[pos]) || buf[pos] == '_')
	lex_name();
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
      else
	throw Parse_error("Syntax error.", line_number);
    }
}

void parser::parse(std::string const& file_name)
{
  enum class state {
    config,
    function,
    basic_block,
    instruction
  };

  std::ifstream in(file_name);
  if (!in)
    throw Parse_error("Could not open file.", 0);

  state parser_state = state::config;
  while (in.getline(buf, max_line_len)) {
    line_number++;
    lex_line();
    if (tokens.empty())
      continue;

    if (parser_state == state::config)
      {
	parse_config();
	parser_state = state::function;
      }
    else if (parser_state == state::function)
      {
	parse_function();
	parser_state = state::basic_block;
      }
    else if (parser_state == state::basic_block)
      {
	parse_basic_block();
	parser_state = state::instruction;
      }
    else if (parser_state == state::instruction)
      {
	Op opcode = parse_instruction();
	if (opcode == Op::RET)
	  {
	    // We are done with the functions. I.e. all basic blocks are
	    // created, so we can add the branches.
	    for (auto bb : current_func->bbs)
	      {
		if (bb == current_bb)
		  continue;

		br_inst br_args = bb2br_args.at(bb);
		if (br_args.args.size() == 0)
		  {
		    assert(br_args.labels.size() == 1);
		    Basic_block *dest_bb =
		      get_bb_from_string(br_args.labels[0]);
		    bb->build_br_inst(dest_bb);
		  }
		else
		  {
		    assert(br_args.args.size() == 1);
		    assert(br_args.labels.size() == 2);
		    Instruction *cond = br_args.args[0];
		    Basic_block *true_bb =
		      get_bb_from_string(br_args.labels[0]);
		    Basic_block *false_bb =
		      get_bb_from_string(br_args.labels[1]);
		    bb->build_br_inst(cond, true_bb, false_bb);
		  }
	      }

	    current_func = nullptr;
	    current_bb = nullptr;
	    id2bb.clear();
	    id2inst.clear();
	    parser_state = state::function;
	  }
	else if (opcode == Op::BR)
	  {
	    current_bb = nullptr;
	    parser_state = state::basic_block;
	  }
      }
  }

  if (in.gcount() >= max_line_len - 1)
    throw Parse_error("line too long", line_number);
  if (parser_state != state::function)
    throw Parse_error("EOF in the middle of a function", line_number);

  for (auto func : module->functions)
    {
      for (auto bb : func->bbs)
	{
	  if (!bb->last_inst)
	    throw Parse_error("basic block ." + std::to_string(bb2id.at(bb))
			      + " is not defined", 0);
	  if (bb->preds.empty() && bb != func->bbs[0])
	    throw Parse_error("basic block ." + std::to_string(bb2id.at(bb))
			      + " is not used", 0);
	}
      reverse_post_order(func);
    }
}

} // end anonymous namespace

Module *parse_ir(std::string const& file_name)
{
  parser p;
  p.parse(file_name);
  Module *module = p.module;
  p.module = nullptr;
  validate(module);
  return module;
}

} // end namespace smtgcc
