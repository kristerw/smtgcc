#include <cassert>

#include "smtgcc.h"

using namespace std::string_literals;

namespace smtgcc {

namespace {

Inst *extract_vec_elem(Basic_block *bb, Inst *inst, uint32_t elem_bitsize, uint32_t idx)
{
  if (idx == 0 && inst->bitsize == elem_bitsize)
    return inst;
  assert(inst->bitsize % elem_bitsize == 0);
  Inst *high = bb->value_inst(idx * elem_bitsize + elem_bitsize - 1, 32);
  Inst *low = bb->value_inst(idx * elem_bitsize, 32);
  return bb->build_inst(Op::EXTRACT, inst, high, low);
}

} // end anonymous namespace

ParserBase::ParserBase(std::map<std::string, Inst *, std::less<>>& sym_name2mem)
  : sym_name2mem(sym_name2mem) {}

void ParserBase::skip_line()
{
  while (buf[pos++] != '\n')
    ;
}

void ParserBase::skip_whitespace()
{
  while (buf[pos] == ' ' || buf[pos] == '\t')
    pos++;
}

uint32_t ParserBase::translate_base64_char(uint8_t c)
{
  if ('A' <= c && c <= 'Z')
    return c - 'A';
  if ('a' <= c && c <= 'z')
    return c - 'a' + 26;
  if ('0' <= c && c <= '9')
    return c - '0' + 52;
  if (c == '+')
    return 62;
  if (c == '/')
    return 63;
  if (c == '=')
    return 0;
  throw Parse_error("Invalid .bas64 string", line_number);
}

std::string_view ParserBase::parse_cmd()
{
  const size_t start_pos = pos;
  if (buf[pos] == '.' && isalnum(buf[pos + 1]))
    {
      pos += 2;
      while (isalnum(buf[pos]) || buf[pos] == '_')
	pos++;
      assert(buf[pos] == ' '
	     || buf[pos] == '\t'
	     || buf[pos] == '\n'
	     || buf[pos] == ':');
    }
  return std::string_view(&buf[start_pos], pos - start_pos);
}

std::optional<std::string_view> ParserBase::parse_label_def()
{
#if defined(SMTGCC_SH)
  // TODO: Adjust sym maps to always use this extra `_` instead of stripping
  // it here.
  if (buf[pos] == '_')
    pos++;
#endif
  size_t start_pos = pos;
  while (buf[pos] != ':' && buf[pos] != '\n')
    pos++;
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

bool ParserBase::parse_data(Basic_block *bb, std::vector<Inst *>& data)
{
  size_t orig_data_size = data.size();
  for (;;)
    {
      const size_t start_pos = pos;

      if (pos == buf.size() - 1)
	break;
      assert(pos < buf.size());

      skip_whitespace();
      if (buf[pos] != '.')
	{
	  pos = start_pos;
	  break;
	}
      std::string_view cmd = parse_cmd();
      if (cmd == ".xword"
	  || cmd == ".dword"
	  || cmd == ".long"
	  || cmd == ".word"
	  || cmd == ".half"
	  || cmd == ".short"
	  || cmd == ".hword"
	  || cmd == ".2byte"
	  || cmd == ".byte")
	{
	  int size;
	  if (cmd == ".byte")
	    size = 1;
	  else if (cmd == ".half"
		   || cmd == ".short"
		   || cmd == ".hword"
		   || cmd == ".2byte")
	    size = 2;
	  else if (cmd == ".word" || cmd == ".long")
	    size = 4;
	  else
	    size = 8;

	  skip_whitespace();

	  if (buf[pos] == '_' || isalpha(buf[pos]))
	    {
	      size_t sym_start_pos = pos;
	      while (buf[pos] == '_' || buf[pos] == '.' || isalnum(buf[pos]))
		pos++;

#if defined(SMTGCC_SH)
	      // TODO: Adjust sym maps to always use this extra `_` instead of
	      // stripping it here.
	      if (buf[sym_start_pos] == '_')
		sym_start_pos++;
#endif

	      std::string_view sym_name(&buf[sym_start_pos],
					pos - sym_start_pos);
	      auto I = sym_name2mem.find(sym_name);
	      if (I == sym_name2mem.end())
		throw Parse_error("unknown symbol " + std::string(sym_name),
				  line_number);
	      Inst *ptr = I->second;
	      if ((buf[pos] == '+' || buf[pos] == '-')
		  && isdigit(buf[pos + 1]))
		{
		  Op op = buf[pos++] == '+' ? Op::ADD : Op::SUB;
		  uint64_t value = 0;
		  while (isdigit(buf[pos]))
		    {
		      value = value * 10 + (buf[pos] - '0');
		      pos++;
		    }
		  Inst *value_inst = bb->value_inst(value, ptr->bitsize);
		  ptr = bb->build_inst(op, ptr, value_inst);
		}
	      if (buf[pos] == '@')
		throw Parse_error("'@' in symbol expression", line_number);
	      if (buf[pos] != '\n')
		throw Parse_error(std::string(cmd)
				  + " unknown symbol expression",
				  line_number);

	      for (int i = 0; i < size; i++)
		data.push_back(extract_vec_elem(bb, ptr, 8, i));

	      skip_line();
	    }
	  else
	    {
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

	      for (int i = 0; i < size; i++)
		{
		  data.push_back(bb->value_inst(value & 0xff, 8));
		  value = value >> 8;
		}

	      assert(buf[pos] == '\n');

	      skip_line();
	    }
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
		    data.push_back(bb->value_inst('\b', 8));
		  else if (c == 'f')
		    data.push_back(bb->value_inst('\f', 8));
		  else if (c == 'n')
		    data.push_back(bb->value_inst('\n', 8));
		  else if (c == 'r')
		    data.push_back(bb->value_inst('\r', 8));
		  else if (c == 't')
		    data.push_back(bb->value_inst('\t', 8));
		  else if (c == '"')
		    data.push_back(bb->value_inst('\"', 8));
		  else if (c == '\\')
		    data.push_back(bb->value_inst('\\', 8));
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
		      data.push_back(bb->value_inst(val, 8));
		    }
		  else
		    throw Parse_error("unknown escape sequence \\"s + c,
				      line_number);
		}
	      else
		data.push_back(bb->value_inst(buf[pos], 8));
	      pos++;
	    }
	  pos++;
	  assert(buf[pos] == '\n');
	  skip_line();

	  if (cmd == ".string")
	    data.push_back(bb->value_inst(0, 8));
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
	    data.push_back(bb->value_inst(0, 8));

	  assert(buf[pos] == '\n');

	  skip_line();
	}
      else if (cmd == ".base64")
	{
	  skip_whitespace();

	  if (buf[pos++] != '"')
	    throw Parse_error("expected '\"' after " + std::string(cmd),
			      line_number);

	  while (buf[pos] != '"')
	    {
	      uint32_t val = 0;
	      val |= translate_base64_char(buf[pos]) << 18;
	      val |= translate_base64_char(buf[pos + 1]) << 12;
	      val |= translate_base64_char(buf[pos + 2]) << 6;
	      val |= translate_base64_char(buf[pos + 3]);

	      data.push_back(bb->value_inst((val >> 16) & 0xff, 8));
	      if (buf[pos + 2] != '=')
		data.push_back(bb->value_inst((val >> 8) & 0xff, 8));
	      if (buf[pos + 3] != '=')
		data.push_back(bb->value_inst(val & 0xff, 8));

	      pos += 4;
	    }
	  pos++;
	  assert(buf[pos] == '\n');
	  skip_line();
	}
      else
	{
	  pos = start_pos;
	  break;
	}
    }

  return orig_data_size != data.size();
}

void ParserBase::parse_rodata(Basic_block *bb)
{
  enum class state {
    global,
    memory_section
  };

  pos = 0;
  state parser_state = state::global;
  for (;;)
    {
      const size_t start_pos = pos;

      if (pos == buf.size() - 1)
	break;
      assert(pos < buf.size());

      skip_whitespace();
      const std::string_view cmd = parse_cmd();
      if (cmd == ".section")
	{
	  skip_whitespace();
	  if (buf[pos] == '"')
	    pos++;

	  const size_t first_pos = pos;
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
      if (cmd == ".align"
	  || cmd == ".p2align"
	  || cmd == ".type"
	  || cmd == ".size"
	  || cmd == ".local"
	  || cmd == ".global"
	  || cmd == ".globl"
	  || cmd == ".comm")
	{
	  skip_line();
	  continue;
	}
      if (cmd == ".set")
	{
	  skip_whitespace();

	  size_t start = pos;
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

	      parse_data(bb, sym_name2data[label_name]);
	      continue;
	    }
	}
      else
	skip_line();

      parser_state = state::global;
    }
}

}  // end namespace smtgcc
