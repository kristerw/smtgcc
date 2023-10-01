#include <cassert>
#include <cstring>

#include "smtgcc.h"

using namespace smtgcc;

void print_help()
{
  fprintf(stderr, "USAGE: smtgcc-opt [-c] filename\n");
  exit(1);
}

int main(int argc, char **argv)
{
  const char *file_name = nullptr;
  bool check = false;

  for (int i = 1; i < argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
	{
	  if (i != argc - 1)
	    print_help();
	  file_name = arg;
	}
      else if (!strcmp(arg, "-c"))
	check = true;
      else
	print_help();
    }

  if (!file_name)
    print_help();

  try {
    Module *module = parse_ir(file_name);
    simplify_insts(module);
    dead_code_elimination(module);
    module->print(stderr);

    if (check)
      {
	Module *orig = parse_ir(file_name);
	size_t nof_funcs = module->functions.size();
	assert(orig->functions.size() == nof_funcs);
	for (size_t i = 0; i < nof_funcs; i++)
	  {
	    assert(orig->functions[i]->name == module->functions[i]->name);
	    Module *m = create_module(module->ptr_bits, module->ptr_id_bits,
				      module->ptr_offset_bits);
	    Function *src_func = m->clone(orig->functions[i]);
	    src_func->rename("src");
	    Function *tgt_func = m->clone(module->functions[i]);
	    tgt_func->rename("tgt");

	    Solver_result result = check_refine(m);
	    if (result.status != Result_status::correct)
	      {
		assert(result.message);
		fprintf(stderr, "%s", (*result.message).c_str());
	      }

	    destroy_module(m);
	  }
      }

    destroy_module(module);
  }
  catch (Parse_error error)
    {
      fprintf(stderr, "%s:%d: Parse error: %s\n", file_name, error.line,
	      error.msg.c_str());
    }
  catch (Not_implemented error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
    }

  return 0;
}
