#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#include "smtgcc.h"

using namespace smtgcc;

std::vector<std::string> opts;

void print_help(FILE *f)
{
  const char* help_message = R"(
Usage: smtgcc-opt [OPTION]... [FILE]
Perform optimizations on the specified FILE.

Options:
  -h, --help         Display this help message and exit.
  -c                 Check that the optimizations are correct.
  -simplify_inst     Run instruction simplification optimization.
  -simplify_cfg      Run control flow graph simplification optimization.
  -dce               Run dead code elimination optimization.
  -loop_unroll       Run loop unrolling optimization.
  -convert           Run the conversion transformation pass.

Examples:
  smtgcc-opt -simplify_inst example.ir
      Run the instruction simplification optimization on 'example.ir'.

  smtgcc-opt -c -simplify_inst -dce example.ir
      Run instructions simplification and dead code elimination optimizations
      on 'example.ir' and check the result of each pass.
)";

  fprintf(f, "%s", help_message);
}

void check(Module *orig_module, Module *module, const char *opt)
{
  size_t nof_funcs = module->functions.size();
  assert(orig_module->functions.size() == nof_funcs);
  for (size_t i = 0; i < nof_funcs; i++)
    {
      if (has_loops(orig_module->functions[i]))
	{
	  fprintf(stderr, "warning: "
		  "%s(%s): Functions with loops cannot be checked with -c\n",
		  opt, orig_module->functions[i]->name.c_str());
	  return;
	}

      assert(orig_module->functions[i]->name == module->functions[i]->name);
      Module *m = create_module(module->ptr_bits, module->ptr_id_bits,
				module->ptr_offset_bits);
      Function *src_func = orig_module->functions[i]->clone(m);
      src_func->rename("src");
      Function *tgt_func = module->functions[i]->clone(m);
      tgt_func->rename("tgt");

      Solver_result result = check_refine(m);
      if (result.status != Result_status::correct)
	{
	  assert(result.message);
	  module->functions[i]->print(stderr);
	  fprintf(stderr, "Error: %s(%s): %s", opt,
		  module->functions[i]->name.c_str(),
		  (*result.message).c_str());
	  exit(1);
	}

      destroy_module(m);
    }
}

int main(int argc, char **argv)
{
  const char *file_name = nullptr;
  bool flag_c = false;

  for (int i = 1; i < argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
	{
	  if (i != argc - 1)
	    {
	      print_help(stderr);
	      exit(1);
	    }
	  file_name = arg;
	}
      else if (!strcmp(arg, "-h") || !strcmp(arg, "--help"))
	print_help(stdout);
      else if (!strcmp(arg, "-c"))
	flag_c = true;
      else if (!strcmp(arg, "-simplify_inst")
	       || !strcmp(arg, "-simplify_cfg")
	       || !strcmp(arg, "-dce")
	       || !strcmp(arg, "-loop_unroll")
	       || !strcmp(arg, "-convert"))
	opts.push_back(arg);
      else
	{
	  print_help(stderr);
	  exit(1);
	}
    }

  if (!file_name)
    {
      print_help(stderr);
      exit(1);
    }

  try {
    Module *module = parse_ir(file_name);

    for (const std::string& opt : opts)
      {
	Module *orig_module = nullptr;
	if (flag_c)
	  {
	    if (opt == "-convert")
	      fprintf(stderr, "warning: -convert cannot be checked with -c\n");
	    else
	      orig_module = module->clone();
	  }

	if (opt == "-simplify_inst")
	  simplify_insts(module);
	else if (opt == "-simplify_cfg")
	  simplify_cfg(module);
	else if (opt == "-dce")
	  dead_code_elimination(module);
	else if (opt == "-loop_unroll")
	  loop_unroll(module);
	else if (opt == "-convert")
	  {
	    for (auto func : module->functions)
	      {
		if (has_loops(func))
		  {
		    fprintf(stderr, "error: "
			    "-convert cannot transform functions with loops\n");
		    exit(1);
		  }
	      }
	    convert(module);
	  }

	if (orig_module)
	  {
	    check(orig_module, module, opt.c_str());
	    destroy_module(orig_module);
	    orig_module = nullptr;
	  }
      }

    module->canonicalize();
    module->print(stdout);

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
