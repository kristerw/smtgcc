#include <cassert>

#include "smtgcc.h"

using namespace smtgcc;

int main(int argc, char **argv)
{
  if (argc != 2)
    {
      fprintf(stderr, "Error: argc == %d\n", argc);
      exit(1);
    }

  try {
    Module *module = parse_ir(argv[1]);

    if (module->functions.size() != 2)
      {
	fprintf(stderr, "Error: Nof functions != 2\n");
	exit(1);
      }
    Function *src = module->functions[0];
    Function *tgt = module->functions[1];
    if (src->name != "src")
      std::swap(src, tgt);
    if (src->name != "src" || tgt->name != "tgt")
      {
	fprintf(stderr, "Error: The function names must be src and tgt.\n");
	exit(1);
      }

    Solver_result result = check_refine(module);
    if (result.status != Result_status::correct)
      {
	assert(result.message);
	fprintf(stderr, "%s", (*result.message).c_str());
      }

    destroy_module(module);
  }
  catch (Parse_error error)
    {
       fprintf(stderr, "Parse error: line %d: %s\n",
	      error.line, error.msg.c_str());
    }
  catch (Not_implemented error)
    {
      fprintf(stderr, "Not implemented: %s\n", error.msg.c_str());
    }

  return 0;
}
