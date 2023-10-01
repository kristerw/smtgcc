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

    for (auto func : module->functions)
      {
	Solver_result result = check_ub(func);
	if (result.status != Result_status::correct)
	  {
	    assert(result.message);
	    fprintf(stderr, "%s: %s", func->name.c_str(),
		    (*result.message).c_str());
	  }
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
