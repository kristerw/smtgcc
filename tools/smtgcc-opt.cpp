#include "smtgcc.h"

using namespace smtgcc;

int main(int argc, char **argv)
{
  if (argc != 2)
    {
      fprintf(stderr, "Error: argc == %d\n", argc);
      exit(1);
    }

  const char *file_name = argv[1];

  try {
    Module *module = parse_ir(file_name);
    simplify_insts(module);
    dead_code_elimination(module);
    module->print(stderr);
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
