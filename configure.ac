AC_INIT([smtgcc],[0.0])

AC_PROG_CXX()
AC_LANG([C++])

AC_MSG_CHECKING([Compiler to compile plugins for])
AC_ARG_WITH([target-compiler],
            [AS_HELP_STRING([--with-target-compiler=PATH],
                            [Path to the compiler to use with the plugins])],
            [TARGET_COMPILER="$withval"],
            [TARGET_COMPILER=$(which gcc)])
AC_MSG_RESULT([$TARGET_COMPILER])
AC_SUBST([TARGET_COMPILER])

AC_MSG_CHECKING([for -lz3])
save_LIBS=$LIBS
LIBS="-lz3 $LIBS"
AC_LINK_IFELSE([AC_LANG_PROGRAM([[
	  #include<z3++.h>
	]], [[
	  z3::context ctx;
	]])], [AC_DEFINE([HAVE_LIBZ3], [1], [Define to 1 if you have z3.])
               AC_MSG_RESULT([yes])],
	      [LIBS=$save_LIBS
               AC_MSG_RESULT([no])])

AC_MSG_CHECKING([for -lcvc5])
save_LIBS=$LIBS
LIBS="-lcvc5 $LIBS"
AC_LINK_IFELSE([AC_LANG_PROGRAM([[
          #include <cvc5/cvc5.h>
	]], [[
          cvc5::Solver solver;
          solver.setOption("produce-models", "true");
	]])], [AC_DEFINE([HAVE_LIBCVC5], [1], [Define to 1 if you have cvc5.])
               AC_MSG_RESULT([yes])],
	      [LIBS=$save_LIBS
               AC_MSG_RESULT([no])])

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT()
