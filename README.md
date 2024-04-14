# smtgcc
This is an implementation of translation validation for GCC (similar to LLVM's [Alive2](https://github.com/AliveToolkit/alive2)), used to find bugs in the compiler.

The main functionality is in a plugin, which is passed to GCC when compiling:
```
gcc -O3 -fplugin=smtgcc-tv file.c
```
This plugin checks the IR (Intermediate Representation) before and after each optimization pass and reports an error if the IR after a pass is not a refinement of the input IR (which means the optimized code doesn't do the same thing as the input source code — that is, GCC has miscompiled the program). While the tool is somewhat limited, it has already found several bugs in GCC. A partial list of bugs found includes:
[106513](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106513),
[106523](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106523),
[106744](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106744),
[106883](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106883),
[106884](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106884),
[106990](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=106990),
[108625](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108625),
[109626](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109626),
[110434](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110434),
[110487](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110487),
[110495](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110495),
[110554](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110554),
[110760](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110760),
[111257](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111257),
[111280](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111280),
[111494](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111494),
[112736](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=112736),
[113588](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=113588),
[113590](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=113590),
[113630](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=113630),
[113703](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=113703),
[114032](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114032),
[114056](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114056),
[114090](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114090).

The implementation is described in a series of blog posts. The first posts describe a previous version of this tool ([pysmtgcc](https://github.com/kristerw/pysmtgcc)), but the general ideas are the same for both tools:
1. [Writing a GCC plugin in Python](https://kristerw.github.io/2022/10/20/gcc-python-plugin/)
2. [Verifying GCC optimizations using an SMT solver](https://kristerw.github.io/2022/11/01/verifying-optimizations/)
3. [Memory representation](https://kristerw.github.io/2023/07/17/memory-representation/)
4. [Address calculations](https://kristerw.github.io/2023/07/18/address-calculations/)
5. [Pointer alignment](https://kristerw.github.io/2023/07/20/pointer-alignment/)
6. Problems with pointers
7. Uninitialized memory
8. Control flow

# Compiling smtgcc
You must have the Z3 SMT solver installed. For example, as
```
sudo apt install libz3-dev
```
Configuring and building `smtgcc` is done by `configure` and `make`, and you must specify the target compiler for which to build the GCC plugins
```
./configure --with-target-compiler=/path/to/install/bin/gcc
make
```

# plugins

## smtgcc-tv
smtgcc-tv compares the IR before/after each GIMPLE pass and complains if the resulting IR is not a refinement of the input (i.e. if the GIMPLE pass miscompiled the program).

For example, compiling the function `foo` from [PR 111494](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111494)
```c
int a[32];
int foo(int n) {
  int sum = 0;
  for (int i = 0; i < n; i++)
    sum += a[i];
  return sum;
}
```
with a compiler where the bug is not fixed (for example, the current trunk GCC) using `smtgcc-tv.so`
```
gcc -O3 -fno-strict-aliasing -c -fplugin=/path/to/smtgcc-tv.so pr111494.c
```
gives us the output
```
pr111494.c: In function 'foo':
pr111494.c:2:5: note: ifcvt -> dce: Transformation is not correct (UB)
.param0 = #x00000005
.memory = (let ((a!1 (store (store (store ((as const (Array (_ BitVec 64) (_ BitVec 8)))
[...]
```
telling us that the output IR of the dce pass is not a refinement of the input that comes from ifcvt (in this case the error is in the vectorizer pass, but we are treating vect followed by dce as one pass because of [PR 111257](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111257)) because the result has more UB than the original, and the tool give us an example for `n = 5` where this happens.

## smtgcc-tv-backend
smtgcc-tv-backend compares the GIMPLE IR from  the last GIMPLE pass with the generated assembly code, and complains if the resulting assembly code is not a refinement of the GIMPLE IR (that is, if the backend has miscompiled the program).

This was just a quick experiment, so it has far too many limitations to be useful:
 * Only RISC-V
 * The source code must have exactly one function, called `foo`
 * The output must go to a file `k.s`
 * The function must not access global memory
 * The ABI is not correctly implemented, so the function must have a few parameters of an integral type.

## smtgcc-check-refine
smtgcc-check-refine requires the translation unit to consist of two functions named `src` and `tgt`, and it verifies that `tgt` is a refinement of `src`.

For example, testing changing the order of signed addition
```c
int src(int a, int b, int c)
{
  return a + c + b;
}

int tgt(int a, int b, int c)
{
  return a + b + c;
}
```
by compiling as
```
gcc -O3 -fno-strict-aliasing -c -fplugin=/path/to/smtgcc-check-refine.so example.c
```
gives us the output
```
example.c: In function 'tgt':
example.c:6:5: note: Transformation is not correct (UB)
.param2 = #x9620d6eb
.param0 = #x7edbb92a
.param1 = #x062be612
```
telling us that `tgt` invokes undefined behavior in cases where `src` does not,
and gives us an example of input where this happens (the values are, unfortunately, written as unsigned values. In this case, it means `[c = -1776232725, a = 2128329002, b = 103540242]`).

**Note**: smtgcc-check-refine works on the IR from the ssa pass, i.e., early enough that the compiler has not done many optimizations. But GCC does peephole optimizations earlier (even when compiling as `-O0`), so we need to prevent that from happening when testing such optimizations. The pre-GIMPLE optimizations are done one statement at a time, so we can disable the optimization by splitting the optimized pattern into two statements. For example, to check the following optimization
```
-(a - b)  ->  b - a
```
we can write the test as
```
int src(int a, int b)
{
  int t = a - b;
  return -t;
}

int tgt(int a, int b)
{
  return b - a;
}
```
Another way to verify such optimizations is to write the test in GIMPLE and pass the `-fgimple` flag to the compiler.

It is good practice to check with `-fdump-tree-ssa` that the IR used by the tool looks as expected. 

# Environment variables
 * `SMTGCC_VERBOSE` — Print debug information while running. Valid value 0-2, higher value prints more information (Default: 0)
 * `SMTGCC_TIMEOUT` — SMT solver timeout (Default: 120000)
 * `SMTGCC_MEMORY_LIMIT` — SMT solver memory use limit in megabytes (Default: 10240)

# Limitations
Some of the major limitations in the current version:
* Function calls are not implemented.
* Exceptions are not implemented.
* Only tested on C and C++ source code.
* Irreducible loops are not handled.
* Memory semantics is not correct
  - Strict aliasing does not work, so you must pass `-fno-strict-aliasing` to the compiler.
  - Handling of pointer provenance is too restrictive.
  - ...
