# Limitations

## Loops

### Limitations on loop iterations
smtgcc cannot handle loops directly, so it unrolls them for a fixed number of iterations (currently 16). Any iterations beyond this are treated as UB.

For example, a loop like:
```c
for (int i = 0; i < n; i++)
{
  ...
}
```
is only checked for n < 16. Bugs that only appear in later iterations are not detected by smtgcc.

A loop like:
```c
for (int i = 0; i < 100; i++)
{
  ...
}
```
always exceeds the limit, so it is treated as UB and skipped entirely.

### Backend checking: Loops from structure assignment, etc.
The backend may lower structure assignments, `memcpy`, `memset`, etc., into loops that iterate beyond smtgcc's unroll limit. smtgcc will then report a false alarm because it considers the assembly code to have more UB than the original (as loops iterating beyond smtgcc's unroll limit are treated as UB).

This issue can be observed with the code below when compiled for RISC-V with -march=rv64gcv:
```c
struct S {
  unsigned char a[320];
} s1, s2;

void foo()
{
  s1 = s2;
}
```
The code is generated as:
```
foo:
	lui	a5,%hi(s1)
	lui	a4,%hi(s2)
	addi	a5,a5,%lo(s1)
	addi	a4,a4,%lo(s2)
	li	a2,320
.L2:
	vsetvli	a3,a2,e8,m1,ta,ma
	vle8.v	v1,0(a4)
	sub	a2,a2,a3
	add	a4,a4,a3
	vse8.v	v1,0(a5)
	add	a5,a5,a3
	bne	a2,zero,.L2
	ret
```
which requires 20 iterations to copy the structure, and smtgcc reports:
```
foo.c:foo: Transformation is not correct (UB)
```

### Backend checking: Loop canonicalization
The backend may rewrite loops in a way that causes problems for smtgcc, as it currently does not perform loop canonicalization.

To illustrate the problem, consider a loop:
```c  
do {
  ...
} while (a || b);
```
Depending on how the backend lowers this, the assembly might look like two nested loops:
```asm
.L2:
        ...
        bne     a0,zero,.L2
        bne     a1,zero,.L2
```
or a single loop:
```asm
.L2:
        ...
        or      a2,a0,a1
        bne     a2,zero,.L2
```
The first case is unrolled as two nested loops, effectively unrolled 16*16 iterations, while the second case is unrolled for only 16 iterations.

Simple loops like these are usually handled correctly by the tool, but more complex loops can cause issues where smtgcc may interpret the original source code as having two nested loops, while the assembly has only one.

## Memory
TODO

## Floating-point
TODO
