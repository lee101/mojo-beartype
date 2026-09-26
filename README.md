# mojo-beartype

`mojo-beartype` is not a port of `beartype`. It is a port of the one loop
`beartype` runs over a byte array — the PEP 484 type-code operand-stack machine
— and this README leads with that because `beartype` has no numeric core and
pretending otherwise would be the dishonest choice.

## This package has no numeric core, and that is the finding

`beartype` is runtime type checking. There is no array kernel in it, no
arithmetic beyond a type code's stack effect, and no numerical surface of any
size. A coverage table with a big kernel in it would be a fabrication; the
honest table is the one below, and it has one row.

What does exist is the operand-stack machine. A PEP 484 type hint compiles to a
flat array of type codes terminated by a terminator code, and the validator
walks that array maintaining an operand stack. Each code declares how many
operands it **pops** and how many it **pushes**; a bare name pops nothing and
pushes one, a subscript and a set operation each pop two and push one, and the
terminator closes the array with exactly one operand on the stack. That walk is
the validator, and a code's pops and pushes are the only arithmetic in the
package.

So the kernel is a compiled structural pass over a code array: it reports
whether the array is well formed, which code first violates the grammar, the
operand depth there, and the deepest the stack got. The code tables are supplied
by the caller, so the same kernel serves PEP 484, PEP 585 and PEP 604 without a
recompile.

Modelling pops and pushes separately, rather than as one net arity, is
deliberate. A single net arity cannot express "pop two, push one" without a
special case for the codes that do it, and a special case is exactly where this
kind of machine grows an off-by-one. There is a test for that specific bug.

## Parity with the real `beartype` is impossible on this toolchain

`beartype` 0.22.9 cannot be imported under the pinned Python 3.13.13. It
vendors a `typing` compatibility shim at `beartype/typing/`, and importing it
puts that package ahead of the standard library, so the `from typing import ...`
inside beartype resolves to its own shim and dies:

```
ImportError: cannot import name 'TYPE_CHECKING' from 'typing'
```

That happens on `import beartype` itself, before any machinery under discussion
is reached. There is no upstream package to compare against. **Every test in
this repository is therefore analytic**: it checks the PEP 484 grammar the
machine implements, not agreement with something that cannot be loaded. That is
weaker evidence than parity, and it is stated here rather than buried.

The tests are still substantive. The machine has seven distinct failure modes
plus a depth profile, and the strongest test is a direct Python transcription of
the same rules run over 3000 random code arrays, compared field by field —
verdict, fault code, fault index, depth and peak.

## Covered subset

| area | implemented API | where the work happens |
| --- | --- | --- |
| Code-array structure | `CodeTable.check`, `check_codes` | Mojo: the operand-stack walk with per-code pops and pushes |
| Depth trace | `CodeTable.profile`, `profile_codes` | Mojo: depth after every code, plus the peak |
| Dialect tables | `CodeTable`, `simple_table` | Python metadata, validated at construction |
| Fault taxonomy | `FAULTS` | Python constants, one per failure mode |

Not implemented, and not invented: the entire rest of `beartype`. Value
validation, `isinstance` and the type-hint metaclass, code *generation* from a
type hint, the forward-reference resolver, the O(1) validator cache, the
violation-reason machinery, decorators, and the `beartype.door` API. None of
that is numeric, and a kernel invented to cover it would be a fabrication.
`beartype` has no numeric core to miss.

This package does not import or shadow `beartype`.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-beartype.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task.

## Performance

Best-of-N wall clock in one process. The reference is a plain Python
transcription of the same rules, which is both the strongest available baseline
and the thing the tests check against. There is no NumPy formulation of a
stack machine, and no upstream loop to time.

| case | python walk | mojo-beartype | ratio |
| --- | ---: | ---: | ---: |
| stack check, 8 codes | 2.14 us | 57.60 us | 0.04x |
| stack check, 64 codes | 15.36 us | 73.82 us | 0.21x |
| stack check, 512 codes | 149.16 us | 93.22 us | 1.60x |
| stack check, 4096 codes | 1102.73 us | 298.71 us | 3.69x |
| 200 arrays x 32 codes | 1.84 ms | 13.69 ms | 0.13x |
| 2000 arrays x 32 codes | 20.93 ms | 145.77 ms | 0.14x |

Read that table as it is. **This kernel is a loss for realistic workloads.**
The machine is a handful of table lookups per code with no arithmetic to
amortise over, so the compiled version only overtakes the Python walk once a
single array is long enough for the loop to matter, somewhere past a few hundred
codes. Real type hints are tens of codes, which is the first three rows and the
worst of them: a ctypes crossing plus three NumPy scratch allocations costs
tens of microseconds against a loop that takes two.

The last two rows are the case that should have been the good one and is not.
A validator checks one hint at a time, so the per-call cost is the only cost
that matters, and it is dominated by fixed overhead rather than by the walk.
Threading does not help because the work per call is negligible. If this kernel
were going to earn its place, the thing to fix would be the call overhead — one
caller-side buffer reused across a block of arrays — not the inner loop, which
is already fine.

This is the result the port produces, and it is reported as a loss rather than
quietly dropped. A long precompiled array of codes, where the grammar is
checked once and reused, is the one workload where the compiled version wins.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit.
`build/build.sh` compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-beartype.so`.

The Python layer owns every array. Scratch buffers are allocated per call, which
keeps the exported symbols free of allocation and therefore free of `raises` —
an `@export ... abi("C")` function cannot be `raises`. Buffers cross the C ABI as
64-bit addresses and are rebuilt in Mojo as `Pointer[T, AnyOrigin[mut=True]]`.

The loop is serial on purpose: it is a scan with a single accumulator, so there
is nothing to parallelise, and 1.2.0 cannot carry a pointer into a parallel
body in any case.

Four details are pinned by tests rather than left to inspection:

- **A terminator must be the last code.** A hint truncated after its terminator
  would otherwise validate the part before the truncation, which is a real
  failure mode for a type hint assembled from several fragments.
- **The stack must hold exactly one operand at the terminator.** That is the
  invariant that says the array describes one type rather than a partial
  expression.
- **A code's stack effect is pops and pushes, not a net arity.** Reading a
  subscript as "push two" grows the stack where it should shrink it, and a
  three-deep hint still validates while a one-deep one does not.
- **The reported index is the first fault, not the last.** A machine that
  validated the whole array and then reported where it ended up would give a
  plausible index pointing at the wrong code.

The machine is exact integer and index work, so the tests assert equality and
no tolerance is used.

## Tests

```bash
pixi run test
```

20 tests: each of the seven failure modes with its own fault code, the depth
profile, the peak tracked independently of the final depth, 64-deep nesting,
the first-fault index, the table's own metadata validation, and 3000 random code
arrays compared field by field against a Python transcription of the rules.

## License

MIT
