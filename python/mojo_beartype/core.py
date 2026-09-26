"""PEP 484 type-code structural checking, with the stack machine in Mojo.

## This package has no numeric core, and this module says so first

`beartype` is runtime type checking. There is no array kernel in it, no
arithmetic beyond a type code's pops and pushes, and no numeric surface of any
size. A port that presented a large kernel here would be inventing work to look
complete, which is worse than a small honest one.

What exists is the operand-stack machine. A PEP 484 type hint compiles to a flat
array of type codes terminated by a terminator code, and the validator walks
that array maintaining an operand stack: each code declares how many operands it
pops and how many it pushes, the stack must never underflow, and it must hold
exactly one operand when the terminator is reached. That walk is the validator.

So that is what is ported: a compiled structural pass over the code array. It
verifies a code array's shape, reports the first fault and where it is, and can
emit the depth after every code. The code tables are supplied by the caller, so
the same kernel serves PEP 484, PEP 585 and PEP 604.

## Parity with the real `beartype` is not possible on this toolchain, and here
is why

`beartype` 0.22.9 cannot be imported under the pinned Python 3.13.13. It vendors
a `typing` compatibility shim at `beartype/typing/`, and importing it puts that
directory ahead of the standard library, so the `from typing import ...` inside
beartype resolves to beartype's own shim and dies on

    ImportError: cannot import name 'TYPE_CHECKING' from 'typing'

That happens on `import beartype` itself, before any of the machinery under
discussion is reached. There is no upstream package to compare against, so the
tests check the PEP 484 grammar this machine implements rather than checking
agreement with something that cannot be loaded.
"""

from __future__ import annotations

from ._lib import (
    FAULTS,
    OPERAND_NONE,
    OPERAND_SET,
    OPERAND_SINGLE,
    OPERAND_TERM,
    check_codes,
    profile_codes,
)

# The two code values the grammar gives special meaning, and the defaults the
# convenience constructors use. A dialect's real table assigns these its own
# values; nothing here depends on these particular numbers.
CODE_SET = 254
CODE_TERM = 255

__all__ = [
    "CODE_SET",
    "CODE_TERM",
    "CodeTable",
    "FAULTS",
    "OPERAND_NONE",
    "OPERAND_SET",
    "OPERAND_SINGLE",
    "OPERAND_TERM",
    "check_codes",
    "profile_codes",
]


class CodeTable:
    """Pops, pushes and operand kind for each code of one type-hint dialect.

    This is the only per-dialect state the kernel needs, and keeping it in
    Python means a new dialect is a table, not a recompile.
    """

    def __init__(self, names, pops, pushes, kinds, *, set_code=None,
                 term_code=CODE_TERM):
        self.names = list(names)
        self.pops = [int(v) for v in pops]
        self.pushes = [int(v) for v in pushes]
        self.kinds = [int(v) for v in kinds]
        size = len(self.names)
        if len(self.pops) != size or len(self.pushes) != size or len(self.kinds) != size:
            raise ValueError("names, pops, pushes and kinds must be the same length")
        if self.names != list(range(size)):
            raise ValueError("code names must be the integers 0..n-1, in order")
        for code in range(size):
            if self.kinds[code] == OPERAND_TERM and code != term_code:
                raise ValueError("only the terminator may have the term kind")
            if self.pops[code] < 0 or self.pushes[code] < 0:
                raise ValueError(f"code {code} has a negative pops or pushes")
            # A code may legitimately shrink the stack: a subscript and a set
            # operation each pop two and push one. What it may not do is ask
            # for more operands than exist, and that is a property of the array
            # rather than of the table, so the machine checks it per code.
        if set_code is not None and self.kinds[set_code] != OPERAND_SET:
            raise ValueError("the set code must have the set operand kind")
        self.set_code = set_code
        self.term_code = term_code

    def check(self, codes):
        """Structural verdict for one code array."""
        return check_codes(codes, self.pops, self.pushes, self.kinds)

    def profile(self, codes):
        """Depth after each code, or None when the array is malformed."""
        return profile_codes(codes, self.pops, self.pushes, self.kinds)

    def is_well_formed(self, codes) -> bool:
        return bool(self.check(codes)["ok"])

    def __repr__(self) -> str:
        return f"CodeTable({len(self.names)} codes, term={self.term_code})"


def simple_table(spec, *, term_code=CODE_TERM):
    """Build a table from `{code: (pops, pushes)}` plus a set code.

    Every code not mentioned defaults to "pop nothing, push one", which is what
    a bare type name does; that is the overwhelming majority of a real table and
    spelling it out for each one would be noise.
    """
    size = max([*spec, term_code, len(spec)]) + 1
    pops = [0] * size
    pushes = [1] * size
    kinds = [OPERAND_NONE] * size
    for code, (need, give) in spec.items():
        pops[code] = need
        pushes[code] = give
    for code in spec:
        if code == term_code:
            kinds[code] = OPERAND_TERM
    pops[term_code] = 0
    pushes[term_code] = 0
    kinds[term_code] = OPERAND_TERM
    return CodeTable(
        range(size), pops, pushes, kinds, set_code=None, term_code=term_code
    )
