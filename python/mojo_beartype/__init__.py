"""mojo-beartype: PEP 484 type-code structural checking with the stack machine
in Mojo.

`beartype` is runtime type checking and has no numeric core. This package ports
the one loop it runs over a byte array, the operand-stack machine, and says so
plainly rather than inventing a kernel to fill a coverage table. It does not
import or shadow `beartype`; see the README for why parity against the real
package is not possible on this toolchain.
"""

from .core import (
    CODE_SET,
    CODE_TERM,
    FAULTS,
    OPERAND_NONE,
    OPERAND_SET,
    OPERAND_SINGLE,
    OPERAND_TERM,
    CodeTable,
    check_codes,
    profile_codes,
    simple_table,
)

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
    "simple_table",
]
__version__ = "0.1.0"
