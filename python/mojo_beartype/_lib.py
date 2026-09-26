"""ctypes bridge to the compiled PEP 484 type-code stack machine.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults. Scratch buffers are allocated here, on the Python
side, which keeps the exported Mojo symbols free of allocation and of `raises`.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-beartype.so"

# Operand classifications, mirroring the kernel's constants.
OPERAND_NONE = 0
OPERAND_SINGLE = 1
OPERAND_SET = 2
OPERAND_TERM = 3

# Fault codes reported by the kernel, in the order it can raise them.
FAULTS = {
    0: "well-formed",
    1: "negative code count",
    2: "empty code array",
    3: "code after the terminator",
    4: "stack depth is not one at the terminator",
    5: "code needs more operands than the stack holds",
    6: "inconsistent pops and pushes",
    7: "operand stack underflow",
    8: "missing terminator",
}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    i = ctypes.c_int64
    lib.bt_stack_check.restype = i
    lib.bt_stack_check.argtypes = [i, i, i, i, i, i]
    lib.bt_stack_profile.restype = i
    lib.bt_stack_profile.argtypes = [i, i, i, i, i, i]
    return lib


lib = _load()


def check_codes(codes, pops, pushes, kind) -> dict:
    """Walk a type-code array and report its structural verdict.

    `pops`, `pushes` and `kind` are indexed by code value, so the caller
    supplies the dialect's table. Returns a dict with `ok`, `code` (0 when
    well-formed, else the fault), `message`, `index` (the offending code, -1
    when there is none), `depth` there and `peak`.
    """
    arr = np.asarray(codes, dtype=np.uint8)
    pops_arr = np.asarray(pops, dtype=np.int32)
    pushes_arr = np.asarray(pushes, dtype=np.int32)
    kind_arr = np.asarray(kind, dtype=np.int32)
    result = np.zeros(4, dtype=np.int64)
    ok = lib.bt_stack_check(
        arr.ctypes.data if arr.size else 0, arr.size,
        pops_arr.ctypes.data, pushes_arr.ctypes.data, kind_arr.ctypes.data,
        result.ctypes.data,
    )
    code = int(result[0])
    return {
        "ok": ok == 0,
        "code": code,
        "message": FAULTS.get(code, "unknown fault"),
        "index": int(result[1]),
        "depth": int(result[2]),
        "peak": int(result[3]),
    }


def profile_codes(codes, pops, pushes, kind):
    """Operand depth after each code, plus the peak, or None if malformed."""
    arr = np.asarray(codes, dtype=np.uint8)
    pops_arr = np.asarray(pops, dtype=np.int32)
    pushes_arr = np.asarray(pushes, dtype=np.int32)
    kind_arr = np.asarray(kind, dtype=np.int32)
    depths = np.zeros(max(arr.size, 1), dtype=np.int32)
    peak = lib.bt_stack_profile(
        arr.ctypes.data if arr.size else 0, arr.size,
        pops_arr.ctypes.data, pushes_arr.ctypes.data, kind_arr.ctypes.data,
        depths.ctypes.data,
    )
    if peak < 0:
        return None
    return [int(v) for v in depths[: arr.size]], int(peak)
