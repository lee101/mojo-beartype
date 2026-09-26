"""Correctness-gated benchmark for mojo-beartype.

`beartype` cannot be imported on this toolchain, so there is no upstream loop to
time against. The reference is therefore a plain Python transcription of the
same stack machine, which is the strongest available baseline and also the
thing the tests check against.

Expect parity or a loss. This is a handful of table lookups per code, the
machine is latency-bound rather than compute-bound, and there is no NumPy
formulation of it. The honest result is reported either way.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_beartype as bt  # noqa: E402

NAME, SUBSCRIPT, OR, TERM = 0, 1, 2, bt.CODE_TERM
SPEC = {NAME: (0, 1), SUBSCRIPT: (2, 1), OR: (2, 1)}
TABLE = bt.simple_table(SPEC)


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _python_check(codes, spec, term):
    """The same rules, in Python, as the reference."""
    depth = peak = 0
    for i, code in enumerate(codes):
        if code == term:
            if i != len(codes) - 1:
                return 3, i, depth, peak
            return (0, -1, depth, peak) if depth == 1 else (4, i, depth, peak)
        need, give = spec[code]
        if depth < need:
            return 5, i, depth, peak
        depth = depth - need + give
        peak = max(peak, depth)
    return 8, len(codes) - 1, depth, peak


def _make(n_codes: int, seed: int):
    """A well-formed code array of about `n_codes` codes.

    Built from fragments that each return the stack to depth one, so the result
    is valid by construction. Appending names to a random body would not do: a
    name only ever grows the stack, so there is no way back down to one.
    """
    rng = np.random.default_rng(seed)
    fragments = (
        (NAME, OR),
        (NAME, NAME, SUBSCRIPT, OR),
        (NAME, NAME, OR, OR),
        (NAME, NAME, NAME, SUBSCRIPT, OR, OR),
    )
    codes = [NAME]
    while len(codes) < n_codes:
        codes += list(fragments[int(rng.integers(0, len(fragments)))])
    codes.append(TERM)
    return codes


def bench_check(n_codes: int, repeats: int):
    codes = _make(n_codes, n_codes)
    mine = TABLE.check(codes)
    assert mine["ok"], mine
    theirs = _python_check(codes, SPEC, TERM)
    assert (mine["code"], mine["index"], mine["depth"], mine["peak"]) == theirs
    return (
        f"stack check {n_codes}",
        _time(lambda: _python_check(codes, SPEC, TERM), repeats),
        _time(lambda: TABLE.check(codes), repeats),
    )


def bench_many(n_arrays: int, n_codes: int, repeats: int):
    """A block of code arrays, which is how a validator actually runs."""
    arrays = [_make(n_codes, seed) for seed in range(n_arrays)]
    for codes in arrays[:8]:
        assert TABLE.check(codes)["ok"]
    return (
        f"{n_arrays} arrays x {n_codes}",
        _time(lambda: [_python_check(c, SPEC, TERM) for c in arrays], repeats),
        _time(lambda: [TABLE.check(c) for c in arrays], repeats),
    )


def main():
    plan = [(8, 2000), (64, 800), (512, 200), (4096, 50), (32768, 10)][:4]
    print(f"{'case':<24}{'python walk':>14}{'mojo-beartype':>16}{'ratio':>9}")
    print("-" * 63)
    for n_codes, repeats in plan:
        label, ref, ours = bench_check(n_codes, repeats)
        print(f"{label:<24}{ref*1e6:>12.2f}us{ours*1e6:>14.2f}us{ref/ours:>8.2f}x")
    for n_arrays, n_codes, repeats in ((200, 32, 20), (2000, 32, 5)):
        label, ref, ours = bench_many(n_arrays, n_codes, repeats)
        print(f"{label:<24}{ref*1e3:>12.2f}ms{ours*1e3:>14.2f}ms{ref/ours:>8.2f}x")
    print()
    print("ratios above 1.00x favour mojo-beartype; below 1.00x is a loss.")
    print("This kernel is a table lookup per code, so parity or a loss is the")
    print("expected result and the honest one to report.")


if __name__ == "__main__":
    main()
