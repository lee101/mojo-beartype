"""Analytic tests for the PEP 484 type-code operand-stack machine.

`beartype` 0.22.9 cannot be imported under this toolchain's Python 3.13.13: it
vendors a `typing` shim at `beartype/typing/` that shadows the standard library
module of the same name, so `import beartype` itself fails with

    ImportError: cannot import name 'TYPE_CHECKING' from 'typing'

There is therefore no upstream package to compare against, and every test here
checks the machine against the PEP 484 grammar it implements. That is weaker
evidence than parity and the README says so. It is not weak evidence, though:
the machine has seven distinct failure modes and a depth profile, and a
plausible bug in any of them changes the verdict on a hand-built array.
"""

import numpy as np
import pytest

import mojo_beartype as bt

# A miniature dialect: a name pops nothing and pushes one, a subscript and a
# set operation each pop two and push one, and 255 terminates.
NAME, SUBSCRIPT, OR, TERM = 0, 1, 2, bt.CODE_TERM
TABLE = bt.simple_table({NAME: (0, 1), SUBSCRIPT: (2, 1), OR: (2, 1)})


def test_a_single_name_hint_is_well_formed():
    verdict = TABLE.check([NAME, TERM])
    assert verdict["ok"]
    assert verdict["code"] == 0
    assert verdict["index"] == -1
    assert verdict["peak"] == 1
    assert verdict["depth"] == 1


def test_a_union_of_three_names_is_well_formed():
    verdict = TABLE.check([NAME, NAME, OR, NAME, OR, TERM])
    assert verdict["ok"]
    assert verdict["peak"] == 2


def test_a_subscript_pops_two_and_pushes_one():
    """`list[int]` is Name, Name, Subscript: two operands in, one out.

    A machine that read the subscript's arity as a net push would grow the stack
    where it should shrink it, and a three-deep hint would still validate while
    a one-deep one would not. Both are caught below.
    """
    codes = [NAME, NAME, NAME, SUBSCRIPT, OR, TERM]
    verdict = TABLE.check(codes)
    assert verdict["ok"]
    assert verdict["peak"] == 3
    depths, _ = TABLE.profile(codes)
    assert depths == [1, 2, 3, 2, 1, 1]


def test_profile_reports_the_depth_after_every_code():
    depths, peak = TABLE.profile([NAME, NAME, OR, NAME, OR, TERM])
    assert depths == [1, 2, 1, 2, 1, 1]
    assert peak == 2


def test_profile_agrees_with_check_on_a_well_formed_array():
    codes = [NAME, NAME, NAME, SUBSCRIPT, OR, NAME, OR, TERM]
    verdict = TABLE.check(codes)
    depths, peak = TABLE.profile(codes)
    assert verdict["ok"] and depths is not None
    assert peak == verdict["peak"]
    assert depths[-1] == verdict["depth"]


def test_profile_returns_none_for_a_malformed_array():
    assert TABLE.profile([NAME, OR, TERM]) is None
    assert TABLE.check([NAME, OR, TERM])["ok"] is False


# -- the failure modes -----------------------------------------------------


def test_an_empty_array_is_rejected():
    verdict = TABLE.check([])
    assert not verdict["ok"]
    assert verdict["code"] == 2
    assert verdict["message"] == "empty code array"


def test_a_missing_terminator_is_rejected():
    verdict = TABLE.check([NAME, NAME, OR])
    assert not verdict["ok"]
    assert verdict["code"] == 8
    assert verdict["index"] == 2


def test_a_terminator_must_be_last():
    """A hint truncated after its terminator would otherwise validate the part
    before the truncation, which is what this check exists to catch."""
    verdict = TABLE.check([NAME, TERM, NAME])
    assert not verdict["ok"]
    assert verdict["code"] == 3
    assert verdict["index"] == 1


def test_the_stack_must_hold_exactly_one_operand_at_the_terminator():
    verdict = TABLE.check([NAME, NAME, TERM])
    assert not verdict["ok"]
    assert verdict["code"] == 4
    assert verdict["depth"] == 2


def test_a_set_operation_needs_two_operands():
    verdict = TABLE.check([NAME, OR, NAME, OR, TERM])
    assert not verdict["ok"]
    assert verdict["code"] == 5
    assert verdict["index"] == 1
    assert verdict["depth"] == 1


def test_a_bare_set_operation_underflows():
    verdict = TABLE.check([OR, TERM])
    assert not verdict["ok"]
    assert verdict["code"] == 5
    assert verdict["index"] == 0


def test_a_subscript_with_one_operand_underflows():
    verdict = TABLE.check([NAME, SUBSCRIPT, OR, TERM])
    assert not verdict["ok"]
    assert verdict["code"] == 5
    assert verdict["index"] == 1


def test_every_fault_code_has_a_message():
    for code, message in bt.FAULTS.items():
        assert isinstance(message, str) and message
    assert bt.FAULTS[0] == "well-formed"


# -- the machine is a real walk, not a shape check --------------------------


def test_the_peak_is_tracked_independently_of_the_final_depth():
    """A hint that goes three deep and comes back to one must report a peak of
    three. A machine that only checked the final depth would pass it, and so
    would one that reported the peak in the depth slot."""
    codes = [NAME, NAME, NAME, OR, OR, TERM]
    verdict = TABLE.check(codes)
    assert verdict["ok"]
    assert verdict["depth"] == 1
    assert verdict["peak"] == 3
    depths, peak = TABLE.profile(codes)
    assert depths == [1, 2, 3, 2, 1, 1]
    assert peak == 3


def test_deeply_nested_hints_are_well_formed():
    codes = [NAME]
    for _ in range(64):
        codes += [NAME, NAME, SUBSCRIPT, OR]
    codes.append(TERM)
    verdict = TABLE.check(codes)
    assert verdict["ok"]
    assert verdict["depth"] == 1
    # Two names push the stack to three before each subscript folds it back.
    assert verdict["peak"] == 3


def test_the_reported_index_is_the_first_fault_not_the_last():
    """A machine that validated the whole array and then reported where it ended
    up would give a plausible index pointing at the wrong code, which is the
    difference between a useful diagnostic and a misleading one."""
    codes = [NAME, NAME, OR, OR, NAME, OR, OR, NAME, OR, TERM]
    verdict = TABLE.check(codes)
    assert not verdict["ok"]
    assert verdict["code"] == 5
    assert verdict["index"] == 3


def test_the_kernel_agrees_with_a_plain_python_walk():
    """A direct transcription of the same rules, over random arrays.

    This is the strongest check available without the upstream package: if the
    two disagree on a verdict, a fault code, an index, a depth or a peak, one of
    them is wrong.
    """
    rng = np.random.default_rng(20260926)
    pool = np.array([NAME, SUBSCRIPT, OR], dtype=np.uint8)
    spec = {NAME: (0, 1), SUBSCRIPT: (2, 1), OR: (2, 1)}

    def reference(codes):
        depth = peak = 0
        for i, code in enumerate(codes):
            if code == TERM:
                if i != len(codes) - 1:
                    return False, 3, i, depth, peak
                if depth != 1:
                    return False, 4, i, depth, peak
                return True, 0, -1, depth, peak
            need, give = spec[code]
            if depth < need:
                return False, 5, i, depth, peak
            depth = depth - need + give
            peak = max(peak, depth)
        return False, 8, len(codes) - 1, depth, peak

    table = bt.simple_table(spec)
    for _ in range(3000):
        n = int(rng.integers(1, 14))
        codes = [int(v) for v in rng.choice(pool, size=n)] + [TERM]
        got = table.check(codes)
        ok, code, index, depth, peak = reference(codes)
        assert got["ok"] == ok, (codes, got)
        assert got["code"] == code, (codes, got)
        assert got["index"] == index, (codes, got)
        assert got["depth"] == depth, (codes, got)
        assert got["peak"] == peak, (codes, got)


# -- the table itself -------------------------------------------------------


def test_a_table_rejects_inconsistent_metadata():
    with pytest.raises(ValueError):
        bt.CodeTable([0, 1], [0, 1], [1, 1], [0])
    with pytest.raises(ValueError):
        bt.CodeTable([0, 2], [0, 0], [1, 1], [0, 0])
    with pytest.raises(ValueError):
        bt.CodeTable([0, 1], [0, -1], [1, 1], [0, 0])
    # A code may shrink the stack: a subscript pops two and pushes one.
    assert bt.CodeTable([0, 1], [0, 2], [1, 1], [0, bt.OPERAND_TERM], term_code=1)
    # Only the terminator may carry the term kind.
    with pytest.raises(ValueError):
        bt.CodeTable([0, 1], [0, 0], [1, 0], [bt.OPERAND_TERM, 0], term_code=1)


def test_check_codes_and_profile_codes_agree_directly():
    codes = [NAME, NAME, OR, TERM]
    size = TERM + 1
    pops = [0] * size
    pushes = [1] * size
    kinds = [bt.OPERAND_NONE] * size
    pops[OR] = 2
    kinds[OR] = bt.OPERAND_SET
    pushes[TERM] = 0
    kinds[TERM] = bt.OPERAND_TERM
    verdict = bt.check_codes(codes, pops, pushes, kinds)
    assert verdict["ok"]
    depths, peak = bt.profile_codes(codes, pops, pushes, kinds)
    assert peak == verdict["peak"]
    assert depths[-1] == verdict["depth"]
    assert depths == [1, 2, 1, 1]
