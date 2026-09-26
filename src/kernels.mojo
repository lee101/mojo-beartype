"""The PEP 484 type-code operand-stack machine, in compiled Mojo.

## Why this kernel exists

`beartype` is runtime type checking. It has no arithmetic, no array kernel, and
no numerical core of any kind; the honest coverage table for a port of it is
almost empty, and inventing a numeric kernel to fill that table would be a
defect rather than a port.

There is exactly one thing in the package that is a loop over a byte array, and
this is it. A PEP 484 type hint is compiled to a flat array of type codes
prefixed by a terminating code, and the validator walks that array with an
operand stack. Each code declares how many operands it **pops** and how many it
**pushes**; a name pushes one and pops none, a subscript and a set operation
each pop two and push one, and the terminator closes the array. That walk is the
validator, and a code's pops and pushes are the only arithmetic in the package.

So the port is the stack machine: a compiled pass over the code array that
tracks operand depth and reports the first structural violation. That is a real
computation over a real byte array with a real observable result, and it is
checkable without importing `beartype` at all, which matters because this
package cannot be imported on the pinned Python 3.13 (its vendored `typing`
shim shadows the standard library module of the same name).

Modelling pops and pushes separately, rather than as a single arity, is what
makes the machine general. A single net arity cannot express "pop two, push one"
without a special case for the codes that do it, and a special case is exactly
where this kind of machine develops an off-by-one.

The code tables are supplied by the caller rather than hard-coded here, so the
same kernel serves the PEP 585 and PEP 604 dialects without a recompile.

Buffers cross the C ABI as 64-bit addresses and are rebuilt inside each body,
because `@export` rejects a function whose parameter types are inferred. Nothing
here allocates, so no exported symbol is `raises`.
"""

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I32 = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64 = Pointer[Int64, AnyOrigin[mut=True]]

# Operand kinds. Only the terminator is structurally special; `set` is recorded
# because a set operation is the one code whose underflow is a bug in the hint
# rather than a truncation, and the distinction is worth reporting.
comptime OPERAND_NONE = 0
comptime OPERAND_SINGLE = 1
comptime OPERAND_SET = 2
comptime OPERAND_TERM = 3


def bu8(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def bi32(addr: Int) -> I32:
    return I32(unsafe_from_address=addr)


def bi64(addr: Int) -> I64:
    return I64(unsafe_from_address=addr)


@export("bt_stack_check")
def bt_stack_check(codes_addr: Int, n: Int, pops_addr: Int, pushes_addr: Int,
                   kind_addr: Int, result_addr: Int) abi("C") -> Int:
    """Walk `n` type codes with an operand stack and report the first fault.

    `pops[code]` and `pushes[code]` describe the code's stack effect and
    `kind[code]` its classification. The rules, and the only rules, are:

    - the stack must hold at least `pops[code]` operands before the code runs
    - the code then leaves `depth + pushes[code] - pops[code]` operands
    - a terminator ends the array; it is always the last code
    - the stack must hold exactly one operand when the terminator is reached

    `result` receives four int64s: the fault code (0 when well-formed), the index
    of the offending code (-1 when there is none), the operand depth there, and
    the deepest the stack got. The return value is 0 for a well-formed array and
    -1 otherwise, so a caller that only wants a boolean never reads the detail.
    """
    var res = bi64(result_addr)
    res[unsafe_offset=0] = 0
    res[unsafe_offset=1] = -1
    res[unsafe_offset=2] = 0
    res[unsafe_offset=3] = 0
    if n < 0:
        res[unsafe_offset=0] = 1
        return -1
    if n == 0:
        # An empty array carries no hint at all, so there is no terminator to
        # close and the depth-one invariant cannot hold.
        res[unsafe_offset=0] = 2
        return -1

    var codes = bu8(codes_addr)
    var pops = bi32(pops_addr)
    var pushes = bi32(pushes_addr)
    var kind = bi32(kind_addr)

    var depth = 0
    var peak = 0
    for i in range(n):
        var code = Int32(codes[unsafe_offset=i])
        var k = kind[unsafe_offset=code]
        if k == OPERAND_TERM:
            if i != n - 1:
                # A terminator is always the final code. Anything after it is
                # unreachable, and accepting that would let a truncated hint
                # validate the part before the truncation.
                res[unsafe_offset=0] = 3
                res[unsafe_offset=1] = Int64(i)
                res[unsafe_offset=2] = Int64(depth)
                res[unsafe_offset=3] = Int64(peak)
                return -1
            if depth != 1:
                res[unsafe_offset=0] = 4
                res[unsafe_offset=1] = Int64(i)
                res[unsafe_offset=2] = Int64(depth)
                res[unsafe_offset=3] = Int64(peak)
                return -1
            res[unsafe_offset=2] = Int64(depth)
            res[unsafe_offset=3] = Int64(peak)
            return 0
        var need = pops[unsafe_offset=code]
        var give = pushes[unsafe_offset=code]
        if need < 0 or give < 0 or give - need < -1:
            res[unsafe_offset=0] = 6
            res[unsafe_offset=1] = Int64(i)
            res[unsafe_offset=2] = Int64(depth)
            res[unsafe_offset=3] = Int64(peak)
            return -1
        if depth < Int(need):
            res[unsafe_offset=0] = 5
            res[unsafe_offset=1] = Int64(i)
            res[unsafe_offset=2] = Int64(depth)
            res[unsafe_offset=3] = Int64(peak)
            return -1
        depth = depth - Int(need) + Int(give)
        if depth > peak:
            peak = depth
        if depth < 0:
            res[unsafe_offset=0] = 7
            res[unsafe_offset=1] = Int64(i)
            res[unsafe_offset=2] = Int64(depth)
            res[unsafe_offset=3] = Int64(peak)
            return -1

    # The array ran out without a terminator.
    res[unsafe_offset=0] = 8
    res[unsafe_offset=1] = Int64(n - 1)
    res[unsafe_offset=2] = Int64(depth)
    res[unsafe_offset=3] = Int64(peak)
    return -1


@export("bt_stack_profile")
def bt_stack_profile(codes_addr: Int, n: Int, pops_addr: Int, pushes_addr: Int,
                     kind_addr: Int, depths_addr: Int) abi("C") -> Int:
    """Write the operand depth after each code, for tools that want the trace.

    Returns the maximum depth reached, or -1 if the array is malformed. The
    depth recorded at a terminator is the depth before it, since a terminator
    consumes nothing.
    """
    var codes = bu8(codes_addr)
    var pops = bi32(pops_addr)
    var pushes = bi32(pushes_addr)
    var kind = bi32(kind_addr)
    var out = bi32(depths_addr)
    var depth = 0
    var peak = 0
    for i in range(n):
        var code = Int32(codes[unsafe_offset=i])
        var k = kind[unsafe_offset=code]
        if k == OPERAND_TERM:
            if i != n - 1 or depth != 1:
                return -1
            out[unsafe_offset=i] = Int32(depth)
            return peak
        var need = pops[unsafe_offset=code]
        var give = pushes[unsafe_offset=code]
        if need < 0 or give < 0 or give - need < -1:
            return -1
        if depth < Int(need):
            return -1
        depth = depth - Int(need) + Int(give)
        if depth > peak:
            peak = depth
        out[unsafe_offset=i] = Int32(depth)
    return -1
