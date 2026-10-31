# Explained: Type Promotion Ledger — result dtype and the write-back gate

## Origin

pytorch/pytorch `docs/source/tensor_attributes.md` (anchor `(type-promotion-doc)=`): `:106-118` are the three promotion rules (a scalar or 0-dim operand participates only when its category is strictly higher; otherwise the dimensioned operands alone decide the type), `:120-123` states that a Python float takes `torch.get_default_dtype()`, a non-boolean integral scalar takes `torch.int64`, and that values are never inspected; `:165-168` lists the three write-back prohibitions and `:170-188` gives the allowed/disallowed in-place list. On the tutorial side this is the dtype section of *Learn the Basics*.

Two parts of that document lag behind the implementation, and this problem follows torch 2.14.1 behavior instead: `:122` "Complex types are not yet supported" is no longer true (complex has its own two-tier lattice), and `:114-116` only covers "a 0-dim tensor of higher category" — it never describes the branch where *all* operands are 0-dim or bare numbers.

## Key points and pitfalls

1. **The "strictly higher" rule for wrapped numbers is the one people invert**: `float16 tensor ⊗ 2.0 → float16`, `default_float` never enters — a Python float and float16 are both floating, so it is not strictly higher. This is the standard answer to "why didn't my half-precision multiply widen?".
2. **uint8 is integral, and unsigned does not mean wider**: `int8 ⊗ uint8 → int16` — the smallest integral type whose range holds both [-128,127] and [0,255].
3. **Integers never widen floats**: `int64 ⊗ float16 → float16` (same as the doc's `torch.add(long_tensor, float_tensor) → float32`, `:159-162`). Across categories the lower-category operands are simply ignored.
4. **float16 and bfloat16 share one tier**: there is no "bfloat16 ⊃ float16" containment; each survives alone, and only their combination escalates to float32. Treating floats as one linear chain is the most common wrong model.
5. **The hidden all-0-dim branch** (undocumented, measured): with no dimensioned tensor, 0-dim tensors stop being wrapped and join the lattice in full, so `0-dim uint8 ⊗ 0-dim int8 → int16`; bare Python numbers keep the strict-higher rule, so `0-dim uint8 ⊗ 3 → uint8` but `0-dim bool ⊗ 3 → int64`.
6. **Promotion and the write-back gate are separate questions**: `uint8 ← int16` passes (the doc's own `uint_tensor *= int_tensor` at `:181`) and truncates silently; the three prohibitions only block cross-category directions (integral out, bool out, non-complex out).
7. **`copy_` bypasses the gate**: `torch.zeros(3, torch.uint8).copy_(float32_tensor)` succeeds while `torch.add(..., out=uint8_tensor)` raises `RuntimeError: result type Float can't be cast to the desired output type Byte` — "assignment works, arithmetic does not" is normal here.
8. **The complex lattice is escalated only by a participating float64**: `complex64 ⊗ float64 tensor → complex128`, but `complex64 ⊗ 0-dim float64 → complex64` (not strictly higher than complex), and `complex64 ⊗ 2.0` stays complex64 whatever `default_float` is.

## Real behavior excluded from the domain

- Allowing float16/bfloat16/complex as 0-dim operands produces dtypes that appear in no documentation: measured, `torch.ones(3, float16) + torch.ones((), complex64)` → `torch.complex32`, and bfloat16 → `torch.bcomplex32`. Neither is one of the 12 public dtypes, so the statement restricts wrapped operands to 8 dtypes (bool, five integers, float32, float64).
- A Python complex works only in the `result_type(number, number)` and `result_type(number, tensor)` overloads and `torch.result_type(1j)` alone raises TypeError, so the scalar domain is only `bool/int/float`.
- `torch.result_type` accepts exactly two positional arguments (no varargs, no lists — measured TypeError), which is why this problem is specified over binary operands: no fold-order ambiguity and every grid cell maps one-to-one onto a real torch call.

## Cross-check basis

The solution is pure Python rule arithmetic and calls no torch promotion API. `test_cases.py` uses torch itself as the oracle:

- Promotion grid: 23×23 operand kinds (12 dimensioned dtypes + 8 0-dim dtypes + 3 scalar kinds, same domain on both sides) × {float32, float64} defaults = 1058 cells, each compared with `torch.result_type` (switching via `torch.set_default_dtype`, restored in `__exit__`).
- Write-back grid: 276 operand pairs (dimensioned/0-dim/scalar × 12 dimensioned) × 12 output dtypes = 3312 cells, each compared with whether `torch.add(a, b, out=)` raises `RuntimeError`.
- All operand values are 1 so that value-overflow checks never interfere with dtype-only behavior.

The 24 example lines of the statement were measured from the reference implementation and re-parsed back through `solve` for verification.

## Complexity

O(1) per call; the full cross-check is 4370 torch calls and runs in under a second.
