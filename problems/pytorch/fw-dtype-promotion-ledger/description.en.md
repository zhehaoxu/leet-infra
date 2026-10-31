# Type Promotion Ledger: Result dtype and the Write-back Gate

## Problem

`int8 tensor * uint8 tensor` does not give int8; `float16 tensor * 2.0` does *not* widen to float32; and `torch.add(a, b, out=c)` raises `result type Float can't be cast to the desired output type Char` as soon as c's dtype "cannot hold" the result. All three behaviors are specified in the Type Promotion Mechanics section of the official *Tensor attributes* page — required reading in the PyTorch dtype / tensor-attribute chapters of the beginner tutorials, and the quietest footgun when debugging mixed precision and quantized inference.

This problem turns the rules into a pure-dtype ledger: given the kind and dtype of the two operands of a binary arithmetic operation, plus an optional output dtype, report the result dtype and whether writing back into it is accepted. No tensor values are involved anywhere.

## Rules

Implement `solve` (the parameter and invalid-input contract lives in the starter docstring). Key points:

1. The category order is `bool < integral < floating < complex`. uint8 is integral — being unsigned does not make it wider.
2. Only **bare Python numbers and 0-dim tensors** (the docs' "wrapped numbers") get the "participate only when strictly higher" treatment: when their category is not strictly higher than every dimensioned tensor's, they have no effect at all — hence `float16 tensor ⊗ 2.0 → float16`, `uint8 tensor ⊗ 3 → uint8`, `int8 tensor ⊗ 0-dim int64 → int8`. As soon as one is strictly higher, the result only has to hold those numbers (`bool tensor ⊗ 2.0 → float32`), and the lower-category dimensioned tensors stop constraining the result.
3. When neither operand is a dimensioned tensor, 0-dim tensors stop being wrapped and join the lattice in full (`0-dim uint8 ⊗ 0-dim int8 → int16`); only bare Python numbers keep the "strictly higher" rule (`0-dim uint8 ⊗ 3 → uint8`, but `0-dim bool ⊗ 3 → int64`).
4. Values never matter: a Python int counts as int64, a Python float as `default_float` (i.e. `torch.get_default_dtype()`).
5. Inside the highest category, take the smallest type that holds the participating operands: integers by range containment (`int8 ⊗ uint8 → int16`); in the float lattice float16 and bfloat16 sit on the **same tier** — alone they keep themselves, together they escalate to float32; integers never widen floats (`int64 ⊗ float16 → float16`); the complex lattice is escalated only by a **participating** float64 (`float64 ⊗ complex64 → complex128`, but `complex64 ⊗ 0-dim float64 → complex64`).
6. The write-back gate (shared by `out=` and in-place arithmetic) has exactly three prohibitions: an integral out rejects floating/complex results; a bool out accepts only bool; a non-complex out rejects complex results. Everything else passes even when it silently truncates (`int8 ← int64` and `float16 ← float32` are both `ok`).

## Examples

Each line is `lhs rhs out default_float -> result`; `w` = wrapped (0-dim tensor), `t` = tensor (dimensioned), `s` = bare Python scalar.

```
('t','int8')   ('t','uint8')  None 'float32' -> ('int16',   'ok')
('t','int64')  ('t','float16') None 'float32' -> ('float16', 'ok')
('t','float16') ('t','bfloat16') None 'float32' -> ('float32', 'ok')
('t','bool')   ('s','int')    None 'float32' -> ('int64',   'ok')
('t','bool')   ('s','float')  None 'float32' -> ('float32', 'ok')
('t','uint8')  ('s','int')    None 'float32' -> ('uint8',   'ok')
('t','uint8')  ('s','float')  None 'float32' -> ('float32', 'ok')
('t','int8')   ('w','int64')  None 'float32' -> ('int8',    'ok')
('t','float16') ('s','float') None 'float32' -> ('float16', 'ok')
('t','float16') ('s','float') None 'float64' -> ('float16', 'ok')
('t','float64') ('s','float') None 'float64' -> ('float64', 'ok')
('w','uint8')  ('w','int8')   None 'float32' -> ('int16',   'ok')
('w','uint8')  ('s','int')    None 'float32' -> ('uint8',   'ok')
('w','bool')   ('s','int')    None 'float32' -> ('int64',   'ok')
('s','bool')   ('s','int')    None 'float32' -> ('int64',   'ok')
('t','float64') ('t','complex64') None 'float32' -> ('complex128', 'ok')
('t','complex64') ('w','float64') None 'float32' -> ('complex64',  'ok')
('t','int8')   ('t','float32') 'int8'  'float32' -> ('float32',  'cast_error')
('t','int8')   ('t','int64')   'int8'  'float32' -> ('int64',    'ok')
('t','uint8')  ('t','int8')    'uint8' 'float32' -> ('int16',    'ok')
('t','float32') ('t','float32') 'float16' 'float32' -> ('float32', 'ok')
('t','bool')   ('s','int')     'bool'  'float32' -> ('int64',    'cast_error')
('t','complex64') ('t','float64') 'float64' 'float32' -> ('complex128', 'cast_error')
('t','float64') ('t','float64') 'complex64' 'float32' -> ('float64', 'ok')
```

Line 8: in `('t','int8') ⊗ ('w','int64')` the 0-dim int64 is integral, the same category as the dimensioned int8, so it is not strictly higher and does not participate — int8 alone decides the result. Line 12 is the reverse: with no dimensioned tensor at all, 0-dim uint8 and 0-dim int8 both participate and the smallest holder is int16.

## Constraints

- Every dtype is one of 12 lowercase strings: `bool uint8 int8 int16 int32 int64 float16 bfloat16 float32 float64 complex64 complex128`; `status` is either `"ok"` or `"cast_error"`.
- Purely discrete comparison — no float tolerance, no randomness.
- Do not cheat by looking the answer up with `torch.result_type` / `torch.promote_types` / `torch.can_cast`.
