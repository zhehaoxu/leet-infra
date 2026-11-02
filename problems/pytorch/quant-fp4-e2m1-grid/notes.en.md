# notes (quant-fp4-e2m1-grid)

## Source

- Element format: FP4 `E2M1` from the OCP **MX v1.0** spec — 1 sign + 2 exponent + 1 mantissa,
  `bias=1`, **no inf / no NaN**, largest finite value `6`, and exactly one non-zero subnormal
  code (`0.5`). Bit-field decoding: `e=0` gives `mag = m/2`; `e>0` gives `mag = (1 + m/2)·2^(e-1)`.
- Scaling: MXFP4 uses one E8M0 shared exponent per 32 elements (single level); NVFP4 (Blackwell)
  uses one E4M3 block scale per 16 elements times one FP32 per-tensor scale (two levels).
  This problem takes **already-decoded scale values** as inputs and only tests the element
  format plus the two-level product. E8M0 encoding itself belongs to `quant-mx-microscaling`(612),
  and E4M3 round-trips belong to the FP8 problems (125/445), so neither is repeated here.
- The judge environment has no FP4 hardware and no `torch.float4_e2m1fn_x2`, so the reference
  table is rebuilt from the bit-field formula.

## Traps in the upstream implementation

1. **`6.0` does not saturate, `6.0000001` does**: the gate is `|scaled| > 6`. Using `>=` marks
   an exact `6` as saturated and corrupts both `sat_mask` and `block_sat`.
2. **No `inf` code and no `NaN` code**: `inf` can only saturate to `6`; `NaN` has no code, so
   the upstream contract is "input must not contain NaN" and this problem raises `ValueError`.
3. **The only subnormal is `0.5`**: at `e=0` the mantissa contributes `m/2`, so `0.1 / 0.2 / 0.25`
   can only round to `0`. Writing the grid as "starts at `0.25`, step `0.25`" is the classic bug.
4. **RTNE "even" means an even mantissa bit, not an even magnitude**: `1.25 → 1.0` and
   `3.5 → 4.0` (both winners have mantissa bit 0). Because the code index parity equals the
   mantissa bit here, "tie to even" can be read directly off the index.
5. **`-0.0` has its own code `8`**: testing the sign with `v < 0` misses negative zero;
   `torch.signbit` / `math.copysign` are needed, and any negative value below `0.25` becomes `-0.0`.
6. **The two-level scale is a product**: `eff = global · block`, not `(v / block) / global`.
   The two differ in the last few ULPs for non-power-of-two scales; the product is the pinned
   contract here, and the `eff_scale` field makes the difference observable.

## Normalizations in this problem

- Code layout is fixed as `code = (sign << 3) | (exp << 1) | mant`, and `dequant = ±GRID[idx] · eff`.
- Tie detection uses **exact float equality** (every midpoint `0.25/0.75/1.25/1.75/2.5/3.5/5.0`
  is binary-exact) and exposes `tie_mask`, so a right answer reached by a different process
  cannot pass silently.
- Validation gates: `block_size` is a positive integer (`bool` rejected, tested separately so
  the length check cannot mask it), `block_scales` length equals the block count with finite
  positive entries, `global_scale` finite and positive, `values` free of `NaN`; empty input is legal.
- All eight returned tensors own their storage; mutating `dequant` in place leaves `codes` and
  `eff_scale` untouched.

## Mutations

All 16 mutants are CAUGHT, with no equivalent mutant:

```
m-tie-odd / m-tie-to-lo      odd / lower tie pick     → RTNE table checks
m-sat-ge / m-sat-code        >= gate / code 6          → sat_mask + codes
m-maxv                       6.0 → 6.5                 → 6.0000001 stops saturating
m-grid-subnormal             0.5 → 0.25                → grid + subnormal checks
m-no-negzero                 copysign → v < 0          → negative-zero check
m-eff-sum / m-eff-no-global  additive / dropped global → two-level checks
m-dequant-sign               sign dropped on dequant   → negative dequant check
m-nearest-lower              always round down         → 0.4 / 0.26 checks
m-block-wrap                 block id via modulo       → block-wall check
m-no-len-check               length check removed      → validation check
m-bs-bool                    bool rejection removed    → dedicated bool check
m-nan-ok                     NaN allowed               → NaN check
m-block-ties-sat             tie counter tracks sat    → block_ties check
```

## Overlap check

- `quant-mx-microscaling`(612) tests the E8M0 shared-exponent encoding itself and explicitly
  does not test downstream element formats; this problem never touches E8M0 encoding.
- FP8 round-trip / block scaling (125/445) covers E4M3 and E5M2 widths and scale recursion,
  not a 4-bit element grid.
- `quant-uint4-asym-pack` packs 4-bit asymmetric integers into whole bytes; no byte repacking here.
- `quant-afpq-dual-scale-fp4` treats the FP4 code table as an abstract set and searches two-sided
  scales; this problem pins the E2M1 bit-level semantics and the two-level product, with no search.
