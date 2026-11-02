# notes (quant-fp6-dual-subformat-pack)

## Source

- Paper: arXiv 2401.14112 (FP6-LLM), official implementation `usyd-fsalab/fp6_llm`. The claim:
  weights and activations have different value ranges, so `FP6_E3M2` (wider exponent, `max 28`)
  suits activations with outliers while `FP6_E2M3` (wider mantissa, `max 7.5`, finer step)
  suits narrow weights.
- Bit fields (IEEE-style `1 + E + M`, `e=0` means subnormal, `bias = 2^(E-1) - 1`):
  - `e3m2`: `E=3, M=2, bias=3`; `max = (1 + 3/4)·2^(7-3) = 28`; smallest subnormal `0.25·2^-2 = 0.0625`.
  - `e2m3`: `E=2, M=3, bias=1`; `max = (1 + 7/8)·2^(3-1) = 7.5`; smallest subnormal `0.125·2^0 = 0.125`.
  - Neither has `inf` / `NaN` (all-ones exponent and mantissa is just the maximum 28 / 7.5).
- Packing: the upstream 3-byte path splices four 6-bit codes into 24 bits as a little-endian
  bit stream. This problem writes the three byte formulas into the statement, so the byte
  order is self-contained and does not depend on the exact CUDA kernel in the repo.

## Traps in the upstream implementation

1. **6 bits cross byte boundaries**: `b0` carries the 6 bits of `c0` plus the low 2 bits of
   `c1`; shifting without masking first overflows past 255 and `packed` is no longer bytes.
2. **The two layouts must not be mixed**: in `e3m2` the exponent sits at `bit4..2`, in `e2m3`
   at `bit4..3`, so both `>> mbits` and `& (2^ebits - 1)` must follow the format.
3. **`auto` boundaries are closed**: `amax == 7.5` still picks `e2m3` (it fits and is finer),
   `amax == 28` still picks `e3m2`; strict inequality pushes exactly-fitting tensors to the
   wider format.
4. **An explicit format saturates instead of raising**: only `auto` raises `ValueError` above
   `28`; explicit `e2m3` given `8.0` must saturate to `7.5` and set `sat_mask`.
5. **Subnormal top and smallest normal are adjacent, not duplicated**: in `e3m2` the top
   subnormal is `0.1875` (code 3) and the smallest normal is `0.25` (code 4). While designing
   this problem I assumed `0.25` had both a subnormal and a normal encoding; enumerating the
   64 code values showed **no duplicate values** (only two zero codes, `±0`), so the check was
   replaced by the adjacency assertion above.
6. **Padding codes must be truncated**: `unpacked` has to cut back to the original length, or
   a length-1 input returns 3 extra zero codes.

## Normalizations in this problem

- `codes` are 6-bit values (sign in `bit5`); `packed` follows its own little-endian bit-stream
  formula; `unpacked` is produced by unpacking `packed`, so the ledger proves the round trip.
- `tie_mask` means "two codes with **different values** are equidistant from the scaled value";
  an exact grid hit is not a tie (only `±0` has two codes, and the sign preference separates them).
- Validation gates: `fmt` whitelist and `NaN` rejection; empty input returns an empty ledger
  with `pad_count == 0`.
- Returned tensors never share storage; `codes` and `unpacked` are independent.

## Mutations

All 19 mutants are CAUGHT:

```
m-spec-bias / m-spec-mbits    wrong bias or mantissa width  → extreme-value checks
m-limit-formula               saturation point as 2^(2^E - bias) → 28 / 7.5 checks
m-tie-ignore-parity           tie ignores mantissa parity   → 0.09375 check
m-tie-no-sign                 no ±0 sign preference         → -0.0 check
m-tie-any-exact               exact hit counted as a tie    → 0.1875 / 0.25 check
m-subnormal-shift             subnormal loses one 2^(1-bias) → 0.0625 check
m-pack-b0/b1/b2               one byte formula each         → byte assertions
m-unpack-c2/c3                unpack shift off by one       → unpacked == codes
m-pad                         padding as n % 4              → pad_count check
m-auto-le / m-auto-range      auto boundary/upper bound     → selection checks
m-sat-ge / m-sat-nosign       saturation gate or sign       → saturation checks
m-fmt-gate / m-nan-gate       whitelist and NaN gates       → validation check
```

Equivalent mutant (excluded, not counted): flipping the last `_pick` tiebreak from `code` to
`-code` — neither format has duplicate code values (only `±0`, decided earlier by the sign
preference), so that tiebreak is unreachable.

## Overlap check

- `quant-mx-microscaling`(612) and `quant-fp4-e2m1-grid`(740) cover 4-bit element formats and
  shared exponents, with no non-byte-aligned packing.
- `quant-uint4-asym-pack` packs 4-bit integers into whole bytes, with no cross-byte splicing.
- The two FP8 problems (125/445) use 8-bit codes that divide bytes evenly.
- The only subject here is "two 6-bit subformats plus 4-in-3 cross-byte repacking".
