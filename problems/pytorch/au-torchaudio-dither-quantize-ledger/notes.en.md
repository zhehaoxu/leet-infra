# 870 Implementation Notes

## Verbatim Official Environment

`official870.py` extracts the bodies of `_add_noise_shaping`, `_apply_probability_distribution` and `dither` from `filtering.py` with `ast.get_source_segment` and `exec`s them into one shared namespace WITHOUT A SINGLE CHARACTER CHANGED. It injects real `torch` (`set_default_dtype(float64)`), `math` / `warnings` / `typing` / `F` / `Tensor`. The only stand-in is an RNG spy for `torch.randint`: it consumes an explicit injected sequence in CALL ORDER, records `high` / `size` / `value`, and reproduces the official RuntimeError message when `high<=0`.

## Three-Layer Independent Oracle

The probe recomputes Bartlett values and the quantization lattice with closed forms and exact `Fraction` rationals. Layer 2 deep-compares official `comp()` against reference `solve()` over 822 configs, type-sensitively and bit-exactly (including `-0.0`). Layer 3 does NOT share the reference algorithms: `round` uses `decimal.Decimal ROUND_HALF_EVEN`, Bartlett uses `Fraction(2*min(i,n-i),n)`, noise shaping uses an explicit `[0]+err[:-1]` list shift, and the GPDF alias uses a dict-based mutable cell store; exact rationals additionally verify that every output lies on the `k/32768` lattice.

## Mutation and Equivalence Evidence

`mut870.py` started with 172 mutants (13 anchor errors + 22 survivors). Anchor errors were fixed with explicit occurrence indices and multi-line replacements. Of the 22 survivors, **10 were judging blind spots** caught by two hardening rounds: `_clean` nan/inf/list branches, the `round` inf guard, three default parameters, nested-empty input, and a shallow census consts copy. `ba-half` (`n//2` to `(n-1)//2`) was found to differ by 1 ulp at `n=98` / `196`: the **11th blind spot, not a true equivalent**, caught after adding large-n constants.

Final: **171 mutants, 11 true equivalents excluded, 160 active all CAUGHT**. `equiv870.py` proves the 11 equivalents with 1214 configs x 9 modes = 617507 bit-exact full `solve()` output comparisons (including exception types / messages): dict key ordering, the dead `_clean`-on-dict branch, redundant empty guards, the `x+0.0` and `1-|x-1| == 2-x` float identities, rows/grid equivalence in non-mutating branches, a comment-only edit, shape-label swaps under the equal-shape domain, and unknown densities already falling through to TPDF. **No mutant is deleted**.

## Engineering Lessons

- An RNG spy must consume its sequence in CALL ORDER; fixed positional indexing shifts when `channel_size=0` or `time_size=0`.
- `torch.round` is half-to-even and preserves `-0.0`; `x < 0` cannot detect signed zero, so `copysign` is required.
- PyTorch windows often use different float expressions for the left and right halves; bit-exact replication must copy that split, and only large n exposes `n//2`-style boundary differences.
- `+=` on a 0-D view mutates the input; if the same function later reuses the original, the contamination propagates across steps.
- Non-finite values must be normalized to stable strings, otherwise fixtures are not valid Python literals.
