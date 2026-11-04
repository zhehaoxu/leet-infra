# Solution notes — Polytropon

## Official account

- HF PEFT `peft/tuners/poly/layer.py` (PolyLinear) + `router.py` (PolyRouter);
  papers Polytropon + MHR.
- Judging domain: init RNG domain, inference routing, einsum mixing, the 1/r-scaled
  forward, parameter budget. The RelaxedBernoulli training branch is out of scope.

## Pitfalls

- **Kaiming float formula**: `math.sqrt(5)**2 = 5.000000000000001`, so the bound
  differs from exact `1/√fan_in` by ~1e-16 and the uniform draws differ at the bit
  level — replays must reuse the identical formula (`1.0/math.sqrt(d)` fails).
- **Loop order is the RNG domain**: skill-outer, split-inner; router logits draw
  after all A blocks; with `init_weights=False` the B blocks continue the same stream.
- **B init argument order**: official kaiming on `empty((d_out, r))` (fan_in = r)
  transposed to (r, d_out). Swapping to `empty((r, d_out))` errors on shape (caught
  by RuntimeError) but semantically it is a wrong fan_in distribution.
- **Routing is not softmax**: sigmoid then skill-wise normalization; EPS=1e-12
  prevents NaN when sigmoids underflow (logits ≈ -800) — invisible in normal numeric
  ranges but judgeable with a tiny-logits case.
- **Combine reassembly**: A reshapes to split-major input columns; B needs
  transpose(1, 2) before reshape (a missing-transpose mutant is caught by shape/values).
- **The scale is exactly 1/r**: not α/r or 1/√r (LoRA habit leads astray).
- Placeholder-anchor lesson: when filling anchors from the FAIL message, the reported
  value is the true value — but only after a manual cross-check confirms the solution
  itself is correct.

## Mutant list (17)

init: skill/split loop swap, B zeros→ones, A/B kaiming argument order (fan_in swap),
router range ×10; route: EPS removal, softmax replacement, view(ns,sk) swap,
boundary >=→>; combine: A mixing coefficient, B missing transpose, A reshape
transpose; delta: 1/r→1/√r, scale removal, swapped bmm (shape error); param_count:
router formula, lora +1. 17/17 CAUGHT.
