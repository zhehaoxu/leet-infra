# notes — int-activation-steering (order 660)

## Source

- Official implementation: `nrimsky/steering-vectors` (the `SteeringVector` class, whose `apply` adds `vector * coefficient` to the residual-stream output of chosen layers; supports `normalize=True` to normalize first). Upstream operates inside a real model forward; per the judging red line this problem takes all hidden states as explicit inputs and covers only the injection and direction ledger.
- Paper context: representation engineering / activation steering (same family as ITI, ActAdd); boundary with 653 `int-tuned-lens-affine` (probe readout): this problem is the **write side**.

## Judging surface

- solve args: hiddens `[L,T,D]`, s `[D]`, alphas (K reals), layer_indices (K ints, may repeat), normalize bool. Returns steered/s_norm/layer_delta/cos_before/cos_after/proj_coef.
- Oracle: `test_cases._py`, independent pure Python (math.sqrt), comparing every field at 1e-9.
- Anchors: (1) h=[3,0], s=[0,4], α=0.5 orthogonal injection (cos_before=0, cos_after=2/√13, layer_delta=2); (2) normalize on/off comparison (injection length differs 4× when |s|=4); (3) two same-layer injections α=1,2 ≡ single α=3 (accumulate semantics); (4) cross-layer injection touches only target rows and cos_before is entirely from the originals; (5) α=−proj_coef zeroes cos_after (orthogonalization self-consistency); (6) float64 input is not mutated in place (`.double()` is the identity map on float64, making in-place corruption observable).
- Invalid inputs: dims/shapes, zero s, K=0, length mismatch, out-of-range/float layer index, alpha nan/inf/bool → ValueError.

## Mutation log (9, all caught)

no-inject, idx-rotate, overwrite (same-layer overwrite instead of accumulate), normalize-drop, normalize-sq (divide by |s|²), cos-no-norm, cos-after-num (numerator from the original h), proj-denom (|s| instead of |s|²), in-place (`clone()` → reference — equivalent under float32 inputs since `.double()` always copies; caught by the float64-input case).

## Edge cases and pitfalls

- `.double()` only guarantees a copy when the input is float32; the "must not mutate the input" contract needs to be pinned by a float64 case.
- cos/proj all in float64 semantics; layer_delta is the norm of the accumulated injection vector — with same-direction injections it equals |Σα|·|s_eff|, summing signed α first then taking the absolute value.
- Red lines: no real-model forward, no generation-behavior comparison, no ITI attention-head-side mechanics.
