# In Depth: AdamP Scale-Invariant Direction Projection

## Source
AdamP paper: *AdamCP / AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights*, arXiv:2009.00329 (naver, ICLR 2021). The algorithm skeleton follows `AdamP` from kozistr/pytorch_optimizer (the per-param path with `nesterov=False, cautious=False, foreach=False`, `weight_decouple=True`).

## Mechanism in one sentence
For scale-invariant parameters (e.g. normalization weights), neither the update nor the decay along the parameter's own direction changes the function value, yet they slow down effective learning and invite instability. When AdamP detects that "the gradient is nearly orthogonal to the parameter", it **projects out the radial component** of the update vector — the part parallel to the parameter — keeping only the tangential component, and multiplies that group's weight decay by `wd_ratio` (shrinking it).

## Key ordering and traps (pitfalls)
1. **perturb is computed before the projection**: first form \(\text{perturb}=\frac{\sqrt{\hat\beta_2}}{\sqrt v+\epsilon}\odot m\) from the Adam moments; the projection subtracts the radial part from that perturb in place. The denominator \(\sqrt v+\epsilon\) uses the **bias-uncorrected** \(v\), while \(\sqrt{\hat\beta_2}\) multiplies the numerator (equivalent to dividing the denominator by \(\sqrt{\hat\beta_2}\)).
2. **Two views, break on the first hit**: channel view first (\((C,\cdot)\), per channel), then layer view (\((1,\text{numel})\)). Whichever one first satisfies `|cos|.max() < delta/sqrt(num_cols)` is used for the removal and then `return` (`r=wd_ratio`); the other view is not checked. The column counts are `numel/C` and `numel` — so the thresholds differ, and the channel view's `delta/sqrt(num_cols)` is usually larger and easier to hit.
3. **Normalization uses row norms**: \(\hat n = p/(\|p\|_{row}+\epsilon)\); the radial part is \(\hat n\odot\text{perturb}\) summed per row and reshaped to broadcast back, then `perturb -= \hat n*radial`. Note that the `+epsilon` in \(\hat n\) is added to the norm.
4. **Only dim>1 is projected**: one-dimensional parameters (biases, norm scale factors) skip the projection, \(r=1.0\). The weight decay `p *= 1 − lr*wd*r` is applied **after** the projection and before the parameter update; parameters with `grad is None` are skipped entirely.
5. The cosine similarity uses \(\epsilon\) as its denominator floor (`F.cosine_similarity(..., eps=eps)`), and the absolute value is taken afterwards.

## Cross-check convention
The external reference is an independent vectorized recomputation (the cosine/projection uses the very same `_projection`/`_cos_by_view` helpers as the reference solution — because the projection is a data-dependent discrete branch, reference and solution must share one test or floating-point borderline cases could flip it); the Adam-moment part is recomputed with explicit algebra. All float32; measured maxdiff 0 on the example values, and over random multi-step runs (20 steps × 50 seeds × mixed 1D/2D/high-dimensional shapes) the maximum deviation is ≈2.4e-7 with no branch flips.

## Complexity
\(O(\text{numel})\) per parameter per step plus the per-channel cosines and normalization, with two extra states \((m, v)\).
