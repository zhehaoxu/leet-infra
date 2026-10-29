# In Depth: CFG-Zero* Zero-Init and Optimal Scale

## Origin

*CFG-Zero\*: Improved Classifier-Free Guidance for Flow Matching Models* (arXiv:2503.18886). The implementation follows HuggingFace diffusers' `src/diffusers/guiders/classifier_free_zero_star_guidance.py`: `forward` (L100–124) lays out the "zero-init window → scale correction → extrapolation" branches and `cfg_zero_star_scale` (L156–164) defines \(\alpha=\langle c,u\rangle/(\lVert u\rVert^2+\varepsilon)\) with `eps=1e-8`. The paper suggests zero initialization over the first 4% of inference steps; diffusers defaults to `zero_init_steps=1`.

## Core points and pitfalls

1. **\(\alpha\) is a least-squares scale, not a cosine similarity**: the denominator is \(\lVert\epsilon_u\rVert^2\), not \(\lVert\epsilon_c\rVert\lVert\epsilon_u\rVert\). The cosine form drops the magnitude ratio between the two branches, and the residual-orthogonality assertion fails immediately.
2. **The window test is strictly less-than**: zeros only while `step < k`; step \(k\) already guides. Using `<=` skips one guided step.
3. **Inside the window the output is an all-zero tensor**, not \(\epsilon_c\) and not the previous prediction — it makes the sampler take no update at that step.
4. **\(\varepsilon\) sits in the squared denominator**, not multiplying \(\alpha\) and not inside a square root; it can only shrink \(\lvert\alpha\rvert\).
5. **\(\alpha\) may be negative**: for opposite-pointing branches \(\langle c,u\rangle<0\), and clamping it changes the result completely.
6. **The extrapolation base is \(\alpha\epsilon_u\)**: `out = ᾱε_u + s(ε_c − ᾱε_u)`, hence \(s=1\) returns exactly \(\epsilon_c\) — a handy self-check.
7. **One scalar per sample**: the reduction must cover every dim but the batch; a single batch-wide inner product lets samples contaminate each other.

## Verification approach

The reference solution reduces with `torch.sum` over all dims but 0 in float64, then casts to float32. The judge's `_oracle` flattens each sample into python lists and computes the inner product and squared norm by hand (no tensor reductions). Five structural groups: the window boundary (first \(k\) steps zero, step \(k\) guided), the \(s=1\) identity, recovering \(\alpha\epsilon_u\) from the output to check residual orthogonality, the orthogonal/parallel degeneracies, and \(\lvert\alpha\rvert\) shrinking monotonically as \(\varepsilon\) grows. Tolerance `allclose(atol=1e-5)`.

## Complexity

Element-wise \(O(N)\) plus one per-sample inner-product and squared-norm reduction (also \(O(N)\)).
