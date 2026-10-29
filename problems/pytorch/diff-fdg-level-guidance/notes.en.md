# Deep Dive: Frequency-Decoupled Guidance (FDG)

## Where it comes from

*Frequency-Decoupled Guidance* (arXiv:2506.19713). The implementation follows HuggingFace diffusers' `src/diffusers/guiders/frequency_decoupled_guidance.py`:

- `forward` (L241–288): runs a CFG per band (high to low frequency) in frequency space, then inverts with `build_image_from_pyramid`;
- `project` (L42–60): splits `shift` into parallel/orthogonal parts relative to `v1 = pred_cond_freq`, normalizing `v1` over every dimension but dim 0;
- `_is_fdg_enabled_for_level` (L319–335): the per-level window and scale-degeneracy test;
- `build_image_from_pyramid` (L63–72): `img = pyrup(img) + pyramid[i]`, i.e. upsample-blur then add, level by level.

The guider depends on kornia's `pyrup` and `build_laplacian_pyramid`. This problem treats pyramid decomposition/reconstruction as an external premise and takes equal-shape bands directly, so the inverse transform is simply "sum the levels" — it keeps all of FDG's numerical structure (per-level scales, projection, rescale) without a dependency on a third-party geometry library.

## Key ideas and pitfalls

1. **The projection reference is the band's own conditional prediction**: `project(shift, pred_cond_freq)`, i.e. \(\hat c_l=c_l/\lVert c_l\rVert\). Using \(u_l\), or normalizing `shift` itself, is a different operator.
2. **The projection only runs when \(w_l\ne1\)**: the source guards it with `if not math.isclose(parallel_weight, 1.0)`. Numerically "\(w=1\) leaves shift unchanged" either way, but the guard makes the default path bit-identical to plain CFG.
3. **The rescale uses the unbiased standard deviation**: `rescale_noise_cfg` (`guider_utils.py` L390–396) calls `std(dim=list(range(1, ndim)), keepdim=True)` and `torch.std` defaults to `correction=1`. Using the population std makes the \(\varphi=1\) "restore the conditional scale" assertion off by \(\sqrt{n/(n-1)}\).
4. **The rescale happens per band, in frequency space**: with `guidance_rescale_space == "freq"` each level runs `rescale_noise_cfg(pred, pred_cond_freq, φ_l)`. Rescaling the summed result once at the end would flatten the relative magnitudes between bands.
5. **A disabled level passes through its own conditional level**: upstream, the `else` branch appends `pred_cond_freq`, but that variable still holds the previous iteration's level — a genuine stale-variable defect (disabling level \(l\) yields level \(l-1\)'s conditional prediction). This problem takes the intended semantics, "pass through `cond_levels[l]`", and the judge uses that corrected rule.
6. **Equal scales collapse to a single CFG**: when all \(s_l\) are equal with \(w_l\equiv1\) and \(\varphi_l\equiv0\), \(\sum_l[u_l+s(c_l-u_l)]=\sum_l u_l+s(\sum_l c_l-\sum_l u_l)\). That identity is the strongest structural assertion in the judge — any level misalignment shows up immediately.
7. **The epsilon for zero conditional bands is 1e-12**: `F.normalize`'s default. If a band's conditional prediction is the zero vector then \(\hat c_l=0\) and the whole `shift` counts as orthogonal (fully preserved regardless of \(w_l\)).

## How the judge cross-checks

The reference solution loops over levels in float64: `torch.linalg.vector_norm` plus a broadcast inner product for the projection, `torch.std(correction=1)` for the per-level rescale, accumulating levels before the final cast to float32. The `_oracle` reuses no tensor reductions — it reshapes each level to `(batch, width)` and recomputes the L2 norm, the inner product, the parallel/orthogonal split, the unbiased std, and the linear mix with python scalars. Six structural assertion groups: the equal-scale collapse, the orthogonality at \(w=0\) and the linear relation at \(w=0.5\), \(\varphi=1\) restoring the conditional std with \(\varphi\) as a linear mixing factor, disabled levels passing through their own conditional level (including a disabled middle level in a three-level stack), four parameter combinations compared value-by-value against the oracle, and invalid inputs. Tolerance `allclose(atol=1e-5)`.

## Complexity

With \(N\) total elements and \(L\) levels: one projection (two reductions) plus one rescale (two stds) per level, so \(O(N)\) overall, with one extra output buffer.
