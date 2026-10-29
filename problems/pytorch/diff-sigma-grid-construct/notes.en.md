# Deep Dive: EulerDiscrete Timestep Grids and Sigma Converters

## Where it comes from

`EulerDiscreteScheduler.set_timesteps` in HuggingFace diffusers' `src/diffusers/schedulers/scheduling_euler_discrete.py`. It is the shared grid-construction entry point for the Euler / Euler-Ancestral / Heun / DPM2 family: `timestep_spacing` decides how \(T\) collapses to \(N\), `interpolation_type` decides how the sigmas are read off, and the Karras / exponential converters reshape the sigma distribution before the timesteps are recovered. The FlowMatch variant in `scheduling_flow_match_euler_discrete.py` reuses the same logic.

## Key ideas and pitfalls

1. **The sigma baseline runs the other way**: `sigmas_full = sqrt((1 - alphas_cumprod) / alphas_cumprod)` increases with the index, so index 0 is the clean end and \(T-1\) the noisy end — while `timesteps` descends from \(T-1\) to 0. Mixing two opposite index conventions is the main mental load here: interpolation uses `timesteps` as indices, whereas Karras takes its endpoints from "entry 0 of the sigma sequence is the noisy end".
2. **Two different divisions**: `leading` uses `T // N` (integer division: a smaller grid stride that lands on 0), `trailing` uses `T / N` (true division: the grid hugs the noisy end). Picking the wrong branch changes the first entry away from \(T-1\) and ruins the whole problem.
3. **`leading`'s grid is zero-based**: `(arange(N) * step_ratio).round()` is \([0, r, 2r, \dots]\), only reversed afterwards. Hence \(T=10,N=3\) starts at 6, not 9.
4. **`steps_offset` applies to `leading` only**: `linspace` and `trailing` ignore it entirely even though the argument may be passed. The tests explicitly cover "still unchanged when an offset is supplied".
5. **`trailing`'s `arange` length can be \(N+1\)**: `ceil(T / (T/N))` evaluates to \(N+1\) whenever the floating-point division is inexact, so the final point sits near 0 and `round() - 1` yields \(-1\). Upstream writes `np.arange(num_train_timesteps, 0, -step_ratio)` and behaves identically. Never assume "`timesteps` has length \(N\)", or inputs such as \(T=60,N=13\) fail immediately.
6. **`log_linear` ignores the timestep grid completely**: `exp(linspace(log σ[-1], log σ[0], N+1))` uses only the two endpoints and the point count, producing \(N+1\) sigmas (not \(N\)). It therefore has two more entries than `timesteps` (the extra point plus the terminal one). This is a genuine upstream quirk; changing `timestep_spacing` does not move it.
7. **Karras with \(\rho=7\) leans towards the noisy end**: for \(\rho>1\) the middle values sit above the geometric mean and below the linear interpolation \((hi+lo)/2\); as \(\rho\to\infty\) the whole sequence approaches hi. Writing \(\rho=1\), or swapping hi/lo, distorts the sequence while leaving the endpoints correct — checking only the first and last entry will not reveal it.
8. **The inversion happens in log-sigma space**: `_sigma_to_t` picks the largest index with \(\ell_{\text{low}} \le u\) (the equivalent of `cumsum(log_sigmas <= log_sigma).argmax()`) and then interpolates **logarithmically**, not linearly. Sigmas span several orders of magnitude, so using the wrong space visibly shifts the middle timesteps.
9. **A `sigma_min` terminal duplicates the second-to-last sigma**: the terminal value is `sqrt((1 - abar[0]) / abar[0])`, which equals the last linear-interpolation entry whenever `timesteps` ends at 0. That is expected upstream behaviour, not a bug.
10. **`use_beta_sigmas` needs scipy**: `_convert_to_beta` calls `scipy.stats.beta.ppf`, but the judging environment only ships torch / numpy / transformers. The problem therefore defines that switch as "true raises `ValueError`" and says so in the statement rather than faking a beta quantile.

## How the judge cross-checks

The judge re-implements the whole pipeline in numpy (`_grid_oracle` / `_sigma_to_t_oracle` / `_oracle`) and compares element-wise against the reference:

- **Example values**: all six configurations at \(T=10,N=3\) (three spacings + Karras + exponential + log_linear) have both `timesteps` and `sigmas` hard-coded to six decimals;
- **Spacing variants**: every branch is cross-checked for \(T=10\) and \(T=9\) with \(N\in\{1,3,4,7\}\), plus explicit assertions that `leading` uses integer division, `trailing` uses true division, and `np.round(4.5) == 4` (banker's rounding);
- **Offset isolation**: `steps_offset=10` shifts only `leading`; the other two branches are bitwise unchanged;
- **Interpolation hits**: on a 60-point training table with \(N\in\{2,5,13\}\), integer timesteps must hit the training-grid sigmas exactly (the extra `-1` point from `trailing` is excluded from that assertion);
- **Terminal sigma**: `zero` is exactly 0, and `sigma_min` equals the cleanest training sigma and coincides with the second-to-last entry;
- **Karras**: element-wise against the explicit \(\rho=7\) formula, plus three shape assertions ("endpoints are σ_max / σ_min", "monotonically non-increasing", "middle above the geometric mean and below the linear interpolation") and the `sigma_min`/`sigma_max` endpoint overrides;
- **Exponential**: constant log-sigma spacing and correct endpoints;
- **log_linear**: \(N+1\) points (\(N+2\) after the terminal one), spacing does not change the sigmas, and attaching a converter restores \(N+1\);
- **Invalid inputs**: 20 groups (non-tensor, integer / 2-D / too short / out-of-range / NaN alphas, \(N\) out of range or non-integer, bad enum values, conflicting converters, beta, negative offset, `sigma_min >= sigma_max`) must all raise `ValueError`.

Nine assertion groups in total.

## Complexity

\(O(T)\) to build the baseline, \(O(N)\) for interpolation and the converters, and \(O(N\log T)\) for the inversion; time and space are both linear (the inversion is vectorised into an \(O(NT)\) boolean comparison, negligible at this scale).
