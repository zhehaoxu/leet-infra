# Deep Dive: EulerAncestral σ_up/σ_down Split and Ancestral Noise Injection

## Source

HuggingFace diffusers, `src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py`: `scale_model_input` (`sample / ((sigma**2 + 1) ** 0.5)`), `step` (\(x_0\) prediction, the `sigma_up` / `sigma_down` split, the Euler update, and noise injection via `randn_tensor`), and `set_timesteps` (the σ grid with a zero terminal). The algorithm is the "ancestral sampling" section of Karras et al. 2206.00364: re-inject noise of size \(\sigma_{\text{up}}\) each step, turning ODE sampling into stochastic SDE sampling.

## Key Points and Pitfalls

1. **The split identity**: \(\sigma_{\text{up}}^2+\sigma_{\text{down}}^2=\sigma'^2\) and \(\sigma_{\text{up}}^2=\sigma'^2\left(1-\sigma'^2/\sigma^2\right)\). How much of the old noise level is re-injected depends only on the ratio of the two σ values; \(\sigma'=\sigma\) gives no random part, and \(\sigma'=0\) zeroes both. Writing `sigma_down` as \(\sigma'\) instead of \(\sqrt{\sigma'^2-\sigma_{\text{up}}^2}\) overshoots every step and also breaks the exact landing on \(x_0\) at the end.
2. **`dt` uses `sigma_down`, not `sigma_to`**: `dt = sigma_down - sigma`. The deterministic part only walks down to \(\sigma_{\text{down}}\); the rest of the drop is carried by the noise term.
3. **The final step is exact**: \(\sigma'=0\Rightarrow\sigma_{\text{up}}=\sigma_{\text{down}}=0\Rightarrow dt=-\sigma\), so `prev = x + (x-x0)/σ·(-σ) = x0`. This is the best self-check for the problem, and it explains why the last step needs no `noise`.
4. **The `v_prediction` branch copies upstream verbatim**: upstream writes `model_output * (-σ/√(σ²+1)) + sample/(σ²+1)`, i.e. the sample coefficient is \(\alpha^2=1/(\sigma^2+1)\) while the noise coefficient is \(\sigma_t=\sigma/\sqrt{\sigma^2+1}\). The textbook formula is \(x_0=\alpha x-\sigma_t v\) (the version used by `LCMScheduler` and `DPMSolverMultistepScheduler`). It can be read as "the correct formula applied to the **scaled** sample": \(\alpha\cdot\text{scaled}-\sigma_t v\), yet the `epsilon` branch in the same `step` uses the **unscaled** `sample`, so the two branches disagree on convention. This problem requires copying the source, and the tests explicitly verify that it differs from the textbook formula and that the gap shrinks as \(\alpha(1-\alpha)\) when \(\sigma\to0\).
5. **`scale_model_input` affects only the model input**: `scaled_input` is what the UNet sees; the \(x_0\) computation and the Euler update inside `step` use the raw `sample`. Scaling both shifts results systematically.
6. **`noise` is an explicit argument**: upstream draws it with `randn_tensor`; this problem takes it from the caller so the comparison is deterministic. When \(\sigma_{\text{up}}=0\) it takes no part — an unconditional `+ noise*0` with NaN/Inf noise would poison the output.
7. **σ must be non-increasing**: \(\sigma'>\sigma\) makes `sigma_from**2 - sigma_to**2` negative and the square root NaN. Upstream does not validate; this problem raises `ValueError`.
8. **Numerics**: upstream upcasts `sample` to float32 and casts back to the model dtype; this problem computes in float64 and returns float32, so the example numbers match a float64 hand calculation.

## Reference Comparison

Judging uses a pure-Python + numpy `_oracle` that redoes scaling, \(x_0\), the split, and the update elementwise, feeding it inputs already rounded to float32 (the same rounding the reference solution sees):

- **Example values**: `scaled` / `x0` / `prev` / `sigma_up` / `sigma_down` for three steps are hard-coded to six decimals, and the last step asserts `prev` is bitwise equal to `x0`;
- **Split identities**: on a 5-level σ grid, check \(\sigma_{\text{up}}^2+\sigma_{\text{down}}^2=\sigma'^2\), \(\sigma_{\text{up}}^2=\sigma'^2(1-\sigma'^2/\sigma^2)\), \(0\le\sigma_{\text{up}}\le\sigma'\), and the two limits \(\sigma'\to\sigma\) and \(\sigma'\to0\);
- **Final and stationary steps**: at \(\sigma'=0\), `prev == x0` regardless of `noise` (even 1e6); at \(\sigma'=\sigma\), `dt=0` and `prev == sample`;
- **Noise scale**: `prev` minus the zero-noise version equals exactly `noise·sigma_up`, independent of `sigma_down`;
- **The `v_prediction` branch**: matches the upstream closed form \(\alpha\cdot\text{scaled}-\sigma_t v\), visibly differs from the textbook \(\alpha x-\sigma_t v\), and the gap shrinks monotonically across four decreasing σ values;
- **`epsilon` round trip**: recompute \(\epsilon\) from a given \(x_0\), take a step, and recover both \(x_0\) and the analytic `prev`;
- **Combined comparison**: 2 shapes × 4 σ grids × every step × 2 prediction types, all matching the oracle;
- **Invalid inputs**: 18 cases (non-tensor / integer / mismatched shapes / sigmas as a list / 2-D / length 1 / increasing / negative / Inf, out-of-range or non-integer `step_index`, unknown or `sample` prediction type, missing `noise` when \(\sigma_{\text{up}}>0\), wrong `noise` shape, zero current σ) must all raise `ValueError`.

Eight assertion groups in total.

## Complexity

A constant number of multiply-adds per sample element: \(O(B\cdot C\cdot H\cdot W)\) time and space (one extra copy each for the scaled input, \(x_0\), and the next sample).
