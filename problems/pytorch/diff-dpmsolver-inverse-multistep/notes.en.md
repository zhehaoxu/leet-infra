# Pitfall Notes (order 832)

## Source-Reading Caveats

- Upstream: `huggingface/diffusers` `scheduling_dpmsolver_multistep_inverse.py` (`DPMSolverMultistepInverseScheduler`, 2025 TSAIL/HF). Pulled with `curl` to `/tmp/dpm_inverse_src.py` and pinned line by line; `/tmp/verify832.py` then `exec`s the official `_sigma_to_alpha_sigma_t` / `convert_model_output` / three update functions / `step` verbatim and compares against the handwritten `solve`: **5 configs x 8 steps = 40 steps, max absolute error 0.0**.
- The update functions are copied verbatim from the **forward** `DPMSolverMultistepScheduler` (source comment `Copied from ...`). The formulas are identical; what makes inversion distinctive is the sigma table direction: `sigmas` strictly increasing implies `lambda = log(alpha) - log(sigma_t)` decreasing, hence `h < 0`, `exp(-h) > 1`, `exp(h) < 1`. The problem therefore hard-validates a strictly increasing sigma table so it cannot degenerate into a forward-sampling problem.
- The flow branch of `_sigma_to_alpha_sigma_t` is `alpha = 1 - sigma` (not the VP `1/sqrt(1+sigma^2)`); under flow `sigma_t = sigma` (unconverted) and `0 < sigma < 1` is required, otherwise `log(1-sigma)` is undefined.
- `convert_model_output`'s `flow_prediction` branch uses the **raw, unconverted sigma**: the official code writes `sigma_t = self.sigmas[self.step_index]; x0_pred = sample - sigma_t * model_output`, shadowing the local name `sigma_t` with the raw sigma rather than the converted one. Copying the first three branches by analogy is wrong; this needed its own pin (and a VP `sigma=0.25` case to expose it, since under flow the two happen to be equal).
- The official code shadows names heavily: `sigma_t, sigma_s = sigmas[i+1], sigmas[i]` is followed by `alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)`, so `sigma_t/sigma_s` uses converted values while `alpha_t/alpha_s` uses alpha values. Splitting `_lam()` to return `(lambda, alpha, sigma_t)` avoids the confusion.
- Lower-order flags differ from the forward `diff-dpm-solver-order-schedule`: the inverse scheduler has **no** `final_sigmas_type == "zero"` term, only `euler_at_final or (lower_order_final and N < 15)`; `lower_order_second` is `(i == N-2) and lower_order_final and N < 15`. The official third-order call omits `noise` (an `sde-dpmsolver++` assertion would fail); this problem has no SDE branch, so it is unaffected.

## Dedup and Judge-Domain Trimming

- `diff-ddim-inverse-step` (498): first-order DDIM inversion with a cumulative-alpha table and optional clipping; unrelated to the DPM-Solver multistep lambda/h coefficient system.
- `diff-dpm-solver-order-schedule` (520): pure integer order scheduling; here selection is embedded in a single `step` state machine focused on coefficients, and the lower-order conditions differ (see above).
- `diff-cosine-dpmsolver-sde-step` (759): Stable Audio's CosineDPMSolver under EDM/VE (alpha==1), orders 1/2 only, with per-step SDE noise; here VP/flow, orders 1/2/3, no noise.
- `diff-dpmpp-2m` (350) / `diff-dpm-x0-conversion-threshold` (521): cover only the 2M second-order update and x0/epsilon conversion plus dynamic thresholding respectively; the third-order `D1_0/D1_1/D1/D2` and `c_d2` appear in no existing problem.
- The official `set_timesteps` sigma-table construction (linspace/leading/trailing, karras/exponential/beta/flow, scipy interpolation) is excluded entirely; the table is an explicit input.

## Test-Design Pitfalls

- Hand-computed anchors use flow sigmas `sigma = 1/(1+2^k)`, giving `lambda = log(2^k)`, `h = h_0 = h_1 = -log 2`, `r0 = r1 = 1`, so `exp(-h) = 2`, `B++ = alpha_t`, and `c_sample = sigma_t/sigma_s0` are all computable by hand.
- **But using only geometric sigma tables makes r0/r1 mutants true equivalents** (`h_0/h` and `h/h_0` both equal 1). Two non-geometric tables were added to each of the `first/second/third` oracle sweeps (flow `[0.05,0.1,0.2,0.5]` / `[0.02,0.05,0.1,0.2,0.5]`, VP `[0.1,0.4,1.0,3.0]` / `[0.1,0.4,1.0,3.0,9.0]`) so `r0 != 1` and `r0 != r1`.
- **`m-conv-dpm-sample` survived the first pass**: every `dpmsolver/sample` case happened to have `alpha_t == sigma_t` (flow `sigma=0.5` or VP `sigma=1`), so `/sigma_t` -> `/alpha_t` was a true equivalence. Adding flow `sigma=0.25` and VP `sigma=2.0` cases with distinct denominators caught it.
- `step` requires `len(sigmas) == N+1` (the official `set_timesteps` appends `sigma_max`); with `state=None` the history is all `None`, warmup 1->2->3 is driven by `lower_order_nums`, and the official counter increments even on lower-order final steps.
- The independent oracle is rewritten from the rules in pure Python `math` and compared elementwise; `close()` normalizes Tensor to list, avoiding the `T(tensor)` UserWarning.

## mut Design (29 Mutants All CAUGHT)

Coverage: three `sigma->(alpha,sigma_t)` mutants (flow linear / VP sqrt / sigma_t conversion), lambda sign, h direction, three first-order, five `convert`, seven second-order (r0 / D1 / pp midpoint / pp heun / dpm midpoint / dpm heun / denominator), five third-order (r1 / D1 / D2 / two c_d2), five `step` (both lower-order flags / order selection / history shift / counter). Anchors use multi-line context for uniqueness (the pp/dpm `c_d1` branches and the identically named third-order coefficients); `m-first-sample` needs a trailing newline, otherwise `sigma_t/sigma_s` prefix-matches three `sigma_s0` occurrences. One true-equivalence candidate was dropped; 29 mutants remain.
