## Upstream

- Repo: `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_heun_discrete.py` (v0.42.x, downloaded 2026-10-31 via contents API; local copy at `.workbuddy/upstream/d93/`)

## Line-by-line mapping

| Upstream | This problem |
| --- | --- |
| `set_timesteps` L348-377 | sigma layout: `self.sigmas = cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])` → length 2N with trailing 0; `timesteps = cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])` → length **2N−1** (one less than sigmas — the last logical step has prediction only, so the corrector never sees σ=0) |
| `state_in_first_order` L538-540 | `state is None` branch (upstream: `self.dt is None`) |
| `step` L588-600 sigma selection | predictor: `sigma = sigmas[i]`, `sigma_next = sigmas[i+1]`; corrector: `sigma = sigmas[i-1]`, `sigma_next = sigmas[i]` (numerically the same (s_j, s_{j+1}) pair as the predictor) |
| `step` L602-616 x0 branches | `sigma_input = sigma_hat` (predictor) vs `sigma_next` (corrector); v_prediction formula kept verbatim |
| `step` L623-648 predictor | `derivative = (sample − x0)/sigma`, `dt = sigma_next − sigma`, store `prev_derivative/dt/sample`, `prev = sample + derivative·dt` |
| `step` L633-646 corrector | `derivative = (sample − x0)/sigma_next`, `avg = (prev_derivative + derivative)/2`, `prev = self.sample + avg·self.dt` (**stored** sample, not current), then free state |
| `step` L618-621 clip | `clip_sample` clamps x0 before the derivative |

## Deliberate deviations (documented)

- **No diffusers dependency**: eval env has no diffusers; `sigmas` (already in the repeated 2N layout) is an explicit input. `set_timesteps` schedule construction (linspace/leading/trailing, Karras/exponential/beta) is out of scope.
- **Explicit state passing**: upstream hides state on `self` (`prev_derivative`/`dt`/`sample` attributes); here the predictor returns the state dict and the corrector takes it — same information, testable as a pure function.
- **Structure validation added**: `sigmas[i] == sigmas[i+1]` for odd i, predictor index even / corrector odd, corrector index ≤ 2N−3, and σ > 0 on the relevant half-step. Upstream enforces none of these (pipeline convention); they encode the actual calling pattern of `pipeline.prompt...` loops over 2N−1 timesteps.
- **`gamma = 0` fixed** (L596-600): upstream supports `sigma_hat = sigma·(gamma+1)` but hardcodes gamma=0; not parameterized.
- float64 internal math; outputs returned as float32.

## Test design

- Independent numpy oracle `_oracle_half` rewritten from the step equations; full 5-call trajectory for N=3 checked element-wise (1e-9), including the final prediction-only step (dt = 0 − s_{N-1}).
- Hand example with hardcoded anchors: predictor x0=[−6,−1], euler=[−2,−1]; corrector avg=[0.5,0.5], prev=[0,−3] (from the **stored** sample [2,−1]).
- Counter-example documented in description: starting from the current sample gives [−4,−3] — off by exactly the Euler half-step.
- v_prediction branch + clip_sample-before-derivative (mo=0, x0=sample clamped to ±0.5 → derivative = ±(10−0.5)/4).
- 9 validation cases (shape, sigma layout parity, index parity, corrector-out-of-range, state keys).
- Mutation testing 5/5 caught: corrector from current sample, corrector divide σ_next+1, predictor divide σ_next, no averaging, dt recomputed instead of reused.
