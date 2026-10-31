## Problem

Euler stepping (the deterministic core of problem 522) is first-order; Heun's method upgrades the same model-call budget to second order via "predict-correct": take a half Euler step to probe, re-evaluate the derivative at the new point, average the two derivatives and walk the **full** step from the start. diffusers' `HeunDiscreteScheduler` implements this as a two-phase state machine — each logical noise level occupies **two** step indices, with intermediate state (derivative, step size, sample) stashed on the scheduler.

Inputs:

- `model_output` / `sample`: same-shape float tensors (the corrector's `sample` is the **Euler half-step result**; its `model_output` is the model at the new point);
- `sigmas`: 1-D float tensor in upstream `set_timesteps`'s repeated layout: `[s0, s1, s1, s2, s2, …, s_{N-1}, s_{N-1}, 0]`, length 2N;
- `step_index`: current index;
- `state`: `None` = predictor; `{"prev_derivative", "dt", "sample"}` = corrector (pass through the predictor's return);
- `prediction_type` ∈ {`epsilon`, `v_prediction`, `sample`}, `clip_sample`, `clip_sample_range`.

## Rules

**Index layout**: logical step j (0 ≤ j ≤ N−2) occupies indices 2j (predictor) and 2j+1 (corrector); the **last logical step has a predictor only** (index 2N−2, σ_next = 0). timesteps has length 2N−1, sigmas 2N — precisely because sigmas has one extra entry, the corrector never sees σ=0.

**Predictor** (`state=None`, even index):

- `σ = sigmas[i]` (> 0), `σ_next = sigmas[i+1]`;
- `x0 = f(sample, mo, σ)`, `derivative = (sample − x0) / σ`;
- `dt = σ_next − σ`, `prev = sample + derivative·dt`;
- stash `state = {"prev_derivative": derivative, "dt": dt, "sample": sample}`.

**Corrector** (`state` dict, odd index ≤ 2N−3):

- requires `sigmas[i] == sigmas[i+1]` (repeated layout); `σ_next = sigmas[i]` (> 0);
- `x0 = f(sample, mo, σ_next)`, `derivative = (sample − x0) / σ_next`;
- `avg = (state.prev_derivative + derivative) / 2`;
- `prev = state.sample + avg·state.dt` — **from the stored sample with the stored step size**, not from the current sample;
- returns `state = None` (cleared).

**x0 formulas** (σ_input: predictor uses σ, corrector uses σ_next):

| prediction_type | x0 |
| --- | --- |
| `epsilon` | `sample − σ_input·mo` |
| `v_prediction` | `mo·(−σ_input/√(σ_input²+1)) + sample/(σ_input²+1)` |
| `sample` | `mo` |

`clip_sample=True` clamps x0 to ±`clip_sample_range` (before the derivative).

Returns `{"prev_sample", "pred_original_sample", "state", "derivative"}`.

## Worked example

```
sigmas = [8, 4, 4, 1, 1, 0] (N=3), epsilon
Predictor step0 (σ=8, next=4): sample=[2,−1], mo=[1,0]
  x0 = [−6, −1]; d = [1, 0]; dt = −4; euler = [−2, −1]
Corrector step1 (σ_next=4): sample'=[−2,−1], mo'=[0,1]
  x0' = [−2, −5]; d2 = [0, 1]; avg = [0.5, 0.5]
  prev = [2,−1] + (−4)·[0.5,0.5] = [0, −3]   ← from the stored sample=[2,−1]
```

**Counter-example (current sample)**: the same corrector starting from `sample'=[−2,−1]` gives `[−4,−3]` — off by exactly the Euler half-step.

**Counter-example (derivative divisor)**: the predictor divides by the **current σ**, the corrector by **σ_next**; writing one σ in both places divides by zero at the pair boundary where σ_next ≠ σ (e.g. the last logical step's predictor has σ_next = 0).

## Boundaries

- `sigmas` length ≥ 3, finite, non-negative, non-increasing; the repeated layout is checked via `sigmas[i] == sigmas[i+1]` (i odd);
- predictor index must be even, corrector odd, and corrector index ≤ 2N−3 (the last logical step has no corrector);
- predictor σ > 0, corrector σ_next > 0;
- `state` must have exactly the three keys with tensors shaped like sample; `dt` a number;
- float64 internally, `prev_sample`/`pred_original_sample`/`derivative` returned as float32; violations raise `ValueError`.

## Relation to existing problems

- 522 `diff-euler-ancestral-step`: also Euler in σ space but with ancestral-noise splitting and stochastic injection (single step, stateless); this problem is a deterministic two-phase state machine, noise-free;
- 519 `diff-edm-preconditioning`: EDM's c_skip/c_out/c_in network preconditioning, not a multi-call state machine;
- 576 `diff-unipc-pc-step`: UniPC's predict-corrector shares the word "corrector" but its coefficients come from multivariate polynomial fitting, with no repeated-σ expansion;
- 494–496 (SD3 training side) and the 501–513 guidance family: unrelated mechanics.
