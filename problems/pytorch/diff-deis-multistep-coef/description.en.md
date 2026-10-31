## Problem

DPMSolver (problem 520) uses a fixed β coefficient table for its multistep update; DEIS (Diffusion Explicit Integral Sampler, 2204.13902) takes another route: the ODE integral is done **exactly** in log-σ space — the interpolation polynomial is integrated analytically per segment, giving `ind_fn` coefficients that vary with the σ interval. This problem reproduces a single step of diffusers' `DEISMultistepScheduler`.

Inputs:

- `sample` / `model_output`: same-shape float tensors;
- `sigmas`: 1-D float tensor of length N+1 (N inference steps; the **last entry is the appended `sigma_last`, not 0**), all positive, non-increasing;
- `step_index`: current step index in `[0, N-1]`;
- `converted_history`: the already-converted outputs `d` of previous steps (newest last) — the visible part of upstream's `model_outputs` shift window;
- `solver_order` ∈ {1,2,3}, `lower_order_final`, `prediction_type`, `use_flow_sigmas`.

## Rules

**Step 1 — α/σ transform** (upstream `_sigma_to_alpha_sigma_t`):

- default: `α = 1/√(σ²+1)`, `σ_t = σ·α`;
- flow (`use_flow_sigmas=True`, σ must lie in (0,1)): `α = 1−σ`, `σ_t = σ`.

**Step 2 — model output conversion** (`convert_model_output`, at `sigmas[step_index]`):

| prediction_type | x0 |
| --- | --- |
| `epsilon` | `sample − σ_c·mo` |
| `sample` | `mo` |
| `v_prediction` | `α_c·sample − σ_c·mo` |
| `flow_prediction` | `sample − sigmas[step_index]·mo` (**raw sigma**, no transform) |

Then project to the deis form: `d = (sample − α_c·x0) / σ_c`.

**Step 3 — order ramp** (`low = len(converted_history)`):

```
lof = (step_index == N-1) and lower_order_final and N < 15
los = (step_index == N-2) and lower_order_final and N < 15
1st: solver_order==1 or low < 1 or lof
2nd: else solver_order==2 or low < 2 or los
3rd: otherwise
```

(Upstream DEIS's drop conditions are simpler than DPMSolver's — no `euler_at_final`, no `final_sigmas_type`.)

**Step 4 — update**:

- 1st (DDIM-equivalent): `h = λ_t − λ_s` with `λ = ln α − ln σ_t`;
  `x = (α_t/α_s)·sample − σ_t·(e^h − 1)·d`;
- 2nd (`m1 = history[-1]`): `ρ = σ/α`;
  `ind2(t,b,c) = t·(−ln c + ln t − 1)/(ln b − ln c)`;
  `coef1 = ind2(ρt,ρ0,ρ1) − ind2(ρ0,ρ0,ρ1)`, `coef2 = ind2(ρt,ρ1,ρ0) − ind2(ρ0,ρ1,ρ0)`;
  `x = α_t·(sample/α_0 + coef1·d + coef2·m1)`;
- 3rd (also `m2 = history[-2]`): `ind3` integrates the quadratic log-ρ polynomial exactly (see solution);
  `x = α_t·(sample/α_0 + c1·d + c2·m1 + c3·m2)`.

Returns `{"converted", "used_order", "x_next", "coefs"}`: `coefs` is `[]` for 1st order, the coefficient list (python floats) otherwise.

## Worked example

```
sample = [2, -1], mo = [1, 1], sigmas = [8, 2, 0.5], step 0 (1st order)
σ_s = 8/√65, α_s = 1/√65 → λ_s = −ln 8; σ_t = 2/√5, α_t = 1/√5 → λ_t = −ln 2
h = ln 4 → e^h − 1 = 3
x0 = [1.007722, −1.992278]; d = [1.889599, −0.758747]
x = √13·[2,−1] − (2/√5)·3·d = [2.140776, −1.569618]
```

**Ramp example** (`sigmas=[10,4,1,0.5]`, N=3, `solver_order=3`):

- step 1 (1 history entry): warmup insufficient → 2nd order, even with `lower_order_final=False`;
- step 2 (last): N=3 < 15 → 1st order; with `lower_order_final=False` and a full 2-entry history → 3rd order;
- an N=15 schedule does **not** drop at the last step (<15 fails); N=14 drops to 1st.

**Conversion pitfall**: `flow_prediction` differs from `epsilon` — the former uses the raw σ (regardless of `use_flow_sigmas`), the latter the transformed `σ_c`; under the flow transform their deis projections differ in `α_c` (`1−σ` vs `1/√(σ²+1)`).

## Boundaries

- `sigmas` must be all **positive** and non-increasing (upstream's last entry is `sigma_last > 0`; a 0 raises `ValueError`);
- `len(converted_history)` must equal `min(step_index, solver_order)` (history-window/step-count contract);
- `use_flow_sigmas=True` requires all σ ∈ (0,1) (guaranteed by the log's domain);
- 2nd order needs `step_index ≥ 1`, 3rd needs `≥ 2` (enforced via the history contract);
- float64 internally, `x_next` returned as float32; violations raise `ValueError`.

## Relation to existing problems

- 520 `diff-dpm-solver-order-schedule`: only DPMSolver's **order table**; this problem tests DEIS's **coefficient values** and update formulas (DEIS's drop conditions are simpler too);
- 350 `diff-dpmpp-2m`: DPM++(2M)'s fixed midpoint coefficient table, not exact log-ρ integration;
- 521 `diff-dpm-x0-conversion-threshold`: DPMSolver++ convert + dynamic thresholding; this problem's convert is the deis projection `(sample−α·x0)/σ` with four different prediction branches;
- 522 `diff-euler-ancestral-step`: single-step Euler with ancestral noise (stochastic) — no overlap with deterministic multistep DEIS.
