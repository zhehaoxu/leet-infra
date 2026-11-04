## Problem

`CosineDPMSolverMultistepScheduler` is the scheduler used by Stable Audio Open (arXiv 2407.14358) in diffusers: it ports the DPMSolver multistep family to the **EDM/VE convention** — the sigma table comes from an exponential / karras schedule with an appended `sigma_last` (0 or `sigma_min`), the network output passes through **EDM preconditioning** (`c_skip`/`c_out` scaling) before entering the solver, and both updates carry an **SDE noise term** (sampled officially by `BrownianTreeNoiseSampler` each step). Each `step()` call runs the fixed sequence "parse state → convert → history shift → order decision → first- or second-order update → state advance"; the `model_outputs` history and the `lower_order_nums` counter chain N calls into one sampling trajectory. Your task is to implement **one call** of this state machine, with state passed explicitly via `state`.

Judge inputs:

- `model_output` / `sample`: floating tensors of the same shape;
- `sigmas`: 1-D floating tensor, the table `[σ0, …, σ_{N−1}, σ_last]` produced by the official `set_timesteps` (length N+1, N ≥ 2), finite, non-increasing, **positive everywhere except possibly the last entry** — the tail is 0 (`final_sigmas_type="zero"`, the default) or `sigma_min`;
- `step_index`: current index, `0 ≤ step_index ≤ N−1`;
- `state`: `None` = fresh start (the official `begin_index` scenario, legal at any `step_index`); otherwise the `state` dict returned by the previous call;
- `noise`: **required on every call** (officially sampled by the Brownian-tree sampler, which depends on generator/seed and lives on the pipeline side; this problem takes it as an explicit argument), same shape as `model_output`;
- `prediction_type` ∈ {`epsilon`, `v_prediction`} (official config default `v_prediction`; note this scheduler does **not** support `sample`);
- `solver_order` ∈ {1, 2}, `solver_type` ∈ {`midpoint`, `heun`}, `lower_order_final`, `euler_at_final`, `final_sigmas_type` ∈ {`zero`, `sigma_min`}, `sigma_data` > 0: EDM data variance, default 1.0.

## Computation rules

**Sigma convention**: α ≡ 1 (inputs are pre-scaled by `c_in = 1/√(σ²+σd²)` before the unet), so λ = log α − log σ_t ≡ **−log σ**; when σ = 0 (the tail) λ = +∞ (the semantics of the official `torch.log(0) = −∞`, with `exp(−∞) = 0` downstream). h = λ_t − λ_s0 ≥ 0; the identity **e^{−h} = σ_t/σ_s0** yields two closed forms: the first-order sample coefficient = (σ_t/σ_s0)², the decay factor = 1 − (σ_t/σ_s0)².

**convert** (EDM preconditioning, verbatim `precondition_outputs`): σ = `sigmas[step_index]`, d = `sigma_data`:

- c_skip = d²/(σ²+d²);
- `epsilon`: c_out = σ·d/√(σ²+d²); `v_prediction`: c_out = **−**σ·d/√(σ²+d²); anything else raises `ValueError` (official message `Prediction type ... is not supported.`);
- x0 = c_skip·sample + c_out·mo.

**History shift**: `model_outputs = old[1:] + [x0]`, length = `solver_order`, unchanged.

**Order decision**: N = len(sigmas) − 1. lower_order_flag = (`step_index` == N−1) and (`euler_at_final` **or** (`lower_order_final` and N < 15) **or** `final_sigmas_type == "zero"`). The first-order condition = `solver_order == 1` or `lower_order_nums < 1` or lower_order_flag; otherwise second order (the official `elif`'s `solver_order == 2` is always true under the {1,2} validation; `lower_order_second` is dead code).

**First-order update** (`dpm_solver_first_order_update`, α_t = 1 folded in): σ_t = `sigmas[step_index+1]`, σ_s0 = `sigmas[step_index]`, h = λ_t − λ_s0:

- prev = (σ_t/σ_s0)·e^{−h}·sample + (1 − e^{−2h})·x0 + σ_t·√(1 − e^{−2h})·noise;
- at the final step with a zero tail, σ_t = 0 and h = +∞: the three coefficients are 0, 1, 0 → **prev is exactly x0** (official torch semantics).

**Second-order update** (`multistep_dpm_solver_second_order_update`): also take σ_s1 = `sigmas[step_index−1]`, h_0 = λ_s0 − λ_s1, **r0 = h_0/h**:

- D0 = x0 (tail of the shifted history), D1 = (1/r0)·(x0 − model_outputs[−2]);
- `midpoint`: d1c = 0.5·(1 − e^{−2h}); `heun`: d1c = (1 − e^{−2h})/(−2h) + 1;
- prev = (σ_t/σ_s0)·e^{−h}·sample + (1 − e^{−2h})·D0 + d1c·D1 + σ_t·√(1 − e^{−2h})·noise;
- h = 0 or h_0 = 0 (adjacent sigmas equal) leaves the official code dividing 0/0 (undefined); this problem rejects it (`ValueError`).

**Returned dict**: `prev_sample` / `pred_original_sample` (float32), `used_first_order` (bool), `this_order` (1 or 2), `r0` / `d1_coefficient` (floats, `None` in first order), `state`. The new `state` = {`model_outputs`, `lower_order_nums` (+1 while below `solver_order`)}. Validation failures always raise `ValueError`.

## Example

```
sigmas = [3.0, 1.0, 0.0] (N=2, zero ending), solver_order=2, midpoint,
prediction_type=v_prediction, sigma_data=1.0, lower_order_final=True

step 0 (state=None): mo = 0.5, sample = 1.0, noise = 0.2
  c_skip = 1/(3^2+1) = 1/10, c_out = -3/sqrt(10)
  x0 = (1/10)*1.0 - (3/sqrt(10))*0.5 = -0.374342
  h = lam(1) - lam(3) = log(3), e^{-h} = 1/3
  c_sample = (1/3)*(1/3) = 1/9, decay = 1 - 1/9 = 8/9
  noise term = 1*sqrt(8/9)*0.2 = 0.188562
  prev = (1/9)*1.0 + (8/9)*(-0.374342) + 0.188562 = -0.033075
  state = {model_outputs: [None, -0.374342], lower_order_nums: 1}

step 1 (state from step 0, last step): mo = 0.3, sample = -0.033075, noise = 0.1
  c_skip = 1/2, c_out = -1/sqrt(2)
  x0 = (1/2)*(-0.033075) - (1/sqrt(2))*0.3 = -0.228670
  lower_order_flag: step_index == N-1 and final_sigmas_type == "zero" -> True
  sigma_t = 0 -> lam_t = +inf, h = inf: e^{-h} = 0, decay = 1
  c_sample = 0, noise term = 0
  prev = 0*sample + 1*x0 + 0*noise = x0 = -0.228670
  state = {model_outputs: [-0.374342, -0.228670], lower_order_nums: 2}
```
