## Problem

PLMS (pseudo linear multistep) is the **skip_prk_steps=True full-run path** of the PNDM scheduler (Liu et al. 2022, arXiv 2202.09778) — Stable Diffusion's default usage: no 4-substep Runge-Kutta warmup, the Adams-Bashforth ladder over the "ets history" starts from the very first call. Its mechanics condense into three things: (a) **ets bookkeeping** (truncate to 3 entries then append the current output; only counter 1 skips the append); (b) the **AB ladder** (ets length 1→2→3→4 selects identity / trapezoid / 2nd / 3rd / 4th order coefficients); (c) **the counter-1 restart pair** (re-walk one full step over (t+δ, t) starting from the sample stashed at counter 0). Landing uses formula (9) of the PNDM paper. This problem asks you to implement one call, with the state passed explicitly via `state`.

Judge inputs:

- `model_output` / `sample`: floating-point tensors of the same shape;
- `alphas_cumprod`: 1-D float tensor, the upstream ᾱ table (length ≥ 2, all in (0, 1], non-increasing; **ᾱ(0) = 1−β₀ ≈ 0.999, not 1** — precisely what keeps the formula(9) denominator non-zero);
- `timestep`: the current training-timestep index, `0 ≤ timestep ≤ len−1`;
- `step_ratio = num_train_timesteps // num_inference_steps ≥ 1` (the stride δ);
- `counter`: the global call count (the official `self.counter`, explicit input);
- `state`: `None` ⇔ `counter = 0`; otherwise {`ets`, `cur_sample`};
- `prediction_type` ∈ {`epsilon`, `v_prediction`}.

## Rules

**State contract** (the natural shape of the official append/pop bookkeeping): the pre-call ets length is exactly [c1→1, c2→1, c3→2, c4→3, c≥5→4] (the `[-3:]` truncation happens inside the call, so from c≥5 on the caller holds 4 entries); `cur_sample` must be non-None only at counter = 1 (and must be None at counter ≥ 2).

**1. ets bookkeeping**: when counter ≠ 1, `ets = ets[-3:]` truncates and then appends the current output (3 old + 1 new = capped at 4); at counter = 1 there is **no append** (the ets tail stays at the previous call's output).

**2. The AB ladder** (branch selected by the post-append ets length):

- len 1 and counter 0: combined = mo (identity), and stash `cur_sample = sample`;
- len 1 and counter 1: combined = (mo + ets[-1]) / 2 (two-point trapezoid), and **sample is replaced by `state["cur_sample"]`** (restart from the sample stashed at counter 0);
- len 2: (3·ets[-1] − ets[-2]) / 2;
- len 3: (23·ets[-1] − 16·ets[-2] + 5·ets[-3]) / 12;
- len ≥ 4: (55·ets[-1] − 59·ets[-2] + 37·ets[-3] − 9·ets[-4]) / 24.

**3. timestep bookkeeping**: counter 1's (timestep, prev_timestep) = (t+δ, t) — the same adjacent pair as counter 0's combined output; all others are (t, t−δ). At counter 1, t+δ must not run past the ᾱ table tail.

**4. Landing with formula (9)**: α_t = `alphas_cumprod[timestep]`, α_prev = `alphas_cumprod[prev_timestep]` (**when prev_timestep < 0, take the table head `alphas_cumprod[0]`**, the official `final_alpha_cumprod`); v_prediction first converts the network output back to ε: ε = √α_t·mo + √(1−α_t)·sample (sample has already been replaced by cur_sample at counter 1 — the conversion happens after the bookkeeping);

- sample_coeff = √(α_prev/α_t);
- denom = α_t·√(1−α_prev) + √(α_t·(1−α_t)·α_prev);
- prev = sample_coeff·sample − (α_prev − α_t)·ε / denom.

**Return dict**: `prev_sample` / `model_output_combined` (float32), `prev_timestep` (int), `state` = {`ets`, `cur_sample`} (float64). On the final step timestep = 0 → prev_timestep = −δ < 0 → α_prev takes the table head; if timestep is also 0 (t == prev), formula (9) degenerates to the identity (sample_coeff = 1, alpha gap = 0). Validation failures always raise `ValueError`.

## Example

```
alphas = linear-beta cumprod (T=1000), timesteps = [999, 500, 500, 250, 0], delta = 200

counter 0 (state = None): t = 500, mo = 0.3, sample = 2.0
  ets = [0.3] (append), combined = 0.3 (identity), cur_sample = 2.0 (stashed)
  (timestep, prev_timestep) = (500, 300)
  prev = formula(9) with alpha_t = a[500], alpha_prev = a[300]

counter 1 (state from counter 0): t = 500, mo = -0.2
  no append (ets stays [0.3])
  combined = (0.3 + (-0.2)) / 2 = 0.05   (two-point trapezoid)
  sample = cur_sample = 2.0 (restart from the stashed sample!)
  (timestep, prev_timestep) = (700, 500)  <- same adjacent pair, rewalked

counter 2: ets = [m0, m2] (append) -> combined = (3*m2 - m0) / 2
counter 3: ets = [m0, m2, m3] -> combined = (23*m3 - 16*m2 + 5*m0) / 12
counter 4: ets = [m0, m2, m3, m4] -> combined = (55*m4 - 59*m3 + 37*m2 - 9*m0) / 24
counter >= 5: ets[-3:] truncates first, then appends -> stays at 4 entries
final step (t = 0): prev_timestep = -200 < 0 -> alpha_prev = alphas_cumprod[0]
```
