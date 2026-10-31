## Upstream

- Repo: `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_deis_multistep.py` (v0.42.x, downloaded 2026-10-31 via contents API; local copy at `.workbuddy/upstream/d93/`)
- Paper: DEIS — Zhang et al., "Fast Sampling of Diffusion Models with Exponential Integrator" (2204.13902); original repo `qsh-zh/deis`

## Line-by-line mapping

| Upstream | This problem |
| --- | --- |
| `convert_model_output` L581-639 | convert step: four `prediction_type` branches + deis projection `(sample − α_t·x0)/σ_t`; note `flow_prediction` uses raw `self.sigmas[step_index]` (L625), not the transformed `σ_t` |
| `_sigma_to_alpha_sigma_t` L437-457 | `alpha_sigma()`: `use_flow_sigmas → (1−σ, σ)`, else `(1/√(σ²+1), σ/√(σ²+1))` |
| `deis_first_order_update` L641-700 | first-order: `x = (α_t/α_s)·sample − σ_t·(e^h − 1)·d`, `h = λ_t − λ_s` |
| `multistep_deis_second_order_update` L702-773 | `ind_fn(t,b,c) = t(−ln c + ln t − 1)/(ln b − ln c)` (L763-765, the exact integral of `(ln t − ln c)/(ln b − ln c)` in t); coef1/coef2 by endpoint differences L767-770 |
| `multistep_deis_third_order_update` L775-861 | `ind_fn(t,b,c,d)` L840-851 (quadratic log-rho polynomial integral); coef1/2/3 rotate `(ρ0,ρ1,ρ2)` as (b) with (c,d) pairs L853-855 |
| `step` L917-979 | order ramp: `lower_order_final = (step_index == len(timesteps)-1) and config.lower_order_final and len < 15` (L951-956, no `euler_at_final`/`final_sigmas_type` unlike DPMSolver); `model_outputs` shift L959-961; `lower_order_nums` bump L970-971 |

## Deliberate deviations (documented)

- **No diffusers dependency**: eval env has no diffusers; `sigmas` is an explicit input (caller reproduces `set_timesteps`'s `interp + append sigma_last` if needed). Schedule construction (linspace/leading/trailing, Karras/exponential/beta sigmas, flow sigmas) is out of scope — shared schedule-family mechanics covered by 517/518.
- **History window contract**: upstream keeps a fixed-length `model_outputs` list with `None` sentinels plus a separate `lower_order_nums` counter. Here `len(converted_history) == min(step_index, solver_order)` is enforced — exactly the information the ramp uses (`low < 1`, `low < 2`).
- **thresholding omitted**: `config.thresholding` defaults False and dynamic thresholding is already covered by 521 on the DPMSolver++ side.
- **`solver_type` only `"logrho"`**: upstream silently rewrites other values back to `logrho` (L231-235), so only this path exists in practice.
- float64 internal math (upstream float32); `x_next` returned as float32.

## Test design

- Independent numpy oracle (`_oracle_solve`) rewritten from the paper's integral formulas, cross-checked element-wise on 8 random 3rd-order configs (1e-9).
- Hand-computed first-order example with hardcoded anchors (λ_s = −ln 8, h = ln 4, e^h−1 = 3 → clean multiplier).
- Ramp matrix: warmup (step 1 → 2nd order even without lof), final-step drop at N=3, `lower_order_final=False` escape, N=15 (no drop) vs N=14 (drop to 1st) at the last step.
- 4 `prediction_type` branches incl. flow (`σ ∈ (0,1)` — outside it `ln α` is undefined since `α = 1−σ < 0`).
- Mutation testing 8/8 caught: `exp(h)−1 → h`, first-order base sample→d, coef endpoint swap, ind2 missing `−1` term, ind3 constant `2→1`, m1 vs m2 history slot, `<15 → <20`, convert minus→plus.
