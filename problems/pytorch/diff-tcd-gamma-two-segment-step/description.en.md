# Diffusion: TCD Gamma Intermediate Noise Level and Two-Segment Consistency Step

## Problem

TCD (Trajectory Consistency Distillation, arXiv 2410.05964) introduces gamma interpolation and two-segment sampling on top of LCM-style consistency models. Implement `solve(model_output, sample, alphas_cumprod, timesteps, step_index, noise=None, eta=0.3, prediction_type="epsilon")`:

```text
prev_t = timesteps[step_index + 1]        # 0 past the table end
s      = floor((1 − eta) · prev_t)        # gamma intermediate level (paper's gamma, code's eta)
# parameterization (epsilon; sample / v_prediction derive x0 and ε analogously):
x0 = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t
x_s = √ᾱ_s·x0 + √(1−ᾱ_s)·ε               # segment 1: jump to the intermediate level s
# segment 2 (eta > 0 and not the final step):
prev = √(ᾱ_prev/ᾱ_s)·x_s + √(1−ᾱ_prev/ᾱ_s)·noise
# final step (step_index == len(timesteps)−1) or eta == 0: prev = x_s
```

Return `dict(prev_sample, pred_noised_sample, pred_original_sample, pred_epsilon, timestep_s, prev_timestep, noise_applied)`.

## Key Semantics

- **Two segments**: TCD does not go directly from `x_t` to `x_prev`; it first jumps to the gamma intermediate level `x_s` (re-noising with the same ε), then from `x_s` to `prev` — the marginal-distribution splicing of trajectory consistency.
- **eta has two roles**: ① shortens the intermediate level `s = floor((1−eta)·prev_t)` (at eta=0, s equals prev_t and the two segments merge); ② controls the segment-2 noise injection (eta=0 is fully deterministic). The final step never injects noise.
- **Three parameterizations**: epsilon / sample / v_prediction each derive `x0` and `ε`, but the `x_s` and `prev` segment formulas are fully shared.
- **prev_t location**: `timesteps[step_index+1]`, or 0 past the table end (official `torch.tensor(0)`); the official `final_alpha_cumprod` branch (prev<0) is unreachable — dead code.
- **Validation**: timesteps must be integer, **strictly decreasing** (the official custom-path check), and within the ᾱ index range; `eta ∈ [0,1]` (official assert, judgement-domain ValueError); noise must be given explicitly when `eta > 0` and not the final step, and is ignored otherwise.

## Judgement-domain Conventions

- timestep/prev_t/timestep_s are indices into `alphas_cumprod` (the training table, 1-D float); the judgement domain validates the index range explicitly (the official code would raise IndexError on overflow).
- The official `randn_tensor` (generator) lives on the pipeline side; noise is an explicit argument.
- The official eta check is an `assert` (AssertionError); the judgement domain uses ValueError uniformly while keeping the "gamma must be less than or equal to 1.0" keyword.

## Example

Training table ᾱ from cumprod of β=linspace(1e-4, 0.02, 50); inference table `[49, 36, 23, 10, 0]` (N=5), eta=0.3: step 2 has `prev_t = 10`, `s = floor(0.7·10) = 7`; at eta=0.25, `s = floor(7.5) = 7` (round would give 8 — a discriminating case for floor semantics). The one-step table `[49]` makes step 0 the final step: `prev_t = 0`, `prev == x_s`.

Data scale: ᾱ length 50, table length 1–5, tensor shapes `(1, 2)`–`(2, 2)`, float64.
