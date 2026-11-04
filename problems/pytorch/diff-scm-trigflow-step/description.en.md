# Diffusion: SCM Trigflow Consistency Step (Jump to Data End then Re-noise)

## Problem

sCM (Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models) replaces the diffusion timestep with an **angle**: the trigflow parameterization defines

```text
x(θ) = cos(θ)·x0 + sin(θ)·σ_data·ε,    θ = arctan(σ/σ_data) ∈ [0, π/2]
```

Implement `solve(model_output, sample, timesteps, step_index, noise=None, sigma_data=0.5, prediction_type="trigflow")`: given an angle table (monotonically non-increasing, length N+1) and the current step index, perform one consistency step

```text
s = timesteps[step_index]        # current angle
t = timesteps[step_index + 1]    # next angle (cleaner)
pred_x0     = cos(s)·sample − sin(s)·model_output
prev_sample = cos(t)·pred_x0 + sin(t)·(σ_data·noise)
```

Return `dict(prev_sample, pred_original_sample, s_angle, t_angle, noise_term)` where `noise_term = sin(t)·σ_data·noise`.

## Key Semantics

- **Consistency = jump back to the data end, then re-noise**: each step first recovers `x0` from the model output, then re-noises directly from `x0` to the next angle. This is not a small Markov-chain Euler step but sCM's non-Markovian jump.
- **Single-step sampling is noise-free by construction**: with Sana-Sprint's one-step table `[θ_max, 0]`, the final step has `t = 0`, and `sin(0) = 0` zeroes the noise coefficient, so `prev == pred_x0`. No explicit gate is needed — the trig functions do it.
- **Legality of noise=None**: noise may be omitted only when `t` is **exactly** 0.0; when `t != 0` it must be given explicitly. noise not None with `t == 0.0` is legal (officially randn is still sampled but multiplied by zero), and `prev` is then independent of the noise value.
- **σ_data scaling**: applies only to the noise term (officially `randn_tensor(...) * sigma_data`), never to `pred_x0`.
- **Adjacent equal angles are legal**: the official `index_for_timestep` specifically handles duplicates in the table; the judgement-domain non-increasing check allows adjacent equals and rejects only strictly increasing pairs.

## Judgement-domain Conventions

- `step_index` is an explicit argument (the official `_step_index` state machine and `begin_index` boilerplate are external), with `0 ≤ step_index ≤ len(timesteps) − 2`.
- `noise` is an explicit argument (the official `randn_tensor` lives on the pipeline side); σ_data is multiplied inside this function.
- `prediction_type` supports only `"trigflow"` (the sole official implementation; anything else raises `Unsupported parameterization`).
- The official `len(self.timesteps) > 1` gate's else branch is reachable only under a degenerate `num_inference_steps=0` table — dead code, not ported.

## Example

Two-step Sana-Sprint default table `[1.57080, 1.3, 0]` (`set_timesteps(2, intermediate_timesteps=1.3)`):

- step 0: `s = 1.57080, t = 1.3`, `cos(s) ≈ 0.0000007967` — the start is almost pure noise;
- step 1: `s = 1.3, t = 0`, `sin(t) = 0`, the noise term vanishes and `prev == pred_x0`.

Data scale: `timesteps` length 2–32, tensor shapes `(1, 2)`–`(2, 2)`, float64.
