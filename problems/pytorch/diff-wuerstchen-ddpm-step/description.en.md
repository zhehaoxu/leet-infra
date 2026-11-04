## Problem

`DDPMWuerstchenScheduler` is the scheduler of two-stage diffusion pipelines such as Wuerstchen / Kandinsky / Stable Cascade (the prior and decoder each hold an instance, distinguished by `scaler`). Its fundamental difference from plain DDPM: **no discrete β/ᾱ table** — ᾱ is an **analytic cosine function of continuous t ∈ [0,1]** (with an s offset, normalized by `_init_alpha_cumprod`, clamped to [0.0001, 0.9999]), warped by a `scaler` power transform; `prev_t` is not an explicit argument but is **looked up** (argmin nearest neighbor on the table produced by `set_timesteps`, taking the next entry); and the posterior step **gates out the noise term** when `prev_t` is exactly 0 (a DDIM-style deterministic finish). Implement one `step` call, with the noise as an explicit argument.

Judge inputs:

- `model_output` / `sample` / `noise`: floating tensors of the same shape, batch dimension B matching `timestep`;
- `timestep`: 1-D floating tensor (batch dim), values in [0, 1], **all equal within the batch** (the official pipeline shares one t per step; only `timestep[0]` is used to look up prev_t);
- `timesteps_table`: 1-D floating tensor, the official `set_timesteps` output `linspace(1.0, 0.0, N+1)`, length ≥ 2, non-increasing;
- `noise`: **required on every call** (officially from `randn_tensor`);
- `scaler` > 0 (default 1.0; three branches >1 / <1 / =1), `s` > 0 (default 0.008).

## Computation rules

**The ᾱ function** (official `_alpha_cumprod` verbatim):

- `_init_alpha_cumprod` = cos(s/(1+s)·π/2)²;
- scaler warp: scaler > 1 → t = 1−(1−t)^scaler; scaler < 1 → t = t^scaler; =1 identity;
- ᾱ(t) = cos((t+s)/(1+s)·π/2)² / _init_alpha_cumprod, clamped to [0.0001, 0.9999] — t=1 → 0 hits the lower bound, t=0 → 1 hits the upper bound.

**Table lookup** (official `previous_timestep` verbatim): index = argmin(|timesteps_table − timestep[0]|) (ties take the first, torch semantics), prev_t = timesteps_table[index+1]; an out-of-range index+1 (timestep already at the table tail) is rejected.

**Posterior step** (official `step` verbatim): ᾱ_t = ᾱ(timestep), ᾱ_prev = ᾱ(prev_t) (each viewed as (B, 1, …)), α = ᾱ_t/ᾱ_prev:

- mu = √(1/α)·(sample − (1−α)·model_output/√(1−ᾱ_t));
- std = √((1−α)·(1−ᾱ_prev)/(1−ᾱ_t))·noise;
- gate: prev = mu + std·(prev_t != 0) — when prev_t is exactly 0 (final step) no noise is added, a float comparison;
- when ᾱ_t = ᾱ_prev (e.g. adjacent table entries map to the same ᾱ), α = 1 → mu = sample, std = 0 → prev = sample (identity step).

**Returned dict**: `prev_sample` / `mu` / `alpha_cumprod` / `alpha_cumprod_prev` / `alpha_ratio` (float32), `noise_applied` (bool), `prev_t` (float scalar). Validation failures always raise `ValueError`.

## Example

```
timesteps_table = [1.0, 0.75, 0.5, 0.25, 0.0], scaler = 1.0, s = 0.008
t = 0.5 (B=1), mo = -0.2, sample = 0.8, noise = 0.1

init = cos(0.008/1.008 * pi/2)^2 = 0.999845
ac  = cos(0.508/1.008 * pi/2)^2 / init = 0.493844
acp = cos(0.258/1.008 * pi/2)^2 / init = 0.847012
alpha = ac / acp = 0.583042   (< 1: larger t, smaller ᾱ)
prev_t = table[argmin(|tab - 0.5|) + 1] = 0.25 -> noise_applied = True

mu  = sqrt(1/0.583042) * (0.8 - (1-0.583042)*(-0.2)/sqrt(1-0.493844))
    = 1.201215
std = sqrt(0.416958*0.152988/0.506156) * 0.1 = 0.035500
prev = mu + std = 1.236716
```
