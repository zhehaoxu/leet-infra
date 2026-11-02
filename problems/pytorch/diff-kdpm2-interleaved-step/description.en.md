## Problem

Heun (problem 643) gets second order by averaging two derivatives; KDPM2 takes the other road — Karras et al.'s **DPM-Solver-2**: re-evaluate the derivative once at the **geometric mean midpoint in log-sigma space**, then take that single derivative from the **step origin** across the whole step, discarding the first call's derivative outright. diffusers' `KDPM2DiscreteScheduler` implements this as a two-call state machine over an **interleaved sigma table**: the base sigma table first produces a midpoint table via a log-space lerp, then both tables are repeat_interleave-expanded to length 2N+2, with each logical step occupying two step indices.

Judge inputs:

- `model_output` / `sample`: same-shape float tensors (the second call's `sample` is the **midpoint sample**, `model_output` the model output at the midpoint);
- `sigmas`: 1-D float tensor, the **unexpanded** base table `[σ0, …, σ_{N−1}, 0]` (length N+1, N ≥ 2), finite, non-negative, non-increasing, first N entries > 0 and last entry exactly 0;
- `step_index`: current index;
- `state`: `None` = first call; `{"sample"}` = second call (pass the first call's return through);
- `prediction_type` ∈ {`epsilon`, `v_prediction`}.

## Rules

**Building the expanded tables** (recomputed on every call):

- Midpoint table: `I[k] = √(σ[k−1]·σ[k])` (k = 1…N), `I[0] = 0`. Numerically equivalent to the official `sigmas.log().lerp(sigmas.roll(1).log(), 0.5).exp()` (√(ab) = exp((log a + log b)/2));
- Two expanded tables (length **2N+2**: head 1 + body repeat_interleave(2) + tail 1):
  - `sigE = [σ0, σ1,σ1, σ2,σ2, …, σ_{N−1},σ_{N−1}, 0, 0, 0]`
  - `sigIE = [0, I1,I1, I2,I2, …, I_{N−1},I_{N−1}, 0, 0, 0]`

i.e. `sigE[2k] = σ_k` (0 ≤ k ≤ N), `sigE[2k+1] = σ_{k+1}` (0 ≤ k ≤ N−1), `sigE[2N+1] = 0`; `sigIE` likewise.

**Index layout**: logical step j (0 ≤ j ≤ N−1) occupies indices 2j (first call) and 2j+1 (second call); **the last logical step has only the first call** (timesteps length 2N−1, final call at index 2N−2, midpoint `sigIE[2N−1] = 0`, dt = −σ_{N−1} lands straight on x₀).

**First call** (`state=None`, even index, 0 ≤ i ≤ 2N−2; σ_hat = σ·(gamma+1), official gamma=0 identity):

- `σ = sigE[i]` (must be > 0), midpoint `m = sigIE[i+1]`;
- `x0 = f(sample, mo, σ)`, `derivative = (sample − x0) / σ`;
- `dt = m − σ`, `prev = sample + derivative·dt` (Euler half-step to the midpoint);
- store `state = {"sample": sample}` — only the **original sample**.

**Second call** (`state` is a dict, odd index, 1 ≤ i ≤ 2N−3; the last logical step has no second call):

- `σ = sigE[i−1] = σ_j`, midpoint `m = sigIE[i] = I_{j+1}` (must be > 0), `σ_next = sigE[i] = σ_{j+1}`;
- `x0 = f(sample, mo, m)` — sigma_input is the **midpoint**;
- `derivative = (sample − x0) / m` — the denominator is the **midpoint**;
- `dt = σ_next − σ` — the **full step**, not a half-step;
- `prev = state["sample"] + derivative·dt` — restart from the **stored sample**; the first call's derivative is discarded;
- return `state = None`.

**The two x0 forms** (sigma_input: σ on the first call, midpoint m on the second):

| prediction_type | x0 |
| --- | --- |
| `epsilon` | `sample − σ_input·mo` |
| `v_prediction` | `mo·(−σ_input/√(σ_input²+1)) + sample/(σ_input²+1)` |

Return `{"prev_sample", "pred_original_sample", "state", "derivative", "dt"}`.

## Worked example

```
base table [4, 1, 0] (N=2) → I1 = √(4·1) = 2
sigE  = [4, 1, 1, 0, 0, 0]; sigIE = [0, 2, 2, 0, 0, 0]
first call i=0 (σ=4, m=2): sample=1, mo=0.1
  x0 = 0.6; d = 0.1; dt = 2−4 = −2; mid = 1 − 0.2 = 0.8; state stores sample=1
second call i=1 (σ=4, m=2, next=1): sample'=0.8, mo'=0.2
  x0 = 0.8 − 0.4 = 0.4; d2 = (0.8−0.4)/2 = 0.2; dt = 1−4 = −3 (full step)
  prev = 1 + 0.2·(−3) = 0.4      ← departs from the stored sample=1
first call i=2 (σ=1, m=0): dt = −1; prev = 0.8 − 1·0.3 = 0.5 (x0 under epsilon)
```

**Two counterexamples**:

- restarting the second call from the current midpoint sample: `0.8 + 0.2·(−3) = 0.2` — off from the correct 0.4 by the whole half-step displacement;
- dividing the second derivative by σ_next=1 instead of the midpoint 2: `d2 = 0.4`, `prev = 1 − 1.2 = −0.2` — blows up immediately.

## Edge cases and constraints

- `sigmas` length ≥ 3, finite, non-negative, non-increasing, last entry exactly 0, first N entries > 0;
- first-call index must be even, second-call odd; second-call index ≤ 2N−3 (the last logical step has no second call); first-call index ≤ 2N−2;
- first-call σ > 0; second-call σ > 0 and midpoint m > 0;
- `state` must contain exactly the `sample` key with the same shape as sample; `step_index` an integer (bool rejected);
- float64 internally; `prev_sample`/`pred_original_sample`/`derivative` returned as float32; violations raise `ValueError`.

## Boundaries with existing problems

- 643 `diff-heun-predictor-corrector`: also a "one logical step, two calls, stored sample" state machine, but Heun's midpoint is the **arithmetic half-step** (same sigma pair, walk half way and average the two derivatives); KDPM2's midpoint is the log-σ **geometric mean**, keeps only the midpoint derivative, and dt spans the full step — three mechanical differences mapping one-to-one onto three mutations;
- 522 `diff-euler-ancestral-step`: sigma-space Euler with ancestral noise splitting, single step, stateless;
- problem 643 takes the **already-expanded** paired sigma table as input; this problem makes the solver **build** the interleaved tables itself (geometric midpoint plus the repeat_interleave head/tail padding) — construction is part of the grading;
- the ancestral KDPM2 variant (σ_up/σ_down split plus noise injection) is out of scope here, mechanically closer to 522;
- 494–513 training-side and guidance families, 576 UniPC: unrelated.
