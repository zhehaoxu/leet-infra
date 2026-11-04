## Source

- `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_ddpm_wuerstchen.py` (269 lines, Apache-2.0 / MIT mixed header, original by Pablo Pernías). Key lines: `_alpha_cumprod` (L121-129: the three scaler branches L122-125, cosine square + normalization L126-128, clamp L129), the `__init__` `_init_alpha_cumprod` (L116), `previous_timestep` (L255-269: `argmin(|timesteps − timestep[0]|)` using only t[0], the `index+1` successor lookup, expand back to batch), `step` (L202-221: posterior mu L212, std L215, the `(prev_t != 0).float()` gate L216), `set_timesteps` (L148-170: `linspace(1.0, 0.0, N+1)`), `add_noise` (L223-250, duplicates `diff-ddpm-forward`, outside the judge domain).
- Origin: the two-stage cascade of Wuerstchen (arXiv 2306.00637, Pernías et al.) — a prior stage producing a low-resolution semantic compression (CLIP space) and a decoder stage reconstructing high resolution on top of it; each stage holds its own scheduler instance, with `scaler` as the "time-axis warp" knob (>1 pushes sampling steps toward the noisy end).
- Formula family: the ε-parameterized DDPM posterior p(x_{t−1}|x_t, x_0) — the same family as `diff-ddpm-step` (see dedup).

## Normalization (judge domain vs official)

- **Noise made explicit**: the official `randn_tensor(mu.shape, generator, ...)` becomes an explicit `noise` argument, required on every call.
- **Table and t made explicit**: the official `self.timesteps` (from `set_timesteps`) and the internal state of `self._alpha_cumprod` become a `timesteps_table` argument plus a pure-function ᾱ.
- **Batch semantics tightened**: the official code looks up prev_t using only `timestep[0]` (`self.timesteps - timestep[0]`) but computes ᾱ_t from the full t vector — with a mixed batch prev_t decouples from the per-element ᾱ, breaking semantics. The official pipeline guarantees equal t within a batch; the judge domain validates "all equal within the batch" and rejects mixes (the official code has no such validation).
- **Table-tail out-of-range becomes a rejection**: the official `index+1` overflow is an IndexError; the judge domain validates and raises ValueError.
- **scaler / s validation**: none officially (scaler ≤ 0 makes pow produce garbage); the judge domain requires positive and finite.
- **Gate keeps the float comparison**: `(prev_t != 0).float()` verbatim — prev_t comes from the table, exactly 0 triggers; no epsilon tolerance (a mutation hotspot).
- **ᾱ clamp kept**: `clamp(0.0001, 0.9999)` verbatim — both ends trigger exactly at t=1 / t=0, preventing √(1−ᾱ) division by zero and the α degeneration at ᾱ=1.

## Key observations

1. **ᾱ is an analytic function, not a table**: the mirror image of `diff-ddpm-step`'s discrete ᾱ table input. The s offset and the normalization denominator `_init_alpha_cumprod` are the same cos expression evaluated at t=s — mutations `m-ac-init` (denominator drops s) and `m-ac-offset` (numerator drops s) go in opposite directions and both must be caught.
2. **The three scaler branches**: >1 maps t→1−(1−t)^scaler (pushing t toward 1, noisier); <1 maps t^scaler (pulling toward 0); =1 identity. The substance of the Wuerstchen "two stages" — per-stage time-axis warping. One mutation per branch, with the identity branch pinned by the closed-form ᾱ.
3. **Both clamp ends have observation points**: t=1 → ᾱ=0 → lower clamp (`check_first_step` asserts 0.0001 exactly); t=0 → ᾱ=1 → upper clamp (`check_identity_step` asserts 0.9999). The upper bound can also be triggered by the table [1, 0, 0] with t=0 (argmin ties take the first → index=1 → prev_t=0 → pinning the gate and the identity step at once).
4. **The lookup is argmin nearest neighbor + 1**: not index-minus-one, not interpolation; ties take the first (torch/numpy argmin agree). t must be a table value (or close) — tests always take t from the table so argmin has no ambiguity.
5. **The gate is an exact float comparison**: only prev_t == 0 turns the noise off; "changing noise leaves the output unchanged at the final step" is the independent assertion (`check_last_step_gate`), with `m-gate` and `m-noise-applied` pinning the computation path and the flag respectively.
6. **α = ᾱ_t/ᾱ_prev < 1**: the table is descending → prev_t < t → ᾱ_prev > ᾱ_t. When ᾱ_t = ᾱ_prev (adjacent entries map to the same ᾱ) it is an identity step — allowed in the judge domain (std=0, prev=sample), no special handling needed.
7. **No state machine**: a pure function; the full loop just walks the table — but each step's prev_t lookup can still misalign (`m-idx-prev`), pinned by the loop's independent assertion "prev_t is exactly the next table value".

## Dedup check

- **`diff-ddpm-step`**: the same DDPM posterior formula family (same mu/std shape). But that one takes ᾱ from an **explicit discrete table** with a different prev_t/x0 path; here ᾱ is an **analytic function** (four layers of processing: s offset / normalization / clamp / scaler power warp) plus the argmin lookup and the gate. The mechanism increment lies in constructing ᾱ and locating prev_t, not in the posterior formula itself.
- **`diff-ddpm-forward`**: the forward q(x_t|x_0) (this scheduler's `add_noise` duplicates it; outside the judge domain).
- **Deterministic DDIM-style variants**: the gated final step ≡ a deterministic finish, but only a single-point behavior at the last step, not a full DDIM.
- **`diff-dpmpp-2m` / `diff-lms` (757) and the multistep family**: multi-step history state machines; this problem is a single-step pure function with no state.
- **`diff-cosine-dpmsolver-sde-step` (759)**: also "cosine"-named but that is the sigma-space EDM/VE convention + SDE noise + multistep; this one is β-space ᾱ posterior + gated noise — the ᾱ/σ formulas and noise semantics are entirely unrelated.

## Tests and mutations

11 cases: first step t=1 (exact lower-clamp assertion + closed-form independent verification of mu/std/prev), final-step gate (prev≡mu + output unchanged when noise changes), mid step (two closed-form ᾱ + α<1 + closed-form mu), identity step (the table [1,0,0] argmin tie + upper clamp + α=1), scaler>1 / <1 / =1 closed forms, B=2 batch broadcast + mixed-batch rejection, N=4 full loop (independent prev_t assertion + final gate), five table checks, five timestep checks, seven parameter checks — rejects all assert error messages to prevent "empty catches".

34 mutations: **33 CAUGHT + 1 equivalent**:

| Equivalent mutation | Evidence |
|---|---|
| `m-argmin-t0` (lookup `t[0]` → `t[-1]`) | 200 random configs (table length 2-11, three scaler branches, random s, B 1-2, half with a 0 tail), all outputs bit-identical. Structural identity: "all equal within the batch" is a precondition, so t[-1] ≡ t[0] |

The other 33 cover: five table checks (length/finiteness/monotonicity/dimension/dtype), five timestep checks (dimension/finiteness/range bounds/mixed batch/dtype), scaler and s range plus bool, noise required and shape, sample shape, both scaler-branch exponents, normalization denominator dropping s, lower/upper clamp and clamp removal, s offset removal, cosine square demotion, lookup index+1→+0, tail-overflow check removal, gate stuck on, the noise_applied flag, the returned prev_t, mu's α factor/sign/ε term, std's (1−ᾱ_t) factor, the α ratio swap, ᾱ_prev replaced by t[0], and the view shape.

## Pitfalls

- **Pin the gate flag and the computation path separately**: `m-noise-applied` (only changes the returned flag) and `m-gate` (only changes the computation) do not cover each other — a wrong-flag implementation compares green against the oracle (prev is computed correctly), and a wrong-gate implementation keeps a correct flag. Each is caught by a different assertion.
- **The upper clamp's observation point is non-obvious**: ᾱ(t) < 1 for all t ∈ (0, 1]; the upper bound triggers only exactly at t=0; and t=0 as the current step requires a successor after 0 in the table — constructible only with an adjacent-equal table ([1, 0, 0], allowed by non-increasing) where argmin ties take the first. With only strictly decreasing tables, the upper-clamp mutation (0.9999→1.0) would blow up at ᾱ=1 (division by zero) or slip past the identity-step assertion.
- **The official argmin uses only t[0]**: with a mixed batch the official prev_t is shared while ᾱ_t varies per element — not a reproducible "feature" but a leak of a pipeline invariant; the judge domain validates and rejects it (mutation `m-t-batch` is caught by the reject case).
- **`t**scaler` and `1−(1−t)**scaler` float powers**: t=0 with scaler<1 gives 0^0.5=0, legal; the warped t' stays in [0, 1] (power functions are monotone), no extra clamp needed.
