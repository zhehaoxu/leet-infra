# Solution Notes · DDIM Inversion Step

## Source

- `huggingface/diffusers`, `src/diffusers/schedulers/scheduling_ddim_inverse.py`: `DDIMInverseScheduler` is the reverse of `DDIMScheduler`; this problem follows its `step` (lines 311–384) line by line. `set_timesteps` matches `DDIMScheduler`, and the docstring of `step` points at equation (12) of the DDIM paper (arXiv:2010.02502).
- The current diffusers repository has no `pipeline_ddim_inversion.py`; the scheduler is the only entry point for inversion, and this problem stays at that layer, with no pipeline.
- The step only reads a handful of entries of `alphas_cumprod`, so the contract takes the 1-D cumulative-alpha sequence directly instead of dragging beta-schedule construction (`trained_betas`, etc.) into the problem.

## Core Points and Pitfalls

**`prev_timestep` is a misleading name.** The source starts with `prev_timestep = timestep` and then overwrites `timestep` with `min(timestep - T//N, T-1)`. After the overwrite, `timestep` is the **prediction** level (cleaner) and `prev_timestep` is the **output** level (noisier). Reading "previous" as "the cleaner neighbor" flips the direction into sampling and gets `prev_sample` completely wrong.

**The two alphas come from different indices.** `alpha_prod_t` uses the prediction level \(t\), `alpha_prod_t_prev` uses the output level `timestep`. They differ by the step ratio \(T//N\), and that ratio is **integer** division: `T=1000, N=50` gives 20, not 20.0.

**The initial alpha only kicks in when \(t<0\).** The first step of an inversion loop usually has `timestep = 0`, so \(t=-r\) and `alpha_prod_t` becomes `initial_alpha_cumprod`. With the default `set_alpha_to_one=True` that is 1.0, hence `beta_prod_t = 0` — harmless for the epsilon branch (the divisor `/sqrt(alpha_prod_t)` is 1), fatal for the `sample` branch, which divides by `sqrt(beta_prod_t)`. The source silently emits inf/nan; this contract raises `ValueError`.

**`pred_epsilon` must be computed before clipping.** The source order is: compute `pred_original_sample` and `pred_epsilon` → clamp `pred_original_sample` → build the direction term from `pred_epsilon`. Moving the clamp earlier and recomputing the v-branch eps from the clamped x0 diverges from the source; the `clip` test targets exactly this.

**The returned x0 is the clamped one.** `pred_original_sample` is overwritten by `clamp` and returned as is, so `prev_sample` and the returned x0 are always self-consistent. Returning the unclamped value breaks the identity `prev == sqrt(ᾱ_prev)*x0 + direction`.

**All three prediction types share one composition formula.** epsilon / sample / v_prediction differ only in how \((x_0,\epsilon)\) is disentangled; the direction term and output formula are identical. Building all three from the same \((x_0,\epsilon)\) must give the same `prev_sample` — a stronger semantic check than comparing numbers one by one.

**Precision.** Compute in float64 and return float32. `alphas_cumprod` may arrive as float32; promote before taking the `0.5` power so float32 rounding is not amplified across steps.

## Division of Labor

- `diff-ddim`: the forward deterministic denoising step producing a cleaner \(x_{t-1}\); this problem runs the other way, with the extra initial-alpha fallback and the "prediction level vs output level" split.
- `diff-v-prediction-convert`: only converts between ε / x₀ / v; here the three prediction types are wired into the same DDIM update, and the test asserts they agree.
- `diff-ddpm-step`: the stochastic posterior step needing variance and noise; DDIM inversion is deterministic and has no variance term.
- `diff-sdedit-strength`: decides at which level inversion starts (truncating the schedule); this problem is the single-step arithmetic after that decision.
