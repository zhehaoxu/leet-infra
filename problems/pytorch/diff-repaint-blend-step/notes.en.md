# Notes: RePaint's Masked Blend Step and the Forward Undo

## Source

- diffusers v0.42.1 `src/diffusers/schedulers/scheduling_repaint.py` (419 lines in total); the class `RePaintScheduler` spans 99-419 and `order = 1` sits at 126.
- RePaint itself is arXiv:2201.09865 (Lugmayr et al., RePaint: Inpainting using Denoising Diffusion Probabilistic Models). The source names its Algorithm 1 in four places: 317-322 explains that the unknown branch swaps the paper's formula (7) for the DDIM one, 340-344 is Line 5, 347-348 is Line 8, and 390-391 is Line 10.
- The posterior itself: x̂₀ at 311 is formula (15) of DDPM (arXiv:2006.11239); `_get_variance` (234-262) follows formulas (6)(7) of DDPM per the comment at 253-259 and states it matches formula (16) of DDIM (arXiv:2010.02502) when `eta=0`; the direction term at 335 is DDIM formula (12).
- Construction time: the four β tables at 141-152 (`linear` / `scaled_linear` / `squaredcos_cap_v2` via `betas_for_alpha_bar` at 47-96 / `sigmoid`), `alphas_cumprod` at 156-157, and `final_alpha_cumprod = torch.tensor(1.0)` at 160 — **hardcoded**, not `alphas_cumprod[0]`.
- The config docs at 116-118 spell out that `eta=0.0` is DDIM and `1.0` is DDPM; the comment at 317-322 adds that with `eta=1.0` the DDIM schedule gives the same result as DDPM.

## Core Points and Pitfalls

- **The known region never goes through the network.** 348 selects cell by cell with `mask*known + (1-mask)*unknown`; it is not "denoise everything, then overwrite". `known` (345) comes from re-noising `original_image`, so `model_output` and `clip_sample` have no effect where `mask=1` — measured bit-for-bit identical across `eta∈{0,0.5,1.0}` and both `clip_sample` values.
- **Two branches use ᾱ_prev, exactly one uses ᾱ_t.** The `beta_prod_t` at 307 only serves x̂₀ (311) and the variance denominator (260); `known` (345) and the direction term (335) both use `alpha_prod_t_prev` from 306. Writing `known` on the ᾱ_t scale is the classic mistake.
- **`std_dev_t` carries a square root and feeds coefficients on both sides of the unknown branch.** 327 is `eta * variance**0.5`; 335 subtracts `std_dev_t**2` (not `variance`) while 331 multiplies by `std_dev_t` itself. In the worked example `variance=0.168672` and `std=0.205349`, so `eta*variance` shrinks both by roughly half.
- **The η gate only covers the unknown branch.** The `if t > 0 and eta > 0` at 329-331 degrades `variance` to the scalar 0, but the `sqrt(1-ᾱ_prev)*noise` at 345 is added unconditionally. Also, 326 draws noise once and the comment at 321-322 says that draw is reused by steps 7. and 8. — which is why the statement hands out a single `noise`.
- **`prev < 0` takes the hardcoded 1.0.** The ternary at 306/249 means the first `n` steps (`t=0..n-1`) have ᾱ_prev=1 ⇒ `variance=0`, direction coefficient `sqrt(1-1-0)=0`, `known=original_image`, so the whole expression collapses to `mask*orig + (1-mask)*x̂₀` and `eta` drops out entirely. Using `ᾱ[0]` instead shows up immediately on those cells.
- **`1 - ᾱ_prev - std_dev_t**2` can go negative and upstream silently produces NaN.** Since `(1-ᾱ_t/ᾱ_prev) <= (1-ᾱ_t)` whenever ᾱ_prev<1, `variance <= 1 - ᾱ_prev`, so `eta=1` is always safe; the ceiling is `sqrt((1-ᾱ_t)/(1-ᾱ_t/ᾱ_prev))`, which equals 1.169961 in the worked example and `eta=1.5` already crosses it. This problem turns it into an explicit `ValueError`.
- **The undo is n sub-steps, one β at a time.** 379-391: `n = T // N`, each step is `sqrt(1-beta)*x + sqrt(beta)*noise` over `betas[t..t+n-1]`, never a ratio of ᾱ values. Telescoped, the coefficient on x is `sqrt(prod alpha) = sqrt(ᾱ_{t+n-1}/ᾱ_{t-1})` (with ᾱ₋₁:=1 when `t=0`) and the i-th noise draw carries `sqrt(beta_{t+i} * prod_{k>i} alpha_{t+k})`; writing `sqrt(ᾱ_{t+n}/ᾱ_t)` multiplies everything by a stray `sqrt(alpha_{t+n})` — an index slip, not a magnitude difference.

## Deviations from Upstream

- Upstream draws its own noise at 326 via `randn_tensor`, and `undo_step` draws another one per round at 383-388 (the mps branch is only a reproducibility workaround and carries no algorithm). Here `noise` and `undo_noise` (the `n` draws) are explicit inputs.
- Upstream builds `alphas_cumprod` from one of four schedules in the constructor; this problem takes a 1-D `betas` directly, which is the same route as `trained_betas` (139-140), so `beta_schedule` is no longer an argument.
- Upstream keeps `num_train_timesteps` in config and `num_inference_steps` on the instance, and neither integer division (302, 379) checks the remainder; this problem requires `len(betas) % N == 0`.
- Upstream's `timestep` comes from the `timesteps` array already scaled by `(T // N)` at 231, so it is a β-table index; the problem keeps that convention but requires `t + n <= T` explicitly — upstream would raise `IndexError` at 382.
- Upstream lets `mask`/`original_image` broadcast and does no shape validation in `step`; here the five tensors must share one shape, `mask` must lie in [0,1], and dimensionality must be at least 1.
- Upstream's `step` returns only `(prev_sample, pred_original_sample)` (350-356) and `undo_step` is an **independent** method that `step` never calls. This problem packs both into one call: `undone` is `undo_step(sample, t)` and has no data dependency on the blend.
- In v0.42.1 the directory `src/diffusers/pipelines/repaint/` no longer exists (GitHub raw returns 404), so `step`/`undo_step` can only be called directly; `add_noise` still raises `NotImplementedError` at 395-416, telling you to use `DDPMScheduler.add_noise` instead.

## Division of Labor

- `diff-repaint-jump-schedule` (188-232 of the same scheduler) only arranges the `jump_length/jump_n_sample` table from `set_timesteps` and contains no per-step math; this problem picks up exactly where that table lands.
- `diff-ddim-inverse-step` also pushes a sample toward a noisier level, but that is a DDIM inversion and needs the network; this problem's `undo_step` is pure forward noising with no `model_output` involved.
- `diff-ddim` / `diff-ddpm-step` own the standard one-step updates; the unknown branch here is the generalized DDIM formula (12) — ᾱ_prev taken at the jumped-to slot — plus the η correction, while the known branch is built from scratch.
- `diff-consistency-decoder-2step` already examines how `betas_for_alpha_bar` builds a table; this problem consumes a ready β table and rebuilds nothing.
- `diff-amused-topk-remask` and `diff-entropy-bound-unmask` belong to the same "mask plus repeated commitment" family but live on the discrete / latent decoding side; this problem is the continuous image-side spatial mix.
- `diff-sigma-grid-construct` handles continuous σ grids, whereas RePaint always walks the integer training β table, so the two do not overlap.
