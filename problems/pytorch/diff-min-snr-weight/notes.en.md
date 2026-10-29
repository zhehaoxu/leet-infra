# Solution Notes · Min-SNR Loss Weighting

## Source

- `huggingface/diffusers`, `src/diffusers/trainers/training_utils.py`: `compute_snr` gathers `alpha = sqrt(ᾱ)[t]` and `sigma = sqrt(1-ᾱ)[t]`, then computes `snr = (alpha / sigma) ** 2`; the docstring attributes the formula to `TiankaiHang/Min-SNR-Diffusion-Training` (`guided_diffusion/gaussian_diffusion.py`).
- Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy* (arXiv:2303.09556) introduces the \(\min(\mathrm{SNR},\gamma)/\mathrm{SNR}\) weighting, commonly with \(\gamma=5\); training scripts multiply the per-step MSE by it.
- This problem covers only the pure "schedule + timesteps → SNR and weight" layer; no loss, backward pass, or noise sampling is involved.

## Core Points and Pitfalls

**The SNR is capped, not the weight.** Writing `w = min(snr, gamma)` produces weights greater than 1 (SNRs can reach thousands); the correct form is `min(snr, gamma) / snr`, equivalently `min(1, gamma / snr)`. Both agree when `snr <= gamma`, so the mistake only shows at the clean end — the tests deliberately include high-SNR timesteps to catch it.

**SNR goes through the square-root path, not `ᾱ/(1-ᾱ)`.** The source computes `sqrt(ᾱ)` and `sqrt(1-ᾱ)`, divides, and squares; mathematically equivalent, but float32 rounds differently. The solution follows the source path and uses `ᾱ/(1-ᾱ)` as an independent oracle with a 1e-5 tolerance.

**Index direction of `alphas_cumprod`.** Index 0 is the cleanest step (largest \(\bar\alpha\), highest SNR) and index \(T-1\) the noisiest; the range check must use `max(t) < T`. PyTorch would happily treat a negative index as counting from the end, silently computing a different value if it were not rejected.

**Strict open interval \((0,1)\).** At \(\bar\alpha=1\), \(\sigma=0\) and the SNR diverges; at \(\bar\alpha=0\), \(\alpha=0\) and the weight becomes \(0/0\). Real schedules usually stay away from the endpoints, but the contract must reject them or random tests hit inf/nan.

**`gamma` is an SNR cap.** A smaller \(\gamma\) caps more timesteps and lowers their weights; as \(\gamma\to\infty\) every weight is 1 (plain MSE). Monotonicity pins the direction: larger \(\gamma\) never decreases any weight.

**Output precision and shape.** Compute in float64, return float32; the shape follows `timesteps` (not `alphas_cumprod`), repeated indices must yield identical values, and an empty tensor is legal.

## Division of Labor

- `diff-zero-terminal-snr`: modifies the endpoints of \(\bar\alpha_t\) so the noisiest end has zero SNR; here the schedule is read-only and only the weight is computed.
- `diff-sd3-timestep-density`: the density that draws timesteps; this problem weights the loss afterwards — together they form the training-side timestep pipeline.
- `to-*` problems (DPO/IPO, etc.): token-level objectives for language models; this is per-step MSE weighting for diffusion, with different dimensions and semantics.
- `diff-ddpm-forward` uses the same `alphas_cumprod` for the posterior mean/variance of the forward process; this problem only consumes the SNR.
