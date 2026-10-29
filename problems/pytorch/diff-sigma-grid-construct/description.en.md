# Diffusion Sampling: EulerDiscrete Timestep Grids and Sigma Converters

## Problem statement

Every continuous-time diffusion sampler (Euler, Euler-Ancestral, Heun, ...) first compresses the \(T\) discrete training timesteps down to the \(N\) inference steps it will actually take, and then reads off a corresponding sequence of noise levels \(\sigma\). diffusers' `EulerDiscreteScheduler` splits this into three independently switchable layers: the timestep grid (`timestep_spacing`), the sigma interpolation (`interpolation_type`) and the sigma converters (Karras / exponential / beta). The order in which those layers compose fixes both the final time scale and the \(\sigma\) sequence, and therefore the step size taken at each sampling iteration.

Implement that whole pipeline: given the training-side `alphas_cumprod` and the number of inference steps \(N\), return the inference timesteps `timesteps` and the sigma sequence `sigmas`.

## Rules

Let \(T = \) the length of `alphas_cumprod` and define the per-element baseline

\[ \sigma_{\text{full}}[i] = \sqrt{\frac{1-\bar\alpha_i}{\bar\alpha_i}},\qquad i = 0,\dots,T-1 \]

It increases with \(i\): index \(0\) is the clean end (small \(\sigma\)) and index \(T-1\) is the noisy end (large \(\sigma\)).

**1. Timestep grid** (`timestep_spacing`, always descending from the noisy end to the clean end):

- `"linspace"`: `linspace(0, T-1, N)` flipped, evenly spaced in floating point;
- `"leading"`: `step_ratio = T // N` (integer division), `(arange(N) * step_ratio).round()` flipped, then `+ steps_offset`;
- `"trailing"`: `step_ratio = T / N` (true division), take `count = ceil(T / step_ratio)` points `grid[k] = T - k * step_ratio`, then `round() - 1`. When the floating-point value of \(T/N\) is not an integer, `count` becomes \(N+1\), matching the length of numpy's `arange(T, 0, -step_ratio)`, so `timesteps` may gain one extra `-1`.

**2. Sigma interpolation** (`interpolation_type`):

- `"linear"`: linearly interpolate \(\sigma_{\text{full}}\) over integer indices \(0..T-1\) at the `timesteps` positions;
- `"log_linear"`: \(\exp\big(\text{linspace}(\log\sigma_{\text{full}}[T-1],\ \log\sigma_{\text{full}}[0],\ N+1)\big)\). This branch **ignores the timestep grid entirely** and produces \(N+1\) points.

**3. Sigma converters** (`use_karras_sigmas` / `use_exponential_sigmas` / `use_beta_sigmas`, at most one enabled):

- Karras: \(\sigma_k = \big(\text{hi}^{1/\rho} + r_k\,(\text{lo}^{1/\rho} - \text{hi}^{1/\rho})\big)^{\rho}\) with \(r_k = \text{linspace}(0,1,N)_k\) and \(\rho = 7\);
- exponential: \(\sigma_k = \exp\big(\text{linspace}(\log\text{hi}, \log\text{lo}, N)_k\big)\).

Their endpoints default to the two ends of the previous \(\sigma\) sequence: \(\text{hi} = \) the first (noisy-end) entry and \(\text{lo} = \) the last (clean-end) entry, each overridable by `sigma_max` / `sigma_min`. Afterwards every \(\sigma\) must be mapped back onto the time axis: let \(u_k = \log\sigma_k\) and \(\ell_i = \log\sigma_{\text{full}}[i]\) (increasing), take the largest index `low` with \(\ell_{\text{low}} \le u_k\) (clamped to \([0, T-2]\)) and interpolate in log space

\[ w = \operatorname{clamp}\!\left(\frac{\ell_{\text{low}}-u_k}{\ell_{\text{low}}-\ell_{\text{low}+1}},\,0,\,1\right),\qquad t_k = (1-w)\cdot\text{low} + w\cdot(\text{low}+1) \]

**4. Terminal sigma** (`final_sigmas_type`): `"zero"` appends \(0\); `"sigma_min"` appends \(\sqrt{(1-\bar\alpha_0)/\bar\alpha_0}\), i.e. the \(\sigma\) at the cleanest end of the training grid. The terminal value is appended at the end, so `sigmas` usually has one more entry than `timesteps`; with `log_linear` and no converter it has two more.

Return `(timesteps, sigmas)`, both float32 tensors.

## Example

With `alphas_cumprod = [0.99, 0.9, 0.7, 0.5, 0.3, 0.15, 0.06, 0.02, 0.005, 0.001]` (\(T=10\)) and \(N=3\):

```
linspace    -> timesteps = [9.0, 4.5, 0.0]       sigmas = [31.606960, 1.954001, 0.100504, 0.0]
leading     -> timesteps = [6.0, 3.0, 0.0]       sigmas = [3.958114, 1.0, 0.100504, 0.0]
trailing    -> timesteps = [9.0, 6.0, 2.0]       sigmas = [31.606960, 3.958114, 0.654654, 0.0]
karras      -> timesteps = [9.0, 5.561106, 0.0]  sigmas = [31.606960, 3.166429, 0.100504, 0.0]
exponential -> timesteps = [9.0, 4.347704, 0.0]  sigmas = [31.606960, 1.782307, 0.100504, 0.0]
log_linear  -> timesteps = [9.0, 4.5, 0.0]       sigmas = [31.606960, 4.647819, 0.683464, 0.100504, 0.0]
```

The sigma baseline here is \(\sigma_{\text{full}} = [0.100504, 0.333333, 0.654654, 1.0, 1.527525, 2.380476, 3.958114, 7.0, 14.106736, 31.606961]\).

- `linspace` falls at 4.5, between indices 4 and 5, giving \((1.527525+2.380476)/2 = 1.954001\);
- `leading`'s integer ratio is \(10//3=3\), so the zero-based grid \([0,3,6]\) flips to \([6,3,0]\) — the first entry is 6, not 9;
- `trailing`'s true ratio is \(10/3\); the grid \([10,\,6.67,\,3.33]\) rounds to \([10,7,3]\) and minus one gives \([9,6,2]\);
- the `karras` / `exponential` sigmas depend only on the two endpoints and the point count, so their timesteps are non-integer values recovered by inversion;
- `log_linear` disregards the \(N\) timesteps altogether and emits \(N+1=4\) sigmas, plus the terminal one for 5 total; its `timesteps` are still the ordinary `linspace` values.

With `final_sigmas_type="sigma_min"` and \(N=4\), `sigmas = [31.606960, 3.958114, 1.0, 0.100504, 0.100504]` — the last two coincide because the terminal value is exactly the \(\sigma\) at the cleanest end of the training grid.

## Constraints

- `alphas_cumprod` must be a 1-D floating-point tensor of length \(T\ge 2\), with all entries finite and strictly inside \((0,1)\);
- `num_inference_steps` is an integer (`bool` does not count) with \(1\le N\le T\); `steps_offset` is a non-negative integer that only affects `leading`;
- an unsupported value of `timestep_spacing`, `interpolation_type` or `final_sigmas_type` raises `ValueError`;
- enabling more than one sigma converter raises `ValueError`; `use_beta_sigmas=True` raises `ValueError` directly (upstream needs `scipy.stats.beta.ppf`, which this problem does not pull in);
- when provided, `sigma_min` / `sigma_max` must be finite positive numbers with `sigma_min < sigma_max`;
- compute in float64 internally and return both `timesteps` and `sigmas` as float32.
