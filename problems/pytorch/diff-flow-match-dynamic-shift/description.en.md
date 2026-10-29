# Flow Match Sampling: Dynamic Time Shifting and Terminal Stretching

## Problem statement

Rectified Flow / Flow Matching samplers treat the time axis \(t\in[0,1]\) as "noise level" (1 is pure noise, 0 is clean data), so the sigma sequence and the timestep scale are two representations of the same numbers (`timesteps = sigmas * num_train_timesteps`). Two steps reshape that distribution: **time shift** pushes the whole grid towards the noisy end based on resolution or a user parameter, and **terminal stretch** then linearly stretches the grid so it terminates at a prescribed sigma (the `shift_terminal` scheme introduced by LTX-Video). Both live in `FlowMatchEulerDiscreteScheduler`, and their order cannot be swapped.

Implement the whole path from "\(N\) evenly spaced sigmas" to the final timesteps and sigmas: static or dynamic shifting (exponential / linear), an optional terminal stretch, an optional sigma inversion, and the terminal sigma append.

## Rules

Let \(T = \) `num_train_timesteps` (default 1000).

**1. Initial grid**: `sigmas = linspace(sigma_max, sigma_min, N)`, descending from the noisy end.

**2. Shifting**:

- `use_dynamic_shifting=False`: static shift \(s' = \dfrac{\text{shift}\cdot s}{1+(\text{shift}-1)\,s}\);
- `use_dynamic_shifting=True`: `mu` is required, applied element-wise per `time_shift_type`:
  - `"exponential"`: \(s' = \dfrac{e^{\mu}}{e^{\mu} + (1/s - 1)}\)
  - `"linear"`: \(s' = \dfrac{\mu}{\mu + (1/s - 1)}\)

The three forms coincide when \(\text{shift}=s\), \(\mu=\ln s\) (exponential) or \(\mu=s\) (linear) — the static \(s\cdot x/(1+(s-1)x)\) equals the linear \(\mu x/(1+(\mu-1)x)\) exactly — but both the statement and the implementation present them as three independent upstream formulas.

**3. Terminal stretch** (when `shift_terminal` is given): \(\text{scale} = \dfrac{1-s'[-1]}{1-\text{shift\_terminal}}\), \(s'' = 1 - \dfrac{1-s'}{\text{scale}}\). This must happen after shifting and before deriving the timesteps: computing the scale first and shifting afterwards leaves the last entry unequal to `shift_terminal`.

**4. Timesteps**: \(t = s''\cdot T\).

**5. Sigma inversion** (`invert_sigmas=True`, used by Mochi): further set \(s = 1 - s''\) and \(t = s\cdot T\), and append \(1\) as the terminal value instead of \(0\).

Return `(timesteps, sigmas)`: `timesteps` has length \(N\), `sigmas` has length \(N+1\) (the last entry being the terminal value), both float32.

## Example

With \(N=4\), `sigma_min=0`, `sigma_max=1`, \(T=1000\):

```
shift=2               -> timesteps = [1000.0, 800.0, 500.0, 0.0]           sigmas = [1.0, 0.8, 0.5, 0.0, 0.0]
exponential, mu=1     -> timesteps = [1000.0, 844.637573, 576.116882, 0.0] sigmas = [1.0, 0.844638, 0.576117, 0.0, 0.0]
linear, mu=2          -> timesteps = [1000.0, 800.0, 500.0, 0.0]           sigmas = [1.0, 0.8, 0.5, 0.0, 0.0]
shift=2, terminal=0.2 -> timesteps = [1000.0, 840.0, 600.0, 200.0]         sigmas = [1.0, 0.84, 0.6, 0.2, 0.0]
shift=2, invert       -> timesteps = [0.0, 200.0, 500.0, 1000.0]           sigmas = [0.0, 0.2, 0.5, 1.0, 1.0]
shift=2, terminal+inv -> timesteps = [0.0, 160.0, 400.0, 800.0]            sigmas = [0.0, 0.16, 0.4, 0.8, 1.0]
```

The initial grid is \([1,\,2/3,\,1/3,\,0]\). Static `shift=2` and linear `mu=2` produce the same row (\(\mu/(\mu+(1/s-1)) = 2s/(1+s)\)), whereas exponential `mu=1` equals `shift` \(=e\approx2.718\) and therefore sits closer to the noisy end in the middle. With `terminal=0.2` the scale is \(1/0.8=1.25\), mapping \([1,0.8,0.5,0]\) onto \([1,0.84,0.6,0.2]\) so that the last entry lands exactly on 0.2. After inversion the grid increases monotonically and the terminal value appended is 1.

One more boundary case: with `sigma_min=0.1`, `sigma_max=0.9`, `shift=2` and \(N=4\), `sigmas = [0.947368, 0.77551, 0.536585, 0.181818, 0.0]` — the endpoints are no longer 1 and 0, which shows they are only pinned when `sigma_min` is left at 0.

## Constraints

- `num_inference_steps` and `num_train_timesteps` are positive integers (`bool` does not count); with `shift_terminal` set, \(N\ge2\) is required (otherwise the scale collapses to 0 and the stretch yields NaN).
- `shift` must be a finite positive number; `sigma_min` / `sigma_max` must satisfy \(0\le\sigma_{\min}<\sigma_{\max}\le1\).
- `use_dynamic_shifting=True` requires `mu` and it must be finite; `time_shift_type="linear"` additionally requires \(\mu>0\) (\(\mu=0\) collapses every sigma to 0).
- `shift_terminal` must lie in the open interval \((0,1)\); `time_shift_type` accepts only `"exponential"` / `"linear"`, anything else raises `ValueError`.
- Compute in float64 internally and return both `timesteps` and `sigmas` as float32.
