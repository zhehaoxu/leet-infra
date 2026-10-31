# Diffusion Sampling: the Consistency Decoder Schedule and Its Two-Step Denoise

## Description

diffusers' `ConsistencyDecoderScheduler` serves the "consistency decoder": at every step the network already outputs a clean \(x_0\) prediction, and sampling consists of just **two steps** — after the first denoise the result must be **re-noised** to the second timestep, while the second denoise hands the answer over directly. The scheduler does not accept an external \(\bar\alpha\) table; it builds a table of length \(T\) at construction time with `betas_for_alpha_bar` and then derives the three coefficient arrays `c_skip / c_out / c_in` from it.

This task tests both halves together: rebuild the whole schedule under the requested transform, then run the two steps. The two network outputs and the Gaussian noise are given as explicit arguments — no real network and no random number generation anywhere.

## Rules

**Schedule** (every intermediate value computed in float64):

Pick \(\operatorname{ab}(t)\) by `alpha_transform`:

- `cosine`: \(\operatorname{ab}(t)=\cos\!\big(\frac{t+0.008}{1.008}\cdot\frac{\pi}{2}\big)^2\)
- `laplace`: \(\lambda=-\frac12\cdot\operatorname{copysign}(1,\,0.5-t)\cdot\ln(1-2|0.5-t|+10^{-6})\), \(S=e^{\lambda}\), \(\operatorname{ab}(t)=\sqrt{S/(1+S)}\)
- `exp`: \(\operatorname{ab}(t)=e^{-12t}\)

For \(i=0\dots T-1\):

\[ \beta_i=\min\!\Big(1-\frac{\operatorname{ab}((i+1)/T)}{\operatorname{ab}(i/T)},\ \text{max\_beta}\Big),\qquad \bar\alpha_i=\prod_{j\le i}(1-\beta_j),\qquad \sigma_i=\sqrt{\frac{1}{\bar\alpha_i}-1} \]

\[ c_{\text{skip}}^{(i)}=\bar\alpha_i^{-1/2}\cdot\frac{\sigma_d^2}{\sigma_i^2+\sigma_d^2},\qquad c_{\text{out}}^{(i)}=\frac{\sigma_i\,\sigma_d}{\sqrt{\sigma_i^2+\sigma_d^2}},\qquad c_{\text{in}}^{(i)}=\frac{\bar\alpha_i^{-1/2}}{\sqrt{\sigma_i^2+\sigma_d^2}} \]

**The two steps**, with \(t_0=\)`timesteps[0]` and \(t_1=\)`timesteps[1]`:

\[ \text{scaled\_first}=x_{\text{init}}\cdot c_{\text{in}}^{(t_0)} \]
\[ x_0^{(1)}=c_{\text{out}}^{(t_0)}\cdot F_1+c_{\text{skip}}^{(t_0)}\cdot x_{\text{init}} \]
\[ x_{\text{second}}=\sqrt{\bar\alpha_{t_1}}\;x_0^{(1)}+\sqrt{1-\bar\alpha_{t_1}}\;\cdot\text{noise} \]
\[ \text{scaled\_second}=x_{\text{second}}\cdot c_{\text{in}}^{(t_1)},\qquad x_{\text{final}}=c_{\text{out}}^{(t_1)}\cdot F_2+c_{\text{skip}}^{(t_1)}\cdot x_{\text{second}} \]

Three traps. First, the re-noising uses the \(\sqrt{\bar\alpha}/\sqrt{1-\bar\alpha}\) of the **next** timestep \(t_1\), not of the current step \(t_0\). Second, the last step **does not re-noise**: \(x_{\text{final}}\) is the \(x_0\) prediction itself, so `noise` influences the result exactly once, through \(x_{\text{second}}\). Third, the \(\bar\alpha^{-1/2}\) inside \(c_{\text{skip}}\) and \(c_{\text{in}}\) is the **reciprocal** square root (the inverse normalisation back to \(x_0\) scale); writing \(\sqrt{\bar\alpha}\) instead deflates both coefficients.

Return the 4-tuple `(scaled_first, x_second, scaled_second, x_final)`, all float32 tensors with the same shape as `x_init`.

## Example

With `T=1024`, `alpha_transform="cosine"`, `sigma_data=0.5`, `timesteps=(1008, 512)` and `max_beta=0.999`, the scheduled quantities at the two timesteps are:

```
t=1008  abar=5.210667e-04  sqrt_abar=0.022827  sqrt_1_minus_abar=0.999739
        sigma=43.7966  c_skip=0.005709  c_out=0.499967  c_in=1.000195
t=512   abar=4.923217e-01  sqrt_abar=0.701656  sqrt_1_minus_abar=0.712515
        sigma=1.0155  c_skip=0.278100  c_out=0.448573  c_in=1.259124
```

Given

```
x_init = [[0.8, -0.3], [1.2, 0.1], [0.0, 1.5]]
f_first = [[-0.5, 1.0], [0.25, -0.75], [1.0, 0.5]]
f_second = [[0.3, -0.2], [-0.6, 0.9], [0.15, 0.45]]
noise = [[1.0, -1.0], [0.5, 0.0], [-0.25, 0.75]]
```

with `num_train_timesteps=1024` (everything else default):

```
scaled_first  = [[0.800156, -0.300059], [1.200235, 0.100020], [0.000000, 1.500293]]
x_second      = [[0.540317, -0.362912], [0.448766, -0.262703], [0.172676, 0.715798]]
scaled_second = [[0.680326, -0.456951], [0.565052, -0.330776], [0.217421, 0.901278]]
x_final       = [[0.284834, -0.190640], [-0.144342, 0.330657], [0.115307, 0.400921]]
```

The raw `beta_1023` of the `cosine` transform is \(1.0\) and gets clamped to `max_beta=0.999`; lowering `max_beta` to \(0.05\) moves both \(c_{\text{in}}\) and the re-noising coefficients, so all four outputs shift.

## Constraints

- `x_init / f_first / f_second / noise` must share one shape (dimension 0 is the batch) and hold only finite values, otherwise `ValueError`.
- `num_train_timesteps` is a Python int (`bool` is illegal) and `>= 2`; `timesteps` holds **exactly two** Python ints inside \([0, T)\) in strictly decreasing order; `sigma_data > 0`; `max_beta ∈ (0, 1]`; `alpha_transform` must be one of the three known names.
- If the schedule degenerates so that some \(\bar\alpha_i \le 0\) (e.g. `cosine` with `max_beta=1.0`, where the tail \(\beta\) hits the ceiling), \(\sigma_i\) diverges and a `ValueError` is raised.
- Intermediates in float64, all four returns in float32. `import diffusers` is forbidden.
