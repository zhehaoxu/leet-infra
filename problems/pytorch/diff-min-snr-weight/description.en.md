# Diffusion Training: Min-SNR Loss Weighting

## Problem Statement

In diffusion training the signal-to-noise ratio spans several orders of magnitude across timesteps: at the clean end (\(\bar\alpha_t\to1\)) the SNR is huge, the model is nearly copying its input, yet its gradient dominates the total loss. Min-SNR weighting suppresses the loss weight of high-SNR timesteps and leaves low-SNR timesteps untouched:

\[ w_t=\frac{\min(\mathrm{SNR}_t,\gamma)}{\mathrm{SNR}_t}=\min\!\left(1,\frac{\gamma}{\mathrm{SNR}_t}\right) \]

where \(\mathrm{SNR}_t=\bar\alpha_t/(1-\bar\alpha_t)\) and \(\gamma\) caps the SNR (SD3-style training scripts commonly use 5.0). What is capped is the **SNR**, not the weight: the weight never exceeds 1 and only decays as \(\gamma/\mathrm{SNR}\) once the SNR passes \(\gamma\).

Given a schedule's cumulative alphas and a batch of timesteps, compute the SNRs and the corresponding Min-SNR weights.

## Rules

Given a 1-D `alphas_cumprod` (length \(T\)) and 1-D integer `timesteps` (indices):

1. take \(\alpha=\sqrt{\bar\alpha_t}\) and \(\sigma=\sqrt{1-\bar\alpha_t}\);
2. \(\mathrm{SNR}=(\alpha/\sigma)^2\) (equal to \(\bar\alpha_t/(1-\bar\alpha_t)\), but computed through the source's square-root path);
3. \(w=\min(\mathrm{SNR},\gamma)/\mathrm{SNR}\).

Return `(snr, weight)`, both float32 tensors with the shape of `timesteps`.

## Example

`alphas_cumprod = [0.75, 0.5, 0.25, 0.125]`, `timesteps = [0, 1, 2, 3]`, `gamma = 2.0`:

```
snr    = [3.0, 1.0, 0.333333, 0.142857]
weight = [0.666667, 1.0, 1.0, 1.0]
```

The first three timesteps have SNR at most 2, so their weights stay 1; only \(\bar\alpha=0.75\) (SNR=3) is pushed down to \(2/3\).

## Constraints

- `alphas_cumprod` must be a 1-D float tensor with no nan / inf, and every element must lie **strictly** in \((0,1)\); the endpoints 0 and 1 would make the SNR zero or infinite.
- `timesteps` must be a 1-D `int32` / `int64` tensor with values in \([0,T)\); repeats and empty tensors are allowed.
- `gamma` must be a positive finite real number (booleans do not count).
- Raise `ValueError` when any of the above fails.
- Properties: \(w\in(0,1]\); \(w=1\) when \(\mathrm{SNR}\le\gamma\) and \(w=\gamma/\mathrm{SNR}\) when \(\mathrm{SNR}>\gamma\); increasing \(\gamma\) can only increase every weight; walking the schedule from clean to noisy (decreasing \(\bar\alpha\)) makes the SNR decrease and the weight increase monotonically.
