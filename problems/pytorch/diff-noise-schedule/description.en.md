# Diffusion: Linear / Cosine Noise Schedule ᾱ Sequence

## Problem

Every coefficient of the forward noising process \(q(x_t|x_0)=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon\) is determined by a single scalar sequence \(\bar\alpha_t\) (the fraction of signal retained after \(t\) steps). It comes from the product of the per-step "remaining variance" \(\beta_t\): \(\alpha_t=1-\beta_t\), \(\bar\alpha_t=\prod_{s=1}^{t}\alpha_s\). This problem generates the \(\bar\alpha\) sequence according to two mainstream \(\beta\) schedules.

- **linear** (Ho et al. DDPM): \(\beta_t\) rises linearly from \(1\times10^{-4}\) to \(0.02\) over \(T\) steps, i.e. `linspace(1e-4, 0.02, T)`.
- **cosine** (Nichol & Dhariwal, arXiv:2102.09672): set the offset \(s=0.008\) and \(f(u)=\cos\!\bigl(\tfrac{u/T+s}{1+s}\cdot\tfrac{\pi}{2}\bigr)^2\); for \(t=0,1,\dots,T\) take the raw cumulative product \(\tilde a_t=f(t)/f(0)\) (\(\tilde a_0=1\)), then \(\beta_t=1-\tilde a_t/\tilde a_{t-1}\) and **clip to \([0,0.999]\)**.

Both schedules first produce a \(\beta\) of length \(T\), then \(\bar\alpha_t=\prod_{s\le t}(1-\beta_s)\).

## Rules

The inputs are a positive integer `T` and a string `kind` (`"linear"` or `"cosine"`). Return a float32 tensor `alpha_bar` of length \(T\), where `alpha_bar[t-1]` = \(\bar\alpha_t\). Ready-made diffusion libraries are forbidden.

## Examples

`T=4, kind="linear"`:
```
alpha_bar = [0.9998999834060669, 0.9931672811508179, 0.9798920154571533, 0.9602941870689392]
```
`T=4, kind="cosine"`:
```
alpha_bar = [0.8470122218132019, 0.49384361505508423, 0.1442720741033554, 0.00014427020505536348]
```

## Constraints

- `T>=2`, float32; `kind` is only `"linear"` / `"cosine"`.
- `alpha_bar` is monotonically non-increasing and its first entry is `<1` (\(\bar\alpha_1=1-\beta_1\)).
- For cosine you must first generate \(T+1\) values of \(\tilde a\) for \(t=0..T\), compute the \(T\) values of \(\beta\) by adjacent differencing and clip them, and only then multiply back into \(\bar\alpha\) (the clipped \(\beta\) is what feeds the cumulative product).
