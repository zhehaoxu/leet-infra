# Diffusion: SEG Query Smoothing (Square Gaussian Blur vs Mean Collapse)

## Problem Statement

**Smoothed Energy Guidance** (SEG) smooths attention during denoising: the **query projection** of a few self-attention layers is replaced by its smoothed version, that degraded branch plays the role of the bad model, and guidance then pushes away along "conditional minus degraded". Smoothing comes in two settings — **infinite blur** (collapse the query to an equal-weight average, i.e. make every query point the same direction) and **finite Gaussian blur** (a 2-D Gaussian convolution over the square token grid).

This problem isolates the piece that is easiest to get wrong: given one query projection, smooth it into a same-shaped tensor following the upstream rules. What is under test is the exact ordering of grid reshape, kernel odd-ification and clamping, reflect padding, and per-channel convolution.

## Rules

Input `query` has shape \((B, S, D)\): dim 0 batch, dim 1 tokens, dim 2 channels. Following SEG's square-image assumption, only the **first** \(n^2\) tokens are used (\(n=\lfloor\sqrt{S}\rfloor\)); they are reshaped into a \((B, D, n, n)\) feature map and smoothed per channel, then written back in the original order. The remaining \(S-n^2\) tokens are **left untouched**. Return a new float32 tensor of the same shape; the input must not be modified in place.

Let `sigma = blur_sigma` and `thr = blur_threshold_inf`.

**Mean branch** (\(\sigma \le \text{thr}\) — note the "no greater than" goes here): each \((b, d)\) feature map is replaced wholesale by the arithmetic mean of its \(n^2\) entries.

**Convolution branch** (\(\sigma > \text{thr}\)):

\[ k_0 = \lceil 6\sigma\rceil + 1 - \big(\lceil 6\sigma\rceil \bmod 2\big),\qquad k = \min\big(k_0,\ n - (n \bmod 2 - 1)\big) \]

\(k_0\) is first pushed to an odd number; the clamp ceiling is \(n+1\) for even \(n\) and \(n\) for odd \(n\) (so \(k\) stays odd yet within the grid scale). Then

\[ h = \frac{k-1}{2},\qquad x_a = -h + \frac{2h}{k-1}\,a,\qquad p_a = \frac{e^{-\frac12 (x_a/\sigma)^2}}{\sum_b e^{-\frac12 (x_b/\sigma)^2}} \]

\[ \text{out}_{b,d,i,j} = \sum_{a,c} p_a\,p_c \cdot \text{pad}_{\text{reflect}}\big(\text{patch}_{b,d}\big)_{i-\lfloor k/2\rfloor+a,\ j-\lfloor k/2\rfloor+c} \]

That is the outer product of the 1-D Gaussian, applied as a per-channel (depthwise) convolution with \(\lfloor k/2 \rfloor\) reflect padding on each side. For \(k=1\) we get \(x=[0]\), kernel \([1]\), and the output equals the input.

## Example

\(B=1\), \(D=2\), \(S=10\) (\(n=3\): the first 9 tokens are smoothed, the last one is a tail), token \(t\) holding \([t,\ 10+t]\):

```
sigma=1.0, thr=9999 (mean branch)
  token 0..8 = [ 4.000000, 14.000000]
  token 9     = [ 9.000000, 19.000000]     <- tail kept verbatim

sigma=50000, thr=9999 (conv branch, k clamped to 3)
  token 0 = [  2.666667, 12.666667]   token 1 = [  3.000000, 13.000000]
  token 2 = [  3.333333, 13.333333]   token 3 = [  3.666667, 13.666667]
  token 4 = [  4.000000, 14.000000]   token 5 = [  4.333333, 14.333333]
  token 6 = [  4.666667, 14.666667]   token 7 = [  5.000000, 15.000000]
  token 8 = [  5.333333, 15.333333]   token 9 = [  9.000000, 19.000000]
```

## Constraints

- `query` must be a 3-D tensor, else raise `ValueError`; with \(S=0\) there is no grid to build, which also raises `ValueError`.
- `blur_sigma` is a finite non-negative number and `blur_threshold_inf` a finite non-negative number; `bool`, strings, `nan` and `inf` all raise `ValueError`.
- Smoothing is independent per \((b, d)\): different channels have different means, so never average across channels.
- Padding is `reflect` (it does not repeat the edge value) — not `replicate`, and not zero padding.
- For even \(n\) the ceiling is \(n+1\), so \(k\) may exceed the grid side; the padding width is then \(\lfloor k/2\rfloor\).
- Compute in float64 internally if you like, but return float32.
