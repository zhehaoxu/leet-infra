# Quantization: SqueezeLLM Sensitivity-Weighted Non-Uniform Codebook

## Problem

Uniform quantization spaces its grid evenly, but weights do not carry **equal importance**: after fine-tuning, a few weights dominate the output (large Fisher information / second-order sensitivity), and a quantization error on them costs far more than on irrelevant ones. **SqueezeLLM** (Kim et al., ICML 2023, arXiv:2306.07629) assigns each weight a sensitivity \(s_i\) and learns a non-uniform grid with **weighted k-means**: the grid automatically densifies where sensitivity is high and samples are dense.

This problem implements the weighted Lloyd algorithm in 1-D:

\[
\min_{\{c_j\},\ a_i}\ \sum_i s_i\,(w_i-c_{a_i})^2
\]

Alternating two steps:

1. **Assignment**: \(a_i=\arg\min_j |w_i-c_j|\) (ties go to the smaller cluster id);
2. **Update**: \(c_j=\dfrac{\sum_{i:a_i=j}s_i\,w_i}{\sum_{i:a_i=j}s_i}\) (empty clusters keep their previous center).

After each update the centers are **sorted ascending** (assignments are recomputed in the next iteration), so the output centroids are monotone.

**Initialization** (deterministic): sort `w` ascending; the \(j\)-th center is the \(\lfloor (j+0.5)N/k\rfloor\)-th sorted element (positional quantiles, ignoring weights).

## Task

Implement `solve(w, sens, k, iters)`:

- Input: `w` float32 `(N,)`, `sens` float32 `(N,)` (non-negative, positive sum), `k` an integer with `1 ≤ k ≤ N`, `iters` a non-negative integer.
- Output `(centroids, assign, deq)`: `centroids` float32 `(k,)` ascending; `assign` int64 `(N,)` with values `0..k-1` indexing the ascending centers; `deq` float32 `(N,)` satisfying `deq[i] == centroids[assign[i]]`.

**Conventions**: float64 internally, float32 output; tolerance \(10^{-5}\). Out-of-range `k`, `iters < 0`, or `sens` containing negatives / summing to zero raises `ValueError`.

## Example

```
w = [0, 1, 2, 3], sens = [1, 1, 1, 1], k = 2, iters = 1
Initial centers: position floor(0.5*4/2)=1 → w[1]=1; floor(1.5*4/2)=3 → w[3]=3
Assignment: 2 is equidistant from 1 and 3, ties go to the smaller cluster → [0,0,0,1] (left cluster holds 0,1,2)
Updated centers: left mean 1, right mean 3 (all-ones sens degenerates to the arithmetic mean)
Output: centroids=[1.0, 3.0], assign=[0,0,0,1], deq=[1.0, 1.0, 1.0, 3.0]
```

## Constraints

- Centers must be **weighted** means (denominator \(\sum s_i\)), not arithmetic means — only all-ones `sens` degenerates to plain k-means.
- After each update, sort the centers ascending; `assign` is given in terms of the sorted cluster ids.
- With `k == 1`, the center equals the weighted mean of everything (independent of `iters`).
