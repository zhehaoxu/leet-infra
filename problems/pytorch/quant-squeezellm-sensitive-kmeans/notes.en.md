# Quantization: SqueezeLLM Sensitivity-Weighted Non-Uniform Codebook — Algorithm Notes

## 1. Why the grid should be "non-uniform + weighted"

Uniform quantization (`symmetric-quantize` and friends) assumes weights are equally important and uniformly distributed. Two realities break that:

- **Non-uniform distribution**: LLM weights are roughly Gaussian — dense in the middle, sparse in the tails — so an equally spaced grid wastes resolution near 0;
- **Non-uniform importance**: SqueezeLLM (Kim et al., ICML 2023, arXiv:2306.07629) scores each weight by Fisher information (the second-order sensitivity of the output); a quantization error on a highly sensitive weight is amplified. The paper approximates sensitivity as the square of \(\partial \mathcal L/\partial w\) (one backward pass), and builds **sensitivity-aware k-means**: the weighted objective pulls grid points toward high-sensitivity, high-density regions.

## 2. The two steps of weighted Lloyd (judged point)

Objective

\[
\min_{\{c_j\}, a}\ \sum_i s_i\,(w_i-c_{a_i})^2
\]

- **Assignment**: with centers fixed, take the nearest cluster per point — \(a_i=\arg\min_j|w_i-c_j|\), ties to the **smaller** cluster id (`argmin` semantics);
- **Update**: with assignments fixed, each center's optimum is the **weighted mean**

\[
c_j^\star=\frac{\sum_{i:a_i=j}s_iw_i}{\sum_{i:a_i=j}s_i}
\]

  (differentiate w.r.t. \(c_j\): \(-2\sum_{i:a_i=j}s_i(w_i-c_j)=0\)). This is the core assertion of `check_weighted_centroids`: every returned center must equal its cluster's weighted mean (tolerance \(10^{-4}\)) — using the arithmetic mean or forgetting to multiply by \(s_i\) fails immediately.
- **Empty clusters**: keep the previous center (otherwise NaN or center collapse); positional-quantile initialization guarantees no cluster is empty at the start.

## 3. Monotonicity and sorting

1-D Lloyd decreases the objective every round (assignment and weighted-mean steps are each optimal), so the iteration is monotone. After each update the centers are sorted ascending (assignments are recomputed next round), which guarantees:
- `centroids` is monotone, enabling downstream "nearest code by binary search";
- the correspondence `deq == centroids[assign]` is explicit (`check_weighted_centroids` asserts it).

`check_beats_uniform_grid` provides the macro comparison: with the same k, the learned grid's weighted MSE must be far below the equally spaced `linspace(min, max, k)` grid. That is the point of a non-uniform grid, and it also rejects the degenerate "just use k quantiles" solution.

## 4. Targeted assertions for sensitivity semantics

`check_sensitivity_shifts_centroid` compares two sensitivity profiles with k=2 and `w=[0,1,2,3,4]`:

- base: all-ones `sens` → centers at 1 and 3 (left cluster holds 0,1,2);
- hi: raise the sensitivity of 4 only to 100 → the right center is pulled toward 4 (weighted mean \((3+4\cdot100)/101\approx3.99\)), while the left center does not move.

This verifies both the direction of the weighted mean and the locality of "sensitivity only affects its own cluster".

## 5. Division of labor within the collection

- `quant-hqq-scale-solve`: **uniform** grid, optimizes only the scale (closed form);
- `quant-nf4-quantile-grid`: grid fixed by **normal quantiles**, optimizes only the encoding;
- This problem: grid positions are **learned from data + sensitivity** (non-parametric, non-closed-form) — the only "learned" codebook among the three;
- `gptq-column-quantize`: cross-column error compensation, does not change the grid distribution.

## 6. Verification

- `check_example`: 4 points, k=2, one round, hand-computed.
- `check_weighted_centroids`: weighted means + ascending order + `deq` consistency.
- `check_beats_uniform_grid`: weighted MSE < 50% of the uniform grid's.
- `check_uniform_sens_matches_lloyd`: all-ones `sens` matches scalar recomputation.
- `check_sensitivity_shifts_centroid`: targeted shift plus five kinds of invalid arguments (k/iters/sens).
