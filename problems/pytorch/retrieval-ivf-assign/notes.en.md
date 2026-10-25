# Explained: IVF Coarse Quantizer Assignment

## Source
The **coarse quantizer** in Jégou et al., *Product Quantization for Nearest Neighbor Search* (IEEE TPAMI 2011, arXiv version inria-00514462); the assignment stage of Faiss `IndexIVF`/`faiss::search_preassigned`. IVF dispatches vectors to the nearest coarse centroid to prune the search range, and is the representative of the "coarse filter, then fine rank" ANN paradigm.

## Key points and pitfalls
1. **The distance is squared Euclidean**: \(\|x_i-c_j\|^2=\sum_t(x_{i,t}-c_{j,t})^2\). Taking the square root is monotonic and does not change the argmin; using the square directly is more stable and faster.
2. **Expansion-form trap**: \(\|x\|^2+\|c\|^2-2x\cdot c\) suffers catastrophic cancellation in float32 when x≈c and amplifies the error; a sum of per-coordinate squared differences is more reliable.
3. **Ties go to the smallest index**: `argmin` naturally returns the first minimum (smallest index), equivalent to a hand-written "update only if strictly less".
4. Each row is independent, with no cross-row coupling.

## Comparison reference
The reference solution uses `torch.cdist(x, centroids)` followed by `argmin(dim=1)`. The judging `_oracle` uses triple pure-Python loops that explicitly accumulate per-dimension squared differences and update only on strict `<` (different structure, numerically equivalent). There are also assertions that "it agrees with brute-force per-centroid minimization", "with k=1 the result is always 0", and "the result is an integer in {0..k-1}", plus a vector that coincides exactly with one centroid to verify that the centroid is hit.

## Complexity
\(O(n\,k\,D)\).
