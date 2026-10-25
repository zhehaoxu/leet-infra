# Explained: Product Quantization Encoding

## Source
Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search* (IEEE TPAMI 2011, inria-00514462). PQ builds a codebook of size \(2^b\) for each \(d\)-dimensional subspace and runs VQ independently; encoding is simply taking the nearest codeword per segment, exactly what Faiss `IndexPQ::compute_codes` does.

## Key points and pitfalls
1. **Segments must follow the codebook's subspace partition**: segment \(j\) = `x[:, j*d:(j+1)*d]` is compared only with `codebook[j]` (shape \((k_{sub}, d)\)). Using the wrong codebook (e.g. `codebook[0]` for every segment) is a classic bug.
2. **Cartesian product, not a single codebook**: the overall approximation error is the **sum** of the per-segment errors, which is why PQ reaches an effective codebook size of \(k=m\cdot k_{sub}^{m}\) with only \(m\cdot\log_2 k_{sub}\) bits.
3. **Sub-codes are "indices", not "vectors"**: the output is an \((n,m)\) integer index.
4. Squared Euclidean plus smallest index on ties, same origin as IVF assignment, only done independently per segment.

## Comparison reference
The reference solution broadcasts per segment `((sub[:,None,:]-cb[j][None,:,:])**2).sum(-1).argmin(1)` into an `(n,m)` result. The judging `_oracle` uses quadruple pure-Python loops (sample × segment × codeword × dim), explicitly accumulating squared differences and updating only on strict `<` (different structure, numerically equivalent). There are also assertions that "shape=(n,m)", "sub-codes ∈[0,k_sub)", and "a vector assembled from known codewords re-encodes to those codewords (exact reconstruction)".

## Complexity
\(O(n\cdot m\cdot k_{sub}\cdot d)\).
