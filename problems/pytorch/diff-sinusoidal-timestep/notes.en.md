# In Depth: Sinusoidal Timestep Embedding

## Source
In the appendix of Ho et al., *Denoising Diffusion Probabilistic Models* (arXiv:2006.11239), \(t\) is encoded with fixed sinusoidal features from the same lineage as Vaswani's Transformer positional encoding; `timestep_embedding` in openai/improved-diffusion is exactly this implementation. A set of frequencies spanning several orders of magnitude expands the scalar \(t\) into a high-dimensional, smooth, differentiable representation.

## Key Points and Pitfalls
1. **half and the frequency exponent**: \(\text{freq}_i=10000^{-i/\text{half}}\), the denominator of the exponent is `half` (equivalent to \(2i/\text{dim}\)). Using the wrong denominator compresses the frequencies across the board.
2. **Concatenation order**: this problem specifies "the first half all sin, the second half all cos" (`cat([sin,cos])`), not interleaved sin/cos. A different order makes the vectors differ element-wise — follow the statement exactly.
3. **\(t=0\) probe**: \(\sin 0=0\), \(\cos 0=1\), so you can immediately check whether sin/cos were placed in the wrong halves.
4. float32: `arange` must explicitly use `dtype=torch.float32`, otherwise the default integer type raises an error.

## Cross-check Approach
The reference solution is vectorized: `freqs=exp(-log(10000)*arange(half)/half)`, `cat([sin(t*freqs),cos(t*freqs)])`. The grader's `_oracle` uses Python `math` to compute `angle=t/10000**(i/half)` for each i and fills the sin half and the cos half separately (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`. Additional checks cover the `t=0` boundary and shape/consistency across different dims.

## Complexity
\(O(\text{dim})\), one trigonometric evaluation per frequency.
