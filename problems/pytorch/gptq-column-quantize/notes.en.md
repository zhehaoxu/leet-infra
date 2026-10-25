# Solution Notes · GPTQ Column-wise Error Compensation

## From "rounding each independently" to "rounding cooperatively"

RTN rounds every weight independently, so the errors pile up with no one to fix them. But in a linear map $y=Wx$, the weights on the same **input channel** are coupled: the component lost when one weight is quantized can be recovered by adjusting the weights that have **not yet been quantized**, keeping the output $y$ as unchanged as possible. GPTQ turns this "after-the-fact repair" into a **column-wise pipeline**: quantize from left to right, and after each column is quantized, immediately broadcast its error to all unquantized columns on the right.

## Second-order optimal compensation (the OBQ/GPT lineage)

Write "how to correct the remaining columns after quantizing column $i$ so that the output error is minimal" as a quadratic-form minimization with Hessian $H=2XX^\top$. The closed-form solution is an outer-product update along row $i$ of $H^{-1}$, divided by the diagonal entry $d_i=H^{-1}_{ii}$:

$$W_{q,[:,i+1:]} \;-\!=\; \frac{e_i}{d_i}\otimes H^{-1}_{i,i+1:}$$

where $e_i$ is the quantization error of that column. Intuition: $H^{-1}_{ij}$ measures the correlation between channels $i$ and $j$ — the stronger the correlation, the more worthwhile it is to use $j$ to compensate the error of $i$; $d_i$ is the "curvature" of that channel itself, and the larger the curvature, the more "expensive" it is and the more conservative the compensation.

## Why only one inversion is needed

Naive OBQ recomputes the inverse after every weight is quantized, costing $O(C^3)$ and refreshing it with each column. GPTQ's core engineering contribution: fix the column order and use Cholesky / row selection to compress "$n$ inversions" into **one inversion + reading one row per column**. This problem takes the compensation direction directly from row $i$ of $H^{-1}$, sidestepping the Cholesky implementation details to focus on the algorithmic skeleton, and also making it easy to cross-check against an independent pure-Python implementation (Gauss-Jordan inversion).

## The damping term `damp`

$H=2XX^\top$ has only $n$ samples; when $C>n$ or the samples are degenerate it is rank-deficient and the diagonal entries can be 0, causing division by zero. Adding $\texttt{damp}\cdot I$ (the paper uses roughly 1% relative damping) guarantees invertibility and a stable compensation direction.

## Boundary self-checks

- `check_matches_oracle`: 30 random symmetric positive-definite $H$ matrices, compared element-wise against an independent Gauss-Jordan implementation.
- `check_high_bits_near_identity`: at 16-bit the grid is extremely dense, so the output ≈ input.
- `check_output_on_grid`: compensation only decides "which grid point to quantize to"; the final value of every column still lies strictly in the per-tensor legal grid set — verifying that the error feedback does not bypass the quantizer itself.
