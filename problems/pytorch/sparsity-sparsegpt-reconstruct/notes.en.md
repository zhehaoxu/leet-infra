# Solution Notes: SparseGPT Column-Wise Pruning with Hessian Error Compensation

## Origin
The compensation formula is the closed-form OBS update. SparseGPT (Frantar & Alistarh, *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*, 2023, arXiv 2301.00774) turns it into a one-shot, column-by-column procedure that never recomputes the Hessian, maintaining \(H^{-1}\) incrementally with Cholesky. GPTQ (Frantar et al., 2022) uses the same error-compensation skeleton for quantisation, the difference being that a column is replaced by its quantised value instead of 0. In this collection `sparsity-rowwise-prune` is local magnitude pruning without compensation, and `quant-gptq-column-quantize` is full column-wise quantisation — this problem isolates the "mask first, then push the error rightwards" step.

## Key points and pitfalls
1. **Sequential scan with in-place updates**: `cur` must be the **already updated** column, never the original `weight`, otherwise every leftward compensation is lost.
2. **Divide by the diagonal**: \(e\) is divided by \([H^{-1}]_{jj}\), not by \(|W_{:,j}|\) or the Hessian diagonal \(H_{jj}\); writing the reciprocal of the wrong matrix is the most common formula error.
3. **Error uses current minus masked**: \(cur-Q\) equals `cur` itself where the mask is 0 (the pruned value) and 0 where it is 1, so an all-ones mask gives \(e\equiv0\) and the whole algorithm degenerates to identity.
4. **Outer-product orientation**: \(e\in\mathbb{R}^{R}\) and \([H^{-1}]_{j,j+1:}\in\mathbb{R}^{C-j-1}\) produce a rank-1 correction of shape \((R,C-j-1)\); swapping them either crashes on non-square inputs or broadcasts silently.
5. **The last column has nowhere to push**: its error is dropped, which is why pruning is pushed rightwards and real implementations also reorder columns (block-wise) to control error accumulation.
6. **Masked positions stay exactly 0**: compensation only rewrites unmasked columns and must never refill a column it just zeroed.

## Verification
The reference clones the weight and updates column by column in a Python loop. The judge's oracle recomputes the definition on nested Python lists (sharing no tensor ops with the solution) and compares element-wise. Independent assertions: an all-ones mask reproduces the input exactly; an all-zeros mask yields an all-zero matrix (each column is zeroed when its turn comes); with `Hinv=I` the result is exactly `weight*mask` (the no-coupling degeneration); every masked position is exactly 0; and after building a real Hessian \(H=XX^\top+\lambda I\), \(\|W_{out}X-WX\|_F\) must be **strictly smaller** than the naive mask's \(\|(W\odot m)X-WX\|_F\), proving the compensation actually absorbs error; output dtype/shape are checked too.

## Complexity
\(O(RC^2)\) (one outer-product update of length \(C-j\) per column); memory on the order of `weight`.
