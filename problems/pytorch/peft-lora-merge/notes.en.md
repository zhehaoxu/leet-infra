# LoRA Weight Merging (merge_and_unload) Explained

## Source
Hu et al., arXiv:2106.09685; the API name `merge_and_unload` comes from huggingface/peft. Since \(\Delta W=BA\) has the same shape as \(W_0\), it can be folded losslessly: \(W'=W_0+\frac{\alpha}{r}BA\), after which inference no longer needs \(A,B\).

## Core points and pitfalls
1. **The scale applies only to \(\Delta W\)**: \(W'=W_0+s\,(BA)\), \(s=\alpha/r\). Do not multiply \(s\) into \(W_0\) as well.
2. **Order \(BA\)** (\(d_\text{out}\times r\) times \(r\times d_\text{in}\)); the result \((d_\text{out},d_\text{in})\) aligns with \(W_0\); \(AB\) gives \((r,r)\) and broadcasts incorrectly.
3. **Folding reversibility is limited**: after merging, \(A,B\) cannot be exactly recovered (numerically \(W'-W_0\) is exactly that low-rank \(\Delta W\)); this problem only tests forward merging.

## Division of labor with the "LoRA forward" problem
The forward problem (order 199) tests "no merging, step-by-step side-path addition"; this problem tests "merging into a dense \(W'\)". The two produce element-wise equal outputs for the same \(x\) — this problem uses `check_apply_equivalence` to verify on the spot that \(x@W'^\top =\) the step-by-step forward, pinning down the mathematical consistency of the two problems.

## Grading details
The reference solution explicitly computes \(W_0+s(BA)\); the grader's `_oracle` compares against "element-wise \(W_0[i,j]+\frac{\alpha}{r}\sum_k B[i,k]A[k,j]\)" (a triple-loop structure, different from matrix multiplication), with `allclose(atol=1e-5,rtol=1e-5)`. Additional assertions: `alpha=0` identity, shape consistency, and \(x@W'^\top\) being equivalent to the step-by-step forward.

## Complexity
Merging is \(O(d_\text{in}d_\text{out}r)\) and produces a full \(d_\text{out}\times d_\text{in}\) weight (exactly the trade-off of "more memory at deployment, less at training").
