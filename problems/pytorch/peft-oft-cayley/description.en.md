# PEFT: OFT Orthogonal Finetuning via Cayley Transform

## Problem

**OFT** (Orthogonal Finetuning, Qiu et al., arXiv:2306.03889; PEFT library `tuners/oft`) is motivated by the fact that in text-to-image models a neuron's **direction** encodes semantic structure, so fine-tuning should **preserve norms and rotate directions** — instead of learning an additive increment, it learns an **orthogonal matrix** to rotate the weights:

\[W' = R\,W_0,\qquad R^\top R = I\]

To constrain "an arbitrary matrix" into "an orthogonal matrix", OFT uses the **Cayley parametrization**: each block first builds a **skew-symmetric matrix** \(Q\) (\(Q^\top=-Q\)) from a set of upper-triangular parameters \(v\), then applies the Cayley transform

\[R_b = (I+Q)\,(I-Q)^{-1}\]

The Cayley transform of a skew-symmetric matrix is always orthogonal (proof in the notes), and \(v=0\Rightarrow Q=0\Rightarrow R_b=I\) — with zero init the adapter is an identity rotation, echoing the LoRA family's "zero init does not perturb the base".

Real layers are often too wide, so OFT splits the output dimension into several **blocks**, each rotated independently: with block size \(h\) and block count \(n\), the per-block \(R_b\) are assembled along the diagonal into a block-diagonal orthogonal matrix

\[R = \operatorname{blockdiag}(R_1,\dots,R_n)\in\mathbb R^{(n h)\times(n h)}\]

which then multiplies the weight. PEFT implementation details: each block's parameters are its strictly-upper-triangular \(h(h-1)/2\) scalars, in `torch.triu_indices(h, h, 1)` row-major order; `Q` is obtained by "fill the upper triangle with \(v\), then subtract the transpose"; PEFT defaults to a Neumann series \(I+2Q+2Q^2+\cdots\) approximating the Cayley transform (same limit, see the notes); this problem uses the closed form directly.

## Task

Given the block parameters `blocks` (shape `(n_blocks, n_elem)`, `n_elem = h(h-1)/2`, with \(h\) uniquely inferred from it) and the frozen base `W0` (shape `(n_blocks*h, d_in)`):

1. Per block: fill the upper triangle with \(v\) → subtract the transpose to get skew-symmetric \(Q\) → Cayley transform \(R_b=(I+Q)(I-Q)^{-1}\);
2. Assemble the block-diagonal \(R\) and output **`(R, W')`**, where \(W'=R@W_0\).

Both `R` and `W'` are float32 tensors; results are judged at tolerance \(10^{-5}\).

## Hints

- Recovering \(h\): \(h = \dfrac{1+\sqrt{1+8\,\texttt{n\_elem}}}{2}\).
- The inversion may use `torch.linalg.inv` or solve a linear system; the two are numerically equivalent.
- Self-checks: `R @ R.T ≈ I`, and the orthogonal map preserves the Frobenius norm \(\|W'\|_F=\|W_0\|_F\).
