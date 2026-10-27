# PEFT: LoKr Kronecker Factorization

## Problem

**LoKr** (Kronecker Product adaptation, Yeh et al., arXiv:2108.07383; PEFT library `tuners/lokr`) expresses the weight increment as a **Kronecker product** \( \Delta W = W_1 \otimes W_2 \): to split a \(d_\text{out}\times d_\text{in}\) matrix into two factors \(W_1\in\mathbb R^{a\times b}\) and \(W_2\in\mathbb R^{c\times d}\), one must first answer —

\[a\cdot c = d_\text{out},\qquad b\cdot d = d_\text{in}\]

i.e. factorize the dimensions into two multipliers each. PEFT's `factorization(dimension, factor)` does exactly this; its rules are the whole problem (implement strictly as specified):

- **Explicit-factor branch**: if `factor > 0` and `factor` divides `dimension`, return `(factor, dimension // factor)` directly — **keep this order, do not swap by size** (e.g. `dimension=12, factor=6` returns `(6, 2)`; \(6>2\) is legal).
- **Auto-factorization branch** (`factor == -1`): among all factor pairs with \(m\cdot n=\text{dimension}\) and \(m\le n\), return the pair with the **minimum \(m+n\)** (i.e. \(m\) closest to \(\sqrt{\text{dimension}}\), the most balanced split). For example `256 → (16, 16)`, `128 → (8, 16)`, prime `127 → (1, 127)`.
- **Illegal input**: `dimension <= 0`, `factor` neither `-1` nor positive, or `factor > 0` but not dividing `dimension` — raise `ValueError`.

The more balanced the split (the closer \(m,n\) are), the squarer the two Kronecker factors; the LoKr paper notes that this usually corresponds to a better expressivity-vs-parameter trade-off, and PEFT's default `factor=-1` takes exactly the auto-balanced split.

## Task

Implement the `factorization` rules above: given `dimension` (positive integer) and `factor` (`-1` or a positive integer), return the pair `(m, n)` (two Python integers, \(m\cdot n=\text{dimension}\)). Raise `ValueError` on illegal input.

## Hints

- For the auto branch, scan \(m\) from 1 to \(\sqrt{d}\) while keeping the current minimum-\(m+n\) pair; a prime naturally lands on \((1, d)\).
- Do **not** over-sort the explicit branch: `(6, 2)` is `(6, 2)`.
