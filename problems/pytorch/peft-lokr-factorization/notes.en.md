# PEFT: LoKr Kronecker Factorization — In Depth

## 1. Why LoKr needs factorization

LoKr (Yeh et al., arXiv:2108.07383; PEFT `tuners/lokr/layer.py`) writes the increment as a Kronecker product:

\[\Delta W = s\,(W_1\otimes W_2),\qquad s=\alpha/r\]

If \(W_1\in\mathbb R^{a\times b}\) and \(W_2\in\mathbb R^{c\times d}\), then \(W_1\otimes W_2\in\mathbb R^{(a\,c)\times(b\,d)}\). For \(\Delta W\) to share the base's shape, we need \(a\,c=d_\text{out}\) and \(b\,d=d_\text{in}\) — each dimension pair is split into two multipliers, the split is not unique, and `factorization(dimension, factor)` is the deterministic rule that picks one.

## 2. Reading the PEFT source rule step by step

`factorization` (Linear branch, logic transcribed from the source):

1. **`factor > 0` and divides**: return `m = factor, n = dimension // factor` directly, **without** the `m<=n` swap. Hence `factorization(12, 6) == (6, 2)` is a legal output — the caller only needs the product to match and does not care about the order.
2. **`factor == -1` (default)**: start from `m=1, n=d` and repeatedly promote `m` to the next factor, accepting only when \(m+n\) does not grow — and \(m+n\) decreases monotonically while \(m\le\sqrt d\), so the fixed point is the "most balanced" pair; \(m\le n\) is maintained throughout. Equivalent closed-form description: \(m\) is the **largest** factor not exceeding \(\sqrt d\). For a prime \(d\), the only non-trivial pair is \((1,d)\).
3. This problem normalizes "`factor>0` but not dividing / illegal `factor` / illegal `dimension`" into `ValueError` (the original PEFT silently falls to the search branch there; the judge narrows this to an explicit error, as declared in the statement).

## 3. What the split affects

- Parameter count: the full form costs \(m_1 n_1 + m_2 n_2\) (the elements of \(W_1,W_2\)); the more balanced the split, the closer to the \(2\sqrt{d_\text{out}d_\text{in}}\) scale — far smaller than \(d_\text{out}d_\text{in}\).
- \(W_1\) can itself be further low-ranked (`use_w1=False` gives \(W_1=W_{1a}W_{1b}\), the `decompose_both`/`use_w_decomp` switches); not covered here.
- LyCORIS-style init zeroes \(W_2\) (or `w2_b`) → \(\Delta W=0\) at the start, the same idea as LoHa's zeroed degenerate branch.

## 4. Judging focus

- The explicit branch tests the "**original order**": `(6, 2)` is not `(2, 6)`; swapping the order is caught by `check_factor_divides_direct`.
- The auto branch tests "most balanced + \(m\le n\)": \((16,16)\), \((8,16)\), \((6,8)\), prime \((1,d)\).
- The oracle brute-force enumerates all factor pairs and picks the lexicographically smallest — structurally independent from the reference solution's monotone scan.
