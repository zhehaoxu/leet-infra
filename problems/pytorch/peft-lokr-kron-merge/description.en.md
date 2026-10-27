# PEFT: LoKr Kronecker-Product Weight Merge

## Problem

**LoKr** (Kronecker Product adaptation, Yeh et al., arXiv:2108.07383; PEFT library `tuners/lokr`) builds the weight increment with a **Kronecker product** — compressing parameters even more aggressively than LoRA's additive side path: two small factors \(W_1\in\mathbb R^{a\times b}\) and \(W_2\in\mathbb R^{c\times d}\) prop up a large \((a\,c)\times(b\,d)\) increment:

\[\Delta W = s\,\bigl(W_1\otimes W_2\bigr),\qquad s=\frac{\alpha}{r}\]

The **block layout** of the Kronecker product must be memorized (`torch.kron` matches the mathematical definition):

\[(W_1\otimes W_2)\,\bigl[\,i\cdot c+u,\; j\cdot d+v\,\bigr] \;=\; W_1[i,j]\cdot W_2[u,v]\]

i.e. every element of \(W_1\) is blown up into a block of \(W_2\): \(W_1\) is the "outer structure" and \(W_2\) the "inner texture". PEFT's `make_kron(w1, w2, scale)` is one line: `torch.kron(w1, w2) * scale`; merging is \(W' = W_0 + \Delta W\).

LoKr's parameter count is \(a\,b+c\,d\) while \(\Delta W\) has size \(a\,c\times b\,d\) — with both factors \(16\times16\), only 512 parameters express a \(256\times256\) update (65 536 elements), which is where it beats LoHa/LoRA on compactness.

## Task

Given the frozen base \(W_0\in\mathbb R^{(a\,c)\times(b\,d)}\), the Kronecker factors \(W_1\in\mathbb R^{a\times b}\), \(W_2\in\mathbb R^{c\times d}\), the scaling parameter \(\alpha\), and the rank \(r\), return the **merged weight**

\[W' = W_0 + s\,\bigl(W_1\otimes W_2\bigr),\qquad s=\alpha/r\]

**Convention**: no ready-made PEFT/LyCORIS library; `torch.kron` is allowed, but the judge's oracle rebuilds every position with an explicit quadruple loop; results are judged under float32 tolerance \(10^{-5}\).

## Hints

- Check shapes first: `W0.shape == (a*c, b*d)`, where `(a,b)=W1.shape` and `(c,d)=W2.shape`.
- The block layout is "\(W_1\) outer, \(W_2\) inner" — rows are grouped by \(W_1\)'s rows first, then laid out by \(W_2\)'s rows; likewise for columns.
- LyCORIS-style init zeroes \(W_2\): at the start \(W'=W_0\).
