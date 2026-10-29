# Vector Norm: the Full ord Family with Multi-Axis Reduction

## Problem Description

`torch.linalg.vector_norm` looks trivial until the `ord` column is opened: the default is the 2-norm, `0` counts non-zero entries, `±inf` take extreme values of the magnitudes, `1` is the sum of absolute values while `-1` is the harmonic-style $(\sum|x_i|)^{-1}$ reciprocal (any zero inside a segment forces the whole result to 0), and finite $p$ must survive overflow. Gradient clipping, attention weight normalisation and KV importance scoring all sit directly on these definitions.

$$\|x\|_p = \Big(\sum_i |x_i|^{\,p}\Big)^{1/p}, \qquad \|x\|_\infty = \max_i |x_i|, \qquad \|x\|_0 = \#\{i : x_i \ne 0\}$$

For finite $p$ the judge requires the **max-magnitude factored** stable form

$$\|x\|_p = m\Big(\sum_i (|x_i|/m)^p\Big)^{1/p}, \qquad m = \max_i |x_i|$$

otherwise $\sum|x|^p$ overflows to inf as soon as $x\sim 10^3$ and $p=8$.

## Rules

- The full signature and parameter semantics live only in the starter docstring; that is the single source of truth for judging.
- `dim=None` flattens the whole tensor before reducing and returns a **0-dimensional** tensor; `dim` may be an int or a tuple of two or more ints, negatives folded against `x.dim()`.
- `ord` accepts `None` (equivalent to 2), int, float, and the strings `"inf"` / `"-inf"`; `"fro"` and `"nuc"` are matrix-only notations and must raise ValueError for a vector.
- With `dim=None` and `keepdim=True`, this problem keeps **every axis as length 1** (torch returns a 0-d tensor there; the problem defines it as an extension).
- An all-zero vector is 0 under every `ord`; `ord=-1` on a segment containing a 0 yields 0, not inf.

## Example

$X$ is $2\times 3$, reducing along `dim=1` for several `ord` values (numbers taken from the EXAMPLE_* constants filled in from test_cases.py):

```
X = [[ 3, -4,  0],
     [ 0,  0, -1]]

ord=2    -> [5.0, 1.0]
ord=0    -> [2.0, 1.0]      # count of non-zero entries
ord="inf"-> [4.0, 1.0]      # max|x|
dim=None -> 5.099019527435303   # sqrt(26)
```

Row 0 gives $\|\cdot\|_2=\sqrt{3^2+4^2+0^2}=5$; `ord=0` only counts the two non-zeros $3$ and $-4$; `ord="inf"` takes $\max(|3|,|4|,|0|)=4$.

## Constraints

- Input is float32, accumulated in float64, returned as float32; compared against `torch.linalg.vector_norm` with atol=1e-5, rtol=1e-5
- Shape and dtype are asserted too; each illegal `ord`/`dim` (out-of-range axis, duplicate axis, empty tuple, matrix notation) must raise ValueError
