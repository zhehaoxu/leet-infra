# PEFT: GLoRA Weight Merge and Exact Unmerge

## Problem

**GLoRA** (Generalized LoRA, arXiv:2410.01485) writes the low-rank update as an *element-wise multiplicative rescaling of the base weight* plus an *additive low-rank term*:

\[W' = W_0 + W_0 \odot A + B,\qquad A=A_dA_u,\quad B=B_dB_u\]

Unlike LoRA, which only adds \(BA\), the factor \(W_0\odot A\) makes the update follow the magnitude of the base weight element by element: large weights may change more, tiny weights barely move. This gives the low-rank direction a "base-shape" prior. The bias has a symmetric multiplicative side path \(b_0\odot D+E\) plus an extra input-side mixing term \(W_0C\).

Your task is the **merge** semantics: fold several adapters into the base weight and bias at once, then run one plain linear forward; the returned \((W',b')\) must also be exactly invertible back to \((W_0,b_0)\) (which is what PEFT's `unmerge` does).

## Rules

- Each adapter provides low-rank factors \(A_d:(d_\text{out},r)\), \(A_u:(r,d_\text{in})\) and \(B_d:(d_\text{out},r)\), \(B_u:(r,d_\text{in})\), plus bias vectors \(C:(d_\text{in},)\), \(D:(d_\text{out},)\), \(E:(d_\text{out},)\).
- The \(i\)-th adapter contributes \(W_0\odot A_i+B_i\), and **every term multiplies the original \(W_0\)**, never the already-merged weight.
- With no base bias (`b0` is None) the vectors \(C,D,E\) take no part, and the returned \(b'\) is None as well.
- Accumulate in float64, return float32.

## Example

One adapter with \(d_\text{out}=3, d_\text{in}=4, r=2\):

```
Input:
  W0 = [[0.5,-0.2,0.1,0.3],[0.2,0.4,-0.1,0.0],[-0.3,0.1,0.5,0.2]]
  b0 = [0.1,-0.2,0.05]
  A_d = [[0.3,-0.1],[0.2,0.4],[-0.5,0.2]]
  A_u = [[0.1,0.2,-0.3,0.4],[-0.2,0.5,0.1,0.0]]
  B_d = [[0.1,0.2],[-0.3,0.1],[0.0,-0.2]]
  B_u = [[0.4,-0.1,0.2,0.0],[0.3,0.2,-0.5,0.1]]
  C = [0.5,-0.4,0.2,0.1]
  D = [0.2,-0.1,0.3]
  E = [0.05,0.1,-0.15]
  x = [[1.0,0.5,-0.3,0.2],[0.1,-0.4,0.6,0.3]]
Output:
  W' = [[0.625,-0.172,0.01,0.356],[0.098,0.546,-0.208,0.01],[-0.333,0.06,0.685,0.14]]
  b' = [0.55,-0.16,-0.155]
  y  = [[1.1572,0.2754,-0.6355],[0.7941,-0.4904,0.2407]]
```

## Constraints

- `x` may have any leading dimensions; its last dimension is \(d_\text{in}\). `b0` has shape \((d_\text{out},)\).
- Low-rank factors need \(r\ge1\) with matching inner dimensions; the adapter list is non-empty; malformed shapes must raise `ValueError`.
- Inverse formula: \(W_0=(W'-\sum_i B_i)/(1+\sum_i A_i)\) and \(b_0=(b'-\sum_i E_i-\sum_i W_0C_i)/(1+\sum_i D_i)\); the \(C\) term uses the **recovered** \(W_0\).
- A multi-adapter merge is order independent and must equal "sum of the per-adapter unmerged forwards minus \((n-1)\) base forwards".
