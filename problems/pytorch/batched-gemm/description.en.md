# Batched Matrix Multiplication (BMM / matmul semantics)

## Problem

Batched GEMM is the computational backbone of Attention ($QK^\top$, $\text{attn}V$). Given $A: (\dots, n, m)$ and $B: (\dots, m, p)$, following `torch.matmul`'s broadcasting semantics:

$$
\text{out}[\dots, i, j] = \sum_{k=0}^{m-1} A[\dots, i, k]\cdot B[\dots, k, j]
$$

The leading batch dimensions $(\dots)$ are aligned by broadcasting rules: they may be identical, one side may be empty (2D matrix), or contain dimensions of length 1.

## Rules

- Implement `solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`; both are $\ge 2$-D and the inner dimensions must be equal (otherwise raise `ValueError`).
- **Using `torch.matmul` / `bmm` / `mm` / the `@` operator / `einsum` / `dot` is forbidden** — go back to the definition: expand $a$ to $(\dots, n, m, 1)$ and $b$ to $(\dots, 1, m, p)$, multiply element-wise and sum along $m$.
- Handle batch-dimension broadcasting with `torch.broadcast_shapes` + `expand`; expanded views have stride 0, which is fully compatible with element-wise multiplication.
- **Pitfall**: inputs may be non-contiguous transposed views (e.g. `x.transpose(1,2)`), and implementations that rely on reshape/contiguous will fail here; the broadcasting semantics must cover "2D @ 3D" and "batch dimension containing 1".

## Examples

```
Input: a = [[1,2],[3,4]], b = [[5,6],[7,8]]
Output: [[19, 22], [43, 50]]     # 1*5+2*7=19, 1*6+2*8=22, ...
```

## Constraints

- float32/float64; test tensors are small (the $O(nmp)$ memory of the outer-product method is acceptable)
- Must match `torch.matmul(a, b)` exactly (atol 1e-5)
