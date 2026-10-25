# ALiBi Positional Bias (Attention with Linear Biases)

## Problem

ALiBi adds no position vectors; instead it applies a head-specific linear penalty to attention scores based on the **distance between query and key** $|i-j|$, with larger penalties for larger distances, yielding position awareness with stronger extrapolation.

$$ \text{bias}[h, i, j] = -\,m_h \cdot |i - j|, \qquad m_h = 2^{-8h/n} $$

where there are $n$ heads in total and the head index $h$ starts at **1** (1-indexed).

## Rules

- Implement `solve(seq_len, num_heads, dtype=torch.float32)`, returning a bias tensor of shape `(num_heads, seq_len, seq_len)`.
- Slope sequence: head $h$ (1-indexed) has $m_h = 2^{-8h/n}$. For $n=8$ this is exactly $[2^{-1}, 2^{-2}, \dots, 2^{-8}]$.
- **Monotonicity**: the larger the head index, the smaller the slope (geometric decay), so the decay is slower and the head sees farther.
- The diagonal is always 0 ($i=j$); the matrix is symmetric about the diagonal; all elements are non-positive.

## Examples

```
Input: seq_len=3, num_heads=1
   The only slope m=2^(-8*1/1)=2^-8=0.00390625
   |i-j| distance matrix = [[0,1,2],[1,0,1],[2,1,0]]
Output: [[[ 0.0      , -0.00390625, -0.00781250],
        [-0.00390625, 0.0      , -0.00390625],
        [-0.00781250, -0.00390625, 0.0      ]]]
```

## Constraints

- `dtype` supports float32/float64
- Must match the hand-derived formula $-m_h|i-j|$
