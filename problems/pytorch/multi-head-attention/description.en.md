# Multi-Head Attention Forward Pass

## Problem Description

Multi-head attention splits the feature dimension $D$ into $H$ heads; each head performs scaled dot-product attention independently in a $d_h = D/H$-dimensional subspace, and the per-head outputs are then concatenated back to $D$.

$$ \text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_H),\quad \text{head}_i = \text{Attention}(Q_i,K_i,V_i) $$

## Rules

- Implement `solve(q, k, v, num_heads, mask=None)`, where `q/k/v` have shape `(B, L, D)` and the return value has shape `(B, L_q, D)`.
- `D` must be divisible by `num_heads`.
- **Dimension-ordering trap**: the heads are split as **contiguous blocks** along the last dimension. Split heads with `view(B, L, H, dh).transpose(1, 2)` to get `(B, H, L, dh)`; merge them with `transpose(1, 2).reshape(B, L, D)`, and the output column order must be `[head0 | head1 | ...]` - swapping the head order gives a wrong answer.
- Each head scales by $1/\sqrt{d_h}$ (note it is the sub-head dimension $d_h$, not $D$).
- The mask semantics match scaled dot-product attention (bool: `True=visible`, `False=masked out`), with shape `(..., L_q, L_k)` or `None`.

## Example

```
Input: q=ones(1,2,4), k=ones(1,2,4), num_heads=2 (dh=2)
      v=[[[0,1,10,20],[2,3,30,40]]]
q and k are all 1 → every per-head score is equal → uniform weights (0.5 each) → each head's output = the mean of that head's block of v columns
output: [[[1,2,20,30],[1,2,20,30]]]   # head0=mean(v[...,0:2]), head1=mean(v[...,2:4])
```

## Constraints

- float32/float64 input
- Must match looping over the heads and calling single-head attention each time (same `F.scaled_dot_product_attention`)
