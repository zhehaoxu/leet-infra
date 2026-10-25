# Multi-Query Attention (MQA)

## Problem Description

In standard multi-head attention (MHA) every query head has its own paired set of key/value, so the KV cache grows linearly with the number of heads.
**Multi-Query Attention (MQA)** lets **all query heads share a single group of key/value heads** (number of KV heads = 1), greatly compressing
the KV cache and the memory bandwidth during decoding. It is a classic technique for faster inference (the compromise between MHA and MQA is GQA).

## Rules

- Implement `solve(q, k, v, num_heads)`.
  - `q` has shape `(B, L_q, hidden)` and is split into `num_heads` query heads, each of dimension $d_h = \text{hidden}/\text{num\_heads}$.
  - `k` and `v` have shape `(B, L_k, d_h)` - there is **only one** KV head, shared by all query heads.
- **Broadcast (expand)** that single KV head to `num_heads` copies and run scaled dot-product attention per head:
  $$ \text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}\right)V $$
  then concatenate the per-head outputs back into `(B, L_q, hidden)` in head order.
- No causal mask and no dropout. The result must match `F.scaled_dot_product_attention`
  applied after expanding the single KV head (or `enable_gqa=True`).

## Example

```
Input: num_heads=2 -> head_dim=2
     q = [[[1, 0, 2, 1]]]                shape (1,1,4)  head0=[1,0], head1=[2,1]
     k = [[[0.5, 1], [1, 0]]]            shape (1,2,2)  single shared KV head, L_k=2
     v = [[[1, 2], [0, 1]]]              shape (1,2,2)
   the two query heads each attend over the same K/V, then their outputs are concatenated
Output: [[[0.412521, 1.412521, 0.5, 1.5]]]  shape (1,1,4)
```

## Constraints

- float32 input; `hidden` is divisible by `num_heads`
- The number of KV heads is fixed at 1 (the definition of MQA); the last dimension of `k`/`v` must equal `head_dim`
- Do not use `F.scaled_dot_product_attention` / `nn.MultiheadAttention`; the broadcast and the scaled dot-product must be written by hand
