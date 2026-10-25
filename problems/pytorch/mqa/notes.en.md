# MQA in Detail

## Three Ways of Sharing Attention Heads

| Scheme | Number of KV heads | KV cache | Expressiveness |
|------|---------|---------|--------|
| MHA  | = num_heads | largest | strongest |
| GQA  | 1 < g < num_heads (grouped sharing) | medium | strong |
| **MQA** | **1** (fully shared) | smallest | slightly lower |

MQA comes from Shazeer (2019), "Multi-Query Attention". During decoding every step must store K/V for all historical tokens;
under MHA this cache is proportional to the number of heads and is the memory/bandwidth bottleneck of long-context inference. MQA reduces the K/V projection to a single head,
shrinks the cache to $1/\text{num\_heads}$ and markedly speeds up autoregressive decoding, at the cost of slightly lower quality.

## Implementation Key: Broadcast Rather Than Copy

The single KV head has shape `(B, L_k, 1, d_h)` and has to participate in the attention of `num_heads` query heads, so use
`.expand(B, num_heads, L_k, d_h)` to **broadcast** it (which occupies no extra memory) rather than `repeat`, which really copies it.
After that, each head independently computes $\text{softmax}(Q_h K^{\top}/\sqrt{d_h})V$, and the heads are concatenated back into `(B, L_q, hidden)`.

## Common Mistakes

- **head_dim consistency**: after splitting `q` into heads, $d_h=\text{hidden}/\text{num\_heads}$ must equal the last dimension of `k`/`v`,
  otherwise the `@` shapes do not line up (this was a pitfall hit during this problem's dump phase: writing k with a single-element dimension caused a dimension mismatch).
- **Scale by $\sqrt{d_h}$**, not $\sqrt{\text{hidden}}$.
- **reshape vs view**: after `transpose(1,2)` the tensor is non-contiguous, so use `.reshape` to get back to `(B,L_q,hidden)`
  (or `.contiguous().view` first); calling `.view` directly raises an error.
- Concatenating in head order: `(B, L_q, num_heads, d_h).reshape(B, L_q, hidden)` keeps the heads and the dimensions in order.

## Relation to SDPA

PyTorch's `F.scaled_dot_product_attention(..., enable_gqa=True)` natively supports grouped/multi-query sharing where "the number of KV heads is fewer than the number of Q heads".
This reference answer hand-writes the expand and matches that interface when the number of KV heads = 1 (cross-checked, error ~1e-7).
