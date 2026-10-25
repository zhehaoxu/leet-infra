# Embedding Lookup Explained

## Essence

Embedding is just **row selection by integer index**: `weight` is the $(V,d)$ vocabulary matrix, each value in `indices` is a token id, and the output pulls out the corresponding rows, keeping the shape of `indices` plus one dimension $d$:

$$ \text{out}[\dots]=\text{weight}[\text{indices}[\dots]],\qquad \text{shape}=(\text{indices.shape},\,d) $$

Transformer token embeddings and positional-encoding tables are both this. In the backward pass, gradients flow only to the rows that were looked up (sparse); unused rows get zero gradient.

## The Real Semantics of padding_idx (the big pitfall here)

- **`F.embedding`/`nn.Embedding`'s `padding_idx` does NOT zero that row's output in the forward pass!** It only affects: (1) zeroing that row at initialization; (2) forcing that row's gradient to 0 (no update) in the backward pass. If the padding row in the weight is itself nonzero, the forward pass still returns nonzero values.
- Therefore the convention of this problem is "**wherever indices equals padding_idx, the output row is forcibly zeroed**"—post-lookup zeroing semantics. Do **not** compare directly against `F.embedding(...,padding_idx=)`; build the reference manually: first do an ordinary row lookup, then `torch.where(indices==padding_idx, 0, out)`.

## Implementation Pitfalls

1. **Flatten and then restore the shape**: `weight[indices.reshape(-1)].reshape(*indices.shape, d)`. Using `weight[indices]` directly with multi-dimensional indices yields `indices.shape+(d,)` but is easy to confuse with advanced indexing, so flattening is the safest.
2. `pad_mask.unsqueeze(-1)` extends the $(...)$ mask to the last dimension so it can broadcast with $(...,d)$ in `where`.
3. Indices must be integer and within $[0,V)$ (torch supports negative indices).

## Cross-Checking Advice

Use a manual reference of row selection + padding zeroing; if the problem does not test padding zeroing, the result matches `F.embedding(indices, weight)`.
