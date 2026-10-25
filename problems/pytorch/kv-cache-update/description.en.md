# KV Cache Append (KV Cache Update)

## Problem

During autoregressive decoding, the Key/Value of historical tokens are cached in the KV Cache to avoid recomputing them at every step. Each time a new token is generated, only its K and V need to be **appended along the sequence dimension** to the end of the cache.

$$ K_{\text{cache}} \leftarrow \text{Concat}(K_{\text{cache}},\ K_{\text{new}})\ \text{along seq dim} $$

## Rules

- Implement `solve(cache_k, cache_v, new_k, new_v, dim=-2)`, returning the 2-tuple `(k, v)`.
- Append along the **sequence dimension** `dim=-2` (the $L$ in shape `(B, H, L, dh)`).
- **Empty cache**: when `cache_k` (or `cache_v`) is `None` or a tensor with 0 elements, return the corresponding `new_k` / `new_v` directly (a zero-length empty tensor is equivalent to `None`).
- **Must not modify the input cache tensors in place**: `torch.cat` returns a new tensor, and the old cache contents remain unchanged (`check_input_not_mutated`).

## Examples

```
Input: cache_k=[[[1],[2]]], cache_v=[[[10],[20]]], new_k=[[[3]]], new_v=[[[30]]]
append along dim=-2
Output: k=[[[1,2,3]]] shape (1,3,1),  v=[[[10,20,30]]]
      i.e. k=[[[1.0],[2.0],[3.0]]], v=[[[10.0],[20.0],[30.0]]]
```

## Constraints

- The sequence length may differ at each step (incremental decoding); all dimensions other than `dim` must match
- Consistent with `torch.cat([cache, new], dim=-2)`
