# Scaled Dot-Product Attention

## Problem Description

Scaled dot-product attention is the basic building block of the Transformer: it computes the similarity between Query and Key, turns it into weights via softmax, and then takes the weighted sum of the Values.

$$ \text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + \text{mask}\right)V $$

Here $d_k$ is the last (feature) dimension of Key, and the $\sqrt{d_k}$ scaling suppresses the variance blow-up of dot products as the dimension grows.

## Rules

- Implement `solve(query, key, value, mask=None)` and return a tensor whose batch structure matches the query and whose feature dimension equals `value.shape[-1]`.
- Inputs support arbitrary leading batch dimensions (`(..., L, d)`: batch, head, etc.); softmax is applied row-wise.
- **The mask uses torch's native bool semantics** (consistent with `F.scaled_dot_product_attention`): `True = attend`, `False = block` (masked positions get $-\infty$, giving probability 0 after softmax). If an entire row is masked, that row's output is 0 (torch convention; no NaN may be produced).
- Softmax must run along the key direction (the last axis corresponding to the second-to-last dimension), and the row maximum must be subtracted for numerical stability (large dot products without the max subtraction can overflow).

## Example

```
Input: query=[[[1,0]]], key=[[[1,0],[1,0]]], value=[[[3,0],[1,0]]]
Two identical keys -> equal scores -> weight 0.5 each
Output: [[[2.0, 0.0]]]        # 0.5*3 + 0.5*1 = 2.0
```

## Constraints

- float32/float64 input
- Results match `torch.nn.functional.scaled_dot_product_attention` (bool mask, rows not fully masked)
