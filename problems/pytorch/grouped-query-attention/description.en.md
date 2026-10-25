# Grouped-Query Attention GQA / MQA

## Problem

Grouped-Query Attention is a compromise between multi-head attention (MHA) and multi-query attention (MQA):
**the query has `num_heads` heads while key/value have only `num_kv_heads` heads**, and every several query heads share the same group of
K/V heads, which greatly reduces KV cache GPU memory. Modern large models such as Llama 2 70B / Mistral adopt it.

## Rules

- Implement `solve(q, k, v, num_heads, num_kv_heads)`.
- `q` has shape `(B, Lq, num_heads*dh)`; `k`, `v` have shape `(B, Lk, num_kv_heads*dh)`; return `(B, Lq, num_heads*dh)`.
- `num_heads` must be divisible by `num_kv_heads`; the kv head used by query head `i` is `i // (num_heads // num_kv_heads)`
  (**contiguous groups**, i.e. `repeat_interleave` semantics, not cyclic modulo).
- The scale factor is $1/\sqrt{dh}$, where $dh = \text{last dim}/num\_heads$; compute $\text{softmax}(QK^\top/\sqrt{dh})V$ independently for each query head, then concatenate along the head dimension.
- `num_kv_heads == num_heads` must agree with standard multi-head attention; `num_kv_heads == 1` gives MQA.

## Examples

```
Input: q=[[[1,0,0,1],[0,1,1,0]]] (B=1,Lq=2,H=2,dh=2)
     k=[[[1,0],[0,1]]]        (Lk=2,KV=1)
     v=[[[2,0],[0,2]]]
     num_heads=2, num_kv_heads=1
Output: [[[1.339523, 0.660477, 0.660477, 1.339523],
        [0.660477, 1.339523, 1.339523, 0.660477]]]
```

## Constraints

- float32 input; $B \ge 1$, $Lq, Lk \ge 1$, $dh$ a positive integer
- No causal mask (this problem only tests the mechanism of sharing K/V across groups)
