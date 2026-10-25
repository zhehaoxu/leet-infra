# FlashAttention Tiled Online Softmax

## Problem

The core idea of FlashAttention is to **not explicitly construct the full $L_q \times L_k$ attention matrix**, but instead tile the queries,
accumulate block by block along the key dimension, and use "online softmax" (running max + running sum) to guarantee numerical stability,
ultimately obtaining output **numerically equivalent** to standard attention. This problem asks you to hand-write this tiled accumulation logic.

## Rules

- Implement `solve(q, k, v, block)`, single-head attention: `q` has shape `(B, Lq, d)`, `k` and `v` have shape `(B, Lk, d)`, and the return value is `(B, Lq, d)`.
- The scaling is $1/\sqrt{d}$; there is no mask.
- You must **iterate in blocks** of size `block` over both the query and key dimensions, maintaining per row through online reduction the running
  maximum $m$, the denominator accumulator $l$, and the numerator accumulator $\text{acc}$, and when a larger $m$ appears, rescale the accumulated $l,\text{acc}$ by
  $\alpha=\exp(m_{old}-m_{new})$.
- The result must match the standard $\text{softmax}(QK^\top/\sqrt{d})V$, and must hold for **any value of `block`**,
  as well as when `block` does not divide $L_q/L_k$ and when $L_k \neq L_q$.

## Examples

```
Input: q=[[[1,0],[0,1],[1,1]]]
     k=[[[1,0],[1,0],[0,1]]]
     v=[[[1,2],[3,4],[5,6]]]
     block=2
Output: [[[2.593328, 3.593328],
        [3.510469, 4.510469],
        [3.000000, 4.000000]]]
(consistent with the standard attention result; only the last floating-point digits differ)
```

## Constraints

- float32 input; $B \ge 1$, $L_q, L_k, d \ge 1$, `block >= 1`
- logits may be large (e.g. $\times 20$); the online max reduction must keep them numerically stable and prevent overflow
