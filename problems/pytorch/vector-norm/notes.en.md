# Explained: Vector Norm (the Full ord Family with Multi-Axis Reduction)

## Origin

`torch.linalg.vector_norm` (the new API since 1.9) replaced the legacy `torch.norm(x, p, dim)` but inherited its historical baggage wholesale: the same `ord=1` means a sum of magnitudes for a vector and a maximum column sum for a matrix. Interviewers love asking "what do `p=0` and `p=-1` mean in `torch.norm`", and the miss rate is high. Direct engineering touch points: the norm convention inside `torch.nn.utils.clip_grad_norm_`, per-channel L2 scoring of KV cache entries (`kv-knorm-importance`), and the denominator of many normalisations.

## Full Convention Table and the Stable Form

Write $a_i = |x_i|$ for the magnitudes inside one reduction segment and $m=\max_i a_i$.

| ord | definition | degenerate behaviour |
| --- | --- | --- |
| `None` / 2 | $(\sum a_i^2)^{1/2}$ | all zero → 0 |
| 1 | $\sum a_i$ | all zero → 0 |
| -1 | $(\sum a_i^{-1})^{-1}$ | contains a 0 → 0 |
| 0 | $\#\{a_i \ne 0\}$ | all zero → 0 |
| `inf` | $\max a_i$ | all zero → 0 |
| `-inf` | $\min a_i$ | contains a 0 → 0 |
| finite $p\notin\{0,\pm1,2\}$ | $(\sum a_i^p)^{1/p}$ | all zero → 0 |

The $-1$ row is the least intuitive: it is not a partner of the 1-norm but simply the power formula evaluated at $p=-1$, i.e. $\big(\sum 1/a_i\big)^{-1}$, a scaled harmonic mean. As soon as some $a_i=0$, $1/a_i=+\infty$ and the reciprocal is 0. The reference solution needs no special case for it: in the stable form $(a_i/m)^{-1}=+\infty$ and $\infty^{1/p}=0$ for $p<0$, so multiplying by $m$ still gives 0, matching torch pointwise.

The max-factoring identity holds for every non-zero $p$:

$$\Big(\sum_i a_i^p\Big)^{1/p} = \Big(m^p \sum_i (a_i/m)^p\Big)^{1/p} = m\Big(\sum_i (a_i/m)^p\Big)^{1/p}$$

Since $a_i/m \in [0,1]$, $(a_i/m)^p$ cannot overflow for $p>0$ and is at most $+\infty$ for $p<0$, so an `inf − inf` never appears. An all-zero segment must still be guarded: there $m=0$ makes $a_i/m$ a $0/0$, hence the reference uses `torch.where(m > 0, ...)` to swap the divisor to 1 and the dividend to 0, and the final multiplication by $m=0$ yields 0.

## Core Points and Pitfalls

1. **`ord=0` is not the power formula**: $(\sum a^0)^{1/0}$ is undefined; it counts non-zeros and must take the `(absx != 0).sum()` branch.
2. **Treat the string `"inf"` and `float("inf")` alike**: `torch.linalg.vector_norm` only accepts numbers, this interface additionally accepts the string spelling, and the judge tries both.
3. **Shapes after a multi-axis reduction**: `dim=(0,2)` drops both axes; `keepdim=True` keeps them as length 1 at their original positions — so `keepdim` can only be implemented as `squeeze(dim=axes)`, never by guessing an order with `reshape`.
4. **`dim=None, keepdim=True` is an extension of this problem**: torch ignores `keepdim` there and returns a 0-d tensor, while this problem demands an all-ones shape, asserted separately as `(1,1,1)`.
5. **Cast back to float32**: everything is accumulated in float64; forgetting `.to(x.dtype)` fails the dtype assertion.

## How It Differs from Neighbouring Problems

- `matrix-norm` (order359 in the same batch): there `1/-1/inf` mean "sum first, then take the extreme", here they are plain sums; the pair exists precisely to expose the convention gap.
- `lp-pool-unpool`: an $L^p$ pooling $(\frac1n\sum|x|^p)^{1/p}$ along a named axis, with the $1/n$ mean factor, plus the max-pool index/unpool side; this problem has no normalisation factor.
- `distributed-clip-grad-norm`: the accounting of a global gradient norm for clipping, not the norm itself.
- `rmsnorm` / `layernorm-forward`: they use 2-norms/RMS internally but examine the affine normalisation transform.

## Cross-Checking and Complexity

- The oracle is `torch.linalg.vector_norm` itself, over the Cartesian product of 11 `ord` values and 4 `dim` settings, plus the harmonic `ord=-1` case, multi-axis/keepdim shapes, overflow probes at magnitudes $10^3$ and $10^{-8}$, all-zero segments, and six illegal argument combinations.
- Complexity $O(N)$ for $N$ reduced elements, with one extra division and power in float64; the judging sizes run in milliseconds.
