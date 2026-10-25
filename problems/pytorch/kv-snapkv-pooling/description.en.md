# SnapKV: Window Attention + 1D Average Pooling

## Problem

**SnapKV** (Li et al., 2024, [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)): uses "**the attention of the most recent window_size queries over earlier keys**" as each key's importance score. The core innovation is adding a **1D average pooling** layer (`avg_pool1d` kernel=kernel_size, padding=kernel_size//2, stride=1) that smooths the attention along key positions, mitigating abrupt importance jumps between adjacent positions.

The `window_size` queries at the end of the observation window **do not participate in pruning**—they are forcibly kept with a score "one higher than the global maximum".

## Rules

Given `attn` of shape $(H, W, P)$: H heads (`num_key_value_heads = num_heads`; no kv-group averaging in this problem), the attention weights of the **last W queries** over the **first P keys** in each head (i.e. the result of `attentions[..., -W:, :-W]`, with `softmax` already applied), and `kernel_size` (odd). Return scores of shape $(H, P + W)$:

1. `base = attn.mean(dim=-2)` — mean along W → $(H, P)$
2. `pooled = avg_pool1d(base, kernel_size=K, padding=K//2, stride=1)` — length unchanged (K is odd), **out-of-bounds positions are treated as 0 while the denominator is still K** (PyTorch `avg_pool1d` semantics)
3. `out = pad(pooled, right=W, value=pooled.max() + 1)` — stitch the window positions back with the "guaranteed maximum score"

## Examples

```
H=2, W=3, P=5, kernel_size=3
attn[h=0] = [[0.1, 0.2, 0.05, 0.15, 0.1],
             [0.3, 0.1, 0.1, 0.2, 0.05],
             [0.05, 0.4, 0.1, 0.1, 0.2]]
attn[h=1] = [[0.2, 0.1, 0.3, 0.1, 0.05],
             [0.1, 0.2, 0.1, 0.3, 0.2],
             [0.3, 0.1, 0.1, 0.1, 0.25]]

base[h=0] = mean over W → [0.150, 0.2333, 0.0833, 0.150, 0.1167]
pooled[h=0] = avg_pool(k=3, pad=1, stride=1) →
    [0.1278, 0.1556, 0.1556, 0.1167, 0.0889]
scores[h=0] = pooled ++ [max+1]*3 =
    [0.1278, 0.1556, 0.1556, 0.1167, 0.0889, 1.1667, 1.1667, 1.1667]
scores[h=1] = [0.1111, 0.1667, 0.1556, 0.1667, 0.1111, 1.1667, 1.1667, 1.1667]
```

## Constraints

- CPU float32; `kernel_size` is guaranteed odd (1, 3, 5, 7, ...); `padding = kernel_size // 2`.
- `avg_pool1d` boundary handling: out-of-bounds positions are padded with 0 and do not count toward the numerator; the denominator is always kernel_size.
