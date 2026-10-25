# SimLayerKV: Lazy-Layer Binary Retention

## Problem

Most cross-layer methods (PyramidKV etc.) hand each layer a **numeric budget**; SimLayerKV (Zhang et al., 2024, [arXiv:2410.13846](https://arxiv.org/abs/2410.13846); `SimLayerKVPress` in `kvpress/presses/simlayerkv_press.py` of NVIDIA/kvpress) reduces cross-layer compression to an independent **0/1 decision per layer**: a "lazy" layer can work with head-and-tail information alone — if the attention of the last few queries falls almost entirely onto the **first \(n\_initial\) (attention sinks) plus the most recent \(n\_recent\)** tokens, the layer provably never uses the middle context and its entire middle cache can be dropped; otherwise the layer is kept **in full**. No top-k, no score ordering — just one attention-mass sum and one threshold comparison.

## Rules

Input: `last_attn` of shape \((N, L)\) — for layer \(l\), the attention weights of the **last \(n\_last\) queries** over all \(L\) key positions, already averaged over batch, all query heads and the window rows; integers `n_initial`, `n_recent`, `n_last`; scalar `lazy_threshold` \(\theta\in[0,1]\).

For every layer \(l\):

1. **Disabled branch**: if \(\theta = 1.0\) or \(L\le n\_initial+n\_recent+n\_last\), the layer is kept as-is (all-true mask, ratio 0). The first condition is genuine upstream behaviour: the default \(\theta=1.0\) literally means "compression off" — the threshold must be explicitly lowered for the press to do anything.
2. **Laziness test**:
   \[ s_l=\sum_{j<\min(n\_initial,L)} a_{l,j}+\sum_{j\ge L-\min(n\_recent,L)} a_{l,j} \]
   the layer is lazy iff \(s_l\) is **strictly greater** than \(\theta\) (\(s_l=\theta\) is not lazy).
3. **Lazy retention set**: \([0, n\_initial)\ \cup\ [\,L-(n\_recent-n\_last),\ L)\); everything between is evicted, and the layer's compression ratio is
   \[ r_l=\frac{L-n\_initial-(n\_recent-n\_last)}{L} \]
   Non-lazy layers keep everything with \(r_l=0\). (Reaching step 3 implies \(L>n\_initial+n\_recent+n\_last\), so the two kept blocks never overlap.)

Return two tensors: `mask` (bool, \((N, L)\), true = kept) and `ratios` (float32, \((N,)\)).

## Example

```
N=2, L=10, n_initial=2, n_recent=4, n_last=1, θ=0.5

layer0: [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1]
        s = (0.2+0.2) + (0.0+0.3+0.2+0.1) = 1.0 > 0.5 → lazy
        keep [0,2) ∪ [10-3,10) = {0,1,7,8,9}, r = (10-2-3)/10 = 0.5
layer1: [0.1, 0.05, 0.2, 0.1, 0.2, 0.1, 0.05, 0.1, 0.05, 0.05]
        s = (0.1+0.05) + (0.05+0.1+0.05+0.05) = 0.4 ≤ 0.5 → not lazy
        keep all, r = 0.0

mask   = [[T, T, F, F, F, F, F, T, T, T],
          [T, T, T, T, T, T, T, T, T, T]]
ratios = [0.5, 0.0]
```

## Constraints

- CPU float32; `last_attn` non-negative (an attention mean); \(n\_recent>n\_last\ge1\); \(N,L\ge1\).
- The threshold test is **strictly greater**; with \(\theta=0\) and \(s_l=0\) (all-zero attention) the layer is not lazy.
- If \(n\_initial\) or \(n\_recent\) exceeds \(L\), the sum ranges clamp at \(L\) (such inputs can only hit the disabled branch).
- `mask` is returned as a bool tensor; a lazy layer's mask looks like "left block + right block" with one contiguous run of falses in between.
