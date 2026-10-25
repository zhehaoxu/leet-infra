# TOVA: Cross-Head Averaging of the Last Query's Attention

## Problem

**TOVA** (Otsuka et al., 2024, [arXiv:2401.06104](https://arxiv.org/abs/2401.06104)): uses only **the attention of the last query over earlier keys**, averaged across heads—the importance of each key is simply "how much the model cared about it at the most recent step".

`TOVAPress.score` in kvpress has three steps:
1. `attn_weights = attentions[..., -1:, :-1]` — the last query row, excluding the last key (the self-attention slot)
2. `scores = attn_weights.mean(1)` — average across num_heads (dim=1 is the head dimension when a batch is present)
3. `scores = repeat(num_kv_heads, 1)` plus a trailing pad of `max+1` (always keep the last token)

## Rules

Given `attn` of shape $(H_q, L, L)$ (**each row sums to 1 after softmax**, causal lower triangle already applied) and `num_kv_heads` (the number of KV heads $H$ in GQA): return scores of shape $(H, L)$:

1. Take `attn[:, L-1, :L-1]` → $(H_q, L-1)$
2. Take the mean over the $H_q$ dimension → $(L-1,)$ (**one shared score for all heads**)
3. Copy to $H$ rows → $(H, L-1)$
4. Pad one slot at the tail with value `global max + 1` → $(H, L)$

## Examples

```
Hq=4, L=5, num_kv_heads=2
last row per head attn[:, 4, :4]:
  h=0: [0.1, 0.2, 0.3, 0.3]
  h=1: [0.2, 0.2, 0.2, 0.3]
  h=2: [0.15, 0.15, 0.3, 0.3]
  h=3: [0.1, 0.3, 0.2, 0.3]
mean over heads = [0.1375, 0.2125, 0.25, 0.3]
max+1 = 1.3
scores = [[0.1375, 0.2125, 0.25, 0.3, 1.3],
          [0.1375, 0.2125, 0.25, 0.3, 1.3]]
```

## Constraints

- CPU float32; `attn` is already the post-softmax, causally masked weight matrix.
- $H_q$ need not equal $H$; the scores are **first averaged over $H_q$** and then copied $H$ times.
