# ObservedAttention: Mean Historical Attention (H2O Family)

## Problem Description

**ObservedAttention** (same origin as H2O [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)): the more important a key is, the more attention historical queries have poured into it. The score is **the attention each key received, summed, divided by the number of queries that could see it** — i.e. "on average, how much attention did each past step give it".

kvpress's `ObservedAttentionPress.score` takes three steps:
1. `scores = attentions.sum(2)` — sum over the query dimension → accumulated attention received by each key
2. `scores /= arange(L, 0, -1)` — position $j$ is seen only by the $L - j$ queries with index $\geq j$ (causal); dividing by $L - j$ gives the average
3. `view(H_kv, groups, L).mean(1)` — spread num_heads into num_kv_heads groups and average within each group

## Rules

Given `attn` of shape $(H_q, L, L)$ (**already softmaxed and causally lower-triangular**: `attn[i, j] = 0` if `j > i`) and `num_kv_heads = H`: return scores of shape $(H, L)$.

- `groups = H_q // H`
- For each head $h$ and position $j$:
$$ \text{score}_{h,j} = \frac{1}{\text{groups}} \sum_{g=0}^{\text{groups}-1} \sum_{i=j}^{L-1} \text{attn}_{h\cdot \text{groups}+g,\, i,\, j} \;\Big/\; (L - j) $$

## Example

```
Hq=2, L=4, num_kv_heads=1, groups=2

head0 attn[0]:          head1 attn[1]:
[1.0, 0.0, 0.0, 0.0]    [1.0, 0.0, 0.0, 0.0]
[0.5, 0.5, 0.0, 0.0]    [0.4, 0.6, 0.0, 0.0]
[0.2, 0.3, 0.5, 0.0]    [0.3, 0.3, 0.4, 0.0]
[0.1, 0.2, 0.3, 0.4]    [0.2, 0.1, 0.3, 0.3]

For each key column j, sum along the query i dimension:
  head0 accumulated = [1.8, 1.0, 0.8, 0.4]
  head1 accumulated = [1.9, 1.0, 0.7, 0.3]
Divide by (L-j) = [4, 3, 2, 1]:
  head0 average = [0.45, 0.333, 0.40, 0.40]
  head1 average = [0.475, 0.333, 0.35, 0.30]
kv-group mean = average of the two heads:
  scores = [[0.4625, 0.3333, 0.375, 0.35]]
```

## Constraints

- CPU float32; `H_q` must be divisible by `H` (guaranteed by the problem's input).
- The causal mask has already been applied upstream; out-of-range entries are simply 0 and need no special handling.
