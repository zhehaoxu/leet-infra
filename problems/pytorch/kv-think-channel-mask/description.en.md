# ThinK: Channel Pruning of Keys along head_dim

## Problem

**ThinK** ([arXiv:2407.21018](https://arxiv.org/abs/2407.21018)) compresses not the **token dimension** (sequence length) but the **channel dimension** (`head_dim`): within each head, some dimensions contribute very little to the attention score and can be dropped entirely. It is orthogonal to token-wise methods such as SnapKV/KNorm and can be cascaded with them.

The kvpress scoring formula (**importance of channel $d$ = mean square of the query on that channel × mean square of the key on that channel**):

$$ s_{h,d} = \Big(\frac{1}{W}\sum_{w=1}^{W} q_{h,w,d}^2\Big) \cdot \Big(\frac{1}{L}\sum_{\ell=1}^{L} k_{h,\ell,d}^2\Big) $$

where $q_{h,w,d}$ is the $d$-th dimension of the $w$-th query inside the window **after RoPE**. The $n_\text{pruned} = \lfloor D \cdot R\rfloor$ smallest channels are **set to 0** (the shape is kept and only the values are zeroed, which is memory-equivalent to dropping the dimensions outright).

## Rules

Given `queries` $(H, W, D)$, `keys` $(H, L, D)`, and `compression_ratio` $R \in [0, 1)$:

1. `q_norm = (queries ** 2).mean(dim=1)` → $(H, D)$
2. `k_norm = (keys ** 2).mean(dim=1)` → $(H, D)$
3. `key_scores = q_norm * k_norm` → $(H, D)$
4. `n_pruned = int(D * R)`
5. For each head, select the $n_\text{pruned}$ channels with the **smallest** `key_scores` and **set the corresponding channels of `keys` to 0**

Return `masked_keys` of shape $(H, L, D)$.

## Examples

```
H=2, W=2, D=4, L=3, compression_ratio=0.5  → n_pruned=2

queries[h=0] = [[1.0, 0.5, 0.0, 0.5],   queries[h=1] = [[0.0, 0.0, 1.0, 1.0],
              [0.5, 1.0, 0.5, 0.0]]                     [1.0, 0.5, 0.5, 0.0]]

keys[h=0] = [[1,2,3,4],[0.5,0.5,0.5,0.5],[-1,1,-1,1]]
keys[h=1] = [[2,0,0,4],[0,0,1,0],[1,1,2,3]]

h=0: q_norm=[0.625,0.625,0.125,0.125]·k_norm=[0.75,1.75,3.4167,5.75]
     key_scores=[0.46875, 1.09375, 0.42708, 0.71875]
     smallest 2 → channels {2, 0} → zeroed
h=1: q_norm=[0.5,0.125,0.625,0.5]·k_norm=[1.6667,0.3333,1.6667,8.3333]
     key_scores=[0.8333, 0.0417, 1.0417, 4.1667]
     smallest 2 → channels {1, 0} → zeroed

masked_keys[h=0] = [[0,2,0,4],[0,0.5,0,0.5],[0,1,0,1]]
masked_keys[h=1] = [[0,0,0,4],[0,0,1,0],[0,0,2,3]]
```

## Constraints

- CPU float32; kv-group is treated as 1 (number of query heads = number of key heads).
- `topk(largest=False)` selects the smallest; ties are broken by ascending index.
