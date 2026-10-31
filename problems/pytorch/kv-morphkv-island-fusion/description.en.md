# KV Cache: MorphKV Constant-Cache Island Fusion

## Problem

**MorphKV** ([arXiv:2503.00979](https://arxiv.org/abs/2503.00979), repository [ghadiaravi13/MorphKV](https://github.com/ghadiaravi13/MorphKV)): compresses the KV cache to a **constant size**. Unlike "score every token once and keep the top-K", MorphKV's hierarchical path first keeps the most important tokens as **singletons**; the remaining tokens are split by the important positions into contiguous **islands**, each island is fused into **one slot** via importance-weighted averaging, and then the top-K islands are retained by their mean importance.

The final layout is fixed to `[fused islands | important singletons | recent window]`—fused islands carry a "long-range summary", singletons keep exact spike tokens, and the window keeps generation from losing its recent memory.

## Rules

Implement `solve(attn, keys, values, max_capacity, window_size, imp_budget=0.5, fuse_temperature=1.0, value_norms=None)`.

**Inputs**

- `attn`: $(H, Q, P)$, the **pre-softmax logits** of the `Q` window queries over the first `P` keys (one slice per KV head).
- `keys` / `values`: $(H, P + W, D)$, the **post-RoPE** KV cache; the last `W = window_size` entries are the window and are never compressed.
- `max_capacity`: total slot budget after compression; `window_size`: `W`; `imp_budget`: share of important singletons.
- `fuse_temperature`: temperature of the importance softmax; `value_norms`: $(H, P)$ value norms, defaulting to `values[:, :P, :].norm(dim=-1)`.

**Steps**

1. Importance: `importance = softmax(attn / fuse_temperature, dim=-1).mean(dim=-2) * value_norms`, shape $(H, P)$.
2. Budget: `n_budget = int(max_capacity * imp_budget)`, `n_fused = max_capacity - window_size - n_budget` (must be $\ge 0$), `n_important = min(n_budget, P)`.
3. Singletons: take the top `n_important` positions by **descending** importance (ties pick the smaller index), then return them in **ascending** position order.
4. Islands: the island id of past position $p$ is the number of important tokens to its left (excluding $p$; range $[0, n_{important}]$). Consecutive unimportant tokens share an id; adjacent important tokens leave **holes** in the id range.
5. Fusion: within an island $w_i = \mathrm{imp}_i / \sum_{j \in \text{island}} \mathrm{imp}_j$, `fused_key = Σ w_i · k_i`, `fused_value = Σ w_i · v_i` (per head, per dimension). The island score is its **mean** importance; empty islands score $-\infty$.
6. Island selection: `K = min(n_fused, n_important + 1)`, take `K` islands by descending score (ties pick the smaller island id), then order them by **ascending** island id; remaining slots are zero-filled.
7. Layout: `[fused islands (n_fused) | important singletons (n_important) | window (W)]`, with the window copied verbatim.

**Return** `dict`:

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `keys` | $(H, n_{fused}+n_{important}+W, D)$, input dtype | Compressed K |
| `values` | same | Compressed V |
| `important_idx` | $(H, n_{important})$, int64 | Original positions of the singletons, ascending |
| `island_idx` | $(H, n_{fused})$, int64 | Island id per slot, `-1` for padding |
| `island_counts` | $(H, n_{fused})$, int64 | Tokens absorbed per slot, `0` for padding |

## Examples

```
H=1, Q=2, P=6, W=2, D=2, max_capacity=5, imp_budget=0.5, fuse_temperature=1.0
attn[0] = [[0.0, 0.7, 2.0, -1.0, 1.0, -2.0],
           [0.0, 0.7, 2.0, -1.0, 1.0, -2.0]]
keys[0] = past [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11]]
        + window [[12,13], [14,15]]
values = 10 * keys, value_norms = [1, 1, 1, 1, 1, 1]

n_budget = int(5*0.5) = 2, n_fused = 5-2-2 = 1, n_important = 2
importance = [0.073398, 0.147806, 0.542344, 0.027002, 0.199517, 0.009933]
important_idx = [2, 4]
islands: id0 = {0,1} (mean 0.110602) > id1 = {3} (0.027002) > id2 = {5} (0.009933)
-> pick id0, weights [0.331812, 0.668188]
fused_key   = 0.331812*[0,1] + 0.668188*[2,3] = [1.336376, 2.336375]
fused_value = 10 * fused_key                  = [13.363755, 23.363756]

keys    = [[1.336376, 2.336375], [4,5], [8,9], [12,13], [14,15]]
values  = [[13.363755, 23.363756], [40,50], [80,90], [120,130], [140,150]]
island_idx = [[0]], island_counts = [[2]]; singletons occupy slots 1-2, window the last 2 slots
```

## Constraints

- CPU float32 / float64; every float input (including `value_norms`) must share one dtype; the output keeps that dtype while accumulation happens in float64.
- $P \ge 1$, $Q \ge 1$, $D \ge 1$, $W \ge 0$, and `keys.shape[1] == P + W`.
- `max_capacity ≥ window_size + int(max_capacity * imp_budget)`, otherwise `ValueError`; `0 ≤ imp_budget ≤ 1`, `fuse_temperature > 0`, `value_norms` strictly positive.
- No input may be modified in place.
