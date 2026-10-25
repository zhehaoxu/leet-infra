# MoE Routing: Capacity Dispatch and Token Dropping

## Problem

In a real MoE serving system, each expert can only consume a fixed number of tokens. **Capacity dispatch** sets an upper bound $C$ for each expert and drops the excess in **position order** (keeping the tokens that arrive first in the original batch). This is the classic drop-policy="position" of Shazeer OLNN / GShard.

## Rules

Implement `solve(logits, k, capacity_factor)`:

1. `probs = softmax(logits, -1)`; take `topk(probs, k)` per row to get `top_vals (T,k)` / `top_idx (T,k)`.
2. **renorm** to get `top_w = top_vals / top_vals.sum(-1, keepdim=True)`.
3. `routing_map (T,E) bool`: `routing_map[t, top_idx[t,j]] = True`.
4. `routing_probs (T,E)`: `routing_probs[t, top_idx[t,j]] = top_w[t,j]`, 0 elsewhere.
5. The capacity per expert is $C = \lceil T \cdot k / E \cdot \text{capacity\_factor} \rceil$.
6. **Position-based drop**: scan `routing_map` in token index order $t = 0, 1, \ldots, T-1$, keep the first $C$ True entries per expert, and set the rest to False.
7. `final_map (T,E) bool` = the retained routing mask; `final_probs (T,E)` = `routing_probs * final_map.float()`; `dropped = int(routing_map.sum()) - int(final_map.sum())`.

Return `(final_map, final_probs, dropped)`.

## Examples

```
logits = [[2,0,0,0], [1,0,0,0], [0,2,0,0], [0,3,0,0],
          [0,0,2,0], [0,0,1,0], [0,0,0,2], [0,0,0,3]]
k=1, capacity_factor=0.5

C = ceil(8*1/4*0.5) = 1
final_map = [[1,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
             [0,0,1,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]
dropped = 4
```

Each of the 4 experts wants to consume 2 tokens, but the capacity is only 1, so the first one is kept (t=0/2/4/6) and the second is dropped (t=1/3/5/7).

## Constraints

- CPU float32; `final_map.dtype == torch.bool`; ties are broken in favor of the smaller expert index.
