# MoE Parallelism: All-to-All Dispatch and Combine

## Problem

Under expert parallelism, $E$ experts are distributed over $P = E/\text{EPD}$ devices (each device holds $\text{EPD}$ experts, numbered contiguously). The token sequence is **split evenly along the $T$ dimension** across the $P$ devices; each device first performs local top-k routing, then a single all-to-all sends every (token, slot) to "the device hosting the target expert"; after the local device has executed them, another all-to-all sends them back to the source device, where they are combined with the gate weights.

## Rules

Convention: the expert output is "input + expert id": $\text{expert}_e(x) = x + e$ (the $e$ is the same on every device).

Implement `solve(hidden_states, topk_indices, topk_weights, num_experts, experts_per_device)`:
- `hidden_states` has shape $(T, H)$ float32
- `topk_indices` has shape $(T, k)$ long; `topk_weights` has shape $(T, k)$ float32 (**each row sums to 1**, already renormalized)
- `num_experts = E`; `experts_per_device = EPD`; `P = E // EPD`
- The number of tokens per device is `tokens_per_device = (T + P - 1) // P`; token $t$ starts on `src_device = t // tokens_per_device`
- Expert $e$ lives on `dst_device = e // EPD`

Return `(send_matrix, combined)`:
- `send_matrix` has shape $(P, P)$ long: `send_matrix[i, j]` = the number of **(token, slot) pairs** sent from device $i$ to device $j$ (already flattened by top-k, i.e. each token contributes $k$ entries).
- `combined` has shape $(T, H)$ float: $\text{combined}[t] = \sum_j \text{topk\_weights}[t, j] \cdot (\text{hidden}[t] + \text{topk\_indices}[t, j])$

## Examples

```
T=8, H=2, E=4, EPD=2 → P=2, tokens_per_device=4; k=2
hidden = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
topk_indices = [[0,1],[1,2],[2,3],[3,0],[0,2],[1,3],[2,0],[3,1]]
topk_weights = [[.6,.4],[.5,.5],[.7,.3],[.4,.6],
                [.5,.5],[.3,.7],[.55,.45],[.65,.35]]

send_matrix = [[4,4],[4,4]]
combined = [[1.4, 2.4], [4.5, 5.5], [7.3, 8.3], [8.2, 9.2],
            [10.0, 11.0], [13.4, 14.4], [14.1, 15.1], [17.3, 18.3]]
```

(`combined[t] = hidden[t] + Σ_j w[t,j] * idx[t,j]`, e.g. for t=0: `Σ w*idx = 0.6*0 + 0.4*1 = 0.4` → `1+0.4=1.4`, `2+0.4=2.4`.)

## Constraints

- CPU float32; `send_matrix.dtype == torch.long`.
