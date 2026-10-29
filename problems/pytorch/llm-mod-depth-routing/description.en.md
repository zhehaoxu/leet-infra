# Mixture-of-Depths: Per-Layer Top-k Depth Routing and Residual Flow

## Description

Mixture-of-Depths (Raposo et al., 2024, arXiv 2404.02258) moves compute allocation from the **expert axis** to the **depth axis**: each block carries a scalar router, and only the tokens with the highest router scores actually enter that block while the rest flow through the residual branch. Because each layer computes only a subset of tokens, different tokens travel different depths through the same graph — hence "mixture of depths".

This problem replaces the real Transformer block with a deterministic stand-in: the selected tokens form a subsequence that goes through one **causal self-attention with a shared q/k/v projection**, \(F=P H\). Implement the full per-layer forward pass — scoring, capacity top-k, subsequence attention, score-scaled write-back — and return the routing decisions of every layer as well.

## Rules

- The full signature and parameter meanings live only in the starter docstring; that is the sole contract.
- Layer scores \(s_i=x_i\cdot w_l\) must use the **current state entering that layer**; tokens updated by the previous layer change the next layer's routing.
- Capacity is \(k_l=\min(C_l,T)\): take the first \(k_l\) under \((s_i\text{ descending},\ \text{token index ascending})\). Ties go to the smaller index — a determinism convention added here, since the paper only says surplus tokens are dropped and leaves ties to implementation order.
- Selected indices are re-ordered by **ascending original index** before entering the block, so the causal mask applies to subsequence positions rather than original positions.
- Block: \(H=\tilde X W_l\), \(A=HH^{\mathsf T}/\sqrt d\), \(A_{ij}=-\infty\) for \(j>i\), \(P=\mathrm{softmax}(A)\), \(F=PH\).
- Write-back: \(x_i\leftarrow x_i+s_i\cdot F_{\text{row}(i)}\). The router score is a real-valued **multiplier on the block output** (Eq. (1) of the paper), not a binary gate.
- Unselected tokens must keep their exact value (no multiply-by-zero approximation); a layer with capacity 0 computes nothing but must still report its `router_scores`.
- Return `(output, route_mask, router_scores)`: `output` is `(T, d)` float32, `route_mask` is `(L, T)` bool, `router_scores` is `(L, T)` float32 (scores used by each routing decision); compute internally in float64.

## Example

```
hidden     = [[1, 0], [0, 1], [1, 1]]
routers    = [[1, 0], [0, 1]]
blocks     = [I, I]
capacities = [2, 1]

router_scores = [[1.000000, 0.000000, 1.000000],
                 [0.000000, 1.000000, 1.669762]]
route_mask    = [[1, 0, 1],
                 [0, 0, 1]]
output        = [[2.000000, 0.000000],
                 [0.000000, 1.000000],
                 [5.339523, 4.457865]]
```

Layer 0 scores are \([1,0,1]\); with capacity 2 and ties going to the smaller index the routed set is \(\{0,2\}\). The subsequence is \([[1,0],[1,1]]\), causal attention mixes the second token to \([1.0,0.669762]\), and the write-back with score 1 yields \([2.0,1.669762]\). Layer 1 scores the **updated** state as \([0,1,1.669762]\), capacity 1 picks index 2, and a single-row subsequence degenerates to \(F=x\), so \(x_2\leftarrow x_2+1.669762\cdot x_2=[5.339523,4.457865]\). Index 1 is never routed and stays at \([0,1]\).

## Constraints

- `hidden_states` must be a 2-D float32/float64 tensor `(T, d)` with `T >= 1`, `d >= 1`; `router_weights` is 2-D `(L, d)` and `block_weights` is 3-D `(L, d, d)` with `L >= 1`; all three must be finite.
- `capacities` must be a list/tuple of length `L` whose entries are `int` (`bool`, `float` and strings do not count) and \(\ge 0\); values above `T` are clamped to `T`.
- Any dimension/shape mismatch, non-floating tensor, NaN/±inf, or the invalid `capacities` cases above raise ValueError.
- Per layer the cost is \(O(Td+c^2 d)\) with \(c=k_l\), so \(O(L(Td+c^2d))\) overall, and space is \(O(Td+cd)\).
