# KV Cache: FastKV Token-Selective Propagation Across Layers

## Problem

**FastKV** (Jo et al., 2025, [arXiv:2502.01068](https://arxiv.org/abs/2502.01068)) observes that KV cache compression need not be done independently at every layer. At one layer (the TSP layer, Token-Selective Propagation) it selects a token subset using scores summed across KV heads, and **subsequent layers carry only that subset**. Because those layers see a shorter sequence, the retention rate can be relaxed from `retain_rate` to `retain_rate / tsp_rate`, keeping the absolute budget roughly unchanged. Inside each head it still follows the SnapKV-style rule "keep the highest-scoring past tokens plus the most recent `window_size` tokens".

This problem focuses on the **cross-layer selection and propagation** path only: `attn_caches` already contains per-layer, per-token scores, and `solve` computes budgets, selects tokens per head, chooses the TSP subset, and slices the matching `hidden_states` / `position_ids`.

## Rules

Implement `solve(attn_caches, q_len, tsp_idx, hidden_states, position_ids, window_size=8, retain_rate=0.1, tsp_rate=0.2)`:

### Inputs

- `attn_caches`: a non-empty `list` / `tuple` of length \(L\); item \(l\) is a float32 / float64 tensor of shape \((T_l, H_{\text{kv}})\) holding the importance score of every token for every KV head at that layer. All layers share one dtype and one \(H_{\text{kv}}\); layer 0 must have length `q_len`.
- `q_len`: `int >= 1`, the number of tokens in the current segment.
- `tsp_idx`: `int`, the TSP layer index, \(0 \le \text{tsp\_idx} < L\).
- `hidden_states`: \((q_len, D)\), same dtype as `attn_caches`; rows are selected according to the TSP choice.
- `position_ids`: \((q_len,)\) int64, the original absolute positions.
- `window_size`: `int >= 1`, the number of most recent tokens always kept per layer, default 8.
- `retain_rate`, `tsp_rate`: `int` / `float` in \((0,1]\), defaults 0.1 / 0.2.

First compute
\[
\text{tsp\_length}=\lfloor q\_len \times \text{tsp\_rate}\rfloor,\qquad
\text{gating}=[q\_len > \text{tsp\_length}].
\]
Propagation happens only when `gating` is true. If `gating` and \(\text{tsp\_length}\le \text{window\_size}\), raise `ValueError`.

### Computation

1. **Per-layer length**:
   - \(l \le \text{tsp\_idx}\) or `gating` is false: \(T_l = q\_len\);
   - \(l > \text{tsp\_idx}\) and `gating` is true: \(T_l = \text{tsp\_length}\).
   The supplied cache at layer \(l\) must match this length exactly; otherwise raise `ValueError`.
2. **Per-layer budget**:
   \[
   \text{rate}_l=
   \begin{cases}
   \text{retain\_rate}, & l \le \text{tsp\_idx},\\
   \text{retain\_rate}/\text{tsp\_rate}, & l > \text{tsp\_idx},
   \end{cases}
   \qquad
   \text{capacity}_l=\lfloor T_l \times \text{rate}_l\rfloor .
   \]
   If \(\text{capacity}_l \le \text{window\_size}\), raise `ValueError`.
3. **Per-layer, per-head retention**:
   - If \(T_l < \text{capacity}_l\): the capacity exceeds the cache, so **keep all** indices `0..T_l-1` unchanged.
   - Otherwise sort the first \(T_l-\text{window\_size}\) tokens by score in **descending** order (stable, ties to the smaller index), take the first \(\text{capacity}_l-\text{window\_size}\), merge them with the tail window `T_l-window_size .. T_l-1`, and sort ascending.
4. **TSP selection** (only when `l == tsp_idx` and `gating` is true):
   - **Sum the scores across KV heads** for the first \(T_l-\text{window\_size}\) tokens;
   - take the highest \(\text{tsp\_length}-\text{window\_size}\) (stable, ties to the smaller index);
   - merge with the tail window `q_len-window_size .. q_len-1` and sort ascending to obtain `tsp_indices`.
5. `hidden_out = hidden_states[tsp_indices]` and `position_out = position_ids[tsp_indices]`; positions **keep their original absolute values** and are not renumbered.

### Returns

A 5-key dict:

- `tsp_indices`: `None` (no propagation) or \(( \text{tsp\_length}, )\) int64, ascending;
- `hidden_out`: `None` or \(( \text{tsp\_length}, D)\) with the dtype of `hidden_states`;
- `position_out`: `None` or \(( \text{tsp\_length}, )\) int64, original absolute positions;
- `capacities`: \((L,)\) int64, the retention budget per layer;
- `kept`: a list of length \(L\); item \(l\) is \((H_{\text{kv}}, n_l)\) int64, each row ascending and duplicate-free, where \(n_l=\text{capacity}_l\), or \(n_l=T_l\) when \(T_l<\text{capacity}_l\).

No input may be modified in place.

## Example

```
q_len=12, L=3, tsp_idx=1, H_kv=1, window_size=2,
retain_rate=0.5, tsp_rate=0.25

attn_caches[0] = [[0.1], [0.2], [0.9], [0.3], [0.4], [0.8],
                  [0.5], [0.7], [0.6], [0.85], [0.0], [0.0]]     # (12,1)
attn_caches[1] = [[0.1], [0.5], [0.2], [0.3], [0.95], [0.4],
                  [0.6], [0.2], [0.7], [0.1], [0.0], [0.0]]      # (12,1)
attn_caches[2] = [[0.3], [0.2], [0.1]]                            # (3,1)
hidden_states  = [[0,100], [1,101], ..., [11,111]]                # (12,2)
position_ids   = [100, 101, ..., 111]                             # (12,)

tsp_indices  = [4, 10, 11]
hidden_out   = [[4.0, 104.0], [10.0, 110.0], [11.0, 111.0]]
position_out = [104, 110, 111]
capacities   = [6, 6, 6]

kept[0] = [[2, 5, 7, 9, 10, 11]]
kept[1] = [[1, 4, 6, 8, 10, 11]]
kept[2] = [[0, 1, 2]]
```

`tsp_length = floor(12 × 0.25) = 3`, and `gating = 12 > 3`, so layer 2 holds only 3 tokens. Layers 0 and 1 have length 12 and rate 0.5, giving capacity 6; layer 2 has length 3 and rate \(0.5/0.25=2\), giving capacity 6, which exceeds the cache, so all tokens are kept. Layer 0 picks the 4 highest-scoring tokens among the 10 past tokens (indices 2, 5, 7, 9) and appends the window `[10,11]`; layer 1 similarly yields `[1,4,6,8]`. The TSP layer is layer 1: after summing across KV heads it picks 1 past token (index 4, score 0.95) and appends the window `[10,11]`, producing `[4,10,11]`.

## Constraints

- CPU float32 / float64; every item of `attn_caches` must be a 2D tensor of the same dtype, with the same number of KV heads.
- `q_len >= 1`; `0 <= tsp_idx < len(attn_caches)`; `window_size >= 1`; `retain_rate, tsp_rate ∈ (0,1]`.
- `hidden_states` must be \((q_len, D)\) with the cache dtype; `position_ids` must be a 1D int64 tensor of length `q_len`; layer 0's cache length must equal `q_len`.
- `tsp_length = int(q_len * tsp_rate)` uses floor; `gating = q_len > tsp_length`. When `gating` is true, require `tsp_length > window_size`.
- Per layer, `capacity = int(length * rate)` uses floor; `capacity` must be strictly greater than `window_size`; if `length < capacity`, that layer keeps every token.
- Sorts are stable and ties go to the smaller index; every `kept` row is ascending and duplicate-free; the trailing `window_size` tokens are always kept.
- When no propagation happens, `tsp_indices` / `hidden_out` / `position_out` are all `None`; `capacities` and `kept` are always returned.
- Invalid shapes, dtypes, rates, or budgets (e.g. `tsp_rate=1.0` which disables propagation, `capacity <= window_size`, or a cache length inconsistent with the gating rule) raise `ValueError`.
