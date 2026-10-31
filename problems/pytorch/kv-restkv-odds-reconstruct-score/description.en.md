# KV Cache: ReST-KV Odds Score and Output-Reconstruction Multiplier

## Problem

ReST-KV (*Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing*, arXiv 2605.08840, ICLR 2026) replaces SnapKV's "column-sum of the recent-window attention" with an importance score that is a product of two terms:

- **odds term** `p / (1 − p)`: the closer a history key is watched by the queries in the recent window, the faster this blows up;
- **reconstruction term** `‖W_o·v̄ − W_o·v_l‖₂`: the further this key's value sits from the representative value compressed out of the whole history, the more unique — and the more worth keeping.

The score is then smoothed once along the token axis (spatial smoothing), a top-k picks the history entries, and the `W` window entries are always kept.

Upstream exposes the aggregation order as a switch: `metric_mode="after"` first aggregates the `W` attention rows into `p̄` and only then takes odds and the multiplier, while `"before"` computes the full per-query score first and aggregates over queries afterwards. The two are **not equivalent** — `before` averages `A(q,l)/(1−A(q,l))`, `after` takes the odds of the average, and the representative value is `p̄`-weighted in one branch and a per-query `A(q,·) V` product in the other. Different scores and different rankings from the same input are the main grading signal here.

## Task

Implement `solve(query, key, value, wo, window_size=32, max_capacity_prompt=4096, kernel_size=5, pooling="avgpool", metric_mode="after", use_wo=True, use_norm=False, use_ema=True, alpha=0.3, tau=1.0)`:

- `query` / `key` / `value`: same shape, same dtype, float32 or float64 `(B, H, S, D)`; the call site upstream already expands K/V over query heads, so this problem does no GQA repeat — `H` is always the query-head count.
- `wo`: a 2-D `(C_out, H * D)` output-projection weight of the same dtype; unused when `use_wo=False` but still validated.
- `window_size=W`, `max_capacity_prompt`, `kernel_size` are positive integers; `pooling ∈ {"avgpool","maxpool","none"}`, `metric_mode ∈ {"before","after"}`; `use_wo` / `use_norm` / `use_ema` are booleans; `alpha ∈ [0,1]`, `tau ≥ 0`. Write `L = S − W` and `k = max_capacity_prompt − W`.

Rules (with `S == 7, W == 2, L == 5, k == 3` every index below lives in the `L`-long history region):

1. `A = (query[..., -W:, :] @ key.transpose(2,3)) / sqrt(D)`, shape `(B,H,W,S)`; **add** `finfo(A.dtype).min` on the strict upper triangle of the bottom-right `W×W` block (row `i` inside the window may only see the first `i+1` keys), then `softmax(dim=-1, dtype=float32)` and cast back to the original dtype — masked entries get probability exactly 0.
2. `A_past = A[:, :, -W:, :-W]`, shape `(B,H,W,L)`.
3. `metric_mode == "after"`: `p̄` is the mean of `A_past` over the query axis, or the EMA when `use_ema` (`e_0 = A_past[0]`, `e_i = alpha·A_past[i] + (1−alpha)·e_{i−1}`, and **only the last state is returned**); the representative value is `v̄ = einsum("bhl,bhld->bhd", p̄, value[:, :, :-W])` (its weights sum to `< 1`, it is not a convex combination); `score_l = p̄_l/(1−p̄_l)` times `‖W_o v̄ − W_o V_l‖₂ ** tau` (`use_wo`) or `‖v̄ − V_l‖₂` (without `use_wo`, and **this branch has no tau**). `W_o` is first reshaped to `(C_out,H,D)`, contracted per head over the `(h,d)` axes into output channels, then L2 over the last axis.
4. `metric_mode == "before"`: `O = A @ value` (the full softmax times the **whole** value cache, shape `(B,H,W,D)`); `u = A/(1−A)`, shape `(B,H,W,S)`; the multiplier is per query and per position: `‖W_o O_q − W_o V_l‖₂` with `use_wo`, else `‖O_q − V_l‖₂`; with `use_norm` divide additionally by `‖W_o O_q‖₂` (by `‖O_q‖₂` when `use_wo` is off); **only then** slice `[-W:, :-W]` to `(B,H,W,L)` and average or EMA over the query axis to get `score`. `use_norm` and `tau` only mean something in the `before` branch (the `after` branch never reads `use_norm`).
5. Smoothing: `avgpool` / `maxpool` are 1-D poolings with `kernel_size` (odd), `padding=kernel_size//2`, `stride=1`; `(B,H,L)` is read as `(batch, channel, length)`, i.e. per head independently; `none` leaves the score alone.
6. Selection: only when `S >= max_capacity_prompt`, take the top-`k` of `pooled` (descending score, **ties go to the smaller index**); `kept_positions` is that batch followed by the window positions `L..S-1` ascending. When `S < max_capacity_prompt` nothing is selected (`selected_indices` has length 0, `kept_positions = range(S)`), but steps 1–5 are still computed and returned.

Return 5 keys: `attn_weights (B,H,W,S)`, `important_score (B,H,L)`, `pooled_score (B,H,L)`, `selected_indices` int64 `(B,H,k)`, `kept_positions` int64 `(B,H,k+W)`; the three float keys keep the input dtype, and history indices start at 0 so they are also absolute positions.

## Example

```
setup: B=H=1, S=7, D=2, W=2, L=5, k=3
  query[0,0] = [[0.6,0.2], [0.1,0.9], [-0.4,0.3], [0.8,-0.1], [0.0,0.5], [-0.7,0.4], [0.2,0.2]]
  key[0,0]   = [[0.3,-0.2], [0.7,0.1], [-0.5,0.6], [0.2,0.2], [0.9,-0.3], [-0.1,0.8], [0.4,0.0]]
  value[0,0] = [[0.5,0.5], [-0.2,0.7], [0.9,0.1], [0.0,-0.6], [0.3,0.3], [-0.8,0.2], [0.6,-0.4]]
  wo         = [[0.4,-0.3], [0.2,0.7]]                # (C_out=2, H*D=2)

A: solve(query, key, value, wo, window_size=2, max_capacity_prompt=5, kernel_size=3,
         pooling="avgpool", metric_mode="after", use_wo=False, use_ema=False)
  attn_weights row0 = [0.1375, 0.1228, 0.2562, 0.1618, 0.0993, 0.2224, 0.0]     # q = 5
  attn_weights row1 = [0.136, 0.1502, 0.136, 0.1419, 0.146, 0.1481, 0.1419]     # q = 6
  important_score   = [0.0705, 0.1153, 0.1577, 0.1383, 0.0247]
  pooled_score      = [0.0619, 0.1145, 0.1371, 0.1069, 0.0543]                  # kernel 3, pad 1
  selected_indices  = [2, 1, 3]        kept_positions = [2, 1, 3, 5, 6]

B: same inputs, metric_mode="before"           # per-query odds x diff, then mean
  important_score = [0.0801, 0.1053, 0.191, 0.1352, 0.0314]      # differs from A
  pooled_score    = [0.0618, 0.1254, 0.1438, 0.1192, 0.0555]

C: pooling="none", use_ema=True, alpha=0.5    # alpha=0.5 makes EMA equal the mean
  pooled_score    = [0.0705, 0.1153, 0.1577, 0.1383, 0.0247]     # == A's important_score
  selected_indices = [2, 3, 1]               kept_positions = [2, 3, 1, 5, 6]

D: solve(..., max_capacity_prompt=9)          # S = 7 < 9, nothing is selected
  selected_indices = []                      kept_positions = [0,1,2,3,4,5,6]
  important_score / pooled_score unchanged from A
```

## Constraints

- **The two branches are not equivalent**: `after` aggregates then takes odds, `before` takes odds then aggregates; `p/(1−p)` is convex, so the rankings may differ, and the representative value in `before` also carries the window keys (its `A @ value` runs over the **full** cache) while `after`'s `v̄` is weighted only by the history region.
- **`use_norm` only applies in `before`**: the `after` branch never reads it, so passing `True` changes nothing there.
- **`tau` only applies in `after` with `use_wo=True`**: neither the `before` branch nor `after` without `use_wo` takes a power.
- **EMA returns only the last state**: `alpha` is the weight of the **newest** item, so `alpha=1` means "only the last query in the window" and `alpha=0` means "only the first".
- **Pooling does not change the length**: with odd `kernel_size`, `padding = kernel_size//2` and `stride=1` the output stays `(B,H,L)`; when `L=1` the pooled value is the input itself (both sides are edge padding).
- **The mask is added, not assigned**: after adding `finfo.min` those entries come out of the float32 softmax as exactly 0, rows still sum to 1, and the number of visible keys grows by one per window row.
- **No in-place mutation**; `selected_indices` is in **selection order** (descending score), not ascending; in `kept_positions` the first part follows the selection order and the window part is ascending.
- **Raise ValueError** when: `query/key/value` are not 4-D, or shapes/dtypes disagree, or are not float32/float64; `wo` is not 2-D, has a mismatched dtype, or `shape[1] != H*D`; `S <= W`, `W < 1`, `max_capacity_prompt <= W` (upstream's `assert`), `kernel_size` even or `< 1`; unknown `pooling` / `metric_mode` strings; a switch that is not a bool; `alpha` outside `[0,1]` or `tau < 0`; integer parameters that are not `int` (`bool` counts as illegal).
