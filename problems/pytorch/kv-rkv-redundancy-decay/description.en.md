# KV Cache: R-KV Redundancy-Aware Scoring for Budgeted Eviction

## Problem

Long-context decoding grows the KV cache without bound, so methods such as H2O and SnapKV evict tokens by attention score. The catch is that high-attention tokens often cluster into near-duplicates: in a repetitive passage every token is attended, yet none of them adds information. R-KV (arXiv 2505.24133) therefore splits the eviction score into **attention minus redundancy**, so tokens that shout loudly but say nothing new leave first.

The convention in `rkv/compression/r1_kv.py::_compute_scores` plus `rkv/utils.py::cal_similarity` has four steps:

1. take only the last `window_size=8` queries, softmax over the first `L-window_size` keys, average over the query axis, then `max_pool1d(kernel=7, padding=3, stride=1)` along tokens;
2. normalise keys row-wise, take pairwise cosine, zero the diagonal, treat `> threshold` (default 0.5) as redundancy; for each row the `retain_direction` (default `"last"`) end of the hit indices is the representative, its similarity is zeroed, then take the column mean and a `softmax`;
3. `final = mix_lambda * attn - (1 - mix_lambda) * redundancy` with `mix_lambda` **0.07** by default;
4. keep the `budget - window_size` highest scores per head and always keep the last `window_size` tokens.

That 0.07 is the point of this problem: attention only nudges the ranking, **redundancy decides it** — under the same budget the duplicate twin is traded for a unique token with slightly less attention.

## Task

Implement `solve(k, q, v, budget=128, window_size=8, kernel_size=7, mix_lambda=0.07, threshold=0.5, retain_direction="last")`:

- `k`, `v`: `(L, H, D)`, same dtype (float32/float64);
- `q`: `(window_size, H, D)`, exactly the last `window_size` queries, **no causal mask** (every query may see all `N = L - window_size` keys);
- `L >= budget > window_size >= 1`, `kernel_size` a positive odd integer.

Return a 6-key dict, with `N = L - window_size`:

- `attn`: `(H, N)`, the pooled attention score of step 1;
- `redundancy`: `(H, N)`, the redundancy score of step 2;
- `final`: `(H, N)`, `mix_lambda*attn - (1-mix_lambda)*redundancy`;
- `kept`: `(H, budget)` int64; the first `budget-window_size` entries are indices picked by descending `final` (ties go to the smaller index), the last `window_size` entries are `N..L-1` in original order;
- `kept_keys`, `kept_values`: `(H, budget, D)`, the compressed cache gathered by `kept`.

Raise `ValueError` on: not 3-D, mismatched dtype or non-float, head count / head_dim mismatch, `L < budget` (upstream returns the cache untouched without scoring), `budget <= window_size`, `kernel_size` not a positive odd integer, `q` row count different from `window_size`, `mix_lambda` outside `[0,1]`, `threshold` outside `[-1,1]`, `retain_direction` other than `last`/`first`.

## Example

```
k = [[1,0], [1,0], [0,1], [1,1]]          # L=4 keys
q = [[1,0]]                               # window_size=1: only the last query
v = [[0,1], [2,3], [4,5], [6,7]]
budget=3, window_size=1, kernel_size=1, threshold=0.5, mix_lambda=0.07   # N=3, pick 2

cosine, diagonal cleared                    rep = max hit index in the row
        k0      k1      k2      k3
k0  [ 0.000   1.000   0.000   0.707 ]  -> 3     k0 and k1 are duplicates
k1  [ 1.000   0.000   0.000   0.707 ]  -> 3
k2  [ 0.000   0.000   0.000   0.707 ]  -> 3     k2 only has the window key k3
k3  [ 0.707   0.707   0.707   0.000 ]  -> 2     k3's rep is k2, not k3 itself

column mean after zeroing reps = [0.4268, 0.4268, 0.0000, 0.0000]
redundancy = softmax([0.4268, 0.4268, 0, 0])[:3] = [0.3026, 0.3026, 0.1975]
attn       = softmax([1, 1, 0] / sqrt(2))        = [0.4011, 0.4011, 0.1978]
final      = 0.07 * attn - 0.93 * redundancy     = [-0.2533, -0.2533, -0.1698]
kept       = [2, 0, 3]     # k2 is unique: it wins although k0/k1 have more attention
```

k0 and k1 tie for the highest attention yet both sink to -0.2533 through the redundancy term, so the budget keeps k2 and one duplicate. Also note that the three returned `redundancy` values sum to only 0.8025: the softmax runs over **all 4** keys, so the window key k3 is part of the denominator.

## Constraints

- **The tiny lambda is not a typo**: `final = 0.07*attn - 0.93*redundancy`. Weight the two terms equally or drop redundancy and the selected set changes completely.
- **The representative is the max of the hit-index row**: non-hits are filled with 0, so `retain_direction="last"` returns 0 when a row has no hit, and `"first"`'s min is always 0 (it clears column 0 every time) — that is the literal upstream behaviour, copy it.
- **The softmax happens before the slice**: `redundancy` is `softmax(dim=-1)` first and only then truncated to drop the last `window_size` columns, so the row sums to less than 1; it is not a conditional distribution over the window.
- **threshold is strict**: a cosine of exactly 0.5 is not redundancy; zero the diagonal before comparing.
- **The window is hard-kept**: the tail of `kept` is always `N..L-1`, the head is sorted by descending score (ties to the smaller index, pinning what `torch.topk` leaves undefined); with `cache_len == budget` nothing is evicted.
