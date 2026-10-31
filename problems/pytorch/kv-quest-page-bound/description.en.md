# KV Cache: Quest Page-Level Query-Aware Bounds and Budgeted Page Selection

## Problem

Every decode step has to read the whole KV cache, which is the dominant bandwidth cost of long-context inference. Quest (2406.10774) splits the cache into fixed-size pages and keeps per-page, per-channel key extremes; each decode step uses the query and those extremes to **estimate an attention upper bound**, and only sends the highest-scoring pages into real attention.

The bound comes from a simple fact. For a fixed head and channel \(d\), the keys of one page lie in \([\min_i k_{i,d},\ \max_i k_{i,d}]\), so that page can contribute at most

\[
\sum_{i\in\text{page}} q_d\,k_{i,d}\ \le\ q_d\cdot
\begin{cases}\max_i k_{i,d}, & q_d > 0\\[2pt] \min_i k_{i,d}, & q_d \le 0\end{cases}
\]

Multiplying the sign of \(q\) into the keys and taking a per-page amax tells you which end of the page each channel should contribute; one matmul with \(q\) then scores every page. The upstream CPU reference `quest/tests/test_estimate.py::_ref_cpu_estimate` is exactly this, and this problem copies its conventions line by line.

Three details that are easy to get wrong:

1. \(q_d = 0\) falls into the **negative** branch;
2. both the scoring and the topk **skip the last page** — it is the page just written, it only takes part from the next decode step on, yet it is **always loaded**;
3. the budget is capped by the current page count first, and then reduced by that always-loaded last page.

## Task

Implement `solve(q, k, page_size=16, page_budget=1)`:

- `q`: `(Lq, H, D)` float32/float64, NHD layout, **single decode query only — `Lq` must be 1**;
- `k`: `(Lkv, H, D)` same dtype, `Lkv >= 1`;
- `page_size`, `page_budget`: positive integers.

The page count is \(P=\lceil \mathrm{Lkv}/\texttt{page\_size}\rceil\) and must satisfy \(P\ge 2\). Slots beyond the last real key are padded with the input dtype's `finfo.min`. All arithmetic inside is float64; return values use the input dtype.

Return a 4-key dict:

- `sign`: `(H, D)`, `+1` where `q_d > 0`, `-1` otherwise (including 0);
- `page_max`: `(H, P, D)`, the per-page amax of `k * sign`;
- `scores`: `(H, P-1)`, `(q * sign) @ page_maxᵀ` with the last-page entry **dropped**;
- `selected`: `(H, K+1)` int64, each row ascending, where `K = min(page_budget, P) - 1` page indices are picked by descending score (ties go to the smaller index) and `P-1` is always appended.

Raise `ValueError` on: not 3-D, `Lq != 1`, `Lkv < 1`, head count or head_dim mismatch, differing dtype or a dtype other than float32/float64, `page_size`/`page_budget` not a positive integer, `P < 2`.

## Example

```
q = [[1.0, -2.0]]              # Lq=1, H=1, D=2
k = [[1.0, 3.0],
     [2.0, 1.0],
     [-1.0, -0.5],
     [0.5, -0.2],
     [9.0, 0.5],
     [9.0, -1.0]]              # Lkv=6
page_size   = 2                # P = 3
page_budget = 2                # K = min(2, 3) - 1 = 1

sign     = [[[1, -1]]]         # q_1 = -2 < 0 -> -1
max_key  = k * sign = [[[1, -3], [2, -1], [-1, 0.5], [0.5, 0.2], [9, -0.5], [9, 1]]]
page_max = [[[2, -1], [0.5, 0.5], [9, 1]]]      # per-channel amax of pages 0 / 1 / 2
scores   = (q*sign) @ page_max^T
         = [[[0.0, 1.5, 11.0]]][:, :, :2] = [[[0.0, 1.5]]]
selected = [[[1, 2]]]          # page 1 wins the contest; page 2 rides along
```

Page 2 has a much larger bound (11.0) than the winning page 1 (1.5), but its column never enters `scores`: the last page is loaded anyway, letting it compete would waste a slot.

## Constraints

- **Zero goes negative**: the sign is `(q > 0) + (~(q > 0)) * (-1)`, so `q_d = 0` yields `-1`. Using `torch.sign` gives 0, which collapses the page's bound to 0 and never selects the `min` end.
- **The last page is always loaded but never selectable**: `scores` has one column fewer than there are pages; the last entry of every `selected` row is always `P-1`, and each row is ascending.
- **Cap the budget by the page count, then subtract one**: `K = min(page_budget, P) - 1`. With `page_budget >= P` you get `K = P-1` (load everything); with `page_budget = 1` you get `K = 0` (only the last page).
- **Do not replace the pad sentinel with 0**: slots beyond the last real key are padded with `finfo(dtype).min`. Real values dominate the amax, but a 0 sentinel would lift `page_max` to 0 for a channel whose whole page is negative, and the "upper bound" is no longer an upper bound.
- **The score is a bound, not attention**: no softmax, no division by `√d` — just the sign-weighted dot product, growing linearly in `D` and in the key magnitude.
- **Ties go to the smaller page index**: upstream's `torch.topk` does not define tie order, so this problem fixes it.
