# Solution Notes · Quest Page-Level Bounds

## In one sentence

Replace the query with an **attention upper bound** computed from each page's per-channel key extremes, then select pages by that bound; the last page is always loaded and never competes.

## Why the bound holds

For a single channel \(d\), once the sign is folded into the keys, the extreme inside a page is

\[
\max_{i\in\text{page}}\bigl(s_d\,k_{i,d}\bigr),\qquad s_d=\operatorname{sign}(q_d)
\]

so that channel contributes at most \(q_d s_d \cdot \max_i(s_d k_{i,d})\):

- \(q_d>0\Rightarrow s_d=+1\): the bound is \(q_d\max_i k_{i,d}\), and \(q_d k_{i,d}\le\) it;
- \(q_d\le 0\Rightarrow s_d=-1\): the bound is \(q_d\min_i k_{i,d}\); both factors are non-positive, so the inequality still holds.

Summing over channels gives the page score, and that score is **never below** the true \(q\cdot k\) of any token inside the page — the test asserts exactly this inequality.

As one matmul: `(q*sign) @ page_max^T`, where `page_max` is the per-page amax of `k*sign`. No softmax, no `1/√d`: this is a bound, not an attention score.

## Three conventions that are easy to get wrong

1. **Zero goes negative.** Upstream writes `sign = (q > 0) + (~(q > 0)) * -1`, so `q_d = 0` yields `-1`. `torch.sign` would give 0 and erase that channel's contribution, immediately skewing the selection.
2. **The last page is always loaded but never selectable.** `scores` has one column fewer than there are pages (`[:, :, :cur_page_nums - 1]`), because the last page was just written and only takes part from the next decode step; `selected` nonetheless always carries it. The real contest is therefore for `K = min(page_budget, P) - 1` slots — cap the budget by the page count first, then subtract the always-loaded page.
3. **The pad sentinel.** Slots beyond the last real key are padded with the input dtype's `finfo.min` before the amax. That value cannot affect the bound (real values are always larger), but it must be a very negative number rather than 0: for a channel whose whole page is negative, a 0 sentinel lifts `page_max` to 0 and the bound stops being a bound.

## What is judged

- `sign` takes only ±1 and zero must land on `-1`;
- `page_max` is the amax of `k*sign`, not of `k` — which end of a page the extreme comes from is decided entirely by the query's sign;
- the column count of `scores` and the length of `selected` both follow `min(page_budget, P)`;
- every `selected` row is ascending, duplicate-free, and ends with `P-1`;
- ties go to the smaller page index: upstream's `torch.topk` leaves tie order undefined, so this problem fixes it, and the test constructs an all-equal-scores case with all-ones keys.

## Boundaries

- `P < 2` raises: with a single page, "exclude the last page" leaves zero contest slots, and the upstream controller likewise has `assert cur_page_nums > 1`.
- `page_budget >= P` gives `P-1` slots, i.e. load everything; `page_budget = 1` gives zero slots, i.e. only the last page.
- Single query (`Lq = 1`): the `sign` is broadcast along the kv axis. With `qo_len > 1` each query row would need its own signs, so one `page_max` cannot serve them all — which is why the upstream CPU reference ends with `squeeze(1)`.
- Everything is computed in float64: mixing `finfo.min` with real keys is harmless, but score accumulation over a long `D` is sensitive in float32.

## Complexity

Each step costs \(O(P\cdot D)\) to fold the metadata plus \(O(H P D)\) for the dot products, versus \(O(H L_{\text{kv}} D)\) for full attention — the saving grows with `L_kv`. Upstream ships this as a CUDA kernel; this problem only judges the scoring convention.

## Source

The CPU reference in `mit-han-lab/Quest` (arXiv 2406.10774):

- `quest/tests/test_estimate.py::_ref_cpu_estimate` L17: the whole scoring path;
- L40 `sign = (q > 0) + (~(q > 0)) * -1`, L41 `max_key = k * sign`, L42 `postive_query = q * sign`;
- L44 `padding_length = page_size - ((kv_len - 1) % page_size + 1)`;
- L63 `page_max_key = max_key.reshape(..., page_size, head_dim).amax(dim=-2)`;
- L72 `approx_attn_paged = torch.matmul(postive_query.float(), page_max_key.transpose(1, 2))`;
- L73 `[:, :, :cur_page_nums - 1]  # remove the last page`, L75 `squeeze(1)`.

The budget and always-loaded conventions come from `quest/utils/controller.py`: L112 `inference_page_budget = min(self._page_budget, cur_page_nums)`, L115 `kv_indptr_for_approx_decode = [0, inference_page_budget - 1]`, L118-119 topk buffers of length `inference_page_budget - 1`, L137-142 `need_estimate()` returning `cur_page_nums > inference_page_budget`.

**Additions of this problem** (not written upstream, pinned explicitly): ties go to the smaller page index; `page_size`/`page_budget` are explicit arguments instead of controller state; `k` is an argument instead of first running attention.

## Related problems

- `kv-chunk-independent-topk`: chunked top-k over scores that are **already computed** — pure topk bookkeeping; this problem is about where the scores come from.
- `kv-snapkv-pooling`: SnapKV scores with real attention inside an observation window; this problem never computes attention at all.
- `kv-h2o-heavy-hitters`: evicts tokens by accumulated attention at token granularity; here the unit is a page and nothing is evicted.
