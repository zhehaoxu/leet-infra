# Notes: LLaVA-PruMerge IQR-Adaptive Token Count

## Approach

Validate the input first: `cls_attn` must be a 2-D float32/float64 tensor with `B,N ≥ 1` and all-finite values; no non-negativity constraint — percentiles and the strict-greater comparison are well defined for any reals.

The ratio step aligns line by line with upstream `clip_encoder.py::outlier_dectection`: `attn.to(dtype=torch.float32).cpu().numpy().flatten()` downcasts and flattens the whole `[B,N]`, `np.percentile(·, 25/75)` for Q1/Q3 (linear interpolation), `upper = Q3 + 1.5*(Q3-Q1)`, `np.where(flat > upper)` counts strictly-greater elements, `ratio = len/len`. Three points: the **float32 cast happens BEFORE the percentile** (float64 inputs lose precision observably; a `nextafter(1.0)` boundary test pins it — an outlier in float64, merged into 1.0 after the cast, ratio zero); the **flatten is cross-batch** (multi-batch cases distinguish the per-row mutation); the **comparison is strictly greater** (identical values give `IQR=0`, `upper=Q3`, zero outliers → `ratio=0` → legal empty selection — exactly the split from 536's `ValueError`).

The selection step mirrors 536: `argsort(descending=True, stable=True)` implements "value descending, index ascending"; the first `k` columns are re-sorted ascending before emission; pruned comes from a boolean mask plus stable `argsort`, taking the **LAST** `[:, keep:]` columns — the front segment is kept in ascending order (zeros first, stable), and slicing the first `tokens-keep` columns is wrong (this round's selftest tripped on exactly that, the sample output `[7,0,1,2,3,4,5]` exposed it). With `k=0` return the empty `[B,0]` and the full `0..N-1` directly.

## Deliberate differences from the official code

- **Order**. Upstream `torch.topk` emits kept indices in value-descending order; this task keeps 536's convention of re-sorting kept ascending so token order stays causal.
- **Legalizing `k=0`**. 536 pins the count to an argument, so `k=0` is undefined and raises; here the ratio is computed, `ratio=0` is a genuinely reachable state (uniform attention), the empty selection is observable upstream behaviour, hence legal and documented.
- **Batch axis**. Upstream inference runs `B=1`, so flattening is unobservable; this task pins cross-batch flattening explicitly and keeps the per-row ratio as a caught mutation.
- **Returning ratio**. Upstream passes it straight to `topk`; here it is the first return value, judged with exact equality (both sides are integer count ratios — no float tolerance involved).

## Independent verification

The oracle reuses no numpy/torch percentile or selection. The percentile is hand-written pure Python: `sorted`, virtual position `pos = q/100*(n-1)`, then numpy `_lerp`'s **two-sided** formula (`t ≥ 0.5` goes through `b - (b-a)*(1-t)`) evaluated in **float32 step arithmetic** (`_f32` truncation after the subtraction, the multiplication, and each add) — verified bit-exact against `np.percentile` on a float32 array over 40k random cases; plain double arithmetic (even cast to float32 at the end) differs by 1 ulp on ~3% of cases, which a boundary comparison amplifies into a ratio disagreement. Selection is scalar loops: `sorted(key=(-value, index))` stable descending, re-sorted ascending, complement by set difference. `ratio` is an integer count ratio, so both Python floats compare exactly; kept/pruned use `torch.equal`.

## Complexity

Flattening plus two percentiles cost `O(BN log(BN))` (sorting dominates), counting `O(BN)`; per row the stable sort and the mask argsort are `O(N log N)`, giving `O(BN log N)` overall. The only sizable intermediates are the flattened float32 sequence and the `[B,N]` int64 mask.

## Source and scope

Based on the retrieved `42Shawn/LLaVA-PruMerge`, file `llava/model/multimodal_encoder/clip_encoder.py`: `outlier_dectection`'s float32 flatten, `np.percentile` 25/75, `upper_bound = Q3 + 1.5 * IQR` (the next line is the commented-out lower_bound, out of scope), `np.where(attn_np > upper_bound)` strict comparison, `ratio = len(outlier_indices) / len(attn_np)`, plus the `if if_adaptive: reduction_ratio = outlier_dectection(cls_attn)` wiring and the subsequent `torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)` selection in `token_prune_merge_advanced`. The commented `#*3.5` scaling next to the ratio is disabled upstream and out of scope.

Deliberately out of scope: the cosine assignment and residual merge (covered by 536); the spatially-sampled supplement branch of `token_prune_merge_advanced_plus` (arange-step sampling is another deterministic ledger, a separate task if ever proposed); the real CLIP forward and hook extraction (k/q are caller-guaranteed). The lower IQR tail is unused upstream too.

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-prumerge-iqr-adaptive --mode submit
```

## Mutation testing (7/7 caught)

`>=` bound, dropping the 1.5 factor, swapping Q1/Q3, degrading the float32 cast to float64 (caught by the boundary case), emitting kept in value-descending order (caught once random cases reach `k>1`), replacing `int()` with `int(+0.5)` rounding, and the ascending-re-sort mutation. The first draft of the random cases used uniform(0,1), which almost never produces outliers — everything took the `k=0` empty path and the kept-order mutation survived; switching to a "tight bulk + 20% injected outliers" mixture caught everything. Lesson of this batch: **the random-case distribution must actually trigger the branch under test**.
