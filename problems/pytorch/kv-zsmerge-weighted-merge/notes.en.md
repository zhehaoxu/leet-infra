# Solution Notes · ZSMerge Zero-Discard Weighted Merging

## In one sentence

When the cache is full, do not evict: squeeze out the lowest-scoring slot \(i\) in the non-dense region, fold it into its nearest neighbour \(j\) in the dense region with `counts` weights, and let the new token take slot \(i\).

## Three things that must be right

**1. K and V are merged together with the same weights.** In upstream `cache_merge`

```
w_i = pos_weight[i]         # = counts[i]
w_j = pos_weight[j]         # = counts[j]
k[j] = (w_i * k[i] + w_j * k[j]) / (w_i + w_j)
v[j] = (w_i * v[i] + w_j * v[j]) / (w_i + w_j)
```

`w_i / w_j` is computed once and shared by K and V. Recomputing weights for V (say, by value norm), or merging K but not V, misaligns the attention distribution and the value content—the easiest mistake here that still "looks right".

**2. Counts are inherited, not incremented.** After the merge `counts[j] = w_i + w_j` and `counts[i] = 1` (the new token only represents itself). Writing `counts[j] += 1` would give a slot merged 100 times the same weight as one merged once, so later merges would amplify new tokens again and again and dilute historical tokens into nothing.

**3. The dense mask constrains \(j\), not \(i\).** \(i\) is the slot with the smallest `scores` in the **non-dense** region and \(j\) is the slot nearest to \(k_i\) in the **dense** region. The two are disjoint: a dense slot may never be sacrificed however low its score, and a non-dense slot may never be the anchor however close it is. The tests deliberately include a case where a dense key is closer to \(k_i\), to pin this down.

## Grading points

- `merged_slots`: `[i, j]` per head, with `i` non-dense and `j` dense; \(j=-1\) when nothing is merged.
- `counts`: summed at \(j\), set to 1 at \(i\), unchanged elsewhere.
- `scores`: **summed** at \(j\) (upstream `score_update="sum"`), neither max nor mean.
- `metric`: `l2` uses squared distance and `dot_product` uses the negated dot product; on the example data they pick different \(j\).
- Ties: both \(i\) and \(j\) go to the smaller slot index (upstream `argmin` leaves tie order undefined).
- Inputs are not mutated, the math is float64, and outputs return to the input dtype.

## Boundaries

- **All-False dense mask**: upstream takes the separate `cache_dense == 0` branch—no \(j\) is searched, nothing is merged, and the globally lowest-scoring slot is handed to the new token; this task marks that with \(j=-1\). **All-True dense** leaves no slot to squeeze, so a `ValueError` is raised.
- **\(i\neq j\)**: guaranteed by the dense / non-dense partition, so "write \(j\) first, then \(i\)" never collides; writing \(i\) first and then reading \(k_i\) would read the new token's key instead.
- **Upstream's score decay**: every step multiplies `scores` by `shrink_factor` (0.98) before adding the new score, so scores are a cumulative sum with forgetting. This task treats `scores` as an explicit input—the decay happens at the call site and `solve` only adds—which is what makes exact score assertions possible in the tests.

## Complexity

Each step costs an \(O(THD)\) reduction, \(O(|D_{\text{dense}}| H D)\) neighbour comparisons, and \(O(THD)\) writes; against the \(O(L^2 D)\) of full attention this is only constant-factor maintenance.

## Source

`SusCom-Lab/ZSMerge` (arXiv 2503.10714 — the id comes from the paper landing page; the repository itself ships no README to cross-check it against):

- `mergekv/mergekv.py` L354-428 `cache_merge`: the overall path;
- L362-374 the `cache_dense == 0` branch: no merge, direct replacement;
- L377-378 `mrg_i = argmin(pos_score[..., cache_dense:]) + cache_dense`;
- L384-391: `dot_product` takes the negated `einsum`, `l2` takes squared distance, and \(j\) is the `argmin` inside the dense prefix;
- L395-404: `w_ij = w_i + w_j`, with K and V sharing these weights;
- L411-412: `s_ij = s_i + s_j`; L418-426: scatter the merge result to \(j\) first, then write the new token to \(i\);
- L27-41 and L48-80: `cache_budget / cache_tail / cache_dense` converted through successive `int()` truncations into the dense prefix length;
- the companion entry point `topk_split_new` in the same file performs the slot split; this task abstracts its result into the `dense` mask input and does not judge it separately.

**Additions of this task**: ties to the smaller slot index; the dense mask and `scores` / `counts` as explicit inputs instead of internal state; and \(j=-1\) as an explicit marker for the `cache_dense == 0` case.

## Related problems

- `kv-h2o-heavy-hitters`: also picks victims by score, but discards them outright, merges nothing, and inherits no counts.
- `kv-merging-cos-nearest`: folds evicted tokens' values into their cosine nearest neighbour by key similarity, reducing the slot count; this task picks the anchor by squared L2 / negated dot product **inside the dense region**, merges K and V with one shared count weight, and keeps the slot count fixed.
- `kv-wkvd-least-squares-merge`: projects the evicted values onto the subspace spanned by the surviving values in a least-squares sense; this task is a one-to-one count-weighted average with no regression solve.
- `kv-morphkv-island-fusion`: fuses a contiguous run of positions into one slot, a many-to-one weighted average; this task is one-to-one.
- `kv-snapkv-pooling`: pools inside the observation window; the unit is a window rather than individual slots.
