# TrainableTokens Point Replacement: index_copy Semantics and Disjoint Adapters

## Problem

Adding new words to an LLM (a new language, domain tokens, special tokens) usually means resizing the vocabulary and fine-tuning the whole embedding — expensive. TrainableTokens (huggingface/peft `tuners/trainable_tokens`) takes another route: **keep the vocabulary, train only a handful of token rows**. Each adapter carries a `token_indices` list and a delta matrix with matching rows, applied via PyTorch's `index_copy` — row `k` of the delta **replaces** row `token_indices[k]` of the weight matrix; it is not added.

"Replace, not add" is deliberate: delta is initialized by copying the matching rows of the base weight (kept as `originals`), so delta simply *is* the current value of those rows. Under addition, re-creating a model with freshly randomized embeddings would leave undefined behavior; replacement always means something. Upstream adds one hard constraint: **the token indices of any two adapters must be disjoint** — overlapping merges are undefined and raise. Given an embedding weight and a set of adapters, reproduce this point replacement: return the merged weight and, per adapter, the original rows it hit (for unmerge).

## Task

Implement `solve(weight, adapters)`:

- `weight`: 2-D float tensor `[vocab, dim]` (embedding weight);
- `adapters`: non-empty `list`; each item is a dict with exactly three keys:
  - `"name"`: non-empty string, **must be unique**;
  - `"token_indices"`: non-empty `list[int]`, each in `[0, vocab)`, **unique within one adapter**;
  - `"delta"`: `[len(indices), dim]` numeric (nested `list` or 2-D tensor); row `k` is the new value.

Return a `dict` with keys `merged_weight` and `originals` (`name` → the rows that adapter hit, taken from the **original** `weight`, shape `[len(indices), dim]`).

## Exact rules

1. **Replacement semantics**: `merged` starts as a copy of `weight`; for each adapter **in order**, `merged = merged.index_copy(dim=0, index=indices, source=delta)`. Row `k` of delta becomes row `indices[k]` outright — the original value never influences the result (contrast LoRA/SHiRA's addition).
2. **Disjointness check**: if the `token_indices` sets of any two adapters intersect → `ValueError` (upstream `_check_overlapping_tokens`: overlapping merges are undefined). Duplicate adapter names also raise.
3. **originals**: per adapter, `weight[indices]` (rows **before** merging), for unmerge recovery — grabbing them from the merged weight would return delta itself.
4. delta must have exactly `len(token_indices)` rows of width `dim`.

## Example

```text
W = [[1,1], [2,2], [3,3], [4,4]] (vocab=4, dim=2)
adapter a: indices=[1], delta=[[9,9]]
adapter b: indices=[3, 0], delta=[[8,8], [7,7]]

merged = [[7,7], [9,9], [3,3], [8,8]]   # row 1←a, rows 3,0←b, row 2 untouched
originals = {a: [[2,2]], b: [[4,4], [1,1]]}
```

Semantic anchor: `W=[[100,200],[300,400]]`, indices=[0], delta=[[1,2]] → merged `[[1,2],[300,400]]` — delta fully determines the new row; addition would give `[[101,202],...]`.

## Constraints

- Do not modify inputs; invalid inputs (weight not 2-D / non-finite, adapters empty / not a dict / missing keys / empty or duplicate names, indices empty / non-int / bool / out of range / duplicated within one adapter, delta row-count or width mismatch / non-finite, two adapters overlapping) raise `ValueError`.
