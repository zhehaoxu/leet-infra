# Data Collation: Batch Padding with attention_mask / labels

## Problem

Samples in a batch have different token counts, so before feeding them to the model they must be **padded on the right** to a common length, along with a matching `attention_mask` (so the model does not attend to padding) and `labels` (so the loss is not computed on padding). This is exactly the daily job of a DataLoader's `collate_fn` / HF `DataCollator`. In this problem you write this collation function by hand.

## Rules

Implement `solve(seqs, pad_token_id, ignore_index=-100)`:

- `seqs`: `list[list[int]]`, each entry a sequence of token ids (lengths may differ, all ≥ 1).
- `pad_token_id`: the id used for padding.
- The maximum length in the batch is `L = max(len(s))`; every sequence is padded **on the right** with `pad_token_id` up to `L`.
- Return a **dict** containing three `torch.long` tensors, all of shape `[B, L]`:
  - `input_ids`: the padded tokens.
  - `attention_mask`: `1` at real token positions and `0` at pad positions.
  - `labels`: the same as `input_ids`, but **pad positions are filled with `ignore_index`** (default `-100`, i.e. `F.cross_entropy`'s `ignore_index`; these positions contribute nothing to the loss).

> This problem only does right padding, no truncation. `B` and `L` are determined by the input.

## Examples

```
seqs = [[1,2,3],[4],[5,6]], pad_token_id = 0

input_ids     = [[1,2,3],
                 [4,0,0],
                 [5,6,0]]
attention_mask= [[1,1,1],
                 [1,0,0],
                 [1,1,0]]
labels        = [[1,2,3],
                 [4,-100,-100],
                 [5,6,-100]]
```

## Constraints

- All three output tensors have dtype `torch.int64` (`torch.long`) and shape `[B, L]`
- Do not modify the passed-in `seqs`
- At pad positions: `attention_mask=0` **and** `labels=ignore_index` — both must be right; this is the most common omission
- Purely deterministic integer arithmetic, no randomness
