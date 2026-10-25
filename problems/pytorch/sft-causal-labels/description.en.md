# SFT Data: Causal LM Shifted Labels

## Problem Description

The objective of autoregressive (causal LM) training is "use the prefix to predict the **next** token": the logits at position \(i\) should fit the token at position \(i+1\). So we **shift `labels` left by one** relative to `input_ids`, then overlay the "compute loss only at supervised positions" mask, filling everything else with `ignore_index` (default `-100` in `F.cross_entropy`). The off-by-one in this step, the last position having no successor, and never predicting across pad/segment boundaries are the most classic pitfalls of SFT/pre-training data pipelines.

## Rules

Implement `solve(input_ids, supervised_mask, ignore_index=-100)` and return int64 `labels` of shape \((B, L)\).

- `input_ids`: int64 \((B, L)\).
- `supervised_mask`: \((B, L)\), marking per position whether that **token should serve as a prediction target** (\(>0\) counts as supervised; equivalent to the mask produced by problem 241).
- For each position \(i\): if a successor \(i+1<L\) **within the same row** exists and `supervised_mask[b, i+1] > 0`, then `labels[b, i] = input_ids[b, i+1]` (predict the next supervised token); otherwise `labels[b, i] = ignore_index`.
- In particular, the last column of each row, \(i=L-1\), has no in-row successor and is always `ignore_index` (never reaching into the next batch row).

## Example

\(B=1\), `input_ids=[10,11,12,13,14]`, `supervised_mask=[0,0,1,1,0]` (only indices 2 and 3 are supervised targets):

```
labels:
 i=0 -> next idx1 mask=0 -> -100
 i=1 -> next idx2 mask=1 -> input_ids[2]=12
 i=2 -> next idx3 mask=1 -> input_ids[3]=13
 i=3 -> next idx4 mask=0 -> -100
 i=4 -> last col, no successor -> -100
=> [-100, 12, 13, -100, -100]
```

## Constraints

- CPU int64 input and output, shape strictly \((B, L)\).
- The "prediction target" decision uses `supervised_mask[b, i+1]` (look at the **successor position**), not `i` itself.
- The last column of each row is always `ignore_index`; never treat the first token of the next row as the successor of this row's last position.
- `ignore_index` defaults to `-100`; fill with whatever value is passed in.
