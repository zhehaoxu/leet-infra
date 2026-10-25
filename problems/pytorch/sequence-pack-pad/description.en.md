# Data Preparation: Packing Sequences into Fixed-Length Blocks (pack)

## Problem

Per-sample padding (`data-collator`) wastes a lot of compute — short samples get filled with a pile of pads. Modern LLM training instead uses **sequence packing**: many short sequences are concatenated end to end into one long stream, which is then cut into fixed-length blocks of `block_size`, so only the last block may contain pads. The price is that you must additionally produce `position_ids` (the position of each token **restarts from 0 within its own subsequence**) and `segment_ids` (marking which subsequence each token belongs to), so that attention does **not** cross subsequences. This problem implements that packing function.

## Rules

Implement `solve(seqs, block_size, pad_token_id)`:

- `seqs`: `list[list[int]]`, each of length ≥ 1. Concatenated in the **given order**.
- Join all tokens into a single stream (no pads inserted between subsequences). Let the total number of real tokens be `total`; the number of blocks is `num_blocks = ceil(total / block_size)`.
- Append `num_blocks * block_size - total` copies of `pad_token_id` at the end of the stream (pads have `position=0`, `segment=-1`).
- Reshape the padded stream **row-major** into `[num_blocks, block_size]`.
- Return a **dict** of four `torch.long` tensors, all of shape `[num_blocks, block_size]`:
  - `input_ids`: the packed + padded tokens.
  - `position_ids`: each token's index within its **own subsequence** (every subsequence starts at 0); `0` at pads.
  - `segment_ids`: the index of the subsequence each token belongs to (0, 1, 2, …; **continuous** across blocks); `-1` at pads.
  - `attention_mask`: `1` for real tokens, `0` for pads.

> Key point: a subsequence **may span blocks**, and its `position_ids` keep incrementing across block boundaries; block boundaries do not reset the position, only "a new subsequence starts" does.

## Examples

`seqs = [[10,11,12],[20],[30,31]]`, `block_size = 4`, `pad_token_id = 0` (total=6 → 2 blocks):

```
input_ids     = [[10,11,12,20], [30,31, 0, 0]]
position_ids  = [[ 0, 1, 2, 0], [ 0, 1, 0, 0]]   # doc0:0,1,2 | doc1:0 | doc2:0,1 | pad:0,0
segment_ids   = [[ 0, 0, 0, 1], [ 2, 2,-1,-1]]   # one id per subsequence, pad=-1
attention_mask= [[ 1, 1, 1, 1], [ 1, 1, 0, 0]]
```

`seqs = [[5],[6,7]]`, `block_size = 3` (total=3 → 1 block, no pad):

```
input_ids=[[5,6,7]]  position_ids=[[0,0,1]]  segment_ids=[[0,1,1]]  attention_mask=[[1,1,1]]
```

## Constraints

- `block_size ≥ 1`; all four outputs have dtype `torch.int64` and shape `[num_blocks, block_size]`
- Subsequences are concatenated in input order and `seqs` is not modified; `segment_ids == -1` is the only source of truth for pads
- Pure deterministic integer arithmetic, no randomness
