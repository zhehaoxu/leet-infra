# CTC Greedy Decoding and Collapse Ledger

## Problem

An OCR recognition head outputs per-frame classification logits; CTC (Connectionist Temporal Classification) greedy decoding turns them into text: take the argmax per frame, merge adjacent repeated frames (collapse), then drop blanks. This problem transcribes the strict ledger of PP-OCR's `BaseRecLabelDecode.decode`: given `[T, V]` logits and `blank_id`, output the per-frame argmax path, the decoded token sequence, and each token's frame span.

## Solution

The ledger is pinned by the spec (faithful to upstream):

1. `raw_path[t] = argmax(logits[t])`, **ties pick the smallest class id**;
2. adjacent dedup is decided on the **raw sequence** (blanks included): consecutive identical ids form one run. Hence the two A's in `A blank A` are non-adjacent and both survive — the blank-separator semantics is the core of CTC;
3. blank runs are dropped entirely;
4. `spans[i]` is the frame interval `[start, end]` (**closed**) of output token i's run: it starts at the run's first frame and ends at the last frame of consecutive identical ids.

The official implementation uses a selection mask (`selection[1:] = cur != prev` intersected with `cur != blank`); this problem implements it as a run loop — the two are bit-for-bit equivalent. Returns `{"raw_path", "tokens", "spans"}`.

## Constraints

- 1 ≤ T ≤ 10^4, 2 ≤ V ≤ 10^5, logits are finite numbers.
- Empty logits/rows, ragged rows, non-numeric or NaN/Inf entries, blank_id non-int/out of range → `ValueError`.
