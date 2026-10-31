# notes — ocr-ctc-greedy-decode (order 668)

## Source

- Origin: `PaddlePaddle/PaddleOCR`, `ppocr/postprocess/rec_postprocess.py`, `BaseRecLabelDecode.decode` (inherited by `CTCLabelDecode`, which only prepends `"blank"` to the dict and sets `get_ignored_tokens() → [0]`). `__call__` fixes `is_remove_duplicate=True`.
- The official selection-mask semantics: `selection[1:] = text_index[1:] != text_index[:-1]` (adjacency on the **raw sequence, blanks included**), then the blank filter is intersected. The two masks combine via `&=` and their order does not change the kept set, but "dedup decided on the blank-inclusive raw sequence" is the semantic crux — both A's in `A blank A` survive.
- Boundary with `ctc-loss` (already taken on the loss side): this problem covers only the decode/post-processing path, no forward-backward DP.
- Adds the spans ledger (a simplified version of the frame-interval semantics in `get_word_info`): the run's `[start, end]` closed interval.

## Judging

- solve args: logits [T, V], blank_id (default 0). Returns raw_path/tokens/spans.
- Oracle: `test_cases._py` follows the official selection-mask path (argmax via `index(max())`), 200 random cross-checks (T 1–20, V 2–6, 60% chance of spikes creating duplicate/blank paths).
- Anchors: hand-computed run/blank interleavings (`[1,1,0,2,2,2] → tokens [1,2]`, spans `[[0,1],[3,5]]`); A blank A keeps both; A A blank A A forms two runs; all-blank empty output; leading/trailing blanks; argmax ties pick the smaller id; custom blank_id role swap.
- Invalid inputs: empty/ragged/non-numeric/NaN/Inf, blank_id bool/out-of-range/negative → ValueError.

## Mutation log (9/9 caught)

no-dedup (run-extension condition inverted), step-instead-of-run (per-frame advance duplicates tokens), keep-blank (blanks enter output), span-end-start (end records run start), span-end-exclusive (closed → half-open), span-start-shift (start +1), argmax-tie-max (ties pick larger), blank-fixed-zero (ignores the blank_id arg, always 0), drop-last-frame (final run dropped).

## Red lines

No real images/PIL/torchvision; no softmax/confidence (argmax only); boundary with #108's CTC prefix beam search (`asr`): this is the **greedy path ledger**, no beam/DP.
