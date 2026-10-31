# Walkthrough: KVLink Segment Mask and Link-Token Layout

## In one sentence

Documents precompute their KV independently; a mask of "same segment only, plus segment-0 targets reading globally" keeps documents apart, and per-segment link tokens reconnect the information to the final question.

## Algorithm steps

1. Build `segment_ids`: `prefix_len` zeros; for chunk `j`, `len_j` copies of `j+1` followed by `reencode_num` zeros; append `-1` padding when `pad_to` is given;
2. Create a float32 `(L, L)` `bias` filled with 0;
3. Set the strict upper triangle (`j > i`) to `-inf`;
4. Set every position where `target == -1` or `source == -1` to `-inf`;
5. Set `-inf` where the target segment is non-zero and differs from the source; keep it visible when the target segment is 0.

The **directionality** in step 5 is the core of this problem: `zero_mask` depends only on the target, never on the source.

## Source-level details

Upstream `UCSB-NLP-Chang/KVLink` (master):

- `src/data/titan_preprocessor.py::make_segment_mask` (L591-662):
  - L614-615: `min_dtype = -float("inf")`; L618-627: `torch.triu(torch.full((B,S,S), min_dtype), diagonal=1)`; with `add_causal_lm_mask=False` (L628-633) it becomes all zeros;
  - L579-588 `_make_bool_segment_mask`: `torch.eq(source.unsqueeze(-2), target.unsqueeze(-1))`;
  - L642-645: `target_is_zero = (target_segments == 0).unsqueeze(-1)` and `zero_mask = target_is_zero`—**targets only**;
  - L649-652: `source_invalid_mask = (source_segments == -1).unsqueeze(-2)`, `target_invalid_mask = (target_segments == -1).unsqueeze(-1)`, with `invalid_mask` their OR;
  - L654-656: `all_masks = invalid_mask | ((~bool_mask) & (~zero_mask))`, then `masked_fill_(all_masks, min_dtype)`.
- Layout source: `process_xsum` (L513-548), L530 `segment_ids = segment_ids + [j+1]*len(chunks[j]) + [0]*self.reencode_num`; the evaluation script `scripts/evaluation/wiki_eval.py::preprocess_fn` (L113) does the same at L132, and L134-135 appends the `reencode_num` special link tokens per segment.
- Link-token default 5: `scripts/evaluation/wiki_eval.py` L54 `parser.add_argument("--reencode_num", type=int, default=5, help="Number of the link tokens.")`; `titan_trainer_kvlink.py` also sets `reencode_num=5`.
- Padding semantics: `BlockAttnCollator` (`titan_preprocessor.py` from L548) and `DataCollatorForGeneration` (`wiki_eval.py` from L170) both right-pad `segment_ids` with `-1`.
- The README points to `src/data/titan_preprocess.py` and `src/data/attention.py`, neither of which exists (the former is really `titan_preprocessor.py`, and the latter's `construct_biased_attention_matrix` is a dangling import). This problem uses only the real `make_segment_mask` and ignores the `biased_index` branch.

## Determinism boundaries

- `bias` holds only `0.0` and `-inf`, both exactly representable in float32, so tests compare with `torch.equal` bit for bit; there is no tolerance or rounding issue.
- `torch.triu(..., diagonal=1)` and `torch.eq` are deterministic integer/boolean operations; no randomness and no floating-point accumulation.
- Padding must be trailing: upstream collators right-pad via `segment_ids + [-1]*residual`; left padding or reordering is not allowed here.
- Zero total length is rejected; `chunk_len=0` is rejected too, because upstream chunks come from non-empty slices.

## Differences from other problems

- `sequence-pack-pad`: emits packed `input_ids/position_ids/segment_ids/attention_mask` but **builds no** `(L,L)` mask; this problem does no packing and only builds the segment bias.
- `kv-sepllm-separator-cache`: selects separator tokens by id for KV eviction; this problem builds an attention bias from segment relations and evicts no KV.
- `kv-key-rerotation-rope`: handles RoPE re-rotation after concatenating cross-document KV; this problem handles **visibility** and never touches positional encoding.
- `causal-mask`: pure causality; this problem layers same-segment constraints, the segment-0 global target, and invalid pad positions on top.

## Validation

- `check_example`: the 6-token example in the statement, checking `segment_ids` and the visible columns of every row.
- `check_matches_oracle`: five chunk/prefix/link/pad combinations against an independent numpy double-loop implementation.
- `check_link_tokens_are_global_readers`: a link token sees every preceding token, while non-zero segments cannot see it.
- `check_nonzero_segments_are_isolated`: non-zero segments never see each other, even past the causal lower triangle.
- `check_causal_and_diagonal`: the strict upper triangle is all `-inf`, and valid tokens always see their own diagonal.
- `check_padding_is_fully_masked`: rows and columns of `-1` (diagonal included) are all `-inf`.
- `check_zero_link_tokens`: with `reencode_num=0` there are no link tokens and segments are fully isolated.
- `check_prefix_only`: a prefix-only sequence degenerates to pure causal masking.
- `check_validation`: illegal chunk lengths, negative prefix/link counts, too-small `pad_to`, bools posing as ints, and zero total length all raise `ValueError`.
