# KV Cache: KVLink Segment Mask and Link-Token Layout

## Problem

KVLink (arXiv:2410.06511) splits a long context into documents and **precomputes each document's KV independently**: since documents never attend to each other, each document's KV can be computed offline and reused across requests. To recover cross-document information it appends `reencode_num` trained **link tokens** (default 5 in the README) to each segment and assigns them to **segment 0**; the final question and answer also belong to segment 0. The mask therefore only needs three rules:

1. Mask the future with causality;
2. Only tokens in the same segment may attend to each other;
3. A **segment-0 target** may attend every valid source (link tokens thus act as global summaries readable only by segment-0 targets, while documents stay isolated).

Padding positions carry `segment = -1` and are masked in both directions. This problem implements the layout and mask, i.e. the CPU semantics of upstream `make_segment_mask`.

## Task

Implement `solve(chunk_lengths, prefix_len=0, reencode_num=5, pad_to=None)`:

- `chunk_lengths`: `list[int]`, token count of each document chunk, all ≥ 1;
- `prefix_len`: leading system-prompt tokens, ≥ 0, all in segment 0;
- `reencode_num`: link tokens appended to each segment, ≥ 0;
- `pad_to`: optional, right-pads with `-1` up to that length and must be ≥ the real length.

Layout (real length \(L_0=\texttt{prefix\_len}+\sum_j(\ell_j+r)\)):

```
segment_ids = [0]*prefix_len
for j, len_j in enumerate(chunk_lengths):
    segment_ids += [j+1]*len_j + [0]*reencode_num
if pad_to is not None:
    segment_ids += [-1]*(pad_to - L0)
```

Return a `dict`:

- `segment_ids`: `(L,)` int64;
- `bias`: `(L, L)` float32, where `0` means visible and `-inf` means blocked:
  1. the strict upper triangle (`j > i`, top-left aligned) is always `-inf`;
  2. any position whose `segment == -1` at either end is `-inf` (pad rows and columns are fully masked, diagonal included);
  3. when the target has `segment == 0`, every valid source is visible;
  4. otherwise the target sees only valid sources with the **same segment**.

Raise `ValueError` for invalid input (not a list/tuple, chunk length < 1, negative prefix/link count, `pad_to` below the real length, bool posing as int, or zero total length).

## Example

```
chunk_lengths = [2, 1]
prefix_len = 1
reencode_num = 1

segment_ids = [0, 1, 1, 0, 2, 0]
#                 ↑        ↑     ↑
#              prefix   doc0's  doc1's
#                        link    link

visible columns per row (bias[row, col] == 0):
row 0 (seg 0): [0]
row 1 (seg 1): [1]
row 2 (seg 1): [1, 2]
row 3 (seg 0): [0, 1, 2, 3]
row 4 (seg 2): [4]
row 5 (seg 0): [0, 1, 2, 3, 4, 5]
```

Row 3 is doc0's link token: it belongs to segment 0 and sees every preceding token (0..3), so it can summarize doc0. Row 4 is doc1's first token and sees only itself (col 4)—it sees neither doc0 nor any link token, which is exactly what makes per-document KV reuse valid. Row 5 is the final segment-0 target and sees all six positions.

## Constraints

- `bias` contains only `0.0` and `-inf`, dtype fixed to float32; tests compare element-wise for equality, with no tolerance.
- Segments are numbered from 1 in input order; prefix and all link tokens are 0, and padding is only trailing `-1`.
- "Global" applies only to **segment-0 targets**, not segment-0 sources: a non-zero target must not see link tokens.
- This problem returns no `input_ids`/`position_ids`: concatenated positions are the default `arange` upstream, and link-token ids are trained special tokens, both out of scope.
- No training and no `construct_biased_attention_matrix` (that file is missing from the upstream repo); only `make_segment_mask` is reproduced.
