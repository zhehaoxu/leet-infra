# Reference approach: 2C alignment, front/back pairing, and a closed-form causal-load check

## Implementation

1. `align = 2 * cp_degree`, `padded = ceil(seq_len / align) * align`, `chunk = padded // (2 * cp_degree)`. Alignment **must** be `2C`: zigzag's front/back pairing needs an even number of chunks.
2. Build the `2C` chunks first; chunk `c` holds the original indices in `[c*chunk, (c+1)*chunk)`, with `-1` beyond the end.
3. Concatenate the local sequence by layout: `contiguous` takes `(2r, 2r+1)`, `zigzag` takes `(r, 2C-1-r)`, first chunk first.
4. `local_mask` is `idx >= 0` per slot; `causal_pairs[r] = Σ (idx+1)` over non-negative indices only.
5. `imbalance = max/min`, with min 0 mapped to `1.0` when everything is zero and `inf` otherwise.

Common mistakes:

- **Writing zigzag as "first half forward, second half reversed"**. The real rule pairs chunk $r$ with chunk $2C-1-r$: for $C=2$, rank 0 gets chunks 0 and 3, rank 1 gets 1 and 2. Reversing makes rank 1 take chunks 2 and 1, breaking both the local order and the load.
- **Aligning to `C` instead of `2C`**. `N=6, C=2` aligned only to `C` stays at 6 and cannot be cut into four equal chunks; upstream's metadata builder also uses `alignment = 2 * cp_size`.
- **Counting padding as a query**. The causal load sums over real tokens only; iterating padded global slots counts tail positions 6 and 7 as 7 and 8 keys, turning rank 0's load of 3 into 18 and hiding the imbalance.
- **Dividing when `min == 0`**. With `C > N` some rank necessarily owns no real token.
- **Writing contiguous as "chunk r"**. Each contiguous rank owns chunks `2r` and `2r+1`, which together form one long block; taking only one halves the local length.

## Verification

Seven `check_*` cover: the worked example (zigzag 18/18, contiguous 10/26), 90 cross-checks (`N ∈ 1..13,15,16,17,31,32` × `C ∈ {1,2,3,4,8}` × both layouts, compared against a pure-Python oracle), the padding ledger (increment, total, and the invariant that valid indices are exactly `range(N)`), zigzag balance (every rank's load is **strictly equal** and `imbalance == 1.0` without padding; contiguous is unbalanced for multiple ranks; `C=1` makes the two layouts identical), chunk composition (each rank's front/back boundaries and order), empty ranks (`C > N` gives zeros in `pairs` and `imbalance == inf`), and invalid inputs.

The oracle takes another route: it inverts each original token to `(rank, slot, local position)` and fills the table, then accumulates the load in a separate pass; the reference builds chunks first and concatenates per rank. They share the contract, not code structure.

Mutation testing: swapping zigzag's front/back chunks, degrading zigzag to contiguous, moving padding to the front, an all-True mask, counting tokens instead of `i+1`, and using the mean instead of max/min — all six mutants are caught; "padding at the front" is caught simultaneously by `padding_ledger`, `matches_oracle` and `empty_ranks`.

## Scope and sources

The upstream rules come from NVIDIA Megatron-LM's context-parallel layout code:

- `megatron/core/context_parallel/utils.py::_get_batch_on_this_cp_rank_contiguous`: requires the sequence length to be divisible by `2 * cp_size` and takes `front = narrow(cp_rank * segment_len, segment_len)`, `back = narrow((2*cp_size - cp_rank - 1) * segment_len, segment_len)`, then `cat(front, back)`; for non-attention-mask sequence tensors it takes the contiguous block `narrow(cp_rank * (seq / cp_size), seq / cp_size)`.
- `megatron/core/context_parallel/layout.py::_local_segment_ids`: under zigzag, `front_start = cp_rank * segments_per_cp_half` and `back_start = (2 * cp_size - cp_rank - 1) * segments_per_cp_half` — the front chunk and its symmetric back chunk.
- `layout.py::_build_thd_zigzag_metadata`: `alignment = 2 * cp_size`, each segment is padded to a multiple of `alignment`; `chunk_length = padded_len / (2 * cp_size)`, `destination_cp_rank = min(chunk_id, 2*cp_size - chunk_id - 1)`, `slot = chunk_id >= cp_size`, and the local position is `seq_start // cp_size + slot * chunk_length + offset_in_chunk`. The single-sequence version here applies exactly this logic to one sequence.
- The load-balancing motivation is `zhuzilin/ring-flash-attention` issue #2: on 4 GPUs, `b0 | b1 | b2 | b3` becomes `b0,b7 | b1,b6 | b2,b5 | b3,b4`, which equalises causal load.

Simplifications and differences: this problem is single-sequence and has no `tp_size` (upstream changes the atomic segment count per rank when TP>1 and requires TP to be even); padding is placed at the end of the sequence (upstream pads each variable-length segment separately — see `distributed-cp-varlen-zigzag-plan`); and no communication volume or attention values are computed — ring softmax lives in `distributed-ring-attention`, the closed-form activation/communication ledger in `distributed-sequence-parallel`, and head-dimension splitting (Ulysses) in `distributed-tp-split-rearrange`.
