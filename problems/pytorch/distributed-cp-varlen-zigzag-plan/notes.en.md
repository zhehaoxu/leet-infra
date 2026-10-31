# Reference approach: per-segment padding, local bases, and cross-segment slot mapping

## Implementation

1. Compute each segment length `lengths`, pad every segment to `2C`: `p_b = ceil(n_b / (2C)) * (2C)`; `P = Σ p_b`, `L = P // C`.
2. Prefix sums give `cu_seqlens_padded`.
3. Process each segment: `chunk = p_b // (2C)` and local base `base = q_b // C` (`q_b` is the padded segment start).
4. For chunk `cid`: `rank = min(cid, 2C-1-cid)`, `slot = 0 if cid < C else 1`; write at `base + slot*chunk + k` the value `orig_start + cid*chunk + k` (or `-1` past the segment's real length).
5. `pad_total = P - N`; `pad_between_seqs = any(p_b != n_b)`.

Common mistakes:

- **Padding the whole stream to `2C` and cutting once**. Upstream pads **per segment**: every segment must split into $2C$ equal chunks. For `[0,3,5], C=2`, padding total length 5 to 8 and cutting globally would slice through the segment boundaries `[0,3]` and `[3,5]`, making intra-segment pairing impossible.
- **Using one global chunk length**. Chunk length is `p_b/(2C)` and can differ per segment; using `P/(2C)` overruns short segments.
- **Writing the local base as segment index times a fixed length**. The base is the padded prefix sum divided by `C` (`q_b // C`), not `b * something`.
- **Pairing with `cid % C` or a plain modulo**. It must be `min(cid, 2C-1-cid)`; at `cid = C` that is `C-1`, not `0`.
- **Forgetting to write `-1` past a segment's real length**. Writing `orig_start + offset` unconditionally invents token indices.
- **Testing `pad_between_seqs` only against total padding**. The contract is "any segment padded", equivalent to nonzero total padding here, but semantically tied to segment boundaries; upstream uses `cu_seqlens != cu_seqlens_padded`.

## Verification

Six `check_*` cover: the worked example (`[0,3,5]` with cross-segment padding and `[0,4]` without), 19 cross-checks (varying segment counts/lengths/`C`, compared item by item against a pure-Python oracle), the single-segment degeneration (matches single-sequence zigzag layout), the padding ledger (each segment's padding is a multiple of `2C` and less than `2C`, plus total and flag semantics), chunk ownership (rank $r$'s front/back chunks come from $r$ and $2C-1-r$), and invalid inputs.

The oracle takes another route: it builds each segment's padded stream, slices chunks $r$ and $2C-1-r$, and concatenates them at `base`; the reference computes each slot index directly. They agree only on the contract.

Mutation testing: aligning to `C`, using `p_b/C` as chunk length, skipping the divide by `C` in the base, a constant slot 0, not writing `-1` for padding, a constant `False` flag, and ownership `cid % C` — all seven mutants are caught. The constant-`False` flag is caught only by the example, the oracle cross-check and the padding ledger; `single_sequence` and `rank_assignment` are unrelated to it.

## Scope and sources

The rules follow NVIDIA Megatron-LM's `megatron/core/context_parallel/layout.py::_build_thd_zigzag_metadata` line by line:

```text
alignment = 2 * cp_size
target_lengths = ceil(source_lengths / alignment) * alignment
chunk_lengths = padded_lengths / (2 * cp_size)
chunk_ids = offsets / chunk_lengths
destination_cp_ranks = min(chunk_ids, 2 * cp_size - chunk_ids - 1)
chunk_slots = (chunk_ids >= cp_size)
target_local_positions = target_sequence_starts / cp_size
                       + chunk_slots * chunk_lengths
                       + offsets_in_chunk
rank_order_positions = destination_cp_ranks * (target_token_count / cp_size)
                     + target_local_positions
source_indices = source_sequence_starts + offsets   (offsets < source_length, else -1)
rank_order_indices.scatter_(0, rank_order_positions, source_indices)
```

`megatron/core/context_parallel/utils.py::_get_batch_on_this_cp_rank_padded_zigzag` then takes `rank_order_indices.view(cp_size, -1)[cp_rank]` and `index_select`s with it, filling `-1` slots with 0 (or `padding_mask=True`).

Where this problem differs from upstream:

- **`tp_size = 1` only**. With TP>1 upstream also pads the last segment by `tp_padding * cp_size` tokens so each rank's total divides by TP; this problem does not do that (the invariant here only pads `p_b` to `2C`).
- `cu_seqlens` is a Python list rather than an int32 tensor, and `rank_order` is a nested list rather than a flat tensor — this problem returns `[C, L]` directly.
- It is a plan only: the actual all-to-all move of `contiguous_to_zigzag` / `zigzag_to_contiguous` with its split sizes is out of scope; ring attention values live in `distributed-ring-attention` and the single-sequence load in `distributed-cp-zigzag-partition`.
