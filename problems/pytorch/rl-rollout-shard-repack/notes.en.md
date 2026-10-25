# In Depth: Rollout Re-sharding Across Ranks and Padding-Free Packing

## Origin
- verl's PPO/DAPO trainer calls `rebalance_batch` before handing generation output to the data-parallel training side (the DAPO recipe pairs it with `get_global_partition_idx_and_max_response_length`): sequences are sorted by length, split into a **global partition** that equalizes per-rank token load, and each partition's **longest sequence** is recorded (it fixes the padding width of that partition).
- The padding-free micro-batch split in OpenRLHF / verl lives in `verl/utils/seqlen_balancing.py` (`rearrange_micro_batches` and friends): pack under a token budget and emit `cu_seqlens` boundaries, then feed the chained stream to flash-attention's varlen API.
- "Sequence-level padding waste = count × longest" comes from synchronous DP training: all variable-length sequences of one rank must be padded to the same width to enter the same forward.

## Key conventions (one deviation and everything is wrong)
1. **Balancing order**: process by $(\text{length},\text{original index})$ **ascending**. Ascending (rather than LPT's descending order) combined with "shortest queue first" leaves the largest items for last, so the load spread is bounded by the **largest single sequence**; the index tie-break is required, otherwise the same data would shard differently on two runs.
2. **Greedy objective**: pick the rank with the smallest current **real token total** (ties → smallest rank index). The key is tokens, not sequence count — count-balancing collapses as soon as there is a long tail (see `check_rebalancing_reduces_skew`: 3 of 15 sequences hold 350 tokens).
3. **Micro-batch numbering**: ranks ascending, within a rank in packing order, globally contiguous from $0$. Real frameworks number micro-batches locally per rank (the id only means something inside that rank); here they are flattened into one global numbering so the answer is comparable. Empty ranks consume no id.
4. **Never split an over-budget sequence**: a pack is closed only when it is non-empty and the next sequence would exceed $M$, so every pack is $\le M$ **unless** it holds a single sequence longer than $M$. Padding-free boundaries are whole sequences — cutting a sequence in half would make `cu_seqlens` claim a sequence end that does not exist.

## The pad_saved metric and how it gets written wrong
`pad_saved[r] = count_r × max_len_r − Σ len`. Typical mistakes:
- taking `max_len` as the **global** maximum (across ranks) → overestimates the waste;
- using $N$ instead of that rank's own count;
- forgetting empty ranks → with `count=0` the value must be $0$, and special-casing it wrongly is where bugs hide.
In a correct implementation `pad_saved[r]` $\ge 0$ always, and it is **exactly $0$** when all lengths inside the rank are equal — the strongest self-check available (`check_equal_lengths_round_robin`). What it saves is **wasted compute** (pad tokens still go through forward/backward), not memory; the memory ledger lives in `rl-colocate-memory-timeline` and `training-state-memory`.

## Boundary conventions
- **Initial ownership is contiguous blocking** $\lfloor iW/N\rfloor$, not round-robin $i\bmod W$. Round-robin makes `moved` mostly zero and hides the real transfer volume.
- With $W>N$ at most $N$ ranks get work; empty ranks report `tokens_per_rank`/`pad_saved` equal to $0$ (a missing entry is not the same as a zero).
- With $N=0$ the outputs have different lengths: sequence-level tensors are empty, rank-level tensors are zero vectors of length $W$.

## Division of labour
- `sequence-pack-pad`: packs sequences into fixed-length blocks **inside one stream** and emits token-level `input_ids/position_ids/segment_ids/attention_mask`. This problem **produces no token-level tensor** — only "which sequence belongs to which rank, which micro-batch, the cost difference of the two padding schemes, and how many sequences move". The teaching point is the **cross-rank redistribution**; padding is only the cost metric.
- `distributed-zero-*` / `distributed-pipeline-*`: account for model-state sharding, communication and pipeline bubbles. Here the "shard" is a shard of **data**; the move count is a proxy for communication (deliberately not converted to bytes, which depends on the concrete model and belongs to `distributed-*`).
- `rl-colocate-memory-timeline`: memory and phase timeline; this one: token load and bin packing.
- `to-*`: loss and advantage formulas; this problem does not even look at rewards, only at lengths.

## Complexity
Sorting $O(N\log N)$ plus a rank scan per sequence $O(NW)$; with $N\le 4096, W\le 64$ no heap is needed. The oracle recomputes each rank's load from scratch, packs with an index-pointer while loop, derives initial ownership from ceil boundaries, and accumulates the waste per sequence — four structurally different paths for cross-validation.
