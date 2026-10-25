# Sequence Packing in Detail

## Why Pack

With per-sample padding, if the longest sequence in a batch is 512 and most are only a few dozen tokens, the vast majority of compute is spent on pads (softmax and FFN work that is entirely wasted). Packing splices many short sequences **seamlessly** into fixed-length blocks, so pads appear only at the tail of the last block and token utilization jumps from "average length / max length" to nearly ~100%. Megatron, NeMo, llm.c, and most SFT frameworks enable it by default.

## position_ids Are the Soul of Packing

In ordinary padding, `position_ids` is simply `arange(L)`. After packing, one block contains several subsequences, so positions must **restart from 0 within each subsequence**; otherwise the second subsequence is assigned positions that "continue from the previous one" — and RoPE / positional embeddings are immediately misaligned. So `position_ids` records "the index within this own subsequence", independent of block boundaries: when a subsequence crosses a block boundary, its position **keeps increasing** and is not reset (a reset happens only when "a new subsequence starts"). This is the easiest place to get this problem wrong.

## segment_ids: Keeping Attention in Bounds

Two adjacent subsequences inside a block are semantically unrelated; letting them attend to each other is "document cross-talk". The criterion: attention is allowed only within the **same segment**. Downstream code uses `segment_ids[i] == segment_ids[j]` to build a 4D block-diagonal mask (and then adds causality). This problem only needs to produce `segment_ids`, with pads marked `-1` to distinguish them from all real subsequences.

- Use `segment_ids == -1` as the sole pad criterion, and derive `attention_mask` from it too — avoiding two competing pad logics.
- Subsequence indices are **continuous** across blocks (the 3rd subsequence is 3 no matter which block it lands in), so the parts of one subsequence that span blocks still share an id.

## Trade-offs vs. data-collator

| | data-collator | sequence-pack-pad |
|---|---|---|
| Padding location | End of every sample | Last block only |
| Extra outputs | attention_mask, labels | position_ids, segment_ids |
| Attention | causal + right-pad mask only | also block-diagonal to prevent cross-sequence |
| Utilization | Low | High |

## Related

- Upstream: `bpe-encoding` + `data-collator` (convert to ids first, then decide between per-sample padding and packing).
- Downstream: the packed mask/positions feed `llm-rope` / attention problems; ties into the training-throughput theme.
- Judging: integer tensors, no randomness; `check_example` uses the four matrices from the dump, plus cases for a single sequence dividing evenly, incrementing across blocks, and no pads.
