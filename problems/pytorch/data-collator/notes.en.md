# Data Collation (collate) in Depth

## Why all three tensors are indispensable

- **input_ids**: the model's only input. Variable lengths cannot be stacked into a rectangular tensor for batched matrix multiplication, so they are padded to `L`.
- **attention_mask**: padding consists of "fake tokens"; without a mask, softmax would assign attention to padding, diluting the representation of real tokens and polluting the next generation. The mask sets the scores at pad positions to `-inf` inside attention.
- **labels' -100**: padding must also be excluded when computing cross-entropy. `F.cross_entropy(ignore_index=-100)` skips positions whose value is `-100` — this is the **default value** of `ignore_index` in PyTorch, which is why the whole industry conventionally uses `-100` to mark "no loss".

## Right padding vs. left padding

Training uses **right padding** (this problem): under teacher forcing, future positions are masked anyway, so padding at the tail does not matter. **Generation/inference often uses left padding**: autoregressive decoding continues from the last real token, and if padding were at the end, it would continue from the padding. The attention_mask rule is identical for both; only the side where padding lands differs. This problem fixes right padding — do not take the liberty of padding on the left.

## Two frequently missed points

1. **Forgetting to set pad positions in labels to -100**: copying input_ids directly as labels forces the model to "predict pad", mixing meaningless terms into the loss and inflating the metrics.
2. **Inconsistent padding masks between attention_mask and labels**: the pad determination in both places must come from the same source (both based on the true lengths). This problem slices the prefix with the same `n` to guarantee consistency.

## Why dtype is long

Token ids and labels serve as embedding indices / cross_entropy targets and must be integer tensors (`torch.long`/int64). Float ids will raise an error.

## Related

- Upstream: `bpe-encoding` (text → subwords → ids; this problem organizes them by batch).
- Downstream: loss problems such as `causal-language-modeling` that need label shift + ignore; `attention-mask` (if present) takes the mask here and expands it to 4D.
- Judging: integer tensors, no randomness; `check_example` uses the three dumped matrices, plus cases with equal lengths, a single sequence, and custom pad/ignore.
