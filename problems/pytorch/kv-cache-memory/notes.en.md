# Solution Approach

## Interviewer's intent

"Why is 128K-context inference so memory-hungry?" The core answer is the KV cache. It grows **linearly with sequence length**, and with long contexts
or large batches it can overtake the model weights. This problem also ties in MQA/GQA—attention variants created precisely to shrink the KV cache
(mirroring the `mqa` problem: one is about compute, the other about memory).

## How the size is assembled

The cache stores, for **every layer, every KV head, and every generated token**, one K vector and one V vector, each with
`head_dim` dimensions and `dtype_bytes` bytes per dimension. Hence

$$ \text{bytes} = 2_{(K,V)} \cdot L \cdot \text{kv\_heads} \cdot d_{\text{head}} \cdot \text{seq} \cdot \text{batch} \cdot \text{dtype\_bytes} $$

Just keep your eye on the most commonly missed factor $2$—"K and V are two"—and the rest is a product of dimensions.

## Why MQA / GQA save memory

In MHA every query head has an independent KV, `kv_heads = heads`.

- **MQA**: all query heads share **one set** of KV (`kv_heads=1`), and the cache shrinks directly to $1/\text{heads}$. In the example,
  32 heads → from 1 GiB down to 32 MiB, 32×.
- **GQA**: the compromise, splitting the `heads` query heads into `kv_heads` groups that share KV (e.g. Llama-3 8B uses 8 groups), shrinking by
  $\text{heads}/\text{kv\_heads}$× with a quality loss far smaller than MQA.

This is exactly the ratio relationship that `check_mqa_gqa_ratios` aims to pin down.

## Order-of-magnitude intuition

For a 7B-class model ($L{=}32, n_h{=}32, d_h{=}128$, fp16), each token for each copy is about
$2\cdot32\cdot32\cdot128\cdot2 = 524$KB… multiplied by seq=2048 that is 1 GiB (batch=1). Clearly
**doubling seq → doubles the KV cache**, while the weights stay unchanged—this is the memory bottleneck of long-context batching, and the reason PagedAttention
(vLLM) manages KV memory dynamically.

## Common mistakes

- Missing the $2$ (counting only K or only V).
- Treating `head_dim` as `hidden`; when the problem gives `head_dim` directly, do not divide by heads again.
- Still using `heads` instead of `kv_heads=1` for MQA.

## Grading criteria

`test_cases.py` cross-checks the closed form against an independent reference that **accumulates layer by layer and head by head**, and separately tests the linearity in `seq/batch`, the
`kv_heads=0` boundary, and the shrink ratios of MHA/MQA/GQA.
