# Zipformer Streaming Cache State Ledger

## Background

Zipformer2 (k2-fsa/icefall, `egs/librispeech/ASR/zipformer/zipformer.py`) does not use a traditional KV cache for streaming inference. Instead it keeps a **fixed-length rolling left context**: each layer caches 6 tensors (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2), and every chunk concatenates new frames then keeps the trailing L frames as the new cache. This problem rebuilds the decidable parts of that ledger: cache rolling, relative-position alignment, gated convolution, and the chunk-causal convolution decomposition with its divisibility identity.

## Mechanism breakdown

### 1. Rolling cache (cache_roll / attn_scores / value_out / nonlin_out)

All cache updates share the same idiom (key shown):

```python
k = torch.cat([cached_key, k], dim=0)     # (L+seq, b, dim)
cached_key = k[-left_context_len:, ...]   # trailing L frames
```

- The cache length is always `left_context_len` (conv caches use `left_pad = kernel_size // 2`);
- **Not a KV cache**: old frames are evicted, so anything older than L frames is invisible (the source of streaming accuracy loss);
- `left_context_len = left_context_frames // downsampling_factor` (per stack, after encoder downsampling).

### 2. Relative-position alignment (attn_scores)

In `RelPositionMultiheadAttentionWeights.streaming_forward`, the last axis of pos_scores is a **relative position**, folded into absolute key indices via `as_strided`:

```python
pos_scores = pos_scores.as_strided(
    (h, b, seq, k_len),
    (s0, s1, s2 - s3, s3),
    storage_offset=s3 * (seq_len - 1))
```

The equivalent explicit gather is `indexes[t, j] = (seq-1-t) + j`. Derivation: flattening the contiguous tensor with the official strides shows element `[h,b,t,j]` comes from `orig[h, b, t, (seq-1-t)+j]`. The relative range `[(seq-1)-t, L+2seq-2-t] ⊂ [0, 2seq-2+L]` fits exactly inside the pos_emb length. Scores = `q@k + pos_scores`; padding-masked positions are filled with `-1000` before softmax.

### 3. Gated convolution (conv_gated)

The gating part of `ConvolutionModule.streaming_forward`: in_proj outputs `2*channels`, the second half goes through sigmoid as a per-channel gate `x = x * sigmoid(s)`; padding frames are `masked_fill(..., 0.0)` before the depthwise conv (otherwise padding would leak into the convolution sum).

### 4. Chunk-causal convolution (chunk_causal_conv / chunk_scale / stream_equiv)

`ChunkCausalDepthwiseConv1d` splits a k-tap depthwise convolution into two additive branches:

- **causal branch**: a true causal conv with half kernel (`(k+1)//2` taps), left-padded by `left_pad = k//2` (in streaming that pad comes from the cache);
- **chunkwise branch**: a symmetric k-tap conv (constructed with `padding=k//2`) that only sees neighbors within the chunk, scaled by `chunk_scale`.

`_get_chunk_scale(scale, cs)` returns a `(channels, cs)` table:

```
1.0 + left_edge(head correction, length k, zero-padded right to cs) + right_edge(tail correction, length k, zero-padded left to cs)
if cs < k: truncate — left keeps the first cs, right keeps the last cs
```

When cs and k overlap the two corrections add (columns 3-4 overlap for cs=8, k=5).

**Divisibility identity**: when `seq % chunk_size == 0`, the concatenation of per-chunk streaming calls (chained cache) equals the official one-shot `forward(x, chunk_size)` step by step — the core correctness property of the streaming design, and the strongest assertion in this problem's judge. For non-divisible lengths the official forward pads with `right_pad = -seq % chunk_size` (streaming has no such step).

## Judging domain

- All weights (causal_w/chunk_w/scale/in_proj_w etc.) and inputs are explicit arguments, fp64;
- Every assertion is a closed-form cross-check or identity — no RNG-drift anchor values;
- Modes: cache_roll, attn_scores, value_out, nonlin_out, conv_gated, chunk_scale, chunk_causal_conv, stream_equiv;
- Excluded: real model forwards, the subsample frontend, streaming behavior for non-divisible right_pad, training-time Balancer/Whiten.
