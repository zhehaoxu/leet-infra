# Qwen3-ASR AuT Chunked-Encoder Ledger

## Background

Qwen3-ASR's audio encoder AuT is Whisper-style Conv2d downsampling plus a Transformer, but to support unified streaming/offline inference and memory control on long audio it splits mel frames into fixed windows (chunks) and runs **each chunk independently through three Conv2d layers (kernel 3 / stride 2 / padding 1, 8× downsampling)**; attention is also windowed by cu_seqlens (bidirectional within a segment, no interaction across segments). Chunking is semantics, not an implementation detail: the convolution's boundary padding is paid once per chunk, and the official downsampling length formula hard-codes exactly that. This problem rebuilds the decidable parts of that design.

## Mechanism breakdown

### 1. Per-chunk downsampling lengths (cnn_lengths)

`_get_feat_extract_output_lengths` is not "three whole-sequence rounds of (l-1)//2+1":

```
leave = lens % 100          # hard-coded 100 (full-chunk frames n_window*2), config-independent
l1 = (leave - 1) // 2 + 1
l2 = (l1 - 1) // 2 + 1
l3 = (l2 - 1) // 2 + 1
out = l3 + (lens // 100) * 13   # each 100-frame full chunk independently convs to exactly 13 tokens (50→25→13)
```

Boundary: when leave=0 (a multiple of 100) the three rounds give `(−1)//2+1 = 0`, and the length is carried entirely by full-chunk count × 13 — self-consistent with chunk_plan's "remainder-0 falls back to a full window" (250 → 33, 200 → 26).

### 2. Chunk plan (chunk_plan)

`chunk_num = ceil(lens / chunk_frames)`; chunk_lengths starts as all full windows, the tail chunk is located via `tail_chunk_index = F.pad(chunk_num, (1,0), -1).cumsum(0)[1:]`, overwritten with `lens % chunk_frames`, and **remainder 0 falls back to a full window** (`chunk_lengths[==0] = chunk_frames` — for a divisible sample the "tail chunk" is the last chunk itself).

### 3. Concatenated packing (pack_chunks)

**The mel frames of all samples in a batch are first concatenated into one long sequence and then split by chunk_lengths** — chunks are per-sample-independent units in global frame coordinates. After splitting, `pad_sequence` aligns to (N, Tmax, C) then transposes to (N, C, Tmax) with padding strictly zero; each chunk's post-CNN length is `cnn_lengths(chunk_lengths)`, and a bool mask marks valid positions. The batch-dim concatenated packing shares the same ledger as the single-sample version (which is its special case).

### 4. Positional embedding and flattening (pos_flatten)

The sinusoidal position table is truncated to its first Tmax rows and **added per chunk** — every chunk's positions restart from 0 (per-chunk position encoding; padded positions are also added, then dropped by the mask). `padded_embed[mask]` boolean-flattens the valid positions of all chunks back into one sequence, ordered by (chunk, t) lexicographic order.

### 5. Attention windowing (cu_windows)

`window_aftercnn = padded_t * (infer_window // chunk_frames)` — the window length is derived from the **padded** chunk length (uniformly post-padding, ignoring real tail chunks). Per sample the aftercnn_len splits into `cnn_len // window` full windows plus a remainder window (no remainder window when the remainder is 0); cu_chunk_lens cumsum globally into cu_seqlens.

### 6. Block-diagonal attention (block_attn)

The eager path: q/k/v weights explicit inputs, `softmax(QKᵀ·scaling)V` with scaling=1/√d; standard bidirectional attention within each cu_seqlens segment, **no interaction across segments** (official FA2 uses varlen, eager uses a 4D block-diagonal mask; the semantics match: bidirectional within, isolated across).

## Judge domain

- Mel features, position table, and attention weights are all explicit inputs, fp64; the time-dimension ledger is pure integers; no RNG (block_attn/pos_flatten use cases manual_seed internally);
- 6 modes: cnn_lengths, chunk_plan, pack_chunks, pos_flatten, cu_windows, block_attn;
- Excluded: the mel front-end (audio collection owns signal processing), the MRoPE interleaved layout (omni-tmrope-position already covers it), the full Conv2d forward (the mel-direction dimension formula is ambiguous — the official code carries a Whisper legacy formula), and the Qwen3 decoder side (generic LLM ledger).
