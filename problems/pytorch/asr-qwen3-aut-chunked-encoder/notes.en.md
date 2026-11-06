# Pitfall notes (order 830)

## Source-reading caveats

- Repo `QwenLM/Qwen3-ASR`, `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py` (Qwen3ASRAudioEncoder). grep returned nothing on that file (pure ASCII, reason unknown) — used python line-by-line re.search instead of grep to locate line numbers.
- **The downsampling length formula is per-chunk**: `leave = lens % 100` goes through three rounds of `(l-1)//2+1` on its own, then `(lens//100)*13` is added. This is not three whole-sequence rounds — each 100-frame full chunk convs independently and pays boundary padding once (100 frames independently conv to exactly 13 tokens: 50→25→13). The 100/13 in the formula are hard-coded and do not follow config's n_window (HF config defaults n_window=100 → 200 frames, inconsistent with the weights' actual n_window=50; follow the formula, with chunk_frames an input defaulting to 100).
- chunk_plan's **divisibility quirk**: when the tail chunk's `lens % chunk_frames` leaves remainder 0, it falls back to a full window (`chunk_lengths[==0] = chunk_frames`) — for a divisible sample the "tail chunk" is the last chunk itself, and the remainder formula yields 0, which must be caught.
- Concatenated packing is **batch-concatenate then split uniformly** (`input_features.T.split(...)`): chunk coordinates are global frame coordinates, mixed across samples. The judge domain takes the single-sample version (the batch version is structurally identical), with the docstring pinning the semantics.
- Per-chunk position encoding: the position table is truncated to its first Tmax rows and added per chunk, **padded positions included** (only invisible after the mask drop) — a mutant that "skips padded positions" is unobservable in the flattened output, so the assertion must hit hidden (hidden[0,2] == padded[0,2]+pos[2]).
- cu_windows derives its window from the **padded** chunk length (`padded_t * (infer // chunk_frames)`), ignoring the real tail-chunk length — the official code windows every sample with the same window_aftercnn.
- The eager attention path builds a 4D block-diagonal mask via `_prepare_attention_mask`; FA2 goes varlen with cu_seqlens. Both share the isolated-segments semantics; the judge domain takes "bidirectional softmax(QKᵀ/√d)V within a segment + cross-segment isolation".
- The MRoPE interleaved layout (apply_interleaved_mrope, [THTHWHTHW...TT] frequency interleaving) is already covered by omni-tmrope-position in the omni collection — excluded; get_chunked_index is an MRoPE position-ids chunking helper, also excluded.
- conv_out's mel-direction input-dimension formula `((((mel+1)//2+1)//2+1)//2)` does not match the actual Conv2d derivation (mel=128 → 64→32→16; the formula gives 17) — a Whisper legacy formula; the weight's in_features is the truth. The mel direction is a static dimension, excluded.

## Test-design pitfalls

1. **Hand-computed anchor wrong for 200**: leave=0 → three rounds of (−1)//2+1 = 0, out = 0 + 2×13 = 26; the first hand computation wrote 39 (that is the ledger for 300). Anchors must be derived layer by layer, not "felt".
2. **m-pack-cnn survived (first version)**: changing pack's after_cnn to continuous whole-sequence downsampling `(l-1)//8+1` coincides on 100/50-block data (for chunk length ≤100, per-chunk ≡ continuous) — the divergence point is a chunk length >100 and not a multiple of 100 (a 250-frame chunk with chunk_frames=250: per-chunk 33 vs continuous 16). Added a chunk_frames=250 case to catch it. **Lesson: find the divergence in the formula's domain, don't just shuffle data**.
3. **A mutant dying on shape ≠ dying on semantics**: m-pos-shift (position table shifted by one) hit a slice-out-of-bounds RuntimeError when pos_emb had only Tmax rows — a fake CAUGHT; giving pos_emb one extra row (Tmax+1) keeps the shift legal and the death becomes all-wrong hidden values (semantic). Same for m-attn-leak: the first full-sequence k/v leak mismatched shapes (RuntimeError); changed to `q[a:b], k, v` — legal shapes in-segment, output polluted by out-of-segment tokens, dying on allclose. **Mutants should die on assertions, not crashes**.
4. m-plan-tailidx dies on IndexError (after the tail-index shift, chunk_lengths[tail] goes out of bounds) — acceptable: at test-data scale the shift necessarily overflows, and semantic death is indistinguishable from crash death here.
5. block_attn's cross-segment isolation gets a triple assertion: oracle comparison + mutating the second segment does not affect the first + the single window [0,6] and the two windows [0,2,6] must differ (guards against "whole-sequence attention" coinciding with "block-diagonal" on small data).
6. pack_chunks composes chunk_plan/cnn_lengths (pack internally reuses both modes' outputs) — mutants interact; the m-pack-cnn mutant also verifies the observability of the composition chain.

## mut design (17 mutants, all CAUGHT)

- cnn_lengths family: modulus shift / first-layer kernel semantics / tokens 13→14 — per-layer steps plus hand-computed anchors;
- chunk_plan family: ceil→floor / tail modulus shift / fallback value shift / tail-index pad value — tail and divisibility-fallback dual boundaries pinned;
- pack family: missing transpose / pad value 1 / after_cnn continuous form — the chunk-length >100 divergence case catches the composition chain;
- pos_flatten family: table shift / skipping mask flattening — padded-position observability (on hidden) + flatten length;
- cu_windows family: window losing ×padded_t / remainder-0 empty window — exact cu_chunk_lens list assertions;
- block_attn family: dropping √d / cross-segment leak / dropping output projection — oracle comparison plus the isolation triple.
