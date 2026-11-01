# notes — ocr-mtmd-image-batching (order 690)

## Upstream

- Source: `ggml-org/llama.cpp` `tools/mtmd/mtmd-helper.cpp`, `mtmd_helper_eval_chunks → eval_chunk_single → decode_image_chunk` (GitHub API contents + base64); the batch struct `decode_embd_batch`/`set_position_normal(pos_0 + i)` lives in `mtmd-helper-common.h`.
- Three-stage mechanism (line-by-line verified):
  1. `eval_chunks`: empty sequence returns 0 ("no chunks to eval"); `chunk_logits_last = (i == n_chunks-1) && logits_last`;
  2. **text branch**: outer `while (i < n_tokens)` with inner `n_added < n_batch` filling, per-token `pos = n_past++`, `*new_n_past += n_added`; when `logits_last && is_last_token`, output logits are set on the **last token added in the batch** (idx);
  3. **image branch**: `mtmd_encode_chunk` produces the whole embedding span first, then `decode_image_chunk`: `n_img_batches = (n_tokens + n_batch - 1) / n_batch`, batch loop `pos_offset = i_batch*n_batch`, `n_tokens_batch = min(n_batch, n_tokens - pos_offset)`; positions are set once by `set_position_normal(n_past, ...)` as `pos_0 + i` (batches take views by offset), and **after the batches `n_past += n_pos` (= n_tokens without M-RoPE)**.
- **Official quirk (pinned)**: the image/audio branch never uses the `logits_last` argument — if the last chunk is an image there is no logits output; an empty image chunk (n_tokens=0) yields zero batches and n_pos=0, a full no-op.
- The M-RoPE branch (`n_pos_per_embd = 4`, `set_position_mrope_2d` t/y/x/z) is not sourced into a problem — the omni side owns it (`omni-tmrope-position`); this problem pins the non-M-RoPE path and notes it in the spec.
- Boundary vs `inf-*` scheduling: continuous batching is cross-request scheduling; this problem is the **within-request multimodal chunk batching and position ledger**; vs `mm-mrope-video-text-deltas`: that one covers position_ids deltas, this one covers **batch scheduling and n_past advancement**.

## Judging

- solve inputs: chunks ([{"type": "text"|"image", "n_tokens": int}, ...]), n_batch (positive int; the official GGML_ASSERT(n_batch > 0) becomes a spec ValueError), logits_last (default False), n_past (default 0). Returns batches/new_n_past/logits_pos.
- Oracle: `test_cases._py` recomputes independently via an event-table path — enumerate all batch cut boundaries per chunk first (text greedy full batches, image ceil + tail remainder), then accumulate the pos ledger from the event table; 300 random cross-checks.
- Anchors: [text 5, image 576, text 3] with n_batch=256 → batches 5 / 256(5)/256(261)/64(517) / 3(581), logits_pos=583, new_n_past=584; image-only + logits_last=True → logits_pos=None (quirk); exact division 512/256 → two full batches; 513/256 → single-token tail batch; text 7 with n_batch=3 → 3/3/1 with logits on batch 3; n_past=100 translation; three no-op empties (empty sequence / empty text chunk / empty image chunk); n_batch=1 per-token batches.
- Invalid inputs: chunks not a list / illegal type / negative or non-int n_tokens / n_batch ≤ 0 or bool / non-bool logits_last / negative n_past → ValueError.

## Mutations (12, all caught)

ceil-to-floor, last-batch-overshoot, image-pos-no-offset, image-pos-no-advance, text-logits-every-batch, logits-ignore-last-chunk, text-batch-off-by-one, image-npos-by-batches, text-pos-start-at-end, kind-mislabeled, logits-pos-first-token, mrope-npos-times4.

## Red lines

No embedding/token-id values, no mmproj encoding, no M-RoPE 4-dim positions; audio shares the image path and is not listed separately (noted in spec); continuous batching / cross-request scheduling belongs to the inference family; `omni-tmrope-position` owns the M-RoPE position ledger and `mm-mrope-video-text-deltas` the delta ledger — do not conflate.
