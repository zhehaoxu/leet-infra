# Moonshine streaming verify and cache ledger

Moonshine's (moonshine-ai/moonshine) streaming transcriber
(`core/moonshine-streaming-model.cpp`) chains "audio chunk in → the encoder
commits only stable frames → memory changes invalidate the cross-KV → greedy
speculative verification of the previous draft → rewind at the divergence point
and continue" into one **cross-chunk state-machine ledger**. Delimited from the
speculative-decoding collection (rejection sampling / residuals / tree verify):
this is **greedy argmax verification**, no sampling; delimited from problem 826
(the biaser trie itself): this problem uses the biaser's walk semantics but not
its construction.

## Mechanism

### 1. Stable frames (encode)

- The encoder has `total_lookahead = 16` frames of lookahead: non-final commits
  only `max(0, total − 16)` "stable" feature frames; `is_final` commits
  everything;
- `new_frames = stable − already emitted`, ≤ 0 emits nothing (with little
  audio the lookahead swallows everything);
- The encoder re-runs on a sliding window: a fixed 16-frame per-layer left
  context → `left = 16 × depth`, `window_start = max(0, emitted − left)`,
  `window_size = total − window_start`; the new frames are sliced from the
  window at `start_idx = emitted − window_start`, the adapter continues at
  `pos_offset`, and memory is appended.

### 2. Cross-KV lifecycle (cache_policy)

- Appending to memory (encode producing > 0 frames) → `cross_kv_valid = False`;
- `compute_cross_kv` rebuilds from memory → True;
- **`decoder_reset` clears only the self-attention KV and keeps cross-KV
  valid** (memory unchanged) — the official comment says so explicitly;
- **A frame-less encode never touches the flag** (its early return precedes
  the invalidation).

### 3. Decode budget

`max_tokens = min(ceil(memory_len × 0.020 × 6.5), 448)` — 20 ms per frame,
6.5 tokens per second of speech, capped at max_seq_len.

### 4. Greedy speculative verification (verify_decode)

- Verification: per-row **argmax** over the logits of `[BOS] + draft`
  (teacher forcing); the biaser bonus lands before every argmax;
- **The biaser walks the teacher-forced prefix**: `advance(tokens_with_bos[t+1])`
  uses the draft token, not the prediction — position t is conditioned on the
  prefix itself (a pitfall the official comments call out);
- `diverge_point` = number of leading matching tokens (the first mismatch
  stops the count);
- **Full match**: continue auto-regressively from `predictions[len(draft)]`
  (the bonus row);
- **Divergence**: `cache_reset = True` (the self-KV is cleared and re-run),
  **the biaser walk is rewound to the accepted prefix** (reset + re-walk
  accepted — the walk otherwise still sits on the whole draft, rejected tokens
  included), then the biased argmax of `rerun_logits[diverge_point]` continues;
- The AR loop pushes the current token (**EOS never enters the result**) →
  biaser advance → argmax of the next logits row; it stops at EOS or the
  `max_tokens` budget.

### 5. Interaction pitfalls with 826

- Verification bonuses can flip acceptance: when a rival is stronger, the
  logarithmic depth ramp is what gets the draft token accepted — **whether the
  bonus stacks (max vs sum) directly changes the verdict**;
- Whether the rewind runs directly changes the rerun row's argmax (walk sitting
  on the whole draft vs the accepted prefix).

## Judging domain

Single entry `solve(mode, **kw)`, six modes: `stable_frames` / `encoder_window`
/ `max_tokens` / `cache_policy` / `diverge_point` / `verify_decode`. All logits
are explicit inputs; greedy argmax, no sampling. The incremental mel/conv
frontend (ONNX-internal state), word timestamps, and the context biaser's trie
construction are out of scope.
