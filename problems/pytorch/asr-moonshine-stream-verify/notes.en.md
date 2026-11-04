# Solution notes — Moonshine streaming verify

## Official account

- moonshine-ai/moonshine `core/moonshine-streaming-model.cpp` (encode /
  compute_cross_kv / decode_full / decoder_reset) + `.h` (config constants:
  frame_len = 80, total_lookahead = 16, left context 16×depth,
  max_seq_len = 448); the official comments explain the teacher-forced biaser
  walk and the rewind.
- Judging domain: stable-frames/window ledger, decode budget, cross-KV
  lifecycle, greedy speculative verification with divergence point + rewind +
  biaser interaction. The ORT frontend increments, word timestamps, and the
  trie construction (= 826) are out of scope; no overlap with the sampled
  verification of the speculative-decoding collection (this one is greedy).

## Pitfalls

- **Mutant anchors colliding with the docstring (second replay of the 820
  trap)**: `left = 16 * depth`, `window_start = max(...)`, `cache_reset = True`
  all appear in the module docstring first, so the replace landed on the
  comment and the mutant survived falsely — anchors need indentation context.
- **Two true-equivalence removals**: ① dropping `max(0, ·)` on stable (a
  negative stable is caught by `new_frames <= 0`); ② walking the biaser over
  predictions vs the teacher-forced prefix during verification — divergence
  rewinds and full-match makes them equal, so the difference is not observable
  in the output (the official correctness argument cannot be mutation-judged).
- **The stable-frames assertion itself was miscomputed**: total=30/emitted=14/
  lookahead=16 gives stable=14, new=0 — total=46 is needed for 16 new frames;
  derive expectations step by step via `stable = total − lookahead`.
- **Biaser trie test data must be shape-consistent**: children needs one entry
  per depths entry (leaves get empty dicts); the IndexError fallback would
  otherwise masquerade as CAUGHT-ERR.
- **Rewind observability**: only visible for "divergence + biaser + a rerun
  row whose rival flips between the bonus and no-bonus argmax" — test B3
  builds exactly that (4: 1.0+3.386 vs 7: 2.0); a divergence without a biaser
  cannot detect the rewind.
- **Dedup-max observability**: needs the same token proposed by both the root
  and a deeper node with a rival between the two values (test G:
  3 = 0.5 + max(2, 3.386) = 3.886 < 4.0 rejected; stacked 5.886 would accept).
- **EOS never enters tokens**: the continue_ar while condition checks eos
  first; an eos start point yields an empty result; an eos AR row stops
  without appending.
- cache_policy's `("encode", 0)` is a no-op, not an error (the early return
  precedes the invalidation) — the first draft fell into the unknown-op branch
  and the test caught it.

## Mutant list (17)

stable: is_final full-commit dropped, new≤0 clamp → abs; window: left 16×→16+,
zero clamp dropped, size computed from emitted; max_tokens: ceil→floor, cap
dropped; cache: frame-less encode invalidates, decoder_reset wrongly
invalidates; diverge: point off by one, no break (last match wins); verify:
rewind dropped, cache_reset always False; AR: eos pushed, budget <→≤; bias:
stacking, parent depth. 17/17 CAUGHT + 2 true-equivalence removals
(stable-clamp, predwalk).
