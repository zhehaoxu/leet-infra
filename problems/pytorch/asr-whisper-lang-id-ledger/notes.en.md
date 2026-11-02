# notes — asr-whisper-lang-id-ledger (order 715)

## Sourcing

- Authoritative source: openai/whisper `whisper/decoding.py::detect_language` + `transcribe.py` (main, 2026). detect_language itself is ~40 lines but ledger-dense.
- **Post-suppression softmax = renormalization**: the official code sets `logits[:, mask] = -np.inf` before `softmax(dim=-1)` — non-language positions get exp(−inf)=0, language probabilities equal raw/Σraw(languages). The raw-vs-renormalized gap is this problem's control account (`raw_top_prob`).
- **First-window policy**: transcribe runs detect_language only on `pad_or_trim(mel, N_FRAMES)` (the first 30s window); later windows reuse it — **language switches are not perceived** (the known limitation for long audio with mid-stream language changes). `decode_options["language"] = max(probs, key=probs.get)` — dict ties take the first inserted key.
- **Batch semantics**: `single = mel.ndim == 2` (for mel; 1D for SOT logits) — unsqueeze(0) to batch, unwrap on return. Per-audio softmax/argmax are independent.
- **Encoder shape-collision quirk**: `if mel.shape[-2:] != (n_audio_ctx, n_audio_state): mel = model.encoder(mel)` — mel exactly (1500, 1280) (large tier) is treated as pre-encoded features and skips the encoder. Model-dependent, not made into a problem; noted here.
- Detection runs outside the decode loop specifically **not to interfere with kv-caching** (docstring quote).

## Test Design

- The oracle transcribes the official decoding.py logic; a compact language table (token ids 3–11, vocab 16) replaces the real 50k vocab. Token ids deliberately misordered relative to codes (`[7,3,11,5,9]`) so the argmax tie rule ("smaller index") and the max tie rule ("first inserted key") **disagree** (3 vs zh=7) — the two tie rules are pinned separately.
- 200 randomized cross-checks covering single/batch paths.
- `_make_logits(tie=True)` sets two language logits to the same 2.0: asserts `tokens == min(the two tokens)`, `probs[zh] == probs[en]`, and `top == first inserted key`.

## Mutations (12 caught + 2 equivalent)

- Catching hinges on: mask polarity, −inf vs 0 suppression, softmax before/after (raw==renorm fails immediately), argmax on raw (may select a non-language token), tie flip (**`reversed(list(d))` is a genuine flip — the first draft used `sorted(reverse=True)`, which is a stable sort keeping original tie order, i.e. an accidental equivalent mutation; always verify stability when writing tie mutations**), zip key/value swap, single-unwrap off, sum→max, skip branch still booking, detected always-true, half-temperature, enumerate misalignment.
- 2 equivalent mutations (SURVIVED): (1) the redundant clone in `batch.double().clone()` (double() already returns a new tensor); (2) argmax(suppressed) ↔ argmax(softmax(suppressed)) (softmax is strictly order-preserving, including −inf→0).

## Pitfalls

- Compact language token ids must be < vocab V (the first draft's 50259 > 512 raised IndexError immediately).
- `_make_logits(n=1, single=False)` returns (1, V) — forgetting single=True in single-path checks asserts scalars against lists.
- Boundary with 670 (quality-gate fallback): this problem covers the language-detection step's math and policy; 670 covers decode_with_fallback's temperature descent and three-gate logic. Boundary with 700 (seek loop): detection happens once outside the loop; the loop itself carries no language account.
