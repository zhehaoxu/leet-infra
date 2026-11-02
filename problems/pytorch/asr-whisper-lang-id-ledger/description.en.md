# Whisper Per-Window Language-ID Ledger: Suppression Renormalization and First-Window Policy

## Problem

Whisper's language detection runs **outside** the main decode loop (to avoid interfering with kv-caching): a single-token `[sot]` forward produces SOT-position logits, with all non-language tokens suppressed to −inf before argmax + softmax. This problem hand-writes that account — no model; SOT logits are explicit inputs, and the ledger lives in suppression, renormalization, tie-breaking, and batch semantics.

## Solution

Official semantics (openai/whisper `whisper/decoding.py::detect_language` + `transcribe.py`, v20231105+):

- **Batch semantics**: officially `single = mel.ndim == 2` — a 2D input is a single audio, `unsqueeze(0)` to batch, unwrapped on return (scalar token + single dict); batch input returns a tensor + list of dicts. Here: 1D logits = single;
- **Suppression**: mask all-True with language positions (`all_language_tokens`) set False, then `logits[:, mask] = -inf` — non-language positions are **−inf, not 0** (exactly 0 after softmax);
- **argmax**: taken after suppression; ties take the **first index** (torch.argmax semantics);
- **Renormalization**: `language_token_probs = logits.softmax(dim=-1)` computed after suppression — equivalent to renormalizing over the language set only; the raw softmax (over the original logits) at language positions differs (non-language mass removed); the probs dict holds only language positions with keys inserted in `all_language_codes` order;
- **First-window policy** (transcribe side): detection only when `language=None`, and only on the **first 30s window** (`pad_or_trim(mel, N_FRAMES)`); subsequent windows reuse the first window's result (language switches are not perceived); an explicit `language` skips detection;
- **Selection rule**: `max(probs, key=probs.get)` — Python max takes the **first maximal key** on ties (insertion order);
- **Errors**: empty language table / length mismatch → `ValueError` (official "This model doesn't have language tokens so it can't perform lang id").

## Data Scale

- Compact language table (5 languages, vocab 16/24/32), random logits with explicit seeds.
- Boundaries: two languages exactly tied (argmax vs max tie rules), 1D/2D paths, explicit-language skip, renormalization = raw/Σraw hand check.
- 200 randomized cross-checks (oracle implements the official decoding.py logic).

## Scoring

Every entry matches the official logic: tokens/top exact, probs at 1e-9, prob_sum = 1 (renormalization verified), raw_top_prob as the un-renormalized control (always < renormalized value).

## Constraints

- No whisper / transformers calls;
- Only torch/numpy; logits and the language table are explicit inputs (no model, no real audio).
