# Pitfalls and source-trace notes (order 835)

## Source trace

- Repo `index-tts/index-tts`, default branch **`main`** (not master; that caused a 404 in the 833 batch). Downloaded via GitHub API `contents?ref=main` + `base64.b64decode` (`raw.githubusercontent.com` proved unreliable in the 833 batch). Key sources cached at `/tmp/it_front.py`, `/tmp/it_infer_v2.py`, `/tmp/it_length_regulator.py`.

```
indextts/utils/front.py        TextTokenizer.split_segments_by_token / split_segments / punctuation_marks_tokens
indextts/infer_v2.py           remove_long_silence / generation-loop code_lens / target_lengths
indextts/s2mel/modules/length_regulator.py    (out of scope)
```

- WebSearch hits for "millisecond-level duration control" were mostly marketing / second-hand descriptions; the target-token predictor and latent interpolation are trained model internals with poor determinism, so they are **out of scope**. The problem instead takes three pure sequence/integer ledgers from the repo.
- The source cross-check script `/tmp/verify835.py` extracts the official `split_segments_by_token` verbatim via `ast.get_source_segment` and `exec`s it (stubbing `warnings.warn` as a recorder, taking `List` from typing), calls the official `remove_long_silence` directly (with torch tensors from `.venv-eval`), and aligns `target_lengths` with the official `(code_lens * 1.72).long()`. Run with `.venv-eval/bin/python` (`.venv-backend` has no numpy/torch). **split 720 + remove 30 + mel 18 = 768 groups, all identical.**

## Mechanism highlights

- **Four-way priority in `split_segments_by_token`**: comma recursion > hyphen recursion > under limit (`cur_len <= max_seg`, where only `token in split_tokens and cur_len > 2` splits) > over-limit hard cut. The first two re-run `cur` through the same logic with a new `split_tokens`; inside the recursion `,`/`-` is already in `split_tokens`, so those branches cannot trigger again.
- **The official `for` loop `i += 1` has no effect**: when splitting, the quote token is appended to `cur` and then `i += 1` is written, but `for i in range(...)` still processes that quote on the next round, so the quote token appears twice (pinned by hand anchor 3). The implementation reproduces this faithfully and does not "fix the bug".
- **Merge state machine**: `total_token` accumulates across segments (not the current segment length); `max_seg / 2` is float division; with `quick=0` the `total_token > quick` condition is almost always true, so only the `merge_half` branch exposes the `max_seg/2` threshold; `merge_flags` exposes the branch choice per segment.
- **Observability of hyphen recursion**: the outer loop triggers recursion immediately after appending the first `-`, so for most inputs the recursive split degenerates into "treat `cur` as one segment" (`-` is at the tail and `cur_len` does not exceed `max_seg`). Only when `-` makes `cur_len > max_seg` does the recursion take the hard-cut path and produce two segments, distinguishable from "one segment" (hand anchor 7).
- **Three pitfalls in `remove_long_silence`**: `count` is a whole-row global count (including after stop), so several short silence runs also trigger it; `n` resets to zero on a non-silent token, at most 10 consecutive silent indices survive; `isfix` is batch-level, and **the official code still applies `codes[:, :max_len]` horizontal clipping when `isfix=False`** (easy to miss reading lines 242-291; lines 292-297 confirm it).
- **`target_lengths = (code_lens * 1.72).long()`**: `int` truncation, not `round`; `1 * 1.72 = 1`, not `2`. `has_warned` judges only the boolean `c[-1] != stop`, no warning is actually emitted.

## Test design

- 3 `check_*` functions: `check_split_segments` / `check_remove_long_silence` / `check_mel_length_ledger`; judge `run` and `submit` both `3/3 ACCEPTED`.
- The independent oracle is rewritten in pure Python and **does not import solution**; every mode gets a full grid cross-check plus hand anchors (hand anchors are independent of both oracle and solution). The `split` grid is 11 token lists x 3 `max_seg` x 3 `quick` x 4 `split_tokens`; `remove` is 6 codes x 4 parameter sets; `mel` is 5 codes x 3 parameter sets.
- Intermediate values must be exposed: `pre_merge` / `merge_flags` / `total_token` / `n_segments`, `counts` / `kept_indices` / `isfix` / `rebuilt` / `max_len`, `code_lens` / `truncated_codes` / `target_lengths` / `total_frames` / `has_warned` all appear in the return value, otherwise branch-order and shrinking mutants are unobservable.
- **Two coverage gaps exposed by the first mutation run and their fixes**: (1) in the default-`split_tokens` case `"!"` sat at `cur_len=2` and did not trigger a split, so deleting `"!"` from the default constant went unnoticed -- changed that case so `"!"` triggers at `cur_len=4`, and added hand anchor 6; (2) in the existing `-` case the hyphen recursion was exactly equivalent to one segment, so the `sub = [cur]` mutant survived -- added the grid item `["a","b","c","-"]` with `max_seg=3` (recursion hard-cuts into two segments) and hand anchor 7. After the fix, `38/38 CAUGHT, 0 SURVIVED`.
- Anchor 5 (silence run interrupted by a non-silent token resets `n`) was also added so the `remove-n-reset` mutant dies on an assertion rather than on an unchanged shape.

## Mutation design (38 mutants, all CAUGHT)

- split family (18): missing default punctuation token, empty-input total, comma-variant miss, comma/hyphen recursion degeneration, `-` count threshold, dropping `cur_len > 2`, quote swallowing, hard-cut block size / step, inverting `total_token > quick`, widening `max_seg/2`, resetting `total_token`, swapping first/new flags, blanking `pre_merge` / `n_segments`;
- remove family (12): stop index, wrong `count` / counting only within `len_`, trigger threshold, `n < 10` to `< 11`, disabling `n = 0` reset, inverting `isfix`, pad value, disabling clip, emptying `kept_indices`, inverting `rebuilt`, zeroing `max_len`;
- mel family (8): inverting `has_warned`, `code_lens` not cut at stop, zeroing `max_code_len`, no truncation, `int` to `round`, zeroing `total_frames`, hard-coded / zeroed `ratio` echo, zeroing `max_mel_tokens`.
- True-equivalent mutants removed a priori: `segments.extend(sub)` -> `+=` (same list semantics); the else branch of `len_ = stop_idx[0] if ... else T` is covered by the "with stop / without stop" case pair, so no unreachable branch is constructed.
- Anchor strings must match source line breaks / indentation exactly (lesson from `select-suffix-no-dot` in the 834 batch); all 38 anchors hit `SRC.count == 1` this round.

## Duplicate-avoidance boundaries

- `tts-fastspeech2-length-reg`: phoneme duration -> frame **repeat expansion**; here `target_lengths` is an **integer conversion** of mel code count x 1.72, and the core is segmentation + silence shrinking, so no overlap.
- `tts-mas-monotonic-alignment` / `tts-rvq-residual-quant` / `tts-zh-tn-num-rules`: alignment search / residual quantization / Chinese TN number rules, none has sequence shrinking or recursive punctuation segmentation.
- `data-tokenization` side: `sequence-pack-pad` (concat + pad), `dt-protected-truncation` (protected truncation), `dt-gopher-rules` (filtering), `sp-unigram-segmentation` (unigram segmentation probability) -- **none has "recursive split by punctuation priority + merge state machine"**; `split_segments` is TTS long-text segmentation by semantic punctuation, an independent mechanism.
- The "silence" in `asr-fsmn-vad-endpoint` (VAD endpoint), `asr-whisper-quality-gate` (quality gate), `asr-moonshine-stream-verify` (streaming verify) is endpoint detection / quality wording, different from "shrinking consecutive silence tokens in TTS generated codes".
- Red lines: no real audio IO / mel / waveform / librosa / soundfile; no trained duration predictor, latent interpolation, GPT sampling; warning side judges only the boolean; all parameters are explicit inputs.
