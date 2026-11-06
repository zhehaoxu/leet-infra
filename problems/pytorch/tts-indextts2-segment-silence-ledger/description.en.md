# IndexTTS-2: text segmentation, silence shrinking, and mel-length ledger

## Background

IndexTTS-2 in `index-tts/index-tts` has three pure sequence/integer stages around inference that are independent of the acoustic model itself: recursive text-token segmentation by punctuation priority, shrinking of over-long silence tokens in the generated codes, and converting `code_lens * 1.72` into mel frame counts after generation. This problem distills those three stages into **stateless judging**: tokens are always lists of string ids, `codes` is always a list of lists of ints, and all thresholds / constants / stop tokens are explicit inputs; no real tokenizer, no acoustic model, no audio IO. Implement a single entry point `solve(mode, **kw)` with 3 modes.

"Millisecond-level duration control" involves a trained target-token predictor and latent interpolation, which is not deterministic enough and is out of scope.

## General conventions

Every mode returns a dict; field semantics are in "Modes and rules", and unlisted intermediate values are not judged. The per-token `for` loop follows Python semantics: writing `i += 1` inside `for i in range(...)` **does not change the next iteration variable**; the official source does this and it must be reproduced faithfully. Default constants are below (judging also passes them explicitly; this is only for reference):

```
PUNCTUATION_MARKS_TOKENS = [".", "!", "?", "▁.", "▁?", "▁..."]
DEFAULT_MAX_SEG = 120
DEFAULT_RATIO = 1.72
DEFAULT_SILENT_TOKEN = 52
DEFAULT_MAX_CONSECUTIVE = 30
DEFAULT_MAX_MEL_TOKENS = 1500
```

## Modes and rules

### 1. split_segments: recursive segmentation by punctuation priority

Inputs: `tokenized, max_text_tokens_per_segment=120, split_tokens=None, quick_streaming_tokens=0`. When `split_tokens` is `None`, use a copy of `PUNCTUATION_MARKS_TOKENS`. Returns `{"segments", "pre_merge", "total_token", "n_segments", "merge_flags"}`; `pre_merge` is the segment list before merging, and `merge_flags` has one entry per segment.

Accumulate `cur` / `cur_len` token by token, deciding each round by the following priority (the first two recursively re-run `cur` through the same logic with a new `split_tokens`; inside the recursion `split_tokens` already contains `,` or `-`, so those two branches cannot trigger again):

```
not ("," in split_tokens or "▁," in split_tokens) and ("," in cur or "▁," in cur)
    -> sub = _split(cur, [",", "▁,"], ...)[0]
"-" not in split_tokens and "-" in cur
    -> sub = _split(cur, ["-"], ...)[0]
cur_len <= max_seg:
    token in split_tokens and cur_len > 2 -> split segment
    else continue
cur_len > max_seg -> hard-cut into max_seg blocks
```

- **Punctuation split**: on `token in split_tokens and cur_len > 2`, if `i < len(tokenized) - 1` and `tokenized[i + 1] in ["'", "▁'"]`, first append that quote token to `cur`, then execute `i += 1`; afterwards `segments.append(cur)` and clear `cur`. Because `i += 1` has no effect on a `for` loop, the next round still processes that quote token; reproduce this faithfully.
- **Over-length hard cut**: `for j in range(0, len(cur), max_seg)`; if `j + max_seg < len(cur)` take `cur[j:j + max_seg]`, otherwise take `cur[j:]`. All blocks are `extend`ed into `segments` and `cur` is cleared.
- After the loop, if `cur` is non-empty (assert `cur_len <= max_seg`), `segments.append(cur)`.

**Post-processing merge**: `total_token = 0`, iterate over `segments`:

```
total_token += len(seg)
if len(seg) == 0: continue
if len(merged) == 0:                      append;  flags "first"
elif len(merged[-1]) + len(seg) <= max_seg and total_token > quick:
                                          merge;   flags "merge_quick"
elif len(merged[-1]) + len(seg) <= max_seg / 2:
                                          merge;   flags "merge_half"
else:                                     append;  flags "new"
```

`max_seg / 2` is **float division**; `total_token` accumulates across segments (not the current segment length); with `quick_streaming_tokens=0` the first merge condition is almost always true, so only long or empty-`quick` scenarios expose the second condition's threshold.

### 2. remove_long_silence: silence shrinking and batch rebuild

Inputs: `codes, silent_token=52, stop_mel_token, max_consecutive=30`. Returns `{"codes", "code_lens", "isfix", "counts", "kept_indices", "rebuilt", "max_len"}`: `counts` is the whole-row silent count per row, `kept_indices` is the retained-index list per row, and `rebuilt` records whether a batch-level rebuild happened.

```
len_ = code.index(stop_mel_token) if stop in code else len(code)
count = sum(1 for v in code if v == silent_token)
if count > max_consecutive:
    for k in range(len_):
        if code[k] != silent_token:      keep k; n = 0
        elif n < 10:                    keep k; n += 1
        else:                           drop
    len_ = len(kept_indices)
else:
    kept_indices = list(range(len_))
```

- `count` is a **whole-row global count**, not a consecutive-run length; the part after `stop_mel_token` does not take part in shrinking but does take part in `count`, so "several short silence runs" can also trigger `count > max_consecutive`.
- `n` resets to zero on a non-silent token (it is not a global silence count); at most 10 consecutive silent indices are kept and the rest are dropped. The traversal range is always `range(len_)` (`len_` is already cut at the first stop).
- `isfix` is **batch-level**: if any row triggers shrinking, the whole batch is rebuilt to the longest row using `stop_mel_token` as the padding value (`rebuilt=True`); otherwise `out` is a shallow copy of each row (`rebuilt=False`).
- `max_len = max(code_lens)`; `T = len(codes[0])`. Regardless of `isfix`, whenever `max_len < T` the output is **horizontally clipped** by `[:max_len]` (the official code does this even when `isfix=False`).

### 3. mel_length_ledger: post-generation length ledger

Inputs: `codes, stop_mel_token, max_mel_tokens=1500, ratio=1.72`. Returns `{"code_lens", "max_code_len", "truncated_codes", "has_warned", "target_lengths", "total_frames", "ratio", "max_mel_tokens"}`.

```
has_warned = any(len(c) > 0 and c[-1] != stop_mel_token for c in codes)
code_lens = [c.index(stop) if stop in c else len(c) for c in codes]
max_code_len = max(code_lens) if code_lens else 0
truncated_codes = [c[:max_code_len] for c in codes]
target_lengths = [int(cl * ratio) for cl in code_lens]
total_frames = sum(target_lengths)
```

`has_warned` is the boolean behind the official "exceeding max_mel_tokens" warning (only the boolean is judged, no warning is actually emitted); `target_lengths` uses `int` **truncation**, not `round` (`1 * 1.72 -> 1`, not `2`); `ratio` and `max_mel_tokens` are echoed back unchanged.

## Input and output examples

```
solve(mode="split_segments", tokenized=["a", "b", "c", "."],
      max_text_tokens_per_segment=120, split_tokens=["."],
      quick_streaming_tokens=0)
# {"segments": [["a", "b", "c", "."]],
#  "pre_merge": [["a", "b", "c", "."]], "total_token": 4,
#  "n_segments": 1, "merge_flags": ["first"]}

solve(mode="mel_length_ledger", codes=[[3, 4, 999, 8], [5, 6, 7]],
      stop_mel_token=999, ratio=1.72, max_mel_tokens=1500)
# {"code_lens": [2, 3], "max_code_len": 3,
#  "truncated_codes": [[3, 4, 999], [5, 6, 7]], "has_warned": True,
#  "target_lengths": [3, 5], "total_frames": 8,
#  "ratio": 1.72, "max_mel_tokens": 1500}
```

## Out of scope

Real BPE / tokenizer, the `s2mel` length regulator frame expansion (= FastSpeech2 length-reg), GPT sampling and emovec, BigVGAN / mel / waveform IO, trained duration predictors and latent interpolation, actually emitting `warnings.warn`, and `torch.tensor` numeric forward. Judging covers only the three deterministic ledgers above.
