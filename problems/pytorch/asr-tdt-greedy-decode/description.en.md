# NeMo TDT Token-and-Duration Greedy Decoding Ledger

## Problem

TDT (Token-and-Duration Transducer, arXiv 2304.06795) splits RNN-T's single blank channel into two: the token head still contains blank, and a separate duration head directly predicts "how many frames this emission/pause spans". NeMo's greedy decoder therefore carries a whole extra frame-skipping ledger beyond RNN-T. This problem hand-writes a scalar stateless version of GreedyTDTInfer.torch_impl: the predictor is not modeled (stateless), the joint's two logit streams are passed in per frame, and a pure scalar loop reproduces the full official decoding.

## Solution

Official behavior (NVIDIA/NeMo `nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py`, pinned line by line):

- **Dual argmax**: `token = argmax(token part)`, `duration = model_durations[argmax(duration head)]`; ties resolve to the first index (torch.argmax semantics);
- **Blank forces duration >= 1**: `durations.masked_fill_(durations == 0 & blank_mask, 1)` — blank may not stay in place (in-place dwelling is RNN-T blank semantics; TDT expresses it via duration);
- **Non-blank also skips frames by duration**: this is the essential TDT-vs-RNN-T difference — after emitting, jump forward by duration frames with no trailing blank; duration may push time past the utterance end, and the emission (which happens before the jump) is still valid (official comment *NB: difference between RNN-T and TDT*);
- **Emission ledger**: a token is stamped at the pre-jump timestamp (`time_indices_current_labels` snapshots before advance); durations record the model's predicted raw value (may contain 0);
- **max_symbols anti-loop**: a non-blank with duration=0 emits repeatedly in place (label loop). Official protection via max_symbols — once consecutive non-blank emissions at the same timestamp reach max_symbols, blank is forced, and `time_indices += 1` (**advances exactly 1 frame, ignoring duration** — not duration=1 semantics);
- **Domain constraint**: max_symbols is None with a non-blank duration=0 -> ValueError (the official code would loop forever with a stateless predictor);
- **Termination**: `time >= encoder_output_length` (active_mask all False); there is no RNN-T "must emit blank to finish" — a non-blank + duration that jumps straight out of bounds ends the decode, and final_t may exceed T.

## Data Scale

- Miniature config V=4 (blank + 3 tokens), D=4 (durations [0,1,2,3]), logits constructed so every argmax is pinned;
- Hand-computed anchors: basic (last token with d3 jumps past the end -> final_t > T), zero-duration-domain (ValueError), max_symbols=1 (blank forced after every non-blank), max_symbols=2 (forced after two consecutive emissions), blank-zero (d0 blank coerced to 1), out-of-bounds termination;
- 200 randomized comparisons (oracle = scalar transcription of torch_impl).

## Scoring

All five ledger keys match exactly: tokens / timestamps / durations / final_t / forced_blanks (pure integer ledger).

## Red Lines

- No calls to nemo / numba; pure scalar loop;
- model_durations non-negative; token_logits/duration_logits frame counts and widths must match (else ValueError).
