# Whisper seek loop ledger (notes)

Sources: openai/whisper `whisper/transcribe.py` (main loop, word_timestamps=False) +
`whisper/audio.py` constants.

- **Dual time precision**: token/timestamp math uses 0.02s (HOP 160 / SR 16000), while
  time_offset uses mel-frame precision 0.01s (seek is a mel-frame index). The first
  draft used 0.02 everywhere and was caught immediately by random cross-checks — the
  core precision distinction of this problem.
- **Splitting requires adjacent pairs**: `consecutive = where(ts[:-1] & ts[1:]) + 1` —
  alternating timestamps (T,F,T,F,...) do not split and fall to the else whole-window
  branch (trailing ts sets duration). Several anchors were restructured for this.
- **Unfinished tail dropped**: non-single-ending advances `seek += (tokens[last_slice-
  1]-begin) x input_stride`, ignoring everything after the last timestamp — seek can
  jump to mid-window (not a 3000 boundary).
- **no_speech double gate AND**: `no_speech_prob > 0.6 AND avg_logprob < -1.0` skips
  the window (official comment: no speech AND high decoding failure).
- **Else-branch duration**: with a trailing ts (≠ begin) use `(last-begin) x 0.02`,
  else the whole window `segment_size x 0.01`; ts exactly == begin (0s) -> whole window.
- word_timestamps / hallucination_silence / clip bounds (recovery branch) are out of
  scope. A length-1 token list makes `ts[-2:]` the whole list officially (torch
  semantics, != [False, True]) — anchored.
