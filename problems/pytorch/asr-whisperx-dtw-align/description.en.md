# WhisperX Forced-Alignment Ledger: Trellis DP, Backtracking and Two-Level Merging

## Problem

WhisperX upgrades Whisper's coarse timestamps to phoneme-level alignment: an independent CTC alignment model (emission, (T, V) log probabilities) aligns the transcript tokens via Viterbi forced alignment. The core is four pure algorithms — trellis DP, backtracking, `merge_repeats` (token-level merging) and `merge_words` (word-level merging). This problem hand-writes the whole chain without torchaudio/torio/numba/whisperx.

## Solution

Official semantics (m-bain/whisperX `whisperx/alignment.py`, the torch fallback branch verbatim):

- **trellis** of shape (T+1, N+1) (an extra dim on both axes; the token dim represents `<SoS>`): `trellis[0,0]=0`; column 0 = cumsum of blank emissions (all-blank prefix cost); row 0 token columns = −inf; **the last N rows of column 0 = +inf** (forced-completion mechanism — when backtracking reaches a tail row at j==1, `changed=+inf` always wins); recurrence `max(stay, change)` entirely in log space;
- **Backtracking**: `t_start = argmax(trellis[:, N])` (ties take the **first** row); each step uses strict `changed > stayed` — **ties stay on blank and do not advance**; frame probability `exp(emission[t-1, lbl])` (probability domain, mixed with the trellis's log domain exactly as official; blank frames' probabilities also enter the path); if j never reaches 0 before the loop ends → **None** (guaranteed when T < N: the column is all −inf, argmax returns row 0, empty path);
- **merge_repeats**: consecutive frames of the same token merge into `[first frame, last frame + 1)` (half-open), score = **arithmetic mean** of frame probabilities (blank frames count in the denominator);
- **merge_words**: split on separator `|`; word label = concatenated chars, span = `[first start, last end)`, score = **length-weighted** mean `Σ(score·len)/Σlen` — a different aggregator from merge_repeats' frame mean; leading / consecutive (`i1 == i2` guard) / trailing separators produce no empty words.

## Data Scale

- Hand anchor: a T=4/V=3 emission with the full ledger hand-computed (trellis corner = −3.4, col0_inf=2, path covering only frames 1..2).
- Boundaries: blank frames where stay beats change, exact stay/change ties (strict `>`), T<N → None, separator leading/middle/trailing, both aggregation formulas on multi-frame segments.
- 200 randomized cross-checks (T∈[3,24], V∈[3,8], random separators mixed in) against a verbatim transcription of the official torch fallback.

## Scoring

The full ledger must match the official implementation: ok/col0_inf exactly, path (token, time) exactly with probabilities at 1e-6, segments/words spans exactly with scores at 1e-6.

## Constraints

- No torchaudio / torio / numba / whisperx calls;
- Only torch tensor ops + pure Python loops; emission/tokens/transcript are explicit inputs (no real audio IO).
