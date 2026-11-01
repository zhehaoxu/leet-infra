# notes — asr-whisperx-dtw-align (order 714)

## Sourcing

- Authoritative source: m-bain/whisperX `whisperx/alignment.py` (main, 2026). The four core functions `get_trellis` / `backtrack` / `merge_repeats` / `merge_words` are the torchaudio forced-alignment tutorial algorithm; WhisperX adds a numba branch (`_safe_njit`). The **torch fallback branch matches the numba branch semantically** — the reference here transcribes the fallback.
- **Column-0 tail +inf**: `trellis[-num_tokens:, 0] = float("inf")` — not a typo. Effect: when backtracking reaches a tail row at j==1, `changed = trellis[t-1, 0] + e = +inf` always beats stayed, forcing completion. Column N can never become +inf (inf entering at (r0,1) reaching column N requires r0 ≥ T-N+2 and N-1 steps within T rows → T+1 ≤ T, contradiction), so the fill never pollutes the argmax; net effect: "T ≥ N ⇒ backtrack always ok".
- **ok ⇔ T ≥ N**: for T ≥ N, rows ≥ N of column N are all finite (max propagates finite values), argmax ≥ N, and each backtrack step stays consistent with the DP with enough rows remaining → always completes; for T < N the column is all −inf → argmax=0 → empty path → None. The tests assert this invariant directly.
- **Mixed domains**: the trellis is entirely log-domain (emission added as-is), but frame probabilities are `exp(emission)` back in probability domain; segment scores use the frame mean (merge_repeats) while word scores use length weighting (merge_words) — two aggregators, official as-is, not unified.
- **Strict `>` ties to blank**: backtracking's `changed > stayed` takes the else branch on ties (blank, no advance); the DP recurrence `stay if stay > change else change` keeps stay on ties — consistent strictness; the `>=` mutation is caught.

## Test Design

- The oracle is a **verbatim transcription** of the official torch fallback (not an independent rewrite) — for an algorithm account the anchor is the official code itself; a separate hand-computed anchor (4×3 example, corner/path/segments hand-checked) is also included.
- **Tolerance 1e-6, not 1e-9**: the solution uses a float64 trellis while the official uses float32; cumsum chains drift ~1e-8, so 1e-9 always fails (first-version pitfall).
- Separator case: token index 0 (= CTC blank) doubles as the separator token — harmless in the DP (changing to tokens[j]=0 reads the same column), official semantics intact.
- The randomized checks enforce `ok == (T >= N)` and `col0_inf == N` per group.

## Mutations (13 caught + 2 equivalent)

- Catching hinges on: tail +inf polarity, DP max/min, strict `>` in backtracking, frame probability taken from lbl (blank frames don't masquerade as token probabilities), first-row argmax, segment end +1, frame mean vs length weighting, empty-word guard, path reversal, corner row, col0_inf sign.
- 2 equivalent mutations (SURVIVED): (1) `path.reverse()` ↔ `path = path[::-1]`; (2) ternary max ↔ builtin `max()` (identical values on ties).

## Pitfalls

- Input validation must run **before** the `.shape` access (str input would AttributeError before the intended ValueError).
- `trellis[0, -N:]` with N=0 becomes `trellis[0, 0:] = -inf`, overwriting `trellis[0,0]`'s 0 — the official code is pathological for empty transcripts; the problem guards N ≥ 1.
- Boundary with 670 (Whisper quality gating) / 700 (seek loop): this problem covers the alignment model's timestamp refinement; those two cover the transcription main loop.
