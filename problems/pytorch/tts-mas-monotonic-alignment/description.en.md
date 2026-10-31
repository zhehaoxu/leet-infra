# Monotonic Alignment Search Ledger

## Problem

Non-autoregressive TTS architectures (Glow-TTS / FastSpeech2) align text tokens to mel frames: **Monotonic Alignment Search** finds a single **deterministic monotone** max path over the text-latent log-likelihood matrix and extracts durations. This problem transcribes its core DP ledger: given a `[T_mel, T_text]` log-likelihood matrix, output the best path's log-likelihood, the hard alignment matrix, and per-token frame counts.

## Solution

The ledger is pinned by the spec (the plain monotone form, no skip-search):

- the path runs from `(0,0)` to `(T_mel-1, T_text-1)`: every frame consumes exactly one mel frame (t always +1), and the text pointer either **stays (same token continues) or advances by one**:
  `dp[t][s] = value[t][s] + max(dp[t-1][s], dp[t-1][s-1])`;
- `dp[0][0] = value[0][0]`; the rest of the first row/column is unreachable (-inf); the endpoint being the last token means **every token gets at least one frame** (`T_mel ≥ T_text`, otherwise unsolvable);
- backtracking the parent table yields the hard alignment `attn[t][s] ∈ {0,1}` and `durations[s]` (summing to T_mel).

Boundary with `ctc-loss` (marginalizes over all alignments): MAS is a **deterministic monotone max**. Returns `{"log_prob", "attn", "durations"}`.

## Constraints

- 1 ≤ T_text ≤ T_mel ≤ 10^4; value entries are finite numbers.
- Empty/ragged value, non-numeric or NaN/Inf entries, T_mel < T_text → `ValueError`.
