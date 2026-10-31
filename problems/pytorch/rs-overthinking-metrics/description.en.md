# Overthinking Metrics: Token Efficiency and Accuracy-Length Binning Ledger

## Problem

"Overthinking" evaluation for reasoning models asks one thing: how many reasoning tokens do correct vs incorrect answers spend, and how does accuracy decay across length bins. This problem turns these metrics into a deterministic ledger: given `correct`/`tokens` for N samples and explicit `bin_edges`, return overall accuracy, per-bin count/accuracy/mean_tokens, mean tokens of correct vs wrong answers with the efficiency ratio, and the best bin.

## Solution

All conventions are pinned by the problem statement:

- Binning: `bin_edges` strictly increasing with K+1 entries → K bins; bin j (0 ≤ j < K−1) is `[edges[j], edges[j+1])`, the last bin is `[edges[K-1], +∞)`; samples below `edges[0]` fall into the last bin (pinned).
- Empty bins have `accuracy`/`mean_tokens` = None; `best_bin` is chosen among non-empty bins only, ties to the smallest index (implemented with a strict `>` scan).
- `len_correct`/`len_wrong` are the token means of correct/incorrect samples (None when absent); `efficiency_ratio = len_wrong / len_correct` (None if either is missing or len_correct = 0).
- `best_bin` semantics: the highest-accuracy bin (even all-wrong takes the first non-empty bin, 0.0 > initial −1).

Key points: accuracy is n_correct / N (all samples); per-bin accuracy divides by the bin count; boundaries are left-closed right-open with the last bin right-inclusive.

## Data scale

- 1 ≤ N ≤ 64, tokens ≥ 0, 2 ≤ len(bin_edges) ≤ 8.
- Length mismatch, empty records, non-increasing edges, bool/negative/non-int entries → `ValueError`.
