# CTC Prefix Beam Search Ledger

## Problem

The beam-side CTC decoder works in **prefix space** instead of taking per-frame argmax: each prefix s carries two probabilities — the sum of path probabilities ending in blank (p_b) and ending in a non-blank (p_nb). This problem transcribes Graves' (1504.08129) prefix beam search in its no-LM, linear-probability form: given a per-frame probability matrix `[T, V]`, output the top-k prefixes by probability.

## Solution

The ledger is pinned by the spec:

- each frame expands every prefix s over every class c:
  - **blank class**: `p_b(s) += (p_b(s) + p_nb(s)) * P(blank)` — blank leaves the prefix unchanged and both end-states may come;
  - **non-blank c with s ending exactly in c (repeated character)**: a new character may only be appended from a blank-ending path (`p_nb(s+c) += p_b(s) * P(c)`; otherwise it merges with the just-emitted same character), and the repetition extends in place (`p_nb(s) += p_nb(s) * P(c)`);
  - **non-blank c otherwise**: `p_nb(s+c) += (p_b(s) + p_nb(s)) * P(c)`.
- after each frame, prune to `beam_width` by total probability `p_b + p_nb` descending; **ties break by prefix lexicographic order ascending** (reproducible).
- zero-probability hypotheses are filtered from the output; `k` defaults to `beam_width`.

Initial `beam = {(): (1.0, 0.0)}` (the initial split is consumed as the p_b+p_nb sum on the first frame and rewritten by it — the `(0,1)` initial value is a mathematically equivalent mutant, noted in notes). Returns `{"hypotheses": [{"prefix", "prob"}]}`.

## Constraints

- 1 ≤ T ≤ 10^4, 2 ≤ V ≤ 10^5 (the number of live prefixes is bounded by beam_width).
- Empty/ragged probs, negative or NaN/Inf entries, out-of-range blank_id, non-positive-int or bool beam_width/k → `ValueError`.
