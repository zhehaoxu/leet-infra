# Beam Search: One-step Expansion, Scoring, and Backtracking

## Problem

**Beam Search** maintains $B$ candidate sequences (beams) at each step. In one decoding step: every beam tries appending every token in the vocabulary, producing $B\times V$ candidates; each candidate's cumulative score = the beam's existing cumulative log-probability + the new token's log-probability; then the top $B$ scoring candidates out of $B\times V$ become the beams for the next step.

The key is **backtracking**: the selected top-$k$ are "flattened" global indices, and you must recover which parent beam each came from (`parent = flat // V`) and which token was appended (`token = flat % V`) in order to carry over the correct history sequence and continue it. This step is what makes beam search more than greedy: it "keeps multiple paths and backtracks as needed".

## Rules

Given `input_ids` $(B, L)$ (the token sequences of the current $B$ beams), `beam_scores` $(B,)$ (their cumulative log-probs), `next_log_probs` $(B, V)$ (the log-probability of the next position under each beam), and `num_beams` $k$:

1. Candidate scores $S[b, t] = $`beam_scores[b]` $+$ `next_log_probs[b][t]`, shape $(B, V)$
2. **Flatten** in row-major order to $B\cdot V$; index $f$ corresponds to $b = f // V$, $t = f \bmod V$
3. Take the $k$ highest scores (`torch.topk`); **on ties, keep the smaller flattened index** (topk returns equal values in ascending index order)
4. For each selected entry, recover `parent = f // V`, `token = f % V`
5. New sequence = `input_ids[parent]` with `token` appended; new score = that candidate's $S$ value

Return `(new_input_ids, new_scores)`: `new_input_ids` is `torch.long` of shape $(k, L+1)$ and `new_scores` is float of shape $(k,)$, **sorted by descending score**.

## Examples

```
B=3, L=2, V=4, k=2
input_ids   = [[1,2],[1,3],[4,5]]
beam_scores = [-0.1, -0.2, -1.5]
next_log_probs = log([[0.1,0.4,0.3,0.2],
                      [0.5,0.1,0.2,0.2],
                      [0.2,0.3,0.4,0.1]])

candidates S = beam_scores[:,None] + next_log_probs, flattened then top-2:
  best:  parent=1, token=0 → seq [1,3,0], score -0.893147
  2nd:   parent=0, token=1 → seq [1,2,1], score -1.016291

new_input_ids = [[1,3,0],[1,2,1]]
new_scores    = [-0.893147, -1.016291]
```

## Constraints

- CPU float32; `next_log_probs` is already in log-probabilities (so it can be added directly).
- `torch.topk` returns entries in ascending flattened-index order when scores are equal.
- Return the tuple `(new_input_ids, new_scores)`.
