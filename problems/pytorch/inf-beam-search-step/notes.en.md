# Solution Notes · Beam Search Single Step

## The flatten + topk trick

Flatten the $(B,V)$ candidate scores in row-major order into $B\cdot V$, and a single `topk` accomplishes both "comparing across beams" and "selecting tokens". Recover with integer division / modulo: `parent = f // V` recovers the parent beam and `token = f % V` recovers the new token. This is exactly the core line of `_beam_search` in HF's `BeamSearchScorer`.

## Why backtracking is needed

If each beam independently took its own best next token (per-beam greedy), you would miss cases where "a beam's second-best option is actually globally better". Global top-k allows one parent beam to contribute 0 successors and another to contribute 2, dynamically shifting compute toward more promising paths — that is the point of "backtracking / reordering".

## Relation to finished beams

In real systems, sequences that hit EOS are moved to a finished pool and compete after a **length penalty** $\frac{\text{score}}{((L+5)^\alpha / 5^\alpha)}$. This problem only examines the most error-prone skeleton — "one-step expansion and reordering of live beams" — leaving finished/penalty as a conceptual extension to avoid too many grading branches.

## Tie handling

`torch.topk` does not guarantee index order for exactly equal scores; the fuzzer here uses continuous log-probabilities, so exact float32 equality is almost impossible. The oracle explicitly sorts by `(-score, flat_index)` to define "ties take the smaller index", consistent with the constraint in the problem statement.
