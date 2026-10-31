# notes — rs-overthinking-metrics (order 663)

## Source

- Mechanism origin: the common conventions of overthinking / token-efficiency evaluation (2503.16419 "Do NOT Think That Much", s1's token-vs-accuracy curves are of this kind; no single authoritative implementation), so the ledger conventions are self-defined and pinned in the problem statement — the judge only recognizes the pinned semantics.
- Boundary with 654 `rs-self-consistency-vote` (answer-normalization voting), 655 `rs-prm-best-of-n` (PRM aggregation), 662 `rs-budget-forcing` (budget state machine): this one is the **evaluation-side metric ledger**, no generation semantics.

## Judging surface

- solve args: correct (bool list), tokens (non-negative int list), bin_edges (strictly increasing ints, ≥2). Returns n/accuracy/total_tokens/bins/len_correct/len_wrong/efficiency_ratio/best_bin.
- Oracle: `test_cases._py`, independent pure Python (recomputes binning per sample), 200 random groups comparing every field.
- Anchors: (1) 4 samples 2 bins hand-computed (accuracy 0.75, counts [1,3], efficiency 20/(135/3), best_bin=1); (2) boundary values t=10 → bin0, t=30 → last bin (left-closed right-open + right-inclusive last); (3) empty-bin None and best_bin skipping; (4) all-correct (len_wrong=None, efficiency=None, best_bin=0); (5) all-wrong (best_bin still the first non-empty bin); (6) accuracy ties to the smallest bin; (7) K=1 single bin.
- Invalid inputs: length mismatch, empty records, bool/negative/float entries, edges <2 / not strictly increasing / non-int → ValueError.

## Pitfalls (stepped on while authoring)

- **K = len(edges) − 1**: edges=[10,30] is ONE bin ([10,∞)) — twice wrote anchors treating "2 edges as 2 bins", exposed by the oracle comparison. edges=[10,30,100] is 2 bins.
- Samples below edges[0] have no dedicated bin; the statement pins them into the last bin.
- `efficiency_ratio` None semantics include len_correct == 0 (avoiding division by zero).

## Mutation log (9, all caught)

bin-left-open, eff-inverted, len-wrong-all, best-bin-last, bin-acc-denom-n, bin-mean-correct-only, total-correct-only, acc-wrong-denom, last-bin-excl.

## Red lines

No model generation, no statistical-significance testing, no plotting; correct/tokens are explicit inputs.
