# Solution Notes

## Approach

The official amazon-science/auto-cot run_demo.py (L41-154), taking the deterministic sampling=center path:

1. **Input side**: the official code parses the candidate pool from a zero-shot-CoT log file (the Q:/A:/pred_mode/GT : line protocol plus the Therefore-triggered rationale state machine) — the parsing layer is outside the judging scope and the pool arrives as inputs; SentenceTransformer encoding, KMeans fitting and the PCA plot are likewise excluded (`retrieval-kmeans-lloyd` covers the k-means operator), with `labels_` and the `transform` distance matrix given explicitly;
2. **Judging scope**: cluster → stable distance-ascending intra-cluster sort → per-candidate five filters → one-per-cluster break → rationale cleaning and assembly.

Boundary with existing problems:
- **rs-self-consistency**: folds many candidates into one answer (voting); this one picks representatives to compose a prompt (no aggregation);
- **retrieval-kmeans-lloyd**: the generic k-means operator; this problem consumes its output (assignment + dist);
- The follow-up cot line (② active-prompt ③ self-verify ④ RoT ⑤ cumulative) reuses this cluster-select-assemble skeleton.

## Key implementation points

- **Stable sort is the tie-break**: cluster appends in ascending sid order plus a stable sort by dist, so ties automatically keep the sid order — the explicit `(dist, sid)` double key equals the official single-key stable sort (mutation m-tie-key was removed for this reason).
- **Redundancy in the cleaning chain**: `replace("\n\n","\n").replace("\n"," ")` followed by `" ".join(split())` — `split()` splits on any whitespace, so the replace chain does not change the token sequence; the end result equals `" ".join(r.split())`. The official spelling is kept (fidelity), and mutation m-clean-replace was removed.
- **The safety chain of [-2]**: condition c (endswith ".") guarantees `r.split(".")` has at least two segments, so [-2] cannot overflow — the filter order is part of the semantics, and mutations reordering b/c against e are catchable.
- **[-2] vs the last 10 words**: [-2] is always the last complete sentence (the trailing period guarantees the final segment is empty); the last 10 words ⊆ the last sentence (when it has ≥10 words). "[-2] hits while the tail does not" requires a ≥10-word final sentence with the answer at its head — fixture s0 uses an 18-word sentence with the answer as the first word.
- **The arithmetic check is case-sensitive**: `in` is a literal substring match, so "twelve" does not hit "Twelve" — fixtures and mutation cases must both watch for this (hit twice this round).
- **Dead columns**: the official code never reads dist columns of clusters other than the assigned one, but reading the wrong column (m-dist-col) changes the intra-cluster ordering — the fixture gives cluster 1 two candidates s6/s9 with a deliberately inverted [2] column (s6=0.1 < s9=9.9) so the wrong-column mutation diverges.

## Test design (9 cases)

- `main_oracle`: 10 samples, k=5 (including an empty cluster), full-field comparison on two max_ra_len paths; s2 is closest to the centroid (0.5) but has a five-line rationale — at max_ra_len=2 it yields to s0, at 5 it takes the whole cluster 0 slot (selection order tracks the cap);
- `gold_ans_passthrough`: whitespace-padded gold answers stripped through + the None path;
- `word_count_filter`: 55 words pass, 61 words blocked;
- `ra_len_boundary`: exactly at the cap passes, one line over fails; \n\n counts folded;
- `arithmetic_verify`: four clusters covering four paths ([-2] hit / last-10-words hit / answer absent from text / arithmetic=False lets all through);
- `tie_and_empty`: distance ties pick the smaller index + empty clusters contribute nothing;
- `folding_cleanup`: exact cleaning-chain assertion + line counting by \n;
- `purity`: inputs are not mutated in place;
- `args_validation`: types/lengths/range/bools/row widths/numeric values/cap/messages, including tuple rows and non-numeric elements.

## Mutation coverage

33 designed → **32 CAUGHT + 1 equivalence removal** (m-val-k0: mutually backstopped by the questions non-empty check, unobservable alone):
- wrong dist column / sid-as-assignment / reversed sort; each of the five filters deleted or changed + e's [-1] / tail-3-words + e flipped; break deleted / line filter turned break (cluster-level give-up); default-parameter drift; cleaning unfolded / gold unstripped / gold always None / underreported cluster count / selected vs cluster_order swapped; ten validation mutations (item type / length / bool intrusion / range / row width / numeric value / cap allows 0 / bool cap / arithmetic type / gold length) + two message mutations.

## Engineering remarks

- **The fatal criterion in mutation scripts**: a crashing mutation makes the judge return fatal (passed=0/total=0), and the `passed < total` test misreads it as SURVIVED — use `not accepted or fatal` (a generalization of the 771 lesson).
- **Fixture evolution needs full re-runs**: s2 from three lines to five, s0 rewritten as a long sentence, s9 added to cluster 1 — every change re-verifies main_oracle's two max_ra_len paths, gold and purity in cascade (main_oracle failed five rounds this cycle, all fixture/assertion desync).
- **The per-cluster semantics of break**: the max_ra_len=5 selected assertion was first miswritten as [2,0,9,7,8] (collecting s0 after s2 passed) — after the break, cluster 0 holds only s2: [2,9,7,8].
