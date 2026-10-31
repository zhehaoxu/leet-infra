# notes — ag-generative-memory-retrieval (order 666)

## Source

- Origin: `joonspk-research/generative_agents`, `cognitive_modules/retrieve.py` (`new_retrieve`; paper 2304.03442 "Generative Agents: Interactive Simulacra of Human Behavior").
- Faithful transcription of the official quirk: `extract_recency` uses `decay ** i for i in 1..n` over time-ascending nodes, so the **oldest** memory gets the largest recency value — counterintuitive versus "newer is closer", but that is upstream verbatim; the spec pins it without correction.
- `normalize_dict_floats(d, 0, 1)`: min-max to [0, 1]; when `range == 0` every value becomes the midpoint 0.5 (a single-node memory therefore scores 0.5, not 0/1).
- Weights `gw = [0.5, 3, 2]` (the default combination when recency_w/relevance_w/importance_w are all 1).

## Judging

- solve args: query_emb, memories (emb + importance), k, optional recency_decay/gw. Returns scores/top/components.
- Oracle: `test_cases._py` is an independent rewrite (manual normalization and sorting), 150 random cross-checks (n 1–12, decay ∈ {0.995, 0.99, 1.0}, k may exceed n for truncation).
- Anchors: hand-computed components and totals (cosine exactly 1/2 via emb=[1,√3], q=[1,0]; 1+3=4 → √4=2 keeps the whole chain float-exact); equal importance → range=0 → all 0.5; single node → score=(0.5+3+2)*0.5=2.75; all ties + gw[0]=0 → fully tied totals with top=[0,1,2] (pins stable sorting); k-truncation prefix property.
- Invalid inputs: empty query/memories, dim mismatch, non-numerics, NaN/Inf importance, non-positive/bool k, out-of-range decay, gw length ≠ 3 → ValueError.

## Mutation log (9/9 caught + 1 equivalent)

recency-direction-reversed (new/old order flipped), gw-rel-imp-swap, norm-range0-to-zero (0.5 → 0), cos-no-norm (dot instead of cosine), sort-ascending, k-plus-one, importance-no-norm, recency-no-norm, tie-reverse-index (ties pick the larger index; caught via the gw[0]=0 all-tied-totals case).

**Equivalent mutant**: `decay ** (i+1)` → `decay ** i` is unobservable — a constant scale factor is cancelled by min-max normalization. Replaced with the direction-reversal mutant.

## Red lines

No embedding-model calls, no retrieval engineering (vector DBs/ANN); boundary with `int-*` (interpretability read-outs): this is an **agent memory ledger**, no model-internal signals.
