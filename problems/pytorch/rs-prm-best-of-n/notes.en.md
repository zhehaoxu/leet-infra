# Solution Notes · PRM best-of-N Aggregation Ledger

## Why the four rules exist

The appendix of 2305.20050 compares several ways of collapsing step scores into a scalar and
concludes that **product (multiplication) fits best-of-N best**: a 10-step solution with even one
low-truth-probability step (say, a silent arithmetic slip) exposes that flaw exponentially under
multiplication, while `max` is fooled by a single highlight and `last` ignores the whole process.
The test case `[[0.99]*9 + [0.01], [0.9]*2]` is the minimal reproduction: under product the short
chain 0.81 beats the long chain 0.0092, under max the long chain wins with 0.99.

The `answer_scores` factor corresponds to combining "PRM score × final-answer (ORM) verifier
score": good process and a correct final answer are both necessary, hence multiplication rather
than addition.

## Implementation notes

- **Validation order**: the rule whitelist comes first (an unknown rule should fail before any
  aggregation); `top_m` must exclude both `top_m < 1` and `isinstance(top_m, bool)` (`True` is a
  valid int but semantically absurd); empty `step_scores` and an empty candidate are separate
  checks with distinct messages.
- **Sort key `(-agg[i], i)`**: Python's `sorted` is stable, but putting the index explicitly in the
  key makes the "ties to the smaller index" intent self-documenting; `tie` uses strict counting via
  `agg.count(max_agg) > 1` — float equality is legitimate here because the test cases use exact
  decimals.
- **The `min_len` ledger field** serves best-of-N's "shortest candidate" statistic (e.g. secondary
  rankings preferring fewer steps); a single `min(len(s) ...)` suffices.

## Oracle and mutation testing

- `_oracle` in `test_cases.py` is an independent implementation of the ledger; 40 random cases
  cover all four rules × with/without answer_scores × the full `top_m` range.
- Hand-computed anchors for the anchor case: under product `[0.36, 0.63, 0.001]` → `best=1`,
  `top=[1,0,2]`. Note that under max `best=2` (the maximum of 0.9, 0.9, 1.0 sits in candidate 2),
  and under last `best=1` (0.8, 0.9, 0.1) — an anchor where all four rules disagree is a good one.
- Mutation testing: last→first, min→max, product→sum, drop the answer multiplier, tie always true,
  order by index, min_len→max — 8/8 caught. `no-rule-check` (removing the rule whitelist in solve)
  **survives**: the trailing `raise ValueError(rule)` in `_agg_one` raises the same exception type,
  and `check_invalid_inputs` only asserts the ValueError type — the whitelist check is redundant
  defense (it fires earlier, before the top_m checks, with a clearer message) and is behaviorally
  equivalent; recorded as an equivalent mutation.
