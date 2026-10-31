# Solution Notes · Self-Consistency Majority-Vote Ledger

## Design stance: rules literal, no semantic guessing

The original self-consistency paper uses EqQA-style answer extraction plus exact-match voting; this
implementation condenses the "formatting noise seen in real pipelines" into 8 **independently
decidable rules**, all executed literally:

- Rule 3 (thousands separators) **only inspects local context** (digits on both sides) and never
  validates that the group length is 3 — so "1,2" also becomes "12". This is deliberate: judging
  demands rule predictability, not intelligence.
- Rule 5 (fraction unification) applies only to **pure-integer operands**:
  `(?<!\d)(\d+)\s*(?:/|÷)\s*(?<!\d)(\d+)(?!\d)`. Boundaries like "A/2" and "10 %" exist so the oracle
  and the solution can be compared rule by rule.
- Rule 8 explicitly does **no mathematical equivalence** ("1/2" ≠ "0.5"): introducing numeric parsing
  would drag in float comparison, precision, and infinitely many spellings, all fatal to
  determinism. Real systems hand this step to a stronger answer-extraction model instead of regex.

## Three deterministic details of tallying

1. **Key order = first-occurrence order**: `tally` grows as a plain dict in insertion order
   (order-preserving since Python 3.7); tests assert `list(tally.keys())` directly, so any
   implementation that re-sorts keys alphabetically or by value gets caught.
2. **Ties go to the first occurrence**: scanning `tally.items()`, the winner is overwritten only on
   **strictly greater** weight. A `max(tally, key=...)` implementation happens to agree because `max`
   keeps the first maximal element, but adding any secondary key (e.g. alphabetical) would deviate.
3. **The `vote_share` denominator is the total weight** (`sum(weights)`), not the sample count.
   All-zero weights are valid input: share becomes 0.0 while `winner` still follows the
   first-occurrence rule — division by zero needs an explicit branch.

## Oracle and mutation testing

- `_normalize_oracle` in `test_cases.py` is an **independently hand-written implementation**
  (per-character scanning for thousands separators, span pre-scanning and rebuilding for fractions)
  that cross-validates the solution's regex version, avoiding "self-confirmation by a single regex".
  Note the judge injects `solve` as `check_*(solve)`; a top-level `from solution import _normalize`
  raises `ModuleNotFoundError` in the eval environment, so the oracle must be self-contained.
- Key boundary cases: `"1,2"` → `"12"` (literal rule 3); `"A/2"` and `"a ÷ 2"` stay unmerged
  (letter operands are exempt from the fraction rule); `"10 %"` → `"10 "` (% stripping ignores the
  preceding space; the leftover trailing space is the faithful result of the rule).
- All 9 effective mutations caught (drop strip / drop lowercase / remove all commas / keep trailing
  period / drop fraction rule / keep % / keep $ / ties to last occurrence / share denominator changed
  to sample count); `dict.fromkeys([])` is an ineffective mutation (no observable difference).
