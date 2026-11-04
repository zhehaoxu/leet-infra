# Solution notes — Moonshine context biaser

## Official account

- moonshine-ai/moonshine `core/context-biaser.cpp` + `.h` +
  `context-biaser-test.cpp` (the official doctest pins every behavior);
  `scripts/eval-keyterm-biasing.py` is evaluation orchestration, not part of
  the problem. The repo moved from usefulsensors to moonshine-ai.
- Judging domain: trie registration (shared-prefix reuse / duplicate counting /
  empty sequences), the variant rule, the logarithmic ramp, apply proposals
  with dedup-max, the advance walk with an always-active root, the decode-loop
  simulation. Tokenization (bin-tokenizer) and the streaming speculative
  verification ledger are out of scope (separate problem).

## Pitfalls

- **The official test values are the assertion truth**: depth-1 = 2.0,
  depth-2 = 2(1+ln2), depth-3 = 2(1+ln3), shared prefix boosted once = 3.0,
  two-paths-max = 3(1+ln2) — copy the doctest numbers directly; no RNG anchors
  needed (this problem uses no torch).
- **The apply guard is a true equivalence**: removing the
  `sequence_count == 0` no-op branch changes nothing (an empty trie has no
  candidates to propose, apply still returns {}) — the mutation is
  unjudgeable; recorded in the true-equivalence removals.
- **bias_walk ordering**: bonuses land BEFORE the token is chosen; advance
  consumes the chosen token. Reversing (advance first) loses the entry bonus on
  the first token — the m-walk-order mutant is caught via CAUGHT-ERR.
- **Active nodes still fire on the mismatching step itself**: for
  `[10, 42, 10]`, at the 42 step node 1 is still alive (the depth-2 bonus for
  11 is still proposed); the drop to the root happens after 42 is consumed —
  an expected value of `{10: 2.0}` there would miss `{11: 2(1+ln2)}`; step
  through the official semantics when writing expectations.
- **advance has no dedup and needs none**: root is pushed first, and children
  node indices are globally unique — `next_active` cannot repeat; adding a
  dedup is harmless but not the official behavior.
- **Duplicate sequence registration**: `sequence_count` still increments (the
  count semantics are "registrations", not "distinct sequences"), but no nodes
  are added — assert both branches.
- The `sequence_count == 0` early return in advance is likewise defensive
  (active is already [0]) — same observational-equivalence zone as the apply
  guard; one representative mutant per zone is enough.

## Mutant list (15)

bonus: (1+·) smoothing dropped; apply: parent depth instead of +1, dedup
max→sum (stacking), max→min, range check removed; variants: anchor check
removed, trim removed, empty-string guard removed; build: child depth not
incremented, shared prefixes not reused (forced new nodes), empty sequences
counted; advance: root not pushed, matched child replaced by the node itself
(stalled walk); bonus_for_token: max→min; bias_walk: apply/advance order
swapped (CAUGHT-ERR). 15/15 CAUGHT + 1 true-equivalence removal (the apply
empty guard).
