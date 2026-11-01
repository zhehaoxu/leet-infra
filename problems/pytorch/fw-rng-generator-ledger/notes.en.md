# notes — fw-rng-generator-ledger (order 713)

## Sourcing

- Authoritative source: the local judge environment `.venv-eval/lib/python3.12/site-packages/torch` (v2.14.1) plus `_C.Generator` empirics. No GitHub fetch — RNG semantics drift between versions; empirics win.
- `torch.manual_seed(s)` return value: v2.14 returns a `torch._C.Generator` that **drives the default stream** (`rand(generator=ret)` matches bare `rand()`). The old "returns None" documentation is outdated.
- **Seed wrapping (most-misremembered account)**: `initial_seed` returns `s mod 2**64` as an **unsigned** Python int. `-5 → 18446744073709551611`, `2**63 → 9223372036854775808` (not `-2**63`!). Valid domain `[-2**63, 2**64-1]`; outside → `ValueError: Overflow when unpacking long long`.
- **Fresh Generator default seed = 67280421310721** (the MT19937 golden constant), and two fresh objects produce element-identical sequences — "Generator() is seeded randomly" is a myth.
- `manual_seed` type rejections: `True/1.5/"7"` all raise `RuntimeError` (bool is an int subclass and must be rejected before the int branch — same precedence trap as 710's calculate_gain bool param).
- `get_state()` returns `uint8[5056]` (full MT19937 state + position pointer); mid-stream restore is value-exact.

## Design

- The seven-key generator bundle is built by the judge; `"default"` references `torch.default_generator` itself — the fork account runs on the real default stream.
- `_compare` must **snapshot/restore the default state** between solve and oracle: after the fork account's restore, the post draw advances the default stream by 2 draws, so the two calls would otherwise see desynced default states (the first version died exactly there).
- The oracle calls official `Generator.manual_seed/initial_seed` directly as the hardest anchor; the official `torch.random.fork_rng` semantics cross-check lives in the check functions (the oracle must not call `torch.manual_seed`, which would pollute the default stream).
- 60 randomized cross-checks: random valid seeds plus random mid-stream drain counts (`n_mid ∈ [0,9]`), with the mid generator rebuilt independently so solve/oracle stay isomorphic.

## Mutations (13 caught + 2 equivalent)

- Catching hinges on: unsigned wrapping vs int64 reinterpretation (the `2**63` case), bool rejected first, both domain boundaries (**the exactly-valid `-2**63` case must not be omitted** — the first draft missed it and domain-lower survived), the fresh constant, fork/mid restoration, dtype streams, and consumption primitives/counts.
- 2 equivalent mutations (SURVIVED, recorded here): (1) a redundant `set_state(snap)` before the midstream first draw (restoring to the current state, sequence unchanged); (2) domain upper bound `s <= 2**64-1` ↔ `s < 2**64` (identical over integers).

## Pitfalls

- **The red-line check must be AST-node based**: the first version string-matched `torch.manual_seed`, and docstring mentions hit it → every mutation was falsely judged CAUGHT under the mutation harness (a fake 13/13). After switching to AST and re-running, the honest verdict exposed the real domain-lower survivor.
- `str(seed)` as dict key: `str("7") == "7"` collides with valid int seed 7 (the seed list here never holds both; use repr if extended).
- Boundary with 711: 711 accounts for the DataLoader's per-epoch `_base_seed` draw and permutation drain; this problem accounts for the generator object's own wrapping/snapshot/fork semantics. `randperm(generator=...)` appears in both but at different ledger layers.
