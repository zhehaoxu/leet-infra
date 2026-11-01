# RNG Reproducibility Ledger: Seed Wrapping, Explicit Generators and State Snapshots

## Problem

All reproducibility accounting lives on **generator objects**: `torch.manual_seed` seeds the global default generator, `torch.Generator` is an explicit random source, and `get_state/set_state` are snapshot primitives. This problem hand-writes seed classification and wrapping, the fork semantics of state snapshots, and multi-generator consumption routing — without calling `torch.manual_seed` / `torch.initial_seed` / `torch.random` (including `fork_rng`) or `Generator.manual_seed` / `Generator.initial_seed`. Only `torch.rand/randn/randint/randperm` with an explicit `generator=...` and `get_state/set_state` are allowed.

## Solution

Official semantics (torch 2.14.1 CPU, pinned empirically):

- **Seed wrapping**: valid domain `[-2**63, 2**64-1]`, out of range → `ValueError`; `initial_seed` returns `s mod 2**64` as **unsigned** — `-5 → 2**64-5`, `2**63 → 2**63` (not the int64-reinterpreted negative); `bool/float/str → RuntimeError` (bool is an int subclass and must be rejected before the int branch);
- **Fresh Generator**: an unseeded generator's `initial_seed = 67280421310721` (the MT19937 default seed constant), and two fresh generators produce identical sequences;
- **State snapshots**: `get_state` returns `uint8[5056]`; `set_state` restores exactly to the snapshot moment — re-consuming after a mid-stream restore reproduces identical values; fork semantics: snapshot → consume → side consume → restore → consume == direct continuation from the snapshot;
- **Same seed, different Generator objects** → element-identical sequences; `initial_seed` does not advance state;
- **dtype enters the consumption path**: float32/float64 from the same seed diverge;
- **Consumption is order-sensitive**: `rand → randn → randint → randperm` drain the same generator in turn.

The judge provides a seven-key generator bundle (`"a"/"b"` same seed distinct objects, `"mid"` mid-stream, `"f32"/"f64"`, `"chain"`, `"default"` being `torch.default_generator` itself).

## Data Scale

- 13 seed ledger entries (domain boundaries `±2**63`, `2**64-1`, negatives, 0, ordinary; out-of-range `2**64`/`2**65+7`/`-2**63-1`; type rejections `True/False/1.5/"7"`).
- 4 consumption groups: same-seed pair, mid-stream snapshot restore, default fork (snapshot→rand(2)→rand(4)→restore→rand(2)), dtype dual streams plus a four-primitive consumption chain.
- 60 randomized cross-checks with random valid seeds and random mid-stream drain counts.

## Scoring

Every ledger entry must equal the result of calling the official `Generator.manual_seed/initial_seed` directly (including unsigned wrapping and error classification); fork restoration is verified byte-exact on the default generator's state (exactly one `rand(2)` drain after the snapshot restore).

## Constraints

- No `torch.manual_seed` / `torch.seed` / `torch.initial_seed` / `torch.random` (AST call-node level check);
- No `Generator.manual_seed` / `Generator.initial_seed`;
- All explicit generators are seeded by the judge; solve only routes, snapshots, restores, and consumes.
