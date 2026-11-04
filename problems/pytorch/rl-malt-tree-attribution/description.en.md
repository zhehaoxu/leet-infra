# MALT: Search-Tree Credit Attribution Ledger (791)

**Source**: arXiv:2412.01928 (COLM 2025 v3) sec 4.2 / A.6.1. **No official code release** — the oracle is the paper formula's unique deterministic arithmetic; the sandbox cross-checks it with an independent NumPy translation plus hand-computed anchors.

## Problem

Credit assignment for the multi-agent Generate→Verify→Refine pipeline should not look only at the final outcome: organize each question's n³ sampled trajectories into a three-level search tree, propagate the sparse leaf rewards back through value iteration (bottom-up sibling expectations), binarize with a strict 0.5 threshold (majority-vote alignment), and build DPO preference pairs within the shared parent context. Model sampling stays out of the problem — leaf rewards and the tree structure are explicit inputs.

## What to solve

`solve(mode=..., ...)` five entries (mode kwarg dispatch, stdlib only):

1. `mode="propagate"` (rewards: n×n×n binary list, n) → (verifier_values n×n, generator_values n)
2. `mode="binarize"` (values[, theta=0.5]) → strict greater-than-θ boolean labels
3. `mode="pairs"` (unit_values, parent_ids[, theta]) → (positive index, negative index) pairs
4. `mode="sft"` (unit_values, parent_ids[, theta]) → {positives, counts}
5. `mode="attribute"` (rewards, n[, theta]) → the full per-question pipeline (values/labels/three pairing layers/SFT tally)

## Key paper semantics (see notes.md)

- **No discount factor**: value iteration = two levels of sample means (V(v)=mean of leaves, V(g)=mean of V(v))
- **Strictly greater than 0.5**: V exactly 0.5 is judged incorrect (reachable only for even n; never for n=3)
- **Same-parent pairing**: D_R within a verifier output, D_V within a generator output, D_G within a question; full positive×negative subset combination inside each parent
- **Generator is SFT-only**: D_G is defined but the paper explicitly skips DPO for it
- The SFT-positive criterion differs per layer: R uses the leaf reward V(r)=1 while V/G use the binarized ✓ label

## Scoring

Judging is ledger comparison: the expectation backprop is cross-checked against an independent NumPy translation plus hand anchors (7/9, 2/9, …); pairings and labels are hand-counted.

## Judging environment

torch + numpy; no randomness (fully deterministic ledger, no model calls).
