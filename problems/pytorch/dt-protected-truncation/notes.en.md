# Notes: Allocate Quotas, Then Select Positions in Order

## Reference approach

Validate options and budget first, including whether all protected positions fit. Collect ordinary positions for each sequence. For a single-side strategy, subtract the overflow at once. For `longest_first`, decrement the larger remaining count repeatedly, choosing the second side on ties.

Only after obtaining the final quotas do we decide which positions survive. Head and tail modes select prefixes and suffixes. Middle removal assigns the extra token of an odd quota to the prefix. The reference uses `indices[len(indices)-q:]` for suffixes, making q=0 empty instead of accidentally retaining everything with Python's `indices[-0:]`.

Combine all protected positions and both retained sets, sort them, and use the same indices for all three fields. Position-based selection avoids confusing repeated ordinary IDs with protected tokens.

## Correctness

Each decrement represents one legal ordinary-token deletion, and their total is exactly the overflow. Too many protected positions or insufficient capacity on a required side are the only capacity failures. When either side may be reduced, enough removable tokens exist whenever the protected positions fit.

Each side's prefix and suffix are disjoint because its quota never exceeds its length. Protected positions and ordinary positions of the two sequences form three disjoint groups. Sorting by original index preserves sequence order, relative special-token placement, and alignment of all fields.

## Independent oracle

The reference repeatedly decrements counts. Tests instead enumerate all feasible retained-quota pairs and minimize their absolute difference, favoring a larger first-side quota on ties. This constrained balancing formulation reproduces the rule that an equal-length deletion comes from the second side. Single-side strategies keep the other quota fixed.

Position selection also uses a different formulation: rank positions by distance to their nearest endpoint, prefer the front endpoint on ties, select the required count, then restore original order. Tests compare every returned field and check input immutability.

## Recorded validation

`--selftest dt-protected-truncation --mode submit` returned **15/15 ACCEPTED**. Coverage includes equal-count ties, both single-side strategies, zero quotas, nonbinding budgets, numeric-ID collisions, interleaved ownership, odd middle quotas, empty/all-special inputs, one-sided and unbalanced inputs, every small count/budget combination, maximum length, seeded random cases, impossible budgets, and unknown options.

The statement's example was executed and checked against the independent oracle before its output was documented.

## Complexity and scope

For input length L, decrementing counts takes at most L steps. The reference sorts retained positions, giving `O(L log L)` time and `O(L)` extra space. A sequential scan of retention flags could remove the sorting step without changing the contract.

This is a deterministic educational truncation convention, not a bitwise clone of an external tokenizer. Special positions are explicit; real tokenizers commonly allocate a content budget after subtracting template overhead, then insert the template. Middle retention and interleaved ownership are exercise extensions.

The inspected [tokenizers 0.23.3 source archive](https://files.pythonhosted.org/packages/e0/7c/2cabb2174e772636683008f2c5621949b645da7d303c596589e84516a184/tokenizers-0.23.3.tar.gz), `tokenizers/src/utils/truncation.rs` lines 70–158, and an actual native API probe demonstrate a difference: ordinary lengths (2,2) with content budget 3 retain (1,2) natively, versus (2,1) here. Native allocation classifies the shorter and longer sides by their original lengths, whereas this exercise compares remaining counts each step. Native behavior also differs when exactly emptying a selected side; this exercise permits that according to its own capacity rules. Do not describe the two algorithms as identical.
