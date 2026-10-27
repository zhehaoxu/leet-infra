# KV Cache: Knapsack Allocation of Per-Layer Retention Quotas

## Problem

The "proportional slicing" approach to cross-layer budgets (the CAKE family) only uses each layer's total mass, which implicitly assumes **linear marginal return**: the \(x\)-th position of a layer is worth as much as the \((x+1)\)-th. Real curves are **concave**: the first few tokens of a layer carry most of the value and later positions are close to worthless. Allocating across layers then becomes a standard **discrete resource allocation problem**:

given the gain \(g_l(x)\) of keeping \(x\) positions in layer \(l\) (\(x=0,\dots,M-1\), supplied as a lookup table and **not guaranteed concave**) and a total quota \(B\), solve

\[
\max_{\{x_l\}}\ \sum_{l=0}^{L_c-1} g_l(x_l)
\quad\text{s.t.}\quad \sum_l x_l\le B,\ \ 0\le x_l\le M-1
\]

This is a **multiple-choice knapsack**: each layer is a group of mutually exclusive choices, capacity is the total quota, and every weight is 1. Greedy (repeatedly give one more slot to the layer with the largest current marginal gain) is optimal only when every \(g_l\) is concave; as soon as one layer has a plateau (\(g_l(x+1)=g_l(x)\)) or a non-concave jump, greedy gets stuck in a local optimum and DP is required.

**Tie-breaking**: among equal total gains, take the **lexicographically smallest** allocation vector (compare \(x_0\), then \(x_1\), ...). This rule makes the answer unique—otherwise a large family of equal-value solutions would make judging impossible.

## Task

Implement `solve(gain, budget)`:

- Input: `gain` (float32 `(Lc, M)` where `gain[l][x]` is the gain of keeping `x` positions in layer `l`, \(x\in[0,M-1]\)); `budget` (non-negative integer \(B\)).
- Output `(alloc, total)`: `alloc` is `(Lc,)` int64; `total` is the summed gain (float32).
- The constraint is \(\sum_l x_l\le B\) (**not required to be tight**: positions with zero gain are not worth taking); each \(x_l\) is capped at `M-1`.
- Raise `ValueError` for `budget < 0`, an empty `gain`, or negative entries in `gain`.

## Example

```
gain = [[0, 5, 8, 9],      # layer 0: marginals 5, 3, 1
        [0, 4, 7, 9]]      # layer 1: marginals 4, 3, 2
budget = 4

(1,3)=14  (2,2)=15  (3,1)=13  (2,1)=12  (1,2)=12  (0,3)=9 ...
optimum (2,2) -> alloc = [2, 2], total = 8 + 7 = 15
```

## Constraints

- The quota is an **upper bound**, not an exact amount: if every next slot has zero marginal gain, using fewer slots (`sum(alloc) < B`) is correct.
- DP states must be indexed by the **exact cost** \(b\). With "cost at most \(b\)" states, an equal-value but cheaper prefix is overwritten by an equal-value but more expensive one, removing a feasible branch for later layers and possibly missing the global optimum.
- The final tie-break runs over all states with cost \(\le B\): higher gain wins, and equal gain prefers the lexicographically smaller allocation.
- The gain table is not guaranteed concave, so the "always take the largest marginal" greedy is not a valid substitute for DP.
