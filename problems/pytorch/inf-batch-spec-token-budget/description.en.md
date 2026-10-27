# Token Budget Allocation Across a Batch of Speculative Requests

## Problem

Every step of speculative decoding runs one target forward. How many "columns" fit into that forward is set by the step's token budget — draft positions cost slots and verify positions cost slots. With several requests in flight the budget must be split: giving one request more drafts raises its expected emission, but the marginal return **decays exponentially** (the k-th draft contributes α^k), so handing that slot to another high-acceptance request may pay better.

This is the classic separable-concave-return, unit-cost allocation problem: greedily taking the largest marginal return is optimal. Return the allocation and the expected token counts.

## Task

Implement `solve(alphas, budget)`:

- Inputs: `alphas` `(N,)` float32, per-position acceptance rate for each request (identically distributed over positions); `budget` int, total slots on the target side this step.
- Request i with `γ_i` drafts occupies `γ_i + 1` slots (draft positions plus its own verify column) and expects `e_i(γ) = 1 + α_i + ... + α_i^γ` tokens.
- Maximise `Σ e_i(γ_i)` subject to `Σ(γ_i + 1) <= budget`:
  - give every request the base 1 slot (`γ_i = 0`);
  - then allocate the remaining slots one at a time, each time to the request with the largest marginal gain `α_i^(γ_i + 1)`, ties to the smallest index.
- Outputs: `gammas` `(N,)` int64, `expected` `(N,)` float32, `total_expected` float32 scalar.
- Raise `ValueError` when `budget < N`, `alphas` is empty, out of range, or the wrong rank.

## Example

```
alphas = [0.8, 0.5, 0.2], budget = 5, N = 3

base: 1 slot each -> gamma = [0, 0, 0], 2 slots left
1st spare slot: marginal = [0.8, 0.5, 0.2]      -> r0, gamma = [1, 0, 0]
2nd spare slot: marginal = [0.64, 0.5, 0.2]     -> r0, gamma = [2, 0, 0]

expected = [1 + 0.8 + 0.64, 1, 1] = [2.44, 1.0, 1.0]
total_expected = 4.44
```

## Constraints

- **Never over-allocate**: the final `Σ(γ_i + 1)` must be `<= budget`; a leftover slot is better than exceeding it (one more draft on a tiny α barely helps).
- **The base slot is mandatory**: every request needs its own verify column, so `budget < N` is invalid input rather than "some requests sit out".
- **Ties go to the smaller index**: with all α equal, the first spare slot goes to r0 and the second to r1, because r0's marginal has already decayed to α².
- **α = 1 makes every marginal exactly 1**: the tie rule then piles all slots onto the smallest index — a legal degenerate case.
- **Compute in float64**: the marginal is a long product-and-power chain, and float32 comparisons start calling different powers equal once α is small.
