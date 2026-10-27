# Solution Notes · Per-Layer Knapsack

## Why DP is mandatory

Proportional slicing (split the budget by layer mass share) is equivalent to assuming "the \(x\)-th position of a layer always has the same marginal value", so the allocation only depends on total layer weights. Real curves do not look like that: the first few positions of a layer may carry 80% of the value and the rest almost none. Splitting by mass then wastes slots on layers whose marginal has already hit zero while starving layers whose marginal is still high.

Once the gain table is non-concave (plateaus, jumps), greedy is unreliable: suppose layer A's first slot is worth 10 and its second 0 while layer B gives 6 per slot with two slots—greedy takes A first for a total of 10, but the optimum is 12. Enumeration is needed, and DP is the natural form.

## Why states are indexed by exact cost

This is the easiest place to go wrong. With "cost at most \(b\)" states, `dp[b]` keeps a single best entry, so two prefixes with **equal value but different cost** overwrite each other. Keeping the expensive one throws away the branch where the saved quota goes to later layers, and the global optimum can be missed.

With exact-cost states, each \(b\) only competes with candidates of the same cost, so different-cost paths survive independently and the final answer takes the best over all \(b\le B\). It also makes the lexicographic tie-break clean: among equal-cost, equal-value prefixes the lexicographically smaller one is always preferable, since later coordinates cannot affect an already-fixed prefix.

## Why a tie-break is necessary

Knapsack optima are usually reached by many allocation vectors at once (especially with plateaus in the gain table). Without a tie-break, different implementations would return different `alloc` arrays and the problem could not be judged. Lexicographically smallest is the cheapest deterministic rule—it amounts to "push quota towards later layers"—and implementation is just tuple comparison.

## Complexity

\(O(L_c\cdot B\cdot M)\) time and \(O(B)\) space with a rolling array. In a real system \(L_c\approx 32\), \(B\) is a few thousand and \(M\) is the per-layer cap, so the total is in the millions of operations and fits comfortably in an offline calibration pass.

## Division of labour with CAKE

The CAKE problem tests proportional slicing plus cross-layer propagation, i.e. an **approximation** of the allocation rule; this problem tests the **exact discrete optimum**. Both take per-layer importance/gain as input and neither involves the scorer. kvpress's `per_layer_composed` wrapper and the layer budgets of PyramidKV / SimLayerKV belong to this family, but their production implementations use heuristics—this problem exposes what the optimum actually looks like as a baseline.

## Related problems

- `kv-cake-bidirectional-budget`: proportional slicing plus largest remainder and cross-layer propagation (approximate).
- `kv-pyramid-layer-budget`: a fixed pyramid formula (an even coarser approximation).
- `kv-adakv-headwise`: the same "budget splitting" idea, but over heads instead of layers.
