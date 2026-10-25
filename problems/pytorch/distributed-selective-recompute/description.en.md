# Selective Activation Recomputation: Minimum Recomputation Cost Under a Memory Budget

## Problem

Full gradient checkpointing discards the activations of **every layer** and recomputes the whole layer during backward, saving memory at the cost of expensive recomputation. Korthikanti et al. (2022, *Reducing Activation Recomputation in Large Transformer Models*) proposed **selective activation recomputation**: not all activations cost the same to recompute — element-wise operators such as LayerSoftmax and Dropout occupy little memory and are cheap to recompute, whereas large matrix multiplications such as $SQK$ and $S^2V$ in attention occupy much memory and are expensive to recompute. The strategy is to **keep (store) only the few activations that are the most memory-hungry / most expensive to recompute and recompute everything else**, minimizing total recomputation time under the same memory budget.

Abstract this into a pure integer optimization: given each operator block's activation bytes and recomputation cost, decide which ones to keep within the **total activation memory budget** so that the **total cost of recomputed operators is minimized**.

## Model (follow this definition exactly)

There are $N$ operator blocks, numbered $0,\dots,N-1$. For block $i$:

- $\texttt{mem}[i]$: the memory bytes needed if its activation at this step is **kept** (positive integer);
- $\texttt{cost}[i]$: the time cost of **recomputing** it during backward if it is not kept (positive integer).

Choose a kept set $K\subseteq\{0,\dots,N-1\}$ that simultaneously satisfies:

1. **Memory budget**: $\sum_{i\in K}\texttt{mem}[i]\le \texttt{budget}$;
2. **Minimum keep count**: $|K|\ge \texttt{min\_keep}$ (at least this many operator blocks' activations must be kept).

The objective is to minimize the **total recomputation cost**:

$$ \min_{K}\ \sum_{i\notin K}\texttt{cost}[i] \;=\; \Big(\sum_i \texttt{cost}[i]\Big) - \max_{K}\sum_{i\in K}\texttt{cost}[i] $$

If no $K$ satisfies both constraints (e.g. summing the `min_keep` smallest $\texttt{mem}$ already exceeds `budget`), return **infeasible**.

## Rules

Implement `solve(mem, cost, budget, min_keep)`:

- `mem`, `cost`: equal-length sequences of positive integers (lists are fine), the $i$-th item corresponding to operator block $i$;
- `budget`: non-negative integer, the total activation memory budget;
- `min_keep`: non-negative integer, the lower bound on the number of kept operator blocks.
- Return the **minimum total recomputation cost** (int).
- If the problem is infeasible (no kept set satisfies the constraints), raise `ValueError`.

## Examples

```
mem  = [30, 10, 20, 40, 15]
cost = [50, 20, 30, 60, 10]
budget = 60, min_keep = 2
  keep {0,1,2}: mem=60≤60, |K|=3≥2, kept cost=50+20+30=100 → recompute=170-100=70
  keep {1,3}  : mem=50, |K|=2, kept cost=20+60=80 → recompute=90 (worse)
  optimal recomputation cost = 70
→ 70
```

## Constraints

- `1 ≤ N ≤ 50`; `1 ≤ mem[i], cost[i] ≤ 2000`; `0 ≤ budget ≤ 2000`; `0 ≤ min_keep ≤ N`
- Return a single int; raise `ValueError` when infeasible
- This is a pure combinatorial optimization model; no real computation graph / operator profiler is involved
