# Explained: HNSW Greedy Layer Search

## Source
Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (arXiv:1603.09320 / TPAMI 2018). The paper's `GREEDYSEARCH` (the precursor of Algorithm 4, i.e. `SEARCHLAYER` with \(ef=1\)) is exactly "follow edges to closer neighbors until a local optimum". During insertion, greedy search with \(ef=1\) descends layer by layer to find the entry point, then a queue with \(ef\) refines the search at the bottom layer — this problem only implements the greedy subprocedure.

## Key points and pitfalls
1. **Strict descent, one hop per step**: move only when "the closest neighbor" is closer than the current point; compare with `d < cur_dist` strictly. Distances decrease monotonically → no cycles, guaranteed termination.
2. **Only look at the current point's neighbors**: this is pure greedy (\(ef=1\)), **not** `SEARCHLAYER(ef)` with a candidate queue — the latter keeps \(ef\) frontier points and can escape more local optima. Do not conflate the two.
3. **Greedy search can get stuck in a local optimum**: when the graph is disconnected or the neighbor directions are wrong, the returned node need not be the global nearest — this is the "verifiable trap" of this problem, and the tests include a case specifically designed to get stuck in a local optimum.
4. **The neighbor table uses `-1` sentinels**: skip invalid slots; for equidistant neighbors take the smallest index.
5. Squared Euclidean distance is enough, and `cd` must be recomputed from the new `cur` every round (caching the old value means "closer" is never satisfied and the search oscillates forever).

## Comparison reference
The reference solution is a while loop: each round `cd=dist2(cur)`, scan the valid neighbors in `adj[cur]` for the smallest `dist2`, and if it is `< cd` set `cur=argmin neighbor`, otherwise return `cur`. The judging `_oracle` accumulates per-dimension squared differences in pure Python and picks the first neighbor candidate by `sort key=(dist,index)` (different structure, numerically equivalent). There are also assertions for "a 1-D chain reaches the global nearest", "an entry that is already a local optimum returns immediately", and "a graph where greedy search gets stuck at a suboptimal point (≠ global nearest)".

## Complexity
Worst case visits \(O(n)\) nodes, scans \(O(\text{deg})\) neighbors per node, and spends \(O(D)\) per distance: \(O(n\cdot \text{deg}\cdot D)\); in practice on NSW graphs it is close to \(O(\log n)\).
