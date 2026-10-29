# Retrieval: HNSW Graph Insertion and Neighbor Selection

## Problem

HNSW (Hierarchical Navigable Small World, Malkov & Yashunin, arXiv:1603.09320) organizes vectors in a multi-layer navigable graph: upper layers are sparse and jump quickly to the right region, lower layers are dense and refine the nearest neighbors. During construction, points are inserted one by one. Each new point starts at the entry point, descends greedily through the layers above its own level, runs an `ef_construction`-wide candidate search at its level and below, then uses the Algorithm 4 heuristic from the paper to pick neighbors and connects them bidirectionally. The resulting neighbor lists are exactly the graph traversed later by query-side routines such as the one in `retrieval-hnsw-greedy`.

This task covers the **build / insertion side** only, and replaces the usual random level sampling with an **explicit input**: `levels[i]` gives the highest layer of node \(i\), while `M`, `M0=2M` and `ef_construction` are also caller-specified. The whole process is therefore deterministic and can be checked edge by edge. The difficulty is in the layer-descent boundaries, the ordering of candidates and results, the Algorithm 4 pruning test, bidirectional connection followed by degree-overflow replacement, and deterministic handling of duplicate points and distance ties.

CPU, fully deterministic, no RNG calls.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract. What follows only restates the semantics.

- **Inputs**: points of shape \((N,D)\) (float32 or float64, \(N\ge1\), \(D\ge1\)); `levels` is a 1-D int64 array of length \(N\) with `levels[i] >= 0`, meaning node \(i\) exists on layers \(0\) through `levels[i]` (layer 0 is the bottom); `M >= 1` caps the out-degree on layer 1 and above, while layer 0 is capped at \(M_0=2M\); `ef_construction >= 1` is the candidate-search width. All distances are squared Euclidean.
- **Insertion order and entry point**: nodes are inserted in the order \(0,1,\dots,N-1\). Node 0 becomes the entry point with `max_level = levels[0]`; its neighbor lists on every layer start empty.
- **Layer descent**: when inserting node \(q\) with \(l=\text{levels}[q]\), set `ep=entry` and, for `lc` from `max_level` down to \(l+1\), run a width-\(ef=1\) candidate search on layer `lc`, updating `ep` to the point in the result nearest to \(q\) (distance ties to the smaller node id).
- **Layer search and neighbor selection**: for `lc` from \(\min(\text{max\_level},l)\) down to 0: run a width-`ef_construction` candidate search from `ep` to obtain \(W\); apply the Algorithm 4 heuristic — sort candidates by (squared distance to \(q\), node id), consider them in order, drop candidate \(c\) if its squared distance to any already-selected node \(r\in R\) is **strictly smaller** than its squared distance to \(q\), otherwise add it to \(R\); stop once \(M\) are selected. Then add the bidirectional edges \(q\leftrightarrow e\) for every \(e\in R\) (no self-loops, no duplicate edges), and set `ep` to the point in \(W\) nearest to \(q\) for the next lower layer.
- **Candidate-search ordering**: the visited set starts as `{ep}`; the candidate set is ordered by (distance, node id) and the result set \(W\) keeps at most `ef_construction` nodes best by (distance, node id). Each step pops the best candidate \(c\); if \(\text{dist}(c,q)\) is **strictly greater** than the current worst distance in \(W\), stop (equality keeps expanding); otherwise scan \(c\)'s neighbors on that layer in ascending node id, mark unvisited ones, and if \(|W|<\text{ef}\) or \((\text{dist}(e,q),e)\) beats the worst (distance, node id) in \(W\), add \(e\) to both the candidate set and \(W\), popping the worst when \(|W|>\text{ef}\).
- **Degree-overflow replacement**: for every \(e\in R\), if its out-degree exceeds `cap` (\(2M\) on layer 0, \(M\) otherwise), repeatedly remove the neighbor with the **largest** squared distance to \(e\), breaking ties by the larger node id, until the cap is met. The newly added \(q\) competes equally and may be removed; therefore the final graph **may be asymmetric** (\(q\) lists \(e\) while \(e\)'s list no longer contains \(q\)).
- **Entry update**: only when \(l>\text{max\_level}\) does `entry` become \(q\) and `max_level` become \(l\); when \(l\le\text{max\_level}\) the entry point is unchanged. A new node is isolated on layers above the old `max_level` (it is alone there), so those neighbor lists are empty.
- **Output**: a three-level nested list where `out[lc][i]` is the list of neighbors of node \(i\) on layer `lc`. The outer length is exactly \(L=\max(\text{levels})+1\), the middle length is exactly \(N\), and each inner list is ascending, duplicate-free and self-loop-free; every layer-0 list has length \(\le 2M\), and every higher-layer list has length \(\le M\). Node \(i\) does not exist on layers with `lc > levels[i]`, whose lists are empty.
- **Ties and degeneracies**: every ordering, argmin, worst-element choice and overflow removal uses the key (squared distance, node id), with the smaller id winning. Duplicate points and all-zero distances therefore remain deterministic. When `ef_construction < M`, a new node gets at most `ef_construction` neighbors on that layer; it is not padded up to \(M\). `N=1` is legal and returns \(L\) layers with one empty list each.
- **Errors**: `points` not 2-D, a dtype other than float32/float64, `N=0`, `D=0`; `levels` not 1-D int64, length different from \(N\), or containing negative values; `M` or `ef_construction` not a positive integer (bool is rejected) — all raise `ValueError`.

## Example

Five 1-D points with explicit levels, `M=1` (so \(M_0=2\)) and `ef_construction=2`:

```
points = [[0.0], [0.1], [0.05], [0.02], [4.0]]
levels = [1, 0, 0, 0, 2]
M = 1, ef_construction = 2
```

Insertion trace (all distances are squared Euclidean):

| Insert | Level | Key action | Layer-0 adjacency change |
|---|---|---|---|
| 0 | 1 | becomes entry, max_level=1 | empty |
| 1 | 0 | search from 0, select 0 | 0-1 |
| 2 | 0 | distances to 0 and 1 are both 0.0025; tie goes to smaller id 0 | 0-2; 0's degree is 2, not above \(M_0=2\) |
| 3 | 0 | selects 0; 0's degree becomes 3, so drop its farthest neighbor 1 (0-1 is 0.01, 0-2 is 0.0025, 0-3 is 0.0004) | 0's list becomes [2,3]; the reverse edge 1→0 remains |
| 4 | 2 | layer 1 connects to 0; on layer 0, ef=2 gives \(W=\{2,3\}\) (0 at distance 16 is squeezed out) and the heuristic selects 2 (3 is pruned because its distance to 2 is 0.0009, smaller than its 15.8404 to 4); then entry=4, max_level=2, and layer 2 contains only 4 | 2-4; layer 2 is empty |

Final output (layers 0, 1, 2):

```
layer 0: [[2, 3], [0], [0, 4], [0], [2]]
layer 1: [[4], [], [], [], [0]]
layer 2: [[], [], [], [], []]
```

Two easily missed points: node 1 still lists `[0]`, but node 0's list no longer contains 1 — overflow replacement makes the edge one-way; and node 4, inserted last, becomes the new entry because of its level 2, yet layer 2 contains only itself and is therefore empty.

## Constraints

- Return a three-level nested Python `list` (not a tensor): outer length \(L=\max(\text{levels})+1\), middle length \(N\), inner lists of ascending, duplicate-free, self-loop-free node ids.
- Degree caps: layer 0 \(\le 2M\), layer 1 and above \(\le M\); a node appears only on layers \(0..\text{levels}[i]\), and higher layers must be empty lists.
- Judging compares nested lists exactly; the output must be ascending — do not rely on any other neighbor order.
- No randomness; duplicate points and distance ties must follow the deterministic (distance, node id) rule.
- CPU; judging controls the sizes, and any implementation (heap, sorted list, brute-force scan) is fine as long as the ordering semantics match.
