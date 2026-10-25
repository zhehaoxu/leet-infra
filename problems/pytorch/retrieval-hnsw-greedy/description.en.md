# Retrieval: HNSW Greedy Layer Search

## Problem

HNSW (Hierarchical Navigable Small World, Malkov & Yashunin, TPAMI 2018 / arXiv:1603.09320) is the most mainstream graph-based ANN structure today (Faiss `IndexHNSW`, hnswlib, and many vector databases). It searches a multi-layer navigable small-world graph: **upper layers are sparse and used to quickly locate the right region; lower layers are dense and used for exact nearest neighbors**. The basic move both between and within layers is the same **greedy search (greedy search / `SEARCHLAYER` with \(ef=1\))**: starting from the entry point, repeatedly jump to "the neighbor of the current point that is closest to the query and closer than the current point", until no neighbor is closer (a local optimum). This problem implements that core subprocedure — greedy search on a **single-layer graph**.

## Rules

Implement `solve(points, adj, entry, query)`. The input `points` (float32, \((n, D)\)) holds the node coordinates; `adj` (int64, \((n, \text{maxdeg})\)) is each node's neighbor list with `-1` as padding (meaning the slot has no neighbor); `entry` (int) is the index of the entry node; `query` (float32, \((D,)\)) is the query point. Distance is **squared Euclidean**. Search: `cur=entry`; at each step find the one **closest to query** among `cur`'s valid neighbors, and if it is closer than `cur`, move `cur` there (one point per step), otherwise stop; return the index of the node where it stopped. Return a **0-dim int64** tensor. When neighbors are equidistant, take the smallest index.

## Examples

A 1-D chain \(0-1-2-3-4\), `points=[[0],[1],[2],[3],[4]]`, `query=[3.7]`, `entry=0`:

```
adj = [[1,-1],[0,2],[1,3],[2,4],[3,-1]]
squared distance to query: node0=13.69 node1=7.29 node2=2.89 node3=0.49 node4=0.09
0 ->(neighbor 1 is closer)-> 1 ->(neighbor 2 is closer)-> 2 ->(neighbor 3 is closer)-> 3 ->(neighbor 4 is closer)-> 4
4's neighbor {3}(0.49) is not closer than 4(0.09) -> stop at 4
out = 4   (0-dim int64)
```

## Constraints

- CPU float32; use squared Euclidean distance (monotonic, does not change the argmin).
- **Move only if strictly closer** (neighbor distance \(<\) current), which guarantees no cycles and certain convergence.
- One step moves along a single edge to "the closest among the closer neighbors" (greedy with \(ef=1\)), not a batched \(ef\)-candidate queue.
- Equidistant neighbors: take the smallest node index; skip `-1` slots.
