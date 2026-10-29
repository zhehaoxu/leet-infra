# In Depth: Retrieval: HNSW Graph Insertion and Neighbor Selection

## Source

- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, arXiv:1603.09320 (TPAMI 2018). Algorithm 1 is INSERT, Algorithm 2 is SEARCH-LAYER, Algorithm 3 is simple neighbor selection, and Algorithm 4 is the heuristic.
- Faiss's `IndexHNSW` and hnswlib implement this structure; different implementations shrink an overfull neighbor list differently (the paper's Algorithm 1 reruns the heuristic, while this task follows the statement and simply removes the farthest neighbor — the starter is authoritative).
- The paper samples the level as \(l=\lfloor-\ln(U)\cdot m_L\rfloor\); this task replaces that with the explicit `levels` input, removing randomness so every edge can be cross-checked.

## Full derivation and index convention

**Layer model.** Node \(i\) exists on layers \(0\) through `levels[i]`, and layer 0 contains every node. The output has
\[
L=\max_i \text{levels}[i]+1
\]
layers. During insertion the running maximum must start at `max_level = levels[0]`, not at the global \(\max_i\text{levels}[i]\): node 0 exists only up to its own level. Initializing it to the global maximum makes node 0 appear present on higher layers, so later high-level nodes search from it and create spurious edges. This was actually hit while prototyping: in the example, node 4 (level 2) would incorrectly connect to 0 on layer 2, whereas the correct layer 2 contains only node 4.

**Insertion flow** (the deterministic version of Algorithm 1):

1. For \(q\), set \(l=\text{levels}[q]\) and `ep=entry`.
2. For `lc` from `max_level` down to \(l+1\): run `SEARCH-LAYER(q, ep, ef=1, lc)` and replace `ep` with the result nearest to \(q\). This is the upper-layer greedy descent: one best point per layer.
3. For `lc` from \(\min(\text{max\_level},l)\) down to 0:
   - `W = SEARCH-LAYER(q, ep, ef_construction, lc)`;
   - `R = SELECT-NEIGHBORS-HEURISTIC(q, W, M, lc)`;
   - add a bidirectional edge for every \(e\in R\);
   - shrink every \(e\)'s list if it overflows (below);
   - set `ep` to the point in `W` nearest to \(q\) for the next lower layer.
4. If \(l>\text{max\_level}\), set `entry=q` and `max_level=l`.

The out-degree cap is \(M\) on layer 1 and above, and \(M_0=2M\) on layer 0: the bottom layer carries fine-grained neighbors and needs the larger degree.

**SEARCH-LAYER ordering and termination.** The candidate set \(C\) is a min-heap of points to expand, and the result set \(W\) is a max-heap of the current `ef` nearest points. Each step pops the (distance, node id)-smallest \(c\) and stops if
\[
\text{dist}(c,q) > \max_{w\in W}\text{dist}(w,q).
\]
The comparison is strict: equal distances keep expanding, so the exploration does not depend on heap pop order among ties. When expanding \(c\), its neighbors are scanned in ascending node id; an unvisited neighbor is marked visited and if \(|W|<\text{ef}\) or \((d(e,q),e)\) is lexicographically better than \(W\)'s worst (distance, node id), it is pushed to both \(C\) and \(W\), with the worst popped whenever \(|W|>\text{ef}\). Using (distance, node id) as a total order makes duplicate points and all-zero distances deterministic.

**Why Algorithm 4 prunes.** Candidates are sorted by distance to \(q\). The selected set \(R\) is spread out; if candidate \(c\)'s distance to some selected \(r\) is strictly smaller than its distance to \(q\), then \(c\) lies inside \(r\)'s "coverage": starting from \(q\), \(r\) already reaches the neighborhood of \(c\), so a direct \(c\)-\(q\) edge adds little navigation value and \(c\) is dropped. The test is strict `<`: when \(d(c,r)=d(c,q)\), keep the candidate. Selection stops at \(M\) (or when candidates run out). If `ef_construction < M`, \(W\) itself has at most `ef_construction` points, so a new node gets at most that many neighbors and is not padded.

**Bidirectional connection and overflow replacement.** For each selected \(e\), add the edge once in each endpoint's list on that layer, checking first to avoid self-loops and duplicate edges. Then check \(e\)'s degree: if it exceeds `cap` (\(2M\) on layer 0, otherwise \(M\)), repeatedly remove the neighbor with the largest squared distance to \(e\), breaking ties by the larger node id. The newly added \(q\) competes equally and may be removed immediately; in that case \(q\) still lists \(e\) while \(e\) no longer lists \(q\), so the final graph **may be asymmetric**. This is the most counter-intuitive departure from an "always undirected" mental model.

**Entry update.** The entry changes only when \(l>\text{max\_level}\); for \(l\le\text{max\_level}\) the old entry stays. A new node is isolated on layers above the old `max_level`: those layers had no nodes before and may contain only it afterwards, so its neighbor lists there are empty. Node 4's layer 2 in the example is exactly this situation.

**Output indices.** `out[lc][i]` is node \(i\)'s neighbor list on layer `lc`, sorted ascending. Node \(i\) does not exist on layers `lc > levels[i]`, which must be empty lists. The outer length is the global \(L\), not the running `max_level+1`.

**Measurements** (layer-0 evolution on prefixes, full output, asymmetry):

```
cd /Users/zhehao.xu/Desktop/leet-infra/problems/pytorch/retrieval-hnsw-build/variants/pytorch && \
/Users/zhehao.xu/Desktop/leet-infra/.venv-eval/bin/python -c "
import torch
from solution import solve
pts=torch.tensor([[0.0],[0.1],[0.05],[0.02],[4.0]])
lv=torch.tensor([1,0,0,0,2])
for n in (2,3,4,5):
    out=solve(pts[:n],lv[:n],M=1,ef_construction=2)
    print('n=',n,'layer0=',out[0])
out=solve(pts,lv,M=1,ef_construction=2)
for i,l in enumerate(out): print(i,l)
print('edge 1->0', 0 in out[0][1], 'edge 0->1', 1 in out[0][0])
"
```

Output:

```
n= 2 layer0= [[1], [0]]
n= 3 layer0= [[1, 2], [0], [0]]
n= 4 layer0= [[2, 3], [0], [0], [0]]
n= 5 layer0= [[2, 3], [0], [0, 4], [0], [2]]
0 [[2, 3], [0], [0, 4], [0], [2]]
1 [[4], [], [], [], [0]]
2 [[], [], [], [], []]
edge 1->0 True edge 0->1 False
```

The `n=4` line shows the overflow directly: before node 3, node 0's layer-0 list is `[1,2]`; after adding node 3 it becomes `[1,2,3]`, exceeding \(M_0=2\), so node 1 (distance 0.01, the farthest from 0) is removed and the list becomes `[2,3]`. At the end `1->0` survives while `0->1` is gone, proving that replacement can leave an edge one-way.

Measuring the distance matrix on the same points (`d=((p[:,None,:]-p[None,:,:])**2).sum(2)`) gives: `d[0]=[0.0, 0.01, 0.0025, 0.0004, 16.0]`, `d[2]=[0.0025, 0.0025, 0.0, 0.0009, 15.6025]`, `d[3]=[0.0004, 0.0064, 0.0009, 0.0, 15.8404]`. Every distance quoted in the example table (0.01/0.0025/0.0004/0.0009/15.6025/15.8404/16.0) comes from these rows.

## Key points and common mistakes

1. **`max_level` starts at `levels[0]`, not `max(levels)`**: otherwise node 0 appears to exist on higher layers and high-level insertions create spurious edges (in the example, an extra 0-4 edge on layer 2).
2. **Descent layers and search layers differ**: layers above \(l\) use the \(ef=1\) greedy search; only \(\min(\text{max\_level},l)\) down to 0 uses `ef_construction`. Mixing them changes the candidate sets.
3. **SEARCH-LAYER stops on strict `>`**: when the candidate distance equals the current worst, expansion continues; using `>=` stops early and can miss a tied point with a smaller node id.
4. **Algorithm 4 prunes on strict `<`**: only `d(c,r) < d(c,q)` prunes; equality keeps the candidate. Reversing the comparison prunes almost everything.
5. **Overflow removes the farthest, not the newest**: ties remove the larger node id; the just-added \(q\) may itself be removed, so the final graph can be asymmetric. Do not "repair" it into an undirected graph.
6. **Caps are layer-dependent**: \(2M\) on layer 0 and \(M\) elsewhere; check each layer separately, not with a uniform \(M\).
7. **The entry changes only when \(l>\text{max\_level}\)**: `l==max_level` keeps the old entry; a new node is isolated on higher layers and must output empty lists.
8. **Inner lists must be ascending and duplicate-free**: judging compares nested lists exactly; the order in which neighbors were added (distance order) is not the output order.
9. **`ef_construction < M` is not padded**: the new node's degree is limited by the candidate pool; do not select pruned candidates just to reach \(M\).
10. **Duplicate points and all-zero distances**: every key includes the node id, so comparisons reduce to (0, id); overflow removes the largest id. Tests run all-coincident points twice to verify determinism.

## Differences from neighboring problems

- **vs `retrieval-hnsw-greedy`**: that is the **query side**: given an already-built single-layer graph (a fixed-width tensor padded with `-1`), start at the entry and run \(ef=1\) greedy search, returning the stopping node. This task is the **build / insertion side**: multiple layers, point-by-point insertion, an `ef_construction` candidate queue, Algorithm 4 neighbor selection, bidirectional edges and degree-overflow replacement, returning every layer's adjacency. They share the "greedy descent" subroutine, but here it is only a small part of insertion; the main work is selecting neighbors and maintaining the graph.
- **vs `retrieval-ivf-assign`**: that task assigns vectors to the nearest coarse centroid once, with no graph, layers or neighbors.
- **vs `retrieval-pq-codes`**: that task encodes each subspace against fixed codebooks and builds no graph.
- **vs `retrieval-kmeans-lloyd`**: that task iterates centroids and handles empty clusters; this task updates no centers and only maintains adjacency lists.
- **vs the paper's HNSW**: explicit `levels` replace random levels, and overflow shrinking is simplified to "remove the farthest neighbor"; everything else (SEARCH-LAYER's ef, Algorithm 4, bidirectional edges, entry update) follows the paper.

## Cross-check convention and complexity

- **Primary oracle**: `_oracle_hnsw` uses pure-Python lists and re-sorts each step to take the minimum/worst (no heap), accumulating distances with explicit nested loops; the reference uses `heapq` min/max heaps plus a precomputed distance matrix. The two structures differ, and outputs are compared as nested lists with exact equality.
- **Coverage**: the example (multiple layers, overflow, asymmetric edge), a single-node graph (level 2, three empty layers), layer structure (a node appears only on \(0..levels[i]\)), degree caps and overflow replacement (all-coincident points; with \(M=1\), node 0's degree goes 2 → 3 and is trimmed), duplicate points run twice for determinism, `ef_construction < M`, several 1-D/2-D/float64 oracle cases, and invalid inputs.
- **Complexity**: the distance matrix costs \(O(N^2D)\) time and \(O(N^2)\) memory. Each insertion visits at most \(L\) layers; one SEARCH-LAYER is \(O(\text{ef}\cdot(\bar d+\log\text{ef}))\) where \(\bar d\) is the average degree on that layer, Algorithm 4 is \(O(\text{ef}\log\text{ef}+M\cdot\text{ef})\), and overflow shrinking is at worst \(O(M\log M)\). Overall about \(O(N\cdot L\cdot \text{ef}\cdot M + N^2D)\); judging sizes are small, so correctness comes first.
