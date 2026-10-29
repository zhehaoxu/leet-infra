# In Depth: Retrieval: k-means Lloyd Iterations with Empty-Cluster Reseeding

## Source

- Lloyd, *Least Squares Quantization in PCM*, Bell Labs Technical Report, 1957 (published later in IEEE Trans. Inform. Theory, 1982): alternates "assign to the nearest centroid" and "replace each centroid by its cluster mean".
- MacQueen, *Some Methods for Classification and Analysis of Multivariate Observations*, 1967: named k-means and gave the online update.
- k-means++ (Arthur & Vassilvitskii, 2007) addresses initialization, which this task deliberately avoids: the initial centroids are fixed by the caller, and the exam is the iteration itself plus empty-cluster handling.
- Empty-cluster handling is a mandatory branch in production code; scikit-learn's `_relocate_empty_clusters` and Faiss's Clustering each have their own policy (the exact selection and randomness are version-dependent — **to be traced back to the source**). This task reproduces no library; the starter docstring governs.

## Full derivation and index convention

**Objective and optimality of the two steps.** Let \(a_i\in\{0,\dots,K-1\}\) be the assignment of point \(i\), and
\[
J(\{a_i\},\{c_j\})=\sum_{i=1}^{N}\|x_i-c_{a_i}\|_2^2.
\]
With centroids fixed, every point is independent, and \(a_i=\arg\min_j\|x_i-c_j\|_2^2\) minimizes \(J\); the square root is monotone, so comparing squared distances is enough. With assignments fixed, cluster \(j\) contributes \(\sum_{i:a_i=j}\|x_i-c_j\|^2\); setting the gradient \(2\sum_{i:a_i=j}(c_j-x_i)=0\) gives
\[
c_j=\frac{1}{|S_j|}\sum_{i\in S_j}x_i,
\]
the per-dimension arithmetic mean. Neither step increases \(J\), and \(J\ge 0\), so the iteration converges in the real-valued sense.

**Why the stop test uses displacement, not the objective.** The quantity here is
\[
\text{disp}=\max_{j}\|c_j^{\text{new}}-c_j^{\text{old}}\|_2,
\]
and the test is \(\text{disp}<\text{tol}\) (strict). The change of the objective depends on cluster sizes and distance scales; displacement is a per-centroid geometric quantity. The two are not equivalent, and judging only accepts the displacement rule. With `tol=0` the strict `<` never fires even when \(\text{disp}=0\), so the run executes all `max_iters` rounds — a deliberate edge case covered by tests.

**Why an empty cluster must be reseeded.** If \(S_j=\varnothing\), the mean has a zero denominator. The deterministic rule is:

1. Process empty clusters in ascending cluster index.
2. First compute, for every point, \(r_i=\min_j\|x_i-c_j^{\text{old}}\|^2\), the nearest distance to the **old centroid set used for this round's assignment**.
3. Sort all points by \((-r_i,\ i)\), i.e. farthest first, ties by smaller point index.
4. A `used` set ensures a point seeds at most one empty cluster per round; take the first unused point in that order and set the empty cluster's new centroid to its coordinates.

So "if the farthest point was already taken, take the next farthest" means walking down the same sorted list, not recomputing distances. The condition \(K\le N\) guarantees fewer empty clusters than points (at least one cluster is non-empty), so there are enough candidates. Reseeding changes centroids only, not this round's labels; the seed point still contributes to the mean of its original cluster, because means are computed before step 3. The easy mistake is to delete the seed from its old cluster before averaging.

**Multiple empty clusters and ties.** In the example, c1 is empty in round 1 and the old set is \(\{[0,0],[0,0],[10,0]\}\). The nearest distances are p0=0, p1=4, p2=0, p3=4; sorting by \((-r_i,i)\) gives p1, p3, p0, p2, so c1 takes p1=[0,2]. If c1 and c2 were both empty, c1 would take p1 and c2 would take p3 (the shared `used` set), not both taking p1.

**Why final labels are recomputed.** At stop time the last round's labels were produced by the old centroids, and reseeding can create a cluster out of nothing; returning those labels would leave the reseeded cluster permanently empty and make `labels` inconsistent with `centroids`. The reference therefore performs one more nearest assignment with the returned centroids (ties to the smaller index). A counter-intuitive consequence: the last round's assignment may differ from the final `labels`; judging only accepts the final `labels`.

**Precision.** Distances and means accumulate in float64; final centroids are cast back to the input dtype and final labels are computed from those cast centroids. The tolerance is \(10^{-5}\) for float32 inputs and \(10^{-9}\) for float64; `labels` and `n_iters` are compared exactly.

**Measurements** (example and a no-reseed contrast):

```
cd /Users/zhehao.xu/Desktop/leet-infra/problems/pytorch/retrieval-kmeans-lloyd/variants/pytorch && \
/Users/zhehao.xu/Desktop/leet-infra/.venv-eval/bin/python -c "
import torch
from solution import solve
pts=torch.tensor([[0.,0.],[0.,2.],[10.,0.],[10.,2.]])
init=torch.tensor([[0.,0.],[0.,0.],[10.,0.]])
l,c,it=solve(pts,init); print(l.tolist(), c.tolist(), it)
l1,c1,_=solve(pts,init,max_iters=1); print(l1.tolist(), c1.tolist())
"
```

Output:

```
[0, 1, 2, 2] [[0.0, 0.0], [0.0, 2.0], [10.0, 1.0]] 3
[0, 1, 2, 2] [[0.0, 1.0], [0.0, 2.0], [10.0, 1.0]] 1
```

On the same data, changing the empty cluster to "keep the old centroid, no reseeding" gives measured final labels `[1, 0, 2, 2]` and centroids `[[0.0, 2.0], [0.0, 0.0], [10.0, 1.0]]`: the two left points swap cluster ids and c1 stays at [0,0] forever. Reseeding is not cosmetic; it changes the final partition.

The initial squared-distance matrix was measured too (`d=((p[:,None,:]-c[None,:,:])**2).sum(2)`): `d=[[0.0,0.0,100.0],[4.0,4.0,104.0],[100.0,100.0,0.0],[104.0,104.0,4.0]]`. Hence c1 is empty in round 1, and p1 and p3 both have nearest distance 4.0 to the old centroid set; the tie goes to p1 by point index.

## Key points and common mistakes

1. **The stop quantity is the maximum centroid displacement, not the objective delta**; the test is strict `<`, so `tol=0` runs all `max_iters` rounds.
2. **Empty-cluster distances use the old centroids**: reseeding happens after the mean update, but the ranking is against the centroid set used for this round's assignment, not the freshly computed means.
3. **Reseeding does not change this round's labels**: the seed still contributes to its original cluster's mean; do not remove it before averaging.
4. **Multiple empty clusters: ascending cluster index, distinct seeds**: the `used` set must be shared across empty clusters, and ties go to the smaller point index.
5. **Final labels are recomputed**: run one more argmin against the returned centroids; returning the last round's labels contradicts the reseeded centroids.
6. **Assignment ties go to the smallest index**: `torch.argmin` naturally returns the first minimum; a hand-written loop must update only on strict `<`, never `<=`.
7. **\(K>N\) is an error**: there may be more empty clusters than available points, and the reseeding rule cannot run; `N=0` and `K=0` also raise `ValueError`.
8. **Do not accumulate float32 means in float32**: for large \(N\) the error can exceed \(10^{-5}\); the reference computes in float64 and casts at the end.
9. **Duplicate points are legal**: when all points coincide, reseeding picks seeds by point index; if all centroids remain numerically identical, the first-round displacement is 0 and the run can stop after one round (covered by tests).
10. **On `max_iters` truncation, still return the current centroids and labels based on them**; do not raise "not converged".

## Differences from neighboring problems

- **vs `retrieval-ivf-assign`**: that task only performs one "given coarse centroids, find each vector's nearest centroid" pass — this task's step 1 (the E step), with no centroid update, no empty cluster and no iteration. This is the **training side** with the full Lloyd loop.
- **vs `retrieval-pq-codes`**: that task encodes against fixed per-subspace codebooks that never move; here centroids change every round and global empty clusters must be handled.
- **vs `retrieval-hnsw-greedy` / `retrieval-hnsw-build`**: those are the query side and the build side of a graph index; this is flat vector quantization with no graph, layers or neighbor selection.
- **vs a generic k-means task**: the initial centroids are fixed (no random init), the empty-cluster reseeding order and seed deduplication are explicit, and the stop test is pinned to centroid displacement, so the result is fully reproducible.

## Cross-check convention and complexity

- **Primary oracle**: `_oracle_kmeans` uses pure-Python triple loops to accumulate squared differences explicitly, updates argmin only on strict `<`, computes means as `sum/count`, and reseeds with `sorted(..., key=(-r_i,i))`; its structure differs from the reference's "broadcast distance matrix + `argmin` + `mean` + `bincount`".
- **Coverage**: the example (with real reseeding), early convergence (`tol=10` stops in one round), `max_iters=1/2/3` truncation, `tol=0` running to the limit, a symmetric tie that must pick the smaller index, \(K=1\), \(N=K\) with all points coincident, integer grids, and float64. `labels`/`n_iters` are compared exactly; centroids use \(10^{-5}\) for float32 and \(10^{-9}\) for float64 (`rtol=0`).
- **Complexity**: per round the distance matrix is \(O(NKD)\), assignment \(O(NK)\), means \(O(ND)\), and the empty-cluster sort \(O(N\log N)\); total \(O(\text{max\_iters}\cdot(NKD+N\log N))\) with an \(O(NK)\) temporary distance matrix. Judging sizes are small, so correctness dominates constants.
