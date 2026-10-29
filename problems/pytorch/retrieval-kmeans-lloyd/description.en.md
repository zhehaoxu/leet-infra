# Retrieval: k-means Lloyd Iterations with Empty-Cluster Reseeding

## Problem

k-means is the standard training procedure behind vector quantization and inverted-file indexes: Lloyd's 1957 Bell Labs technical report introduced the "assign, then average" alternation, and MacQueen named it k-means in 1967. Given \(N\) points in \(D\) dimensions and \(K\) initial centroids, each round first performs a hard assignment to the nearest centroid and then replaces every centroid by the mean of its members. The objective
\[
J=\sum_{i=1}^{N}\bigl\|x_i-c_{a_i}\bigr\|_2^2
\]
does not increase in either step, so the iteration converges to a local optimum.

This task pins down two details that production implementations usually gloss over: **the initial centroids are supplied entirely by the caller** (no random initialization), and **empty clusters must be reseeded**. When a centroid receives no point, its mean is undefined; the rule here is to move the point farthest from the current centroid set into that cluster, with an explicit order for multiple empty clusters, distance ties, and seed reuse. A second hidden trap is that the stopping test looks only at the **maximum centroid displacement**, not at the change of the objective.

CPU, fully deterministic, no RNG calls.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring; that is the sole judging contract. What follows only restates the semantics.

- **Inputs**: points of shape \((N,D)\) and initial centroids of shape \((K,D)\), both with the same dtype (float32 or float64); \(N\ge 1\), \(K\ge 1\), \(K\le N\). All distances are squared Euclidean (the square root is monotone and cannot change an argmin). Assignment ties go to the **smallest centroid index**.
- **Four steps per round**: (1) assign every point to its nearest centroid by squared Euclidean distance; (2) set each non-empty cluster's new centroid to the per-dimension arithmetic mean of its members; (3) handle empty clusters as below; (4) compute the Euclidean displacement of every centroid against the previous round, take the maximum \(\text{disp}\), and stop after this round if \(\text{disp}<\text{tol}\); reaching `max_iters` also stops.
- **Empty-cluster reseeding**: process empty clusters in **ascending cluster index**. Candidates are all points not yet used as a seed in this round; sort them by the **descending** distance from the point to the **old centroid set used for this round's assignment**, breaking ties by smaller point index. Take the first unused candidate and set the empty cluster's new centroid to that point's coordinates; a point can seed at most one empty cluster per round. Reseeding changes centroids only, not this round's labels; the seed point still contributes to the mean of the cluster it was assigned to (the means were computed before reseeding).
- **Stopping and counting**: `tol` must be non-negative and the stop test is \(\text{disp}<\text{tol}\) (strict). Hence `tol=0` never stops early even when the displacement is exactly 0, and runs all `max_iters` rounds. The returned iteration count is the number of rounds actually executed, including the round that triggered convergence, so \(1\le n\_iters\le \text{max\_iters}\).
- **Final labels**: the returned `labels` come from one more nearest-centroid assignment using the **returned final centroids** (ties to the smaller index), not from the last round's assignment. This gives reseeded clusters labels and keeps `labels` consistent with `centroids`.
- **Degenerate inputs and errors**: `N=0`, `K=0`, `K>N`, non-2-D tensors, a dtype other than float32/float64, mismatched dtypes, mismatched feature dimensions, a non-positive-integer `max_iters`, or `tol<0` all raise `ValueError`. The condition \(K\le N\) guarantees there are enough distinct seed points.
- **Precision**: distances and means accumulate in float64, final centroids are cast back to the input dtype, and final labels are computed from those cast centroids.

## Example

Four 2-D points and three initial centroids, the first two of which coincide, so the first round necessarily has an empty cluster:

```
points = [[0,0], [0,2], [10,0], [10,2]]
init   = [[0,0], [0,0], [10,0]]
tol = 1e-4, max_iters = 100
```

| Round | Assignment (point → cluster) | Centroids after averaging | Empty-cluster reseed | Max displacement |
|---|---|---|---|---|
| 1 | p0,p1 → c0; p2,p3 → c2; c1 empty | c0=[0,1], c2=[10,1] | c1 ← p1=[0,2] (p1 and p3 are both at distance 4 from the old set; smaller index 1 wins) | 2 |
| 2 | p0 → c0; p1 → c1; p2,p3 → c2 | c0=[0,0], c1=[0,2], c2=[10,1] | none | 1 |
| 3 | same as round 2 | unchanged | none | 0 |

Round 3 has \(\text{disp}=0<10^{-4}\), so the run stops after 3 rounds:

```
labels    = [0, 1, 2, 2]
centroids = [[0,0], [0,2], [10,1]]
n_iters   = 3
```

The reseeded c1 ends up exactly at the coordinates of input point p1, not at any cluster mean. Without reseeding (keeping the old centroid [0,0] in the empty cluster) the final labels would be `[1, 0, 2, 2]` — see the measured contrast in the notes.

## Constraints

- Return the triple `(labels, centroids, n_iters)`: `labels` has shape \((N,)\) and dtype int64; `centroids` has shape \((K,D)\) and the input point dtype; `n_iters` is a Python `int`.
- Judging compares `labels` and `n_iters` exactly and `centroids` with absolute tolerances \(10^{-5}\) for float32 and \(10^{-9}\) for float64.
- The implementation path is free (vectorized distance matrix, per-point loops, `argmin`, or a strict-less-than scan), but the tie, reseeding and stopping rules above must hold.
- No randomness; no objective-monotonicity check and no per-round history is required.
