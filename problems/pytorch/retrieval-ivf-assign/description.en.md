# Retrieval: IVF Coarse Quantizer Assignment

## Problem

The inverted file index (IVF, Inverted File Index) is a core structure for approximate nearest neighbor retrieval (Jégou et al., *Product Quantization for Nearest Neighbor Search*, TPAMI 2011; Faiss `IndexIVF`). It first trains \(k\) **coarse centroids** with k-means, partitioning the vector space into \(k\) Voronoi cells; when building the index, every candidate vector is assigned to its **nearest centroid** (i.e. its inverted list), and a query then only needs to scan a few of the nearest lists. This problem does only this most basic deterministic step: **find the nearest coarse centroid for each vector**.

## Rules

Implement `solve(x, centroids)`. The input `x` (float32, shape \((n, D)\)) holds the vectors to assign, and `centroids` (float32, shape \((k, D)\)) holds the \(k\) coarse centroids. For each vector \(x_i\), compute its **squared Euclidean distance** \(\sum_t (x_{i,t}-c_{j,t})^2\) to each centroid and return the index of the nearest centroid. Return an int64 tensor of shape \((n,)\). If several centroids tie for the minimum, take the **smallest index**.

## Examples

\(n=3, D=2, k=2\), `centroids=[[0,0],[5,5]]`, `x=[[0,0],[3,4],[1,1]]`:

```
x0=[0,0]: distance to c0=0, c1=50 -> 0
x1=[3,4]: distance to c0=25, c1=2 -> 1
x2=[1,1]: distance to c0=2, c1=32 -> 0
out = [0, 1, 0]   shape = (3,)
```

## Constraints

- CPU float32; \(n, k \ge 1\), \(D \ge 1\).
- The **squared** Euclidean distance is enough (the square root is a monotonic transform and does not change the argmin).
- Ties go to the smallest centroid index, ensuring determinism.
