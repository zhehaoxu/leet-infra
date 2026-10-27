# Solution Notes · WKVD Least-Squares Merging

## In one sentence

An evicted value is not thrown away; it is **projected into the subspace spanned by the surviving values and spread back over the surviving rows using the projection coefficients**.

## Get the shape accounting right first

This is where implementations most often get flipped. The objective is

\[
\min_{C}\ \big\|V_{\text{keep}}^{\top}C - V_{\text{evict}}^{\top}\big\|_F,
\quad V_{\text{keep}}^{\top}\in\mathbb R^{d\times K},\
V_{\text{evict}}^{\top}\in\mathbb R^{d\times E}
\]

so the linear system has \(d\) equations and \(K\) unknowns (each column \(C_{:,e}\) is independent), giving \(C\in\mathbb R^{K\times E}\). The alternative formulation \(V_{\text{keep}}W\approx V_{\text{evict}}\) would require \((K\times d)W = E\times d\), i.e. \(W\) of shape \(d\times d\) that only makes sense when \(K=E\)—a different problem, and it does not express "each surviving row absorbs a share of the evicted vectors".

Why the merge is \(V_{\text{keep}} + C V_{\text{evict}}\) becomes obvious per row:

\[
V'_i = V_{\text{keep},i} + \sum_{e=1}^{E} C_{i,e}\, V_{\text{evict},e}
\]

Surviving token \(i\) absorbs a weighted sum of **all** evicted vectors, with the fit coefficients as weights. That is why this is called a merge rather than a drop.

## Why the pseudo-inverse is mandatory

The normal equations \(V_{\text{keep}}V_{\text{keep}}^\top C = V_{\text{keep}}V_{\text{evict}}^\top\) have a \(K\times K\) Gram matrix as coefficient matrix. Rows of \(V_{\text{keep}}\) are frequently linearly dependent in low-rank heads (\(K > \operatorname{rank}\)), the Gram matrix is singular, `inv` blows up and `solve` yields inf/nan.

The Moore–Penrose pseudo-inverse returns the **minimum-norm least-squares solution**:

\[
C^\* = V_{\text{keep}}^{+\top} V_{\text{evict}}^\top
\]

Under rank deficiency it is the optimal solution with the smallest \(\|C\|_F\)—operationally "absorb the evicted information with the smallest possible change", so duplicate basis vectors do not add the same information twice. `torch.linalg.pinv` runs an SVD internally and zeroes singular values below the threshold, so degenerate inputs still return finite values.

## Three physically checkable properties

1. **Full basis, zero residual**: when \(K \ge \operatorname{rank}(V_{\text{evict}}^\top)\) the evicted vectors are exactly representable, the residual is 0 and the information is lossless.
2. **Zero coefficients for orthogonal components**: replace \(V_{\text{evict}}\) with a vector in the orthogonal complement and all fit coefficients are 0, so `merged == V_keep` and the residual equals that vector's norm. This pins down the projection semantics—adding a bias or normalizing \(V_{\text{evict}}\) internally fails this check immediately.
3. **Bilinearity**: with \(V_{\text{evict}}\to 2V_{\text{evict}}\) we get \(C\to 2C\) (homogeneity of linear least squares), while the increment \(C V_{\text{evict}}\to 4C V_{\text{evict}}\). Assuming the increment merely doubles is the most common mistake.

## Boundaries

- `K`/`E` of 0, mismatched `d`, or non-2-D input → `ValueError`; do not "tolerate" these into empty tensors, which would silently turn the downstream merge into a no-op.
- Everything runs in float64: an SVD of a near-singular matrix in float32 amplifies noise through the small singular values, and coefficient/residual errors can reach \(10^{-2}\).
- The residual is the **solve residual**, not a reconstruction error after merging; \(V'\) and \(V_{\text{evict}}\) do not even share a shape.

## Complexity

`pinv` goes through an SVD: \(O(dK^2)\) for \(d\ge K\), plus \(O(dKE)\) for the products. Relative to the memory saved by keeping fewer KV entries, this cost is paid once per compression event and stays off the decoding hot path.

## Related problems

- `kv-cake-bidirectional-budget`: decides **how much each layer is cut**, a budget-allocation view; this problem handles how not to waste what is cut.
- `kv-layer-budget-knapsack`: discrete optimization of cross-layer budgets; also resource allocation, but maximizing retained attention mass.
- `kv-h2o-heavy-hitters`: selects heavy hitters by accumulated attention and simply drops the rest; this problem shows that "dropping" can come after a merge.
