# Solution Notes: Attention Head Importance Scoring and Structured Pruning

## Origin
Head-level structured pruning treats a single head as the smallest deletable unit (Michel et al., *Are Sixteen Heads Really Better than One?*, NeurIPS 2019, arXiv 1905.10650). Saliency can come from a first-order Taylor score \(|\,\partial\mathcal{L}/\partial g_h\cdot g_h\,|\) or from gradient-free weight/activation statistics as here. The product \(\|W_h\|_F\|X_h\|_F\) is the head-granularity generalisation of **Wanda**'s \(|W_{ij}|\cdot\|X_j\|_2\) (Sun et al., 2023, arXiv 2306.11695): Wanda scores columns (input channels) and prunes element-wise, this problem scores head slices and prunes whole slices.

## Key points and pitfalls
1. **Norm scopes**: \(\|W_h\|_F\) is taken over that head's own \((d_\text{head},D)\) slice and \(\|X_h\|_F\) over its \((N,d_\text{head})\) slice; mixing in other heads is a silent bug.
2. **Product, not a weighted sum**: both terms are "larger is more important", so the product automatically kills a head with zero calibration activation no matter how large its weights. A weighted sum would need tuning and loses that semantics.
3. **Global ranking**: `sparsity` is a fraction of **all** heads; a per-layer quota is wrong when layers have different head counts. Flatten before comparing.
4. **Whole-slice zeroing**: the point of structured pruning is that the head disappears; zeroing only a few elements inside it degrades to unstructured pruning and saves no compute.
5. **Deterministic ties**: `torch.argsort(..., stable=True)` keeps equal scores ordered by flattened index, which is what makes bit-exact judging possible.

## Verification
The reference computes both norms with `flatten(2).norm(dim=2)`, multiplies, takes the first \(k\) of a stable ascending argsort and multiplies the weight by a 0/1 mask. The judge's oracle instead sums squares with Python `math.sqrt` per head, sorts by `(importance, index)` and zeroes whole slices by hand — a structurally independent recomputation — then compares `importance` and `pruned_weight` element-wise. Additional assertions cover: the number of fully zeroed heads equals \(k\), `sparsity=0` changes nothing, `sparsity=1` zeroes everything, equal scores drop the smaller flattened index, kept heads are untouched, and a head with zero calibration activation is always dropped first.

## Complexity
\(O(L\cdot H\cdot(d_\text{head}D+N d_\text{head}))\) for the norms plus \(O(LH\log(LH))\) for the sort; memory is linear in the input.
