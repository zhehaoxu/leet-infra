# Deep Dive: Row-wise (Local) Magnitude Pruning

## Origin
Local / layer-wise pruning (as opposed to Han et al.'s global threshold): the sparsity budget is allocated independently within each output neuron (each row of the matrix), which is equivalent to the "local" (per-dimension grouping) idea in `torch.nn.utils.prune`. In engineering it is commonly used to keep the activations of every neuron balanced.

## Core Points and Pitfalls
1. **Each row competes for its own budget**: the only difference from the global problem sparsity-magnitude-prune is the scope of the threshold—here row \(i\) is compared only against itself, and each row prunes \(k=\mathrm{round}(\text{sparsity}\cdot \text{in})\) entries.
2. **Every row prunes the same count → every row's sparsity equals sparsity**: no row ends up emptied and no row stays untouched.
3. **Sort along dim=1 and scatter zeros**: `argsort(dim=1,stable=True)[:k]` followed by `scatter_`; do not flatten into a global sort.
4. Consider only \(|w|\) and keep the original value and sign; break ties by ascending column index to stay deterministic.

## Comparison Against the Oracle
The reference solution uses `abs().argsort(dim=1,stable=True)[:,:k]` → `scatter_(1,idx,0)`. The judge's `_oracle` sorts each row with Python `sorted` by \(|w|`, keeps the tail, and zeros the rest (different structure, numerically equivalent), compared with `allclose(atol=1e-5,rtol=1e-5)`. Additional assertions: "every row has the same non-zero count = ⌈in·(1-sparsity)⌉", "rows are independent", and "results differ from the global variant where the cross-row distribution differs".

## Complexity
\(O(N\log C)\) (sorting per row).
