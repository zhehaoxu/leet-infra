# Deep Dive: Global Magnitude Pruning

## Origin
The global magnitude-threshold pruning of Han et al., *Deep Compression* (arXiv:1510.00149): across the whole tensor, discard the batch of weights with the smallest absolute values. It is the most basic and most interview-frequent unstructured pruning criterion.

## Core Ideas and Pitfalls
1. **Global, not per-row/per-channel**: all elements compete for the same sparsity budget; the threshold is the \(k\)-th smallest \(|w|\), so different rows get pruned unevenly.
2. **Prune exactly the smallest \(k\) elements, not everything below some fixed threshold**: \(k=\mathrm{round}(\text{sparsity}\cdot N)\), corresponding precisely to the sparsity rate.
3. **Look at absolute value, not sign**: \(-9\) matters more than \(1\); sort only by \(|w|\).
4. **Tie determinism**: which element with equal \(|w|\) is pruned first must be fixed (here by ascending flattened index); use a `stable=True` argsort, otherwise different implementations disagree.

## Cross-Checking Against the Oracle
The reference solution does `argsort(|w|.flatten(), stable=True)[:k]`, zeroes the flattened positions, then reshapes. The judge's `_oracle` uses pure Python `sorted(range(N), key=(abs,index))`, takes the first \(k\) to zero out, and recomputes element by element (different structure, numerically equivalent), `allclose(atol=1e-5,rtol=1e-5)`. Additional assertions cover "sparsity=0 keeps as-is / sparsity=1 zeroes everything", "exactly the global smallest \(k\) are pruned", and "unpruned elements keep their value and sign".

## Complexity
Sorting \(O(N\log N)\) (or a `kthvalue` threshold can reach \(O(N)\)).
