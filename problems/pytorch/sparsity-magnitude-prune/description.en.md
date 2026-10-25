# Sparsity: Global Magnitude Pruning

## Problem

**Magnitude pruning** is the most classic and most frequently tested unstructured pruning criterion: the smaller a weight's absolute value, the more limited its contribution to the output, and the more it deserves to be dropped. The "global magnitude threshold" version of Han et al. (Deep Compression, arXiv:1510.00149) zeroes, in one shot, the batch of smallest \(|w|\) elements across the entire tensor, reaching the target sparsity.

Given the weight tensor `weight` and the target sparsity `sparsity` (the fraction of elements to zero out), sort all elements by absolute value in ascending order and zero out the smallest \(k=\mathrm{round}(\text{sparsity}\cdot N)\) of them (\(N\) is the total number of elements), keeping the rest unchanged.

## Rules

Input `weight` (float32 of any shape), scalar `sparsity \in [0,1]`. Return a float32 tensor of the same shape: the \(k=\mathrm{round}(\text{sparsity}\cdot N)\) elements with the smallest absolute values are set to 0, the others keep their original values. On ties (equal \(|w|\)), ties are broken by ascending **flattened index** to decide who gets pruned first, guaranteeing determinism. Calling `torch.nn.utils.prune` directly is forbidden.

## Examples

`weight=[1,-5,3,-2,0.5,-9]`, `sparsity=0.5` (\(N=6,k=3\)):

```
|w| = [1,5,3,2,0.5,9]; the 3 smallest are 0.5(i4), 1(i0), 2(i3) → zero them out
out = [0, -5, 3, 0, 0, -9]
```

## Constraints

- CPU float32; `sparsity` ranges over \([0,1]\), \(k\) uses `round`.
- Global threshold (compared across the whole layer), not per-row/per-channel: different elements compete for the same sparsity budget.
- The output has the same shape as the original tensor; pruned positions are strictly 0 (the values and signs of unpruned elements are not modified).
