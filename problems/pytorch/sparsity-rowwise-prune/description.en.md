# Sparsification: Row-wise (Local) Magnitude Pruning

## Problem

The counterpart of a "global threshold" is **local (layer-wise / row-wise) pruning**: the sparsity budget is allocated independently **inside each output neuron (each row of the matrix)**, which guarantees that every row is pruned by exactly the same fraction and avoids the imbalance where some neurons are emptied entirely while others barely move. It is the more common choice when the layer's output distribution must stay stable.

Given a weight `weight` of shape \((\text{out},\text{in})\) and a sparsity ratio `sparsity`, zero out **per row** the \(k=\mathrm{round}(\text{sparsity}\cdot \text{in})\) elements of smallest absolute value in that row, keeping the rest.

## Rules

Input `weight` (float32, 2-D: \((\text{out},\text{in})\)) and the scalar `sparsity \in [0,1]`. Return float32 of the same shape: within each row prune the smallest \(k\) entries in ascending order of absolute value (the same \(k\) for every row); ties are broken by pruning the smaller column index first. `torch.nn.utils.prune` is forbidden.

## Examples

`weight=[[1,-5,3,-2],[0.5,9,-2,-8]]`, `sparsity=0.5` (per row \( \text{in}=4, k=2 \)):

```
row 0 |w|=[1,5,3,2], prune the 2 smallest (1,2) → [0,-5,3,0]
row 1 |w|=[0.5,9,2,8], prune the 2 smallest (0.5,2) → [0,9,0,-8]
out = [[0,-5,3,0],[0,9,0,-8]]
```

## Constraints

- CPU float32; each row competes only for its own budget (unlike the global problem sparsity-magnitude-prune, which compares across rows).
- The number of kept entries is the same in every row → each row's sparsity is exactly `sparsity`.
- Only pruned positions become 0; unpruned elements keep their value and sign.
