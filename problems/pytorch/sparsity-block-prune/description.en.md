# Sparsity: Block Structured Pruning (by Block Norm)

## Problem

The scattered zeros from unstructured pruning are hard for hardware to exploit; **block (structured) pruning** discards whole, tidy small blocks instead: split the matrix along the last dimension into consecutive blocks of width `block`, measure importance with each block's **sum of magnitudes (L1 norm)**, and prune the least important whole blocks first, thereby preserving a dense substructure that vectorized/blocked GEMM can accelerate.

Given a `weight` of shape \((R,C)\), a sparsity rate `sparsity`, and a block width `block` (\(C\) divisible by `block`). Each row is cut into blocks of width `block`, for a total of \(B=R\cdot (C/\text{block})\) blocks. Compute each block's L1 norm \(\sum |w|\), prune the \(\mathrm{round}(\text{sparsity}\cdot B)\) whole blocks with the smallest norms (every element inside a pruned block is set to zero), and keep the remaining blocks untouched.

## Rules

Input `weight` (float32 2-D), scalar `sparsity`, integer `block`. Return a float32 tensor of the same shape: each selected block is zeroed entirely, unpruned blocks are preserved element by element. A block's importance = the sum of \(|w|\) inside it; on ties, prune first in ascending order of the flattened (row, block-column) index. Off-the-shelf pruning libraries are forbidden.

## Examples

`weight=[[1,-5,3,-2],[0.5,9,-2,-8]]`, `sparsity=0.5`, `block=2`:

```
Block L1: row0-block0=|1|+|-5|=6, row0-block1=|3|+|-2|=5
          row1-block0=|0.5|+|9|=9.5, row1-block1=|-2|+|-8|=10
B=4, prune round(0.5*4)=2 smallest blocks -> block(row0,block1) norm5 and (row0,block0) norm6
out = [[0,0,0,0],[0.5,9,-2,-8]]
```

## Constraints

- CPU float32; \(C \bmod \text{block}=0\); the pruning unit is a whole block (inside a block it is either all kept or all zero).
- Unlike the elementwise/per-row problems: here the comparison uses block-aggregated norms, with all blocks competing globally for the sparsity budget.
- Positions covered by pruned blocks are strictly 0; unpruned blocks are untouched.
