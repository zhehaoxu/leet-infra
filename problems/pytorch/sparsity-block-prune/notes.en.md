# Deep Dive: Block Structured Pruning (by Block Norm)

## Origin
Block / structured pruning (block pruning, cf. Luo et al. *ThiNet*, arXiv:1708.01852; and NVIDIA's motivation for structured sparsity): the discard unit is a tidy small block, importance is measured with an aggregate norm, balancing compression ratio against hardware accelerability (contiguous memory, vectorizable dense sub-blocks).

## Core Ideas and Pitfalls
1. **The pruning unit is a whole block**: a block is either fully kept or fully zeroed; unlike unstructured pruning you cannot zero just a few elements inside — this is the essential difference from the elementwise problem.
2. **Block importance = the sum of \(|w|\) inside the block (L1)**: aggregate first, then compare globally; a single large element does not necessarily save the block it sits in.
3. **Global budget competition**: all blocks (across rows) are sorted together by norm, and the smallest \(\mathrm{round}(\text{sparsity}\cdot B)\) are pruned, where \(B\) is the total number of blocks.
4. **Reshaping into blocks must align with the last dimension**: `reshape(R, C//block, block)`, under the precondition `C%block==0`; the mask is broadcast back over each block with `view(R,nb,1)`.
5. Ties are broken by ascending flattened (row, block-column) index, with `stable=True` to preserve determinism.

## Cross-Checking Against the Oracle
The reference solution does `reshape→abs.sum(dim=2)→argsort(stable)→mask.view(R,nb,1)` broadcast. The judge's `_oracle` uses Python double loops to accumulate each block's L1, `sorted` to take the smallest k blocks, and copies the rest back element by element (different structure, numerically equivalent), `allclose(atol=1e-5,rtol=1e-5)`. Additional assertions cover "pruned blocks are zero as a whole", "kept blocks' elements are unchanged one by one", and "sparsity=0 keeps as-is / =1 zeroes everything".

## Complexity
\(O(N + B\log B)\).
