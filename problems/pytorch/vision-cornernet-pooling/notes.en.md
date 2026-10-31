# Notes: vision-cornernet-pooling

## Source

- `princeton-vl/CornerNet` `models/py_utils/_cpools/src/{right,left,top,bottom}_pool.cpp` forward loops verified line by line: right_pool `output[0]=input[0]`, `output[i+1]=max_out(input[i+1], output[i])`; left_pool starts from W−1 going the other way.
- Combination classes in `models/CornerNet.py`: `class tl_pool(pool): super().__init__(dim, TopPool, LeftPool)`, `br_pool(dim, BottomPool, RightPool)` — each branch pools separately then `pool1 + pool2` elementwise.
- Official implements only tl/br (the network has two corner heatmaps); tr/bl are added by symmetry and declared in the description.

## Key Judgments

- **Filename vs semantics is counter-intuitive**: right_pool.cpp scans rightward (out[j] = max(x[0..j])), so each point takes the **left-side** max. Mutation `right-suffix` (reversing the accumulation direction) pins this.
- **Pool-then-add ≠ add-then-pool**: max does not distribute over addition. Mutation `add-to-max` (add → elementwise max) also pins the add semantics.
- Hand-anchor slips twice: br anchor mis-added 1+1 as 4, and the single-column br forgot to add right=self. Solution and brute-force oracle agreed first, then anchors were fixed — lesson: elementwise-addition hand anchors must be computed cell by cell.
- Running max includes self (endpoints included); `no-self`/`top-no-self` mutations pin endpoint initialization.

## Boundary

- No C++/CUDA, no backward (scatter_add ledger could be a follow-up); no heatmap decoding (dedup vs `vision-heatmap-decode`); no CenterNet's fused pooling variant.
- Same vision-detection batch as `vision-ohem-selection` (676).

## Test Design

- Oracle `_py_pool` recomputes all four directions brute-force O(H·W·n) and is compared against the O(HW) scan solution.
- 8 checks: hand anchors (full-grid tl/br), single row/column degeneration, monotone rows, all-equal, all-negative, 200 random, 6 invalid tiers.
- Mutations 9/9: direction reversal for each of the four pools, two endpoint inits zeroed, tl/br branch swap, add→max. `pool-then-add-reversed` (branch order swap) is an equivalent mutation (addition is commutative); dropped and recorded.
