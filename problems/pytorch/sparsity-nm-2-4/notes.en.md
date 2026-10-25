# In Depth: 2:4 Structured N:M Pruning

## Source
The 2:4 semi-structured pattern of the NVIDIA Ampere Sparse Tensor Core (developer.nvidia.com cusparselt blog; PyTorch `to_sparse_semi_structured`): at most 2 non-zeros per 4 consecutive elements, and the hardware uses a fixed metadata layout to speed up GEMM by nearly 2×.

## Core points and pitfalls
1. **Grouping runs only along the last dimension, 4 consecutive elements per group**: not across rows, and not a per-row overall top-2; `reshape(-1,4)` flattens into groups, and row boundaries do not change the grouping (given last dim %4==0).
2. **An independent top-2 per group**: each group keeps exactly 2 and prunes 2, so the overall sparsity ratio is always 50%, regardless of how element magnitudes are distributed.
3. **Compare \(|w|\) but keep the original value**: the mask selects positions, the elements themselves keep their sign and magnitude; do not replace the retained values with 1.
4. Each 4-element group must be contiguous and aligned — that is exactly why hardware can exploit it more easily than unstructured sparsity.

## Cross-check convention
The reference solution uses `g.abs().topk(2,dim=1)` → scatter mask → `g*mask`. The judging `_oracle` sorts each group in Python by \(|w|\) with `sorted`, takes the first 2 and zeroes the rest (different structure, numerically equivalent), compared with `allclose(atol=1e-5,rtol=1e-5)`. Extra assertions: "exactly 2 non-zeros per group", "sparsity ratio = 0.5", "groups are independent (changing one group does not affect another)".

## Complexity
Per-group top-2, \(O(N)\).
