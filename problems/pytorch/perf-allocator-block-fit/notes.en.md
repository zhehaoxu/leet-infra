# Deep Dive: Best-Fit Block Management in Caching Allocators

## Origin
`find_block`/`release_block`/`get_free_block` in PyTorch's `c10/cuda/CUDACachingAllocator.cpp` (2MB segments, small/large pools, 512B alignment) and the TensorFlow BFC allocator are two implementations of the same idea: **segments + block table + best-fit + coalescing on free**. This problem flattens segments into one contiguous address space and drops stream/event and device-side details, keeping only the integer logic that decides reserved bytes, fragmentation, and growth count — exactly what `torch.cuda.memory_stats()` reports as `reserved_bytes.all.peak`, `segment.all.allocated`, and `fragmentation_bytes`. It differs from `inf-paged-kv-block-alloc` (KV page-table allocation with fixed block size and no coalescing) and `perf-oom-boundary-search` (a closed-form peak formula): here it is a **step-by-step simulation of a dynamic block table**.

## Key points and pitfalls
1. **Best-fit is not first-fit**: among blocks that fit, take the smallest `size`, then the smallest `start`. First-fit eats large blocks early, shatters them, and inflates the growth count.
2. **Re-run best-fit after growing**: once the new segment joins the free set, the optimal block may well be an older, smaller freed block; carving straight from the new segment wastes reserved bytes.
3. **Growth is `max(need, grow_unit)`, aligned up**: requests larger than grow_unit grow by exactly `need` (no waste); smaller ones grow by `grow_unit` (fewer device calls). Both must be aligned to `align`.
4. **Coalesce with both neighbours on free**: skipping either side stops the free set from being maximal, later best-fits miss the merged run, and the growth count explodes — the classic bug in this problem.
5. **A split leaves the remainder free**: when `size > need`, `[start+need, start+size)` goes back to the free set; forgetting it silently swallows memory.
6. **`peak_reserved` updates only on growth**: reserved is monotone, so the peak always sits after the last growth; decrementing reserved on free corrupts both peak and final.

## Verification
The reference keeps a `start`-ordered block table, with `best_fit` scanning linearly and `insert_free` merging right then left. The judge `_oracle` keeps no block table at all: it slices the address space into boolean pages of `align` bytes, recomputes best-fit as "scan runs of free pages, take the shortest run that fits (ties by lower address)", and lets maximal blocks and coalescing emerge from the bitmap; freeing just flips pages back to True. The two structures (block table vs page bitmap) are compared item by item over 60 random op streams. Additional assertions: a single large request grows by exactly `need`; a small request grows by exactly `grow_unit`, aligned; two releases coalesce into one block that is reused in place with no further growth; freeing everything degenerates to one block `[[0, reserved]]`; free blocks never overlap and both `start` and `size` are aligned; the empty op stream returns all zeros.

## Complexity
`O(P·B)` with `P` operations and `B` free blocks (linear scan of the block table). The bitmap oracle is `O(P·reserved/align)` and exists only for small cross-checks.
