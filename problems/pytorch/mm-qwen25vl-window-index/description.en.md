# Multimodal: Qwen2.5-VL Vision Window Index and cu-Window Seqlens

## Task

Qwen2.5-VL's vision tower uses **windowed attention** in some layers: tokens on the spatial grid are reordered into a "window-contiguous" 1-D sequence so a varlen attention only interacts within windows, then restored via the inverse permutation. This task is exactly that **pure index ledger**: given patch-grid sizes `grid_thw` of several images/videos, transcribe transformers' `vision_utils.get_vision_window_index` line by line and produce the reorder index, the cumulative window-boundary ledger, and the inverse permutation.

Every input is an integer grid parameter — there is no float forward pass — and judging uses exact equality. Parameter and exception conventions live in the starter's bilingual contract; the rules below define the computation only.

## Rules

1. **Two scales first**: `h,w` in `grid_thw` are patch-level; dividing by `spatial_merge_size` gives the LLM-token grid `llm_h × llm_w` (each LLM token corresponds to `merge²` patch tokens, the `spatial_merge_unit`). The window edge on the LLM grid is `vws = window_size // spatial_merge_size // patch_size` (official config: `112//2//14 = 4`), and must be ≥ 1.
2. **The padding quirk**: `pad = vws - size % vws` — **when divisible, pad equals vws, NOT 0**! A divisible edge still gains one all-empty window row/column; empty windows contribute no tokens but leave duplicate boundaries on the cu ledger, which `unique_consecutive` then flattens.
3. Per frame, `arange(t·llm_h·llm_w)` is laid out as a `(t, llm_h, llm_w)` grid, padded with `-100` on the right/bottom, and split into windows in (window-row, window-col) row-major order; **inside a window tokens keep (in-row, in-col) row-major order**. Windows unfold frame by frame, window row by window row, window column by window column.
4. `window_index` = the non-`-100` grid indices taken window by window, each **plus the cross-image offset** (the sum of previous images' `t·llm_h·llm_w`), concatenated; it is a permutation of `0..T-1`.
5. `cu_window_seqlens` starts from 0: after each window it accumulates `(that window's non-pad llm-token count) × merge²`. Note that it counts **patch-token level** (the varlen attention runs on the pre-merge patch sequence), so the ledger grows faster than `window_index`. Finally `unique_consecutive` removes adjacent duplicates; result int32.
6. `reverse_indices = argsort(window_index)` — the argsort of a permutation is its inverse, which upstream uses to restore the original order.

The two outputs have **naturally different lengths**: `window_index`/`reverse_indices` are LLM-token level (length `Σ t·llm_h·llm_w`) while `cu_window_seqlens` holds patch-level boundaries; this is not a bug.

## Example 1: single-window anchor

`grid=[[1,4,4]]`, `merge=2`, `patch=14`, `window=112` → `llm 2×2`, `vws=4`, padded to `4×4`, one window:

```text
window_index      = [0, 1, 2, 3]
cu_window_seqlens = [0, 16]        # 4 llm tokens × merge² = 16 patch tokens
reverse_indices   = [0, 1, 2, 3]
```

## Example 2: divisibility creates empty windows (the core quirk)

`grid=[[1,16,16]]` → `llm 8×8`. `pad_h = pad_w = 4` (8 is divisible by 4!), cutting a **3×3 window grid of which the 3rd row and 3rd column — 5 windows — are empty**:

```text
window_index first 8 = [0, 1, 2, 3, 8, 9, 10, 11]   # window (0,0) in (r,c) row-major order
len(window_index)    = 64                            # empty windows contribute nothing
cu_window_seqlens    = [0, 64, 128, 192, 256]        # 4 full windows, duplicates deduplicated
```

Changing the pad to "0 when divisible" gives a **mathematically identical output** (see the equivalent-mutation note); what IS observable is dropping `unique_consecutive`, which leaves the duplicate tail `[0,64,128,192,256,256,256,256,256,256]`.

## Example 3: video frames and non-divisible edges

`grid=[[2,10,6]]`, `merge=2` → per frame `llm 5×3`: window rows `r0-3` (12 tokens) plus `r4` (3 tokens), frame 1 followed by frame 2:

```text
window_index      = [0..29]          # coincidentally the identity (window rows fully cover, no column split)
cu_window_seqlens = [0, 48, 60, 108, 120]   # 12×4 and 3×4 accumulating per frame
```

## Judging

8 `check_*` functions: the single-window anchor, divisible empty windows and dedup, cross-image offsets, video frames with non-divisible edges, the merge-unit scaling of the cu ledger, random grids against an independent oracle, the inverse-permutation property (`window_index[reverse] == arange`), and noncontiguous views plus invalid inputs. The oracle is a pure-Python quintuple loop (frame → window row → window column → in-row → in-col) using no torch reshape/permute; during development a three-way cross-check against local transformers 5.20.2's `get_vision_window_index` ran over 36 random grids with oracle = transformers = solution.

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-qwen25vl-window-index --mode submit
```
