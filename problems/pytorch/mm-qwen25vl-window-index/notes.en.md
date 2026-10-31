# Notes: Qwen2.5-VL Vision Window Index and cu-Window Seqlens

## Approach

The solution is a line-by-line transcription of transformers 5.20.2's `vision_utils.get_vision_window_index` (`modeling_qwen2_5_vl.py` imports exactly this function from `vision_utils`), made explicit in only two places: input validation (`vws ≥ 1`, `h,w` divisible by merge, scalar args reject bool) and the trailing `reverse_indices = torch.argsort(window_index)` that the model computes right after the call.

The core follows upstream verbatim: `arange → reshape(t, llm_h, llm_w) → F.pad(-100) → reshape(t, nwh, vws, nww, vws) → permute(0,1,3,2,4) → reshape(t, nwh*nww, vws, vws)`; `seqlens = (index_padded != -100).sum([2,3]).reshape(-1)` counts llm tokens per window; the flattened non-`-100` indices plus the cross-image offset form `window_index`; the cu ledger uses `seqlens.cumsum(0) * mu + previous-image accumulation`, then `torch.unique_consecutive`. Two easy-to-miss details: **`F.pad`'s `(0, pad_w, 0, pad_h)` order** (the last dim is left/right on w, the second-to-last up/down on h), and **the permute hoisting (window-row, window-col) above (in-row, in-col)** — structurally the same as Swin's `window_partition` (which `mm-window-partition` covers as static tiling without padding or the -100 marker; this task is the full varlen pre-ledger with cross-image offsets and two scale levels — the boundary is documented on both sides).

## Semantic anchors (verified against local transformers)

During development I called transformers 5.20.2's `get_vision_window_index` directly to pin three semantics instead of trusting the 2402.10807 paper: (1) a 16×16-patch (8×8 llm) image at `vws=4` really does cut a 3×3 window grid with 5 empty windows, and the deduped cu ledger keeps only the 4 full windows' 5 boundaries; (2) the cu step size is "llm tokens per window × merge²" (patch level), on a different axis from `window_index`'s llm-level length; (3) with `t>1` windows unfold as `(ti, window-row, window-col)` without re-opening the window grid per frame.

## The equivalent-mutation finding

Of the 7 mutations, all 5 effective ones were caught (missing permute, cu without merge², patch-level cross-image offset, dropping unique_consecutive, replacing reverse with window_index); **the two survivors are mathematically equivalent**: clamping `pad` to "0 when divisible" and an identity rewrite of `num_windows`. The reason: empty windows contribute 0 entries to `window_index` and only append duplicate cu boundaries, and those duplicates are necessarily adjacent (the sequence is non-decreasing), so `unique_consecutive` flattens them regardless — the pad formula is **observably identical** to the clamped version in the final output; upstream's "pad = vws when divisible" quirk is an internal process quirk, not an observable behaviour. What IS observable is **dropping the dedup step** (the no-dedup mutation caught by 5 cases). This joins the repo's prior equivalent-mutation records (no-validation-r, CPU tie stability, …): some upstream "weirdness" in transcription tasks is self-consistent in output space, and the notes must downgrade it from "behavioural contract" to "process quirk".

## Independent verification

The oracle is a pure-Python quintuple loop: frame → window row → window column → in-row → in-col, with both orders hand-written row-major, skipping out-of-bounds logical cells and counting only valid ones × `merge²` into the cu ledger, then hand-written adjacent-dedup and the inverse permutation (`reverse[tok]=pos`). It uses none of torch's reshape/permute/cumsum/argsort. Twenty-five random grids cover merge∈{1,2,4}, patch∈{1,2,7,14}, window∈{28,56,112,224}, G∈1..4, t∈1..3; a separate three-way cross-check (oracle = transformers = solution) ran 36 grids with zero differences. The integer ledger is asserted with exact `==` everywhere — no float tolerance.

## Complexity

Per image with `T_i = t·llm_h·llm_w` llm tokens and `W_i` windows: flattening and the boolean selection cost `O(Σ(T_i + padded grid points))`, the cu accumulation `O(ΣW_i)`, the argsort `O(T log T)`. The sizable intermediates are the padded grid and the window reshaped views (`O(Σ(T_i + pad points))`); with divisible sizes the padding inflates the grid by at most `(1+1/vws)²`.

## Source and scope

Based on transformers 5.20.2's `transformers/vision_utils.py::get_vision_window_index` (imported at `modeling_qwen2_5_vl.py:52`, called at L442), plus `modeling_qwen2_5_vl.py`'s `permute_input_for_window_attn` (`input_tensor[window_index, :, :]`, L419-421) and `reverse_indices = torch.argsort(window_index)` (L479). Line-level alignment points: `vit_merger_window_size = window_size // spatial_merge_size // patch_size`, `spatial_merge_unit = spatial_merge_size**2`, `pad = vws - size % vws`, `F.pad(..., "constant", -100)`, `seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]`, `torch.unique_consecutive`.

Deliberately out of scope: the windowed attention itself (the kernel path that feeds cu_seqlens to flash-attn varlen); the `kwargs` cache branch that pops precomputed `window_index`/`cu_window_seqlens` (pure wiring); boundaries with `mm-window-partition` (Swin static tiling), `mm-mrope-position-ids` (Qwen2-VL position ids) and `mm-flux-2d-rope` (2D RoPE) are documented in each problem's notes.

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-qwen25vl-window-index --mode submit
```

## Mutation testing (5 effective caught + 2 equivalent)

Missing permute (5/8 cases), cu without merge² (7/8), patch-level offset (3/8), dropped unique_consecutive (5/8), replaced reverse (4/8) — all caught; the pad-clamp and num_windows rewrites are output-equivalent, as analysed above.
