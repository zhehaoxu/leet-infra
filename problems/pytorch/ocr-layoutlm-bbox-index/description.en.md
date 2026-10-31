# LayoutLM-Family BBox Normalization and 2D Position Index Ledger

## Problem

The LayoutLM family (LayoutLM/LayoutLMv2/LayoutLMv3, official `microsoft/unilm`, HF `layoutlm*`) takes document-layout text-box coordinates as model input: coordinates are normalized to 0–1000, then looked up in x/y/h/w position embeddings, plus a grid-based 2D position index. This problem extracts the pure **bbox → embedding ids / grid ids** ledger: given pixel-coordinate boxes and page dimensions, output the normalized corners, width/height embedding ids, and corner/center grid ids.

## Solution

The ledger is pinned by the spec (consistent with HF `_calc_spatial_position_embeddings`):

1. **Normalize to 0–1000**: x via `round(1000 * x / page_w)`, y via `round(1000 * y / page_h)` (Python `round`, i.e. **banker's rounding**), then **clamp to [0, 1000]** (HF only raises on out-of-range; the unilm pipeline clamps before feeding — the spec pins the clamp into the ledger);
2. **Width/height embedding ids**: `w_id = x1 - x0`, `h_id = y1 - y0` (after normalization; upstream `bbox[:,:,2]-bbox[:,:,0]` and `bbox[:,:,3]-bbox[:,:,1]`);
3. **2D grid index**: `grid(x, y) = y * 1001 + x` (columns 0..1000). Each box outputs the four corner grid ids `[g(x0,y0), g(x1,y0), g(x0,y1), g(x1,y1)]` and the center `g((x0+x1)//2, (y0+y1)//2)`.

Returns `{"norm_boxes", "wh_ids", "corner_grid", "center_grid"}`.

## Constraints

- 1 ≤ len(boxes) ≤ 10^4; coordinates may be any finite numbers (out-of-range handled by clamp).
- Empty boxes, entries not of length 4, non-numeric or NaN/Inf values, normalized x0 > x1 or y0 > y1, non-positive or bool page_w/page_h → `ValueError`.
