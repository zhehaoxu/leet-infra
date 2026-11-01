# notes — ocr-pix2struct-patch-grid (order 688)

## Upstream

- Source: `huggingface/transformers` v4.51.3 `src/transformers/models/pix2struct/image_processing_pix2struct.py`, `Pix2StructImageProcessor.extract_flattened_patches` (GitHub API contents + base64, fetched by tag).
- Core four lines: `scale = sqrt(M*(ph/H)*(pw/W))` → `R = max(min(floor(scale*H/ph), M), 1)` → `C = max(min(floor(scale*W/pw), M), 1)` → resize then `torch_extract_patches` flatten with `row_ids/col_ids` each +1 (comment verbatim: "Offset by 1 so the ids do not contain zeros, which represent padding") and pad to M rows.
- **R*C ≤ M always holds** (floor product inequality: with x = scale·H/ph the other dimension is M/x, and floor(x)·floor(M/x) ≤ M); 20k random trials, 0 violations; hence `padded ≥ 0` is by design, not coincidence.
- Double-clamp quirk: extreme aspect ratios (e.g. 1000000×1 with M=64) clamp R down to 64 via `min(..., M)`, while C's floor of 0 is rescued to 1 by `max(..., 1)` — both clamps can fire simultaneously (pinned anchor).
- Boundary vs `mm-qwen25vl-window-index`: different algorithms — pix2struct is **single-image variable grid** (ratio-preserving + budget constraint), Qwen VL is window indexing; same for `mm-image-token-expand`.
- Float floor edge: when `scale*H/ph` is mathematically an exact integer, floating point could flip the floor (the 224 case measured exactly 8.0, no flip). The oracle avoids floats entirely: using the identity `floor(sqrt(A/B)) = isqrt(A//B)` (A = M·pw·H, B = W·ph, obtained by squaring away sqrt from floor(scale·H/ph)) it recomputes R/C in pure integers, independently cross-checked against the float solve.
- Pixel resampling (bilinear antialias) and `torch_extract_patches` are excluded: float resampling has no deterministic judging value; the ledger lives at the grid/index layer only.

## Judging

- solve inputs: image_height/image_width/max_patches (required positive ints) + patch_height/patch_width (defaults 16/16). Returns num_rows/num_cols/resized_height/resized_width/total_patches/padded/patch_ids.
- Oracle: `test_cases._py` uses a hand-rolled Newton-iteration `_int_sqrt` (not `math.isqrt`) plus the integer isqrt identity to recompute R/C independently; 200 random cross-checks + a 500-case invariant sweep (R*C ≤ M, padded ≥ 0, clamp bounds).
- Anchors: 224×224/M=64 → R=C=8, total=64, pad=0; 640×320/M=64 → R=11, C=5, 55 patches, pad=9 (patch_ids: first [1,1], row-end [1,2], next row [2,1], last [11,5]); 1000000×1 → R=64 (max clamp), C=1 (min rescue); 1×1 → scale=128, R=C=8; 7×13/M=32 → R=4, C=7, pad=4; non-square patch 8×32.
- Invalid inputs: any argument ≤ 0 / non-int / bool → ValueError.

## Mutations (12 observable all caught + 1 equivalent)

scale-no-sqrt, ceil-instead-of-floor, rows-cols-swapped, clamp-max-off, clamp-min-off, ids-no-offset, ids-col-major, pad-sign-flipped, total-counts-pad, resized-height-uses-cols, resized-forgets-multiply, floor-arg-swapped — all caught. **Equivalent**: scale-patch-swap (swapping the (ph/H)(pw/W) factors in the scale formula leaves the product M·ph·pw/(H·W) unchanged — commutativity).

## Red lines

No real images/PIL/torchvision — sizes are int inputs; bilinear resize values and patch pixel contents excluded (no deterministic judging value); `mm-qwen25vl-window-index` and `mm-image-token-expand` already own the window/resolution ledgers.
