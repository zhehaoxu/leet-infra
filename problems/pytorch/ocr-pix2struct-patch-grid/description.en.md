# Pix2Struct Variable Aspect-Ratio Patch Grid Ledger

## Problem

Pix2Struct is an OCR model that feeds **full-page screenshots** directly to the vision encoder: instead of resizing to a fixed square, the image is scaled aspect-ratio-preserving so that it fills a grid of about `max_patches` patches, then flattened into a sequence and zero-padded. This problem transcribes the grid ledger of transformers v4.51.3's `Pix2StructImageProcessor.extract_flattened_patches`: given the original height/width, the patch budget and patch size, output the row/column counts, resized dimensions, total patches, padding count, and each patch's (row_id, col_id).

## Solution

The ledger is pinned by the spec (faithful to upstream):

1. `scale = sqrt(M * (ph / H) * (pw / W))` — maximize the scale while preserving aspect ratio so the total patch count approaches the budget M;
2. `R = max(min(floor(scale * H / ph), M), 1)` and `C = max(min(floor(scale * W / pw), M), 1)` — clamp to M first, then clamp up to 1;
3. `resized_height = max(R * ph, 1)`, `resized_width = max(C * pw, 1)`;
4. `total_patches = R * C`. Let `x = scale * H / ph`; then `scale * W / pw = M / x`, and `floor(x) * floor(M / x) ≤ x * (M / x) = M` proves **R*C ≤ M always holds**, hence `padded = M - R*C ≥ 0`;
5. `patch_ids` is expanded **row-major**, each item `[row_id, col_id]`, **offset by +1** — 0 is reserved for padding (official comment).

After the grid, the official code bilinearly resizes (antialias) and runs `torch_extract_patches`; this problem only covers the grid and index ledger, not pixel resampling.

## Constraints

- 1 ≤ H, W ≤ 10^4, 1 ≤ M ≤ 10^4, 1 ≤ ph, pw ≤ 64 (all positive ints).
- Any argument non-positive / non-int / bool → `ValueError`.
