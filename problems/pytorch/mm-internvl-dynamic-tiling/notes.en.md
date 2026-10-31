# Notes: InternVL Dynamic Tiling Ledger

## Provenance

OpenGVLab/InternVL `internvl_chat/internvl/train/dataset.py` — `find_closest_aspect_ratio` and `dynamic_preprocess` (fetched line-by-line via the GitHub contents API on 2026-10-31). Token accounting from transformers 5.20.2 `processing_internvl.py` (`image_seq_length = (image_size // patch_size)**2 * scale_factor**2`; `num_image_tokens = 2 + image_seq_length * num_patches`).

## Key points and traps

1. **Candidates come from a triple loop**: for each `n ∈ [min_num, max_num]`, `i, j ≤ n` with `i*j ∈ [min_num, max_num]`. After dedup this equals exactly the pairs with `i*j ≤ max_num` (any `(i,j)` is covered by `n = max_num`). The problem pins the order to `(i*j, i, j)` ascending.
2. **Upstream's set iteration order is unreproducible**: the official code collects into a `set` then `sorted(key=i*j)`; ties depend on CPython set internals — not judgeable across versions. The problem statement pins ties to lexicographic order and this note records the normalization. Mutation testing shows `sorted(ratios)` (no key) is **observationally equivalent** within the testable space: order only matters for ties, and same-area ties get identical area-rule decisions while different-area equidistant pairs are blocked by float exactness (the `13/12` construction differs by 1 ulp on the two sides).
3. **Direction of the area rule**: override iff `area > 0.5 * image_size² * i * j` — "the image area exceeds half the grid's tiled area", not the reverse. For the 896×448 wide image, `r=2` also matches `(4,2)` at diff=0, but `401408 < 802816` so no override — the answer stays `(2,1)`. The first-draft anchor got this wrong (assumed 8 tiles).
4. **Float-exact tie construction**: `13/12` is mathematically equidistant from `2/3` and `3/2`, but in floats the two diffs differ by 1 ulp and the strict-less branch decides. Ties must use binary-exact values: `r=5/4` with `0.5` and `2.0` both at exactly `0.75`. `500×400` (area 200000 < 200704, keeps `(1,2)`) and `505×404` (area 204020 > 200704, overrides to `(2,1)`) cover both sides.
5. **Thumbnail condition**: `use_thumbnail and blocks != 1` — a single-tile image gets no thumbnail (it is its own thumbnail); `use_thumbnail=1` (not a bool) must be rejected.
6. **Three layers of token accounting**: per tile `(448//14)² * 0.5² = 256`; total `256 * total_tiles`; the `<img>` text tokens add 2. Forgetting to square `scale_factor` is the high-frequency error (mutation `no-scale2` caught by 4 cases).

## Mutation testing (9 effective caught + 1 equivalent)

`<` → `<=`, area-rule threshold to `0.5*size²`, ratio row/col swap, drop thumbnail, always thumbnail, drop the scale², drop the `+2`, `best_diff` initial 1.0, sort without key. **Equivalent**: keyless `sorted(ratios)` (point 2 above).

## Complexity

O(max_num²) candidate grids, microseconds.
