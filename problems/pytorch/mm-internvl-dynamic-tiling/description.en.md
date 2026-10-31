# InternVL Dynamic Tiling Ledger: Aspect-Ratio Matching, Tile Grid and Token Counting

## Problem

InternVL handles arbitrary aspect ratios not by brute-force resizing but by splitting the image into 448×448 tiles whose grid shape is chosen from the aspect ratio, plus one whole-image thumbnail tile. How many tiles, which grid, and how many visual tokens reach the vision tower — all determined by one deterministic ledger. This problem asks you to reconcile it item by item.

## Rules

Implement `solve(width, height, min_num, max_num, image_size, use_thumbnail, patch_size=14, scale_factor=0.5)` returning the ledger dict:

| Field | Meaning |
|---|---|
| `aspect_ratio` | `width / height` |
| `target_ratio` | chosen grid `[i, j]` (i columns, j rows) |
| `target_width` / `target_height` | `image_size * i` / `image_size * j` |
| `blocks` | `i * j` |
| `thumbnails` | 1 iff `use_thumbnail and blocks != 1` |
| `total_tiles` | `blocks + thumbnails` |
| `per_tile_tokens` | `(image_size // patch_size)² * scale_factor²` |
| `total_visual_tokens` | `per_tile_tokens * total_tiles` |
| `image_token_count` | `2 + total_visual_tokens` (one `<img>` and one `</img>`) |

Selection rules (sentence-by-sentence upstream):

1. Candidate grids: all `(i, j)` with `i, j ≥ 1` and `min_num ≤ i*j ≤ max_num`, sorted ascending and deduplicated by `(i*j, i, j)`.
2. Iterate candidates with `diff = |aspect_ratio − i/j|`:
   - `diff < best` (best starts at +∞, best ratio at `(1,1)`) → update
   - `diff == best` → override only if `width*height > 0.5 * image_size² * i * j` (the upstream area rule: pick the larger grid only when the image area exceeds half its tiled area)
3. Append one whole-image thumbnail tile iff `blocks != 1 and use_thumbnail`.
4. Raise `ValueError` for any invalid parameter (non-positive ints, `min_num > max_num`).

## Example

```
Input: 896×448, min=1, max=6, size=448, use_thumbnail=True
r=2 hits (2,1) exactly (diff=0); (4,2) also diff=0 but area=401408 < 0.5·448²·8=802816 → no override
Output: target_ratio=[2,1], blocks=2, thumbnails=1, total_tiles=3
        per_tile_tokens=256, image_token_count=2+768=770
```

## Constraints

- `448/14=32` patches per side; InternVL2's 0.5 pixel shuffle → `1024*0.25=256` tokens per tile
- Tie-break constructions must be float-exact (e.g. `r=5/4` with `0.5` and `2.0` both at exactly `0.75`)
