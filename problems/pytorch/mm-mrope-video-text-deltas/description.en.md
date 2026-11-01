# Multimodal: Qwen2.5-VL Full M-RoPE Index (Video Temporal and Interleaving)

## Statement

Qwen2.5-VL's M-RoPE splits position encodings into temporal / height / width axes. Given:

- `tokens`: a single token sequence (list[int], length 1..32);
- `attention_mask`: a 0/1 list where 0 marks padding;
- `image_grid_thw` / `video_grid_thw`: the (t, h, w) grid of each image / video;
- `second_per_grid_ts`: seconds per grid along the temporal axis of each video (None means the default 1.0);
- `spatial_merge_size`, `tokens_per_second`; plus `image_token_id` / `video_token_id` / `vision_start_token_id`.

Implement the full accounting of transformers `get_rope_index` (v4.51.3) and return `{"position_ids": [t, h, w rows], "mrope_position_delta": int}`.

## Rules

1. **Counting**: only an image / video token directly following `vision_start` counts as a segment; bare vision tokens are not counted but the scanning `index` still finds them (official quirk, see example). Segment counts must match grid counts.
2. **Segments**: process vision tokens in order of first occurrence. Each segment emits text first (all axes `arange + (global max so far + 1)`) then vision:
   - temporal axis = `floor(i × second_per_grid_t × tokens_per_second)`, where images always use `second_per_grid_t = 0` (a constant axis!); videos use the parameter value;
   - height / width axes expand in (t, h', w') C order (h' = h // merge, w' = w // merge);
   - all three axes are offset by `text_len + st_idx`; `st` advances to `ed + t·h'·w'`.
3. **Padding**: `position_ids` at `mask == 0` positions carry **1** (the official ones-initialised buffer is never overwritten).
4. **delta** = global max over all three axes + 1 − **L (the full padded length)**.
5. **Fallback**: when both grids are empty, all axes are `cumsum(mask) − 1` with padding positions again carrying 1.

## Examples

```text
Input:  tokens = [7, 102, 100, 100, 100, 100, 8, 9] (102=start, 100=image)
        image_grid_thw = [[1,2,2]], spatial_merge_size = 1, tokens_per_second = 25
Output: t = [0, 1, 2, 2, 2, 2, 4, 5]
        h = [0, 1, 2, 2, 3, 3, 4, 5]
        w = [0, 1, 2, 3, 2, 3, 4, 5]
        delta = 5 + 1 - 8 = -2

Input:  tokens = [102, 200, 200, 200, 9], video_grid_thw = [[4,1,1]],
        second_per_grid_ts = [0.5], tokens_per_second = 25
Output: t = [0, 1, 13, 26, 38]
(temporal steps floor(12.5)=12, floor(37.5)=37 — Python round's half-to-even would give 38; int truncation pinned)
```

## Constraints

- 1 ≤ L ≤ 32; the mask has at least one 1; the three token ids are pairwise distinct.
- Grid t/h/w are positive ints; h and w must be divisible by merge with quotient ≥ 1.
- `second_per_grid_ts` has one entry per video segment, each ≥ 0.
- Malformed inputs (bool posing as int, all-zero mask, segment/grid count mismatch, vision_start as the last effective token, etc.) raise ValueError.
- torch is forbidden; pure Python; inputs are not modified.

## Hints

- The offset chain's `st_idx` is the **global** max over all three axes + 1, not the width axis alone; the image segment's constant temporal axis is the easiest official quirk to miss.
- Padding positions carry 1, not 0, and delta subtracts the full length L — both pinned by the official `ones` initialisation / `total_input_ids` source lines.
