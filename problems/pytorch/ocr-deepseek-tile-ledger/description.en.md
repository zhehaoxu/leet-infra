# DeepSeek-OCR Five-Mode Dynamic Resolution and Tile Ledger

## Problem

DeepSeek-OCR (arXiv 2510.18234) squeezes documents into few vision tokens: five resolution presets plus Gundam dynamic tiling. This problem hand-writes the official vLLM `image_process.py`'s `dynamic_preprocess` tile selection/splitting ledger and `tokenize_with_images`' vision token sequence ledger — pure scalars, no PIL images.

## Solution

Official behavior (deepseek-ai/DeepSeek-OCR `DeepSeek-OCR-vllm/process/image_process.py` + `config.py`, pinned line by line):

- **Five presets** (config comments verbatim): Tiny base/image=512, Small 640, Base 1024, Large 1280, all crop_mode=False; **Gundam base=1024, image=640, crop_mode=True**. Config defaults MIN_CROPS=2, MAX_CROPS=6 (comment: max:9);
- **Small-image gate**: cropping and `width <= image_size and height <= image_size` (**and** — a single dimension overflowing triggers tiling: 500×700 tiles, 640×640 does not); cropping=False gives (1,1) directly;
- **Ratio set quirk**: `{(i,j): n∈[min,max], 1≤i,j≤n, min ≤ i*j ≤ max}` — **the i/j upper bound is the loop variable n, not max_num** (for min=2/max=6 this yields the same set as i,j≤max; an equivalent mutation was verified SURVIVED); min=2 excludes (1,1); sorted ascending by product (stable);
- **Closest match**: `diff = |w/h − i/j|`, strictly `<` updates (first-come-first-served); **on equal diffs, area steals**: `area > 0.5·image_size²·i·j` switches to the current ratio (the image area is constant while the threshold grows with tile count) — at aspect 2.5 the pair (2,1)/(3,1) ties, and area 4×10⁵ does not steal while 2.5×10⁶ does;
- **Tile splitting**: resize **stretches directly** (not aspect-preserving) to `(image_size·nw, image_size·nh)`, then cuts row-major boxes: `x0=(k%nw)·image_size, y0=(k⌿nw)·image_size`;
- **Token sequence ledger**: `q = ceil((image_size//16)/4)` (q image tokens per tile row), `qb` likewise with base_size; the global view = `([qb <image> + 1 sep])·qb` plus a final sep = **(qb+1)·qb+1** (Tiny 73 / Small 111 / Base and Gundam global 273 / Large 421); when `nw>1 or nh>1`, append the local `([q·nw <image> + 1 sep])·(q·nh)` = **(q·nw+1)·q·nh** (tiles concatenated within a row, one sep per row). The README's "vision tokens" (64/100/256/400) are the qb²/q² pure-encoder accounting — the sequence ledger differs by the row seps; both accountings coexist.

## Data Scale

- Hand-computed anchors: Gundam 2000×1500 (4:3 -> (3,2) 6 tiles, 893 tokens, 6 row-major boxes), 2000×2000 ((2,2), 693), 3000×500 ((6,1), 883), small-gate three states (640×640 / 500×700 -> (2,3) / 640×400 tie), the tie double (1000×400 no steal -> (2,1) / 2500×1000 steal -> (3,1)), cropping=False large image, min_crops=1's (1,1) single tile (the official or-condition swallows the single-tile local ledger), five-preset native token counts;
- 200 randomized comparisons (oracle = independent implementation with explicit dict enumeration + hand-written matching).

## Scoring

All seven ledger keys equal exactly: crop_ratio / tiles / tile_boxes / global_tokens / local_tokens / num_image_tokens (pure integer ledger).

## Red Lines

- No PIL images and no PIL import (AST-level check); pure scalar ledger;
- width/height/base_size/image_size must be positive (else ValueError).
