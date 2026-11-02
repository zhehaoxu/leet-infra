# notes — ocr-deepseek-tile-ledger (order 719)

## Source

- Authoritative source: deepseek-ai/DeepSeek-OCR `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py` + `config.py` (main; unchanged since the 2026-10 initial commit). The HF-hub model code (trust_remote_code) is isomorphic; the torchvision/HF dynamic_preprocess shares the same origin (InternLM/XComposer-style anyres).
- **Config pinned**: IMAGE_SIZE=640, BASE_SIZE=1024, CROP_MODE=True, MIN_CROPS=2, MAX_CROPS=6 (comment: max:9, lower to 6 for small GPU memory). Five presets = config edits: Tiny 512/512/False, Small 640/640/False, Base 1024/1024/False, Large 1280/1280/False, Gundam 1024/640/True.
- **Ratio set construction**: `set((i,j) for n in range(min,max+1) for i in range(1,n+1) for j in range(1,n+1) if i*j<=max and i*j>=min)` — the i/j bound is n. **Equivalence**: for min=2/max=6 this equals i,j≤max (any (i,j) with product in [2,6] has i,j≤6; n=6 generates it); confirmed by a SURVIVED equivalent mutation. set -> sorted by product: same-product order comes from set iteration order (deterministic within a process), and the closest-match first-come-first-served tie depends on it — **the oracle rebuilds the same order via explicit dict enumeration** (insertion order = comprehension order; 200 randomized comparisons verified no divergence from the set's iteration order).
- **Tie area steal**: `elif diff == best_diff: if area > 0.5*image_size²*i*j: best = ratio` — the threshold grows with i*j while the area is constant; **the steal is direction-blind** (a vertical (1,n) can be stolen by a horizontal ratio). The aspect-2.5 (2,1)/(3,1) tie chain is the clean separator: area 4×10⁵ (1000×400) does not steal, 2.5×10⁶ (2500×1000) does.
- **Tile splitting**: `image.resize((tw,th))` stretches directly (not aspect-preserving! a long strip gets squashed onto the tile grid) — unlike Qwen-VL's aspect-preserving per-tile resize. Boxes are row-major (k%nw is the column). use_thumbnail defaults False (Gundam's global view is an independent ImageOps.pad, not a thumbnail).
- **Token sequence ledger** (tokenize_with_images): global `([img]*qb + [sep])*qb + [sep]` = (qb+1)·qb+1; local `([img]*(q*nw) + [sep])*(q*nh)` — **tiles are concatenated within a row** (q·nw images share one row-end sep), with no separate trailing local sep. The README's "64/100/256/400 vision tokens" is the qb² pure-encoder accounting (SAM patch 16 -> downsample 4 -> (s/64)²); the sequence ledger adds qb+1 or q·nh seps. num_image_tokens = 273 + local.
- **Small-gate and quirk**: `image.size[0] <= 640 and image.size[1] <= 640` — 500×700 tiles (one dimension over), 640×640 exactly does not (<=). cropping=False skips the gate entirely.
- **min_crops=1 quirk**: (1,1) enters the set -> a large square image can select it with diff 0 -> crop_ratio (1,1) -> `if nw>1 or nh>1` false -> the single tile produced by dynamic_preprocess is **discarded** (images_crop_raw never enters the sequence) — local ledger 0.
- numq's ceil: `ceil((image_size//patch)/down)` — for all supported sizes (512/640/1024/1280 divisible by 64) ceil equals floor; the `//down` mutation is equivalent (SURVIVED; replaced with a substantive downsample×2 mutation, caught). Judging anchors are unaffected.

## Test Design

- Oracle = explicit dict enumeration + hand-written closest match + row/col double-loop boxes (different style from solve's single-k loop).
- All hand-computed anchors are integers: 893/693/883/903/273/73/111/421.
- The tie double (same aspect, different areas) pins two mutations separately: closest-strict-le (<=) and closest-area-thresh (0.5 -> 4.0).

## Mutations (13/13 caught + 1 equivalent)

- Caught: small-gate and->or / <=-><, ratio set max/min bounds, closest strictness, area-steal threshold, box row/col swap, resize nw/nh swap, global sep removal, local or->and, local sep removal, q halved, total-1.
- 1 equivalent mutation (SURVIVED, recorded): ratios-ij-n-bound — relaxing i/j≤n to ≤max_num yields the same set (absorbed by the joint min/max constraints).
- **Lesson**: the numq-floor mutation (ceil -> floor + remainder compensation) is genuinely equivalent on the parameter domain (is//ps always divisible by down) — a SURVIVED result is not bad news; equivalents also get recorded with their accounting.

## Pitfalls

- The README "vision tokens" and the sequence ledger are two coexisting accountings (differing by seps); assertions must name which one.
- set-iteration-order dependence: solve and oracle ratio sequences must be isomorphic (set comprehension vs dict enumeration); backed by 200 randomized comparisons.
- Boundary with 674 (TMRoPE) / 675 (Qwen2-Audio token ledger): this problem owns DeepSeek-OCR's five-preset resolution/tile/token sequence ledger; `mm-qwen25vl-window-index` and `mm-image-token-expand` already own the Qwen VL resolution accounting (todos red line).
