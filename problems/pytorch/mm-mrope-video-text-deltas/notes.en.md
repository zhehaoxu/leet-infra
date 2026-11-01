# Multimodal: Qwen2.5-VL Full M-RoPE Index (Video Temporal and Interleaving)

## Background

Qwen2-VL / Qwen2.5-VL splits rotary position embeddings into temporal / height / width axes: text tokens increment all three axes in lockstep, image tokens expand along a (t, h, w) grid, and video tokens additionally quantise the temporal axis by "seconds per grid × tokens per second". The library's simplified `mm-mrope-position-ids` (order 228) only covers Qwen2-VL text plus a single contiguous image block; this problem implements the full accounting of transformers `Qwen2_5_VLForConditionalGeneration.get_rope_index` (v4.51.3): video temporal quantisation, multi-image / video interleaved segmentation, `mrope_position_deltas`, and padding handling.

## Task

Given a single token sequence, a 0/1 attention mask, image and video `grid_thw` lists, per-video `second_per_grid_ts`, `spatial_merge_size` and `tokens_per_second`, output:

- `position_ids`: a `[3][L]` int nested list (temporal / height / width rows);
- `mrope_position_delta`: a single int, the offset that maps `cache_position` back into M-RoPE coordinates during decoding.

## Mechanism ledger

1. **Counting**: only an `image_token` / `video_token` directly following `vision_start` counts as a segment (official `vision_tokens = input_ids[vision_start_indices + 1]`). Bare vision tokens are not counted, but the scanning `index` call still finds them — the two criteria disagree, an official quirk preserved verbatim (see `check_bare_image_token_quirk`).
2. **Segment scan**: loop `image_nums + video_nums` times; each round finds the first image / video token from `st` and processes whichever appears first. Within a segment, emit the text part first (all three axes `arange(text_len) + st_idx`, where `st_idx` is the global max over all previously emitted axes + 1), then the vision part (temporal axis `floor(i × second_per_grid_t × tokens_per_second)`, height / width axes expanded in C order, all three offset by `text_len + st_idx`). `st` advances to `ed + llm_grid_t·llm_grid_h·llm_grid_w`.
3. **Image temporal axis is constant**: image segments use `second_per_grid_t = 0`, so even with `grid_t = 2` the temporal axis stays at one value (pinned by `check_image_temporal_axis_constant`). Only video segments expand the temporal axis.
4. **Truncation semantics**: temporal steps are float-multiplied then truncated with `int()` (mirroring official `time_tensor.long()`). With `second_per_grid_t = 0.5` and `tokens_per_second = 25`, steps 12.5 → 12 and 37.5 → 37; a Python `round` implementation would turn 37.5 into 38 (half-to-even rounds up), which `check_video_floor_half_odd` pins.
5. **Padding**: positions with `mask == 0` carry the value **1**, not 0 — the official buffer is initialised with `torch.ones` and written only at `mask == 1` positions, so the padding 1s are never overwritten (`check_main_path_padding` / `check_pure_text_padding_mask`).
6. **Full-length delta quirk**: `mrope_position_delta = global max + 1 − L` where L is the **full padded length** (official `len(total_input_ids[i])`), not the effective length. More padding means a more negative delta.
7. **Pure-text fallback**: when both grids are empty, the official second branch applies — all three axes are `cumsum(mask) − 1`, `mask == 0` positions again carry 1, delta uses the same formula. The official entry condition is "grid tensor is not None", but on a pure-text sequence the main path and the fallback coincide (with an all-ones mask the main path's trailing segment is exactly `arange`), so this problem unifies them by observable behaviour: both grids empty → fallback.

## Deliberate deviations from the official code

- The official `if image_token_id in input_tokens and remain_images > 0` tests the **whole sequence** while `index(..., st)` searches **from st**; the mismatch crashes when remain > 0 but all vision tokens lie before st. This problem unifies the test to "from st" (consistent with index).
- The official `h // spatial_merge_size` has no divisibility check; a zero quotient yields an empty vision segment and mis-advances st. This problem explicitly requires h and w divisible by merge with quotient ≥ 1, else ValueError.
- Where the official code raises IndexError or produces shape misalignment (segment/grid count mismatch, vision_start as last effective token), this problem raises ValueError explicitly.
- The official code loops over batch rows; this problem fixes a single row (batch dim omitted), with per-token accounts identical.

## Implementation notes

The solution scans in a single loop: locate tokens with `index` over `eff` (the mask-filtered effective sequence), append each segment's text / vision sub-parts to `segments`, then scatter back into `[3][L]` at the mask-true positions. The global max serves both the next segment's `st_idx` and the delta. The h / w C-order expansion uses nested comprehensions matching official `expand(...).flatten()` semantics.

## Independent verification

The oracle rewrites the official single loop into a two-pass structure: first parse into a `("text", n)` / `("vis", t, gh, gw, sec)` segment list, then generate positions segment by segment, recomputing inter-segment offsets as "current global max over all axes + 1", and expressing truncation explicitly with `math.floor`. It shares no intermediate variables with the solution; of the 15 `check_*`, 13 structural cases are compared element-wise, plus input immutability and 16 groups of invalid inputs.

## Coverage list

Pure-text fallback (all-ones mask / trailing padding / interior hole), single-image four-token expansion, image constant-temporal quirk, video floor steps (including the 37.5 boundary), `second_per_grid_ts=None` default 1.0, image-video interleaved offset chain, merge division, bare image token quirk, two-image offset chain, main-path padding, input immutability, invalid-input groups. All 13 observable mutations CAUGHT; one equivalent mutation analysed: `merge-ceil` (under the divisibility precondition, floor and ceil division coincide), replaced by an h/w axis swap mutation.

## Complexity

Let n be the effective length and v the total vision tokens. Scanning and segmentation cost O(n + 3v); scattering the output costs O(3L). Total O(L + v); besides the output only O(v) of segment bookkeeping.

## Source and scope

Based on transformers v4.51.3 `src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py::get_rope_index` (fetched via GitHub API contents + base64; the main branch has since been refactored, so this problem pins the classic v4.51.3 implementation). Out of scope: rotary frequency computation, the `mrope_section` three-axis split, the vision tower's windowed attention (covered by `mm-qwen25vl-window-index`), and the delta consumption path in `prepare_inputs_for_generation`; this is a deterministic replica of the position-index accounting only. Note the token protocol: vision segments must start with `vision_start` — the official processor's sequence convention.

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-mrope-video-text-deltas --mode submit
```
