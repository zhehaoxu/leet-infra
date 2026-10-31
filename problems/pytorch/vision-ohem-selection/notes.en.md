# Notes: vision-ohem-selection

## Source

- Original repo abhi2610/Ohem-caffe is 404 (github code search requires login); the single authoritative source is **mmseg `mmseg/structures/sampler/ohem_pixel_sampler.py`** (main, all 85 lines verified).
- RoI-level reference **mmdet `task_modules/samplers/ohem_sampler.py`**: `hard_mining` = per-sample loss `topk(num_expected)`, keep all when candidates ≤ quota — recorded in notes, not in the problem (mixing RoI-level with the pixel-level two branches in one problem would blur the ledger).

## Key Judgments

- **Threshold is max, not min**: `max(prob at rank batch_kept, t)`. The naive reading "filter by t first, then top-N" is wrong — official semantics takes the max of the two lower bounds (rank backstop and t).
- **Comparison is strict `<`**: prob == threshold is dropped. Hand-anchor slips twice: 0.5 < 0.5 is False, and when the backstop 0.4 > t=0.25 the threshold is 0.4, not 0.25. Even after solution/oracle agreed, hand-verify anchor direction.
- **probs are correct-class probabilities**: sorted ascending (lower = harder) — opposite direction to the loss branch's descending sort; the direction flip is a proposition point.
- Batch dimension: `batch_kept = min_kept × batch_size` (official `self.min_kept * seg_label.size(0)`).

## Boundary

- No real loss computation (per-sample loss scalars are inputs); no RoI sampler (mmdet semantics in notes); no GHM/focal weighting.
- Bounded against `soft-nms` (308): inference-time score decay vs training-time sample selection.

## Test Design

- Oracle `_py` replays independently (per-branch for-loops, no reuse of solution's sorted/key).
- 9 checks: both anchors, backstop-wins, batch scaling, loss tie stability, quota exceeds, 200 random, 7 invalid tiers.
- Mutations 9/9: rank ±1, max→min, strict→non-strict, batch ignored, desc→asc, tie reversed, branch direction, mask inverted, quota +1. `tie-unstable` (dropping tie-break index i) is an equivalent mutation (Python sort is stable); replaced by `tie-reversed` for full catch.
