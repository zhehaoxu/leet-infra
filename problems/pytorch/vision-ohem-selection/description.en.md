# OHEM Hard-Example Selection Mask Ledger

## Background

Online Hard Example Mining (1604.03540, CVPR 2016) replaces uniform sampling with loss-based hard-example selection. The original caffe repo (abhi2610/Ohem-caffe) is gone (404); the modern authoritative transcription is **mmseg `OHEMPixelSampler`** (pixel-level, two branches) with **mmdet `OHEMSampler`** (RoI-level topk) as RoI reference. This problem transcribes the mmseg pixel-level two-branch ledger.

## Exact Ledger (two branches)

1. **batch_kept = min_kept × batch_size**.
2. **Loss branch** (no threshold): stable descending sort of losses (ties keep the earlier index), mark top `batch_kept` with mask=1; if quota > n, keep all.
3. **Threshold branch**: input is the **correct-class probability** probs (not loss):
   - Sort ascending; `min_threshold = sort_prob[min(batch_kept, n−1)]` (0.0 if n=0);
   - **threshold = max(min_threshold, t)** — the rank backstop guarantees at least batch_kept kept, t guarantees nothing harder than t is missed; take the max of the two lower bounds;
   - `mask = probs < threshold` (strict; equal does NOT keep).
4. Output mask + final threshold (None for the loss branch).

## Constraints

1. min_kept ≥ 1, batch_size ≥ 1, t ∈ [0,1], probs ∈ [0,1], losses ≥ 0 finite.
2. Branches are mutually exclusive: with thresh given pass probs (losses=None), otherwise vice versa.
3. No input mutation; pure sorting/comparison, no float accumulation.

## Data Scale

- n ≤ 10⁵ (O(n log n)); min_kept ≤ 10⁵; batch ≤ 64.

## Acceptance Anchors (excerpt)

- probs `[0.04,0.01,0.05,0.02,0.03]`, t=0.03, min_kept=3 → threshold=0.04, mask `[0,1,0,1,1]`.
- probs `[0.1..0.5]`, t=0.25, min_kept=3 → **rank backstop 0.4 wins**, threshold=0.4, mask `[1,1,1,0,0]`.
- losses `[1,3,2,3,0.5]`, quota 3 → mask `[0,1,1,1,0]` (both 3.0s in, 2.0 fills).
