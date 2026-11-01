# YOLOX SimOTA Dynamic Assignment Ledger

## Problem

YOLOX drops the anchor set and fixed label assignment: SimOTA decides
dynamically, during training, which prediction cells each GT should
supervise. The assignment has three stages: a geometry pre-filter
(center neighborhood), a cost matrix (cls + IoU), and per-gt dynamic-k
lowest-cost picks with conflict arbitration. Given GTs, the anchor grid
and predictions, reproduce the official `get_assignments` ledger.

Given:
- `gt_bboxes`: G x 4 cxcywh (pixels)
- `gt_classes`: G ints (0..num_classes−1)
- `expanded_strides / x_shifts / y_shifts`: A each (anchor grid points)
- `cls_preds`: A x C raw logits (pre-sigmoid)
- `obj_preds`: A raw logits
- `bboxes_preds`: A x 4 cxcywh (pixels)
- `num_classes`: int

## Processing chain

1. **Geometry constraint** (get_geometry_constraint):
   - anchor center `= (shift + 0.5)·stride` (each anchor uses its own
     stride);
   - neighborhood = GT center ± `1.5 · that anchor's stride` (**radius
     scales with stride**, not fixed pixels);
   - hit = min of the four directional deltas **> 0** (strict; on the
     boundary is out);
   - `anchor_filter` = anchors inside ANY gt's neighborhood; only they
     become candidates.
2. **Candidate IoU** (bboxes_iou xyxy=False, cxcywh):
   `tl = max(c−w/2)`, `br = min(c+w/2)`, areas = w·h (no +1, no eps);
   **if any axis fails strict `tl<br` → IoU = 0** (touching does not count).
3. **Cost** (G x n_cand):
   - cls score `p = √(σ(cls_logit)·σ(obj_logit))` per class;
   - cls term = Σ_c BCE(p_c, one_hot(gt_class)) (all C terms);
   - IoU term = 3.0·(−log(iou + 1e-8));
   - geometry term = 1e6 when the anchor is outside that gt's
     neighborhood, else 0.
4. **dynamic-k**: per gt, sum of its top-10 (all if fewer) IoUs,
   `k = max(int(sum), 1)` (**int truncation, not round**); pick the k
   lowest-cost candidates (ties → smaller index).
5. **Arbitration**: a candidate chosen by ≥2 gts is cleared and keeps only
   the cheapest gt (ties → smaller gt index).
6. **Output**: final positives = candidates chosen by some gt; matched
   gt/class/IoU are emitted per anchor (−1/−1/0.0 when unmatched).

## Output ledger

```json
{
  "anchor_filter": [bool×A],
  "is_in_centers": [[bool×A]×G],
  "ious":  [[f64×n_cand]×G],          // candidate order
  "cost":  [[f64×n_cand]×G],
  "dynamic_ks": [int×G],
  "matching_matrix": [[0|1×n_cand]×G], // after arbitration
  "fg_mask": [bool×A],
  "num_fg": int,
  "matched_gt_inds": [int×A],          // -1 = unmatched
  "gt_matched_classes": [int×A],
  "pred_ious_this_matching": [f64×A]   // 0.0 = unmatched
}
```

## Input constraints

- G≥1, A≥1; bbox rows have 4 finite values with w/h>0; class∈[0,C);
  stride≥1; cls row length == num_classes; list lengths consistent;
  violations raise `ValueError`.

## Example

GT `(100,100,20,20)`, anchor `(stride=8, x_shift=12, y_shift=12)`:
center (100,100), dist=12 → hit; concentric pred
`(100,100,14.142,14.142)` → IoU=0.5 (pred ⊂ gt, iou = p²/400). If it is
the only candidate: dynamic_k = max(int(0.5),1) = 1, cost = Σ_c BCE +
3·(−log(0.5+1e-8)) → the anchor is chosen, `num_fg=1`,
`matched_gt_inds=[0]`.
