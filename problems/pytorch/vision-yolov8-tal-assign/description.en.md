# YOLOv8 TAL Task-Aligned Assignment Ledger

## Problem

YOLOv8 drops anchors and IoU-threshold assignment in favor of TAL: the
alignment score `align = cls_score^alpha × CIoU^beta` measures "how
well-classified" and "how well-localized" at once, each GT takes the
top-k in-box anchors by align, multi-gt conflicts arbitrate by overlap,
and the normalized align becomes a **soft label** in the target
classification scores. Given predictions and GTs, reproduce the official
`TaskAlignedAssigner` ledger (single image, all GTs valid).

Given:
- `pd_scores`: A x C predicted probabilities (post-sigmoid)
- `pd_bboxes`: A x 4 xyxy
- `anc_points`: A x 2 anchor centers
- `gt_labels`: G ints; `gt_bboxes`: G x 4 xyxy
- `num_classes`, `topk` (default 13), `alpha` (1.0), `beta` (6.0),
  `stride_val` (16.0), `eps` (1e-9)

## Processing chain

1. **Center candidates**: GT sides below `stride_val` are enlarged
   **symmetrically about the center** to stride_val; the anchor center
   must satisfy all four directional deltas **> eps** (equality out).
2. **Alignment score**:
   - `bbox_scores[k] = pd_scores[k][gt_class]` (gt-class slot, not max);
   - `overlap = CIoU(gt, pd)` **clamped to 0**;
   - `align = score^alpha · overlap^beta` (beta=6 → brutal decay).
3. **top-k**: per gt the topk highest aligns (ties → smaller anchor
   index); `mask_pos = topk ∧ center-candidates`.
4. **Multi-gt arbitration**: an anchor claimed by ≥2 gts keeps the one
   with the **highest overlap** (not align; ties → smaller gt index).
5. **Soft-label normalization**:
   - `pos_align[g] = max_k align[g][k]` over assigned anchors;
     `pos_overlap[g] = max_k overlap[g][k]`;
   - `norm[k] = max_g align[g][k]·pos_overlap[g]/(pos_align[g]+eps)`
     (only for anchors assigned to g; **eps swallows rows with
     align < eps**);
   - `target_scores[k][gt_class] = norm[k]` (other slots 0; unassigned
     anchors all 0).

## Output ledger

```json
{
  "expanded_boxes":    [[x1,y1,x2,y2]×G],   // after enlargement
  "mask_in_gts":       [[bool×A]×G],
  "bbox_scores":       [[f×A]×G],
  "overlaps":          [[f×A]×G],
  "align_metric":      [[f×A]×G],
  "mask_topk":         [[0|1×A]×G],
  "mask_pos":          [[0|1×A]×G],         // pre-arbitration
  "mask_pos_arb":      [[0|1×A]×G],
  "fg_mask":           [bool×A],
  "target_gt_idx":     [int×A],             // -1 unmatched
  "pos_align_metrics": [f×G],
  "pos_overlaps":      [f×G],
  "norm_align_metric": [f×A],
  "target_scores":     [[f×C]×A]
}
```

## Input constraints

- G≥1, A≥1, **A≥topk** (official raises on na<topk); pd_scores rows of
  length C with values in [0,1]; bbox rows of 4 finite values with
  x1<x2, y1<y2; class∈[0,C); anc_points rows of 2 finite values;
  alpha/beta≥0; stride_val/eps>0; violations raise `ValueError`.

## Example

GT `[10,10,40,40]`, anchor27 center (32.5,32.5) predicting a box identical
to the GT with class score 0.9: CIoU=1, align=0.9 → top of topk;
pos_align=0.9, pos_overlap=1 → norm=0.9·1/(0.9+1e-9)≈1 →
`target_scores[27][0]≈1`. GT `[10,10,12,60]` (w=2<16) enlarges
symmetrically to `[3,10,19,60]`.
