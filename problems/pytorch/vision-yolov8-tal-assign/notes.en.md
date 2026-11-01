# YOLOv8 TAL Task-Aligned Assignment Ledger — Implementation Notes

## Source

- Repo: ultralytics/ultralytics main branch, `ultralytics/utils/tal.py`
  `TaskAlignedAssigner` (single image, all GTs valid, topk2==topk so the
  secondary filter is off).
- Paper: TOOD (2108.07755) introduced task alignment; YOLOv8 (2301.05495)
  adopts it.
- Version notes: current code adds stride_val small-side enlargement and
  topk2; forward has **no topk=min(topk,na) clamp** (torch.topk raises
  RuntimeError when na<topk); the count>1 zeroing mainly serves invalid
  GT rows.

## Line-by-line ledger

```python
# select_candidates_in_gts: small-side enlargement + strict eps margins
wh_mask = gt_xywh[..., 2:] < self.stride_val        # strict <
gt_xywh[..., 2:] = where(wh_mask & mask_gt, stride_val, wh)   # about center
mask = (xc - lt_x > eps) & (yc - lt_y > eps) & (rb_x - xc > eps) & (rb_y - yc > eps)

# get_box_metrics
bbox_scores = pd_scores[b, anchor, gt_class]        # gt-class slot score
overlaps = CIoU(gt, pd).clamp_(0)                   # negatives -> 0
align_values = scores^alpha * overlaps^beta         # 1.0 / 6.0

# select_topk_candidates
topk(metrics, self.topk, largest=True)              # no clamp
count_tensor.scatter_add_(...); count_tensor[count>1] = 0

# select_highest_overlaps
max_overlaps_idx = overlaps.max(1).indices          # arbitrate by overlap!
is_max_overlaps.scatter_(1, idx, 1)                 # ties -> first max
mask_pos = where(mask_multi_gts, is_max_overlaps, mask_pos)

# _forward normalization
align_metric *= mask_pos
pos_align_metrics = align_metric.amax(-1, keepdim=True)   # per gt
pos_overlaps = overlaps.amax(-1, keepdim=True)            # per gt
align_metric.mul_(pos_overlaps).div_(pos_align_metrics + eps)
norm_align_metric = align_metric.amax(-2).unsqueeze(-1)   # per anchor
target_scores = target_scores * norm_align_metric         # soft labels
```

## Key subtleties

1. **Small-side enlargement**: `wh < stride_val` (strict; w==stride_val
   makes enlargement an identity — the `<=` mutation is equivalent),
   **symmetrically about the center** ([cx−s/2, cx+s/2], not edge-aligned).
2. **Strict eps margins on all four sides**: `delta > eps`; equality is out.
   With x1=0 an anchor at x=1e-9 compares exactly equal to eps — a
   constructible boundary anchor.
3. **bbox_scores come from the gt-class slot**: `pd_scores[anchor,
   gt_class]`, not the max class. Input pd_scores are post-sigmoid
   probabilities (v8DetectionLoss passes `.sigmoid()`); the problem
   takes probabilities and validates [0,1].
4. **CIoU clamped to 0**: CIoU can be negative (the −ρ²/c² term) → 0 →
   align=0. Fully-overlapping boxes (iou=1, v=0) hit 0/0 — official torch
   yields NaN (clamp does not clean NaN); this problem guards v=0 with
   alpha=0 → CIoU=1, noted as a deliberate deviation.
5. **beta=6 power**: overlap 0.5 → 0.0156; align's dynamic range is huge.
   **eps swallowing**: when pos_align < eps (1e-9), norm =
   align·po/(pa+eps) ≈ align·po/eps ≪ po — soft labels get halved or
   worse. With beta=6, align~1e-10 is common, so this is a real effect,
   not a numerical decoration.
6. **top-k AND in-gts**: topk is chosen first (padding with zero-metric
   slots), then `mask_pos = topk · in_gts · mask_gt` drops out-of-box
   slots. The ledger records topk and pos separately.
7. **Arbitration by overlap, not align**: `overlaps.max(1)` — a GT with
   larger align but smaller IoU loses. Ties go to the first max (smaller
   gt index; torch.max dim semantics guarantee first-on-CPU).
8. **Normalization chain order**: align ×= mask_pos (post-arbitration) →
   pos_align/pos_overlap (per-gt max over anchors) → align·po/(pa+eps) →
   per-anchor max over gts → into the one-hot slot. Unassigned anchors
   get norm=0, scores all 0, idx −1 in the ledger.
9. **Garbage labels in get_targets**: unassigned anchors read gt 0's label
   (flatten index 0) but scores are zeroed by fg_mask; the ledger records
   −1 when fg=False.

## Problem spec

- Inputs: pd_scores (A×C probabilities), pd_bboxes (A×4 xyxy), anc_points
  (A×2), gt_labels (G), gt_bboxes (G×4 xyxy), num_classes, topk=13,
  alpha=1.0, beta=6.0, stride_val=16.0, eps=1e-9.
- Validation: G≥1, A≥1, **A≥topk** (official raises on na<topk), row
  lengths, probabilities in [0,1], ordered xyxy, class in range,
  stride_val>0, eps>0.
- Ledger: expanded_boxes, mask_in_gts, bbox_scores, overlaps, align_metric,
  mask_topk, mask_pos (pre-arbitration), mask_pos_arb, fg_mask,
  target_gt_idx (−1 unmatched), pos_align_metrics, pos_overlaps,
  norm_align_metric, target_scores (A×C soft labels).
- Oracle: explicit enumeration + 200 random cross-checks (seed 708,
  topk∈{3,5,8}, stride_val∈{8,16}).

## Equivalent-mutation notes

- **Enlargement threshold `<` → `<=` is equivalent**: when w==stride_val
  the enlargement is an identity transform (same center, same width).

## Mutation testing

15 caught + 1 equivalent: edge-aligned enlargement, `<=` threshold,
`>=` margins (exact-eps anchor x1=0, ax=1e-9), CIoU without clamp, linear
align, max-class score, reversed topk, missing in-gts AND, arbitration by
align / larger-gt ties / no arbitration, norm without eps / wrong numerator /
extra po division, hard-coded soft label 1, unmatched recorded as gt0.
