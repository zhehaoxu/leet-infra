# YOLOX SimOTA Dynamic Assignment Ledger — Implementation Notes

## Source

- Repo: Megvii-BaseDetection/YOLOX.
- Paths: `yolox/models/yolo_head.py::get_assignments` (calls
  `get_geometry_constraint`, builds the cost, calls `simota_matching`) +
  `yolox/utils/boxes.py::bboxes_iou` (`xyxy=False` path; GT and pred are
  both cxcywh).
- Paper: YOLOX (2107.08430); SimOTA is the simplified OTA (2108.07755).

## Line-by-line ledger

```python
# get_geometry_constraint
x_centers = ((x_shifts + 0.5) * expanded_strides)      # anchor centers
center_radius = 1.5
center_dist = expanded_strides * center_radius          # radius ∝ stride!
c_l = x_centers - (gx - center_dist)                    # 4-direction deltas
...
is_in_centers = center_deltas.min(dim=-1).values > 0.0  # strict >
anchor_filter = is_in_centers.sum(dim=0) > 0            # any-gt neighborhood

# get_assignments
pair_wise_ious = bboxes_iou(gt, pred, False)            # cxcywh
pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
cls_preds_ = (cls_preds.sigmoid_() * obj_preds.sigmoid_()).sqrt()
pair_wise_cls_loss = F.binary_cross_entropy(...).sum(-1)  # per-class sum
cost = cls_loss + 3.0 * ious_loss + 1e6 * (~geometry_relation)

# simota_matching
n_candidate_k = min(10, n_anchor)
topk_ious = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # int truncation
for gt: matching_matrix[gt][topk(cost[gt], k, largest=False)] = 1
anchor_matching_gt = matching_matrix.sum(0)
if (anchor_matching_gt > 1).any():                      # arbitration
    cost_argmin = torch.min(cost[:, mask], dim=0)       # keep cheapest gt
    matching_matrix[:, mask] *= 0
    matching_matrix[cost_argmin, mask] = 1
```

## Key subtleties

1. **Radius scales with the anchor's own stride**: `1.5·stride`. For the
   same GT, stride-8 anchors see a ±12px window, stride-32 ±48px. Unlike
   FCOS's fixed radius and TAL's lack of a geometry gate. Anchor center =
   `(shift+0.5)·stride` (right half-cell).
2. Strict `>0` geometry gate: an anchor exactly on the boundary is out.
3. **cxcywh IoU `en` semantics**: `en = (tl<br).prod(2)`; any axis not
   strictly less → IoU is exactly 0. Touching edges (tl==br) give
   area_i = 0 — same value as a max(0,·) style implementation (the `<=`
   mutation is equivalent: area_i = 0·x = 0; no +1, no eps anywhere).
   Areas are `prod(wh)`.
4. **cls score composition**: `sqrt(σ(cls)·σ(obj))` per class; BCE against
   one-hot summed over classes (`sum(-1)` — negatives contribute log(1−p)).
5. **Cost triple**: cls + 3.0·(−log(iou+1e-8)) + 1e6·(~geometry). The 1e6
   only applies to geometry-missing anchors *inside the candidate set*
   (geometry_relation is G×A_cand; a candidate may still miss a specific gt).
6. **dynamic_k**: `clamp(int(Σ top-10 IoU), min=1)` — int truncation:
   4.585→4 (round would give 5). With fewer than 10 candidates all are
   taken (zero-padded topk).
7. **Per-gt picks**: k lowest costs (largest=False). Ties go to the smaller
   candidate index in this problem's spec (pure-Python stable sort;
   torch.topk tie order is unspecified — anchor test cases avoid cost ties).
8. **Arbitration**: an anchor matched by ≥2 gts is fully cleared and keeps
   only its cheapest gt (min dim=0; ties → smaller gt index per spec).
9. **Retirement semantics**: anchors failing anchor_filter never enter the
   candidate set (matching_matrix has n_cand columns); final positives =
   geometry candidates ∩ chosen by some gt (the `fg_mask[fg_mask.clone()] =
   fg_mask_inboxes` write-back).

## Problem spec

- Inputs: gt_bboxes (cxcywh pixels), gt_classes, expanded_strides/x_shifts/
  y_shifts (A), cls_preds (A×C raw logits), obj_preds (A raw), bboxes_preds
  (A×4 cxcywh), num_classes.
- Validation: G≥1, A≥1, 4 finite columns, wh>0, class∈[0,C), stride≥1,
  cls row length == num_classes; violations raise ValueError.
- Ledger: anchor_filter/is_in_centers, ious, cost, dynamic_ks,
  matching_matrix (G×n_cand after arbitration), fg_mask (A), num_fg,
  matched_gt_inds/gt_matched_classes (A, −1 = unmatched),
  pred_ious_this_matching (A, 0.0 = unmatched).
- Oracle: matrix-style direct computation (explicit sorts, no topk/argmin
  abstractions) + 200 random cross-checks (seed 706, G∈1..3, A∈4..15,
  C∈2..4, stride∈{8,16,32}).

## Equivalent-mutation notes

- **IoU touching-edge `<=` mutation is equivalent**: with one axis touching,
  area_i = 0·h = 0; both branches yield iou = 0 with the same denominator.

## Mutation testing

15 caught + 1 equivalent: radius 1.5→2.5, radius not stride-scaled, min>0→>=0,
IoU +1, cls without sqrt, obj without sigmoid, iou_loss without eps, weight
3.0→1.0, penalty 1e6→1e5, int→round (boundary anchor: top-10 sum 4.585),
floor 1→0, top-10→top-5 (top-5 sum 3.1 → k=3), arbitration picks max, no
arbitration, per-gt picks max.
