# KL-Loss Variance Voting Fusion Ledger — Implementation Notes

## Source

- Official repo: ethanhe42/KL-Loss (yihui-he/KL-Loss redirects here), Caffe2 Detectron lineage.
  Paper: *Bounding Box Regression with Uncertainty for Accurate Object Detection* (CVPR'19, arXiv 1809.08545).
- Path: `detectron/utils/py_cpu_nms.py::soft`, the `cfg.STD.METHOD == 'stdiou'` branch.
  Call chain: `detectron/core/test.py::box_results_with_nms_and_limit` → with `cfg.STD_NMS`,
  `py_nms_utils.soft(dets_j, confidence=confs_j)` where `confs_j = confs[inds, j*4:(j+1)*4]`
  are **4 sigmas per box per class** (per-coordinate localization variance from the KL-Loss head).
- Note: `detectron/utils/boxes.py::box_voting` is the stock Detectron voting (score-weighted
  average, no sigma) — NOT this problem's spec; variance voting lives only in the stdiou
  branch of `py_cpu_nms.soft`.

## Line-by-line ledger (stdiou branch)

```python
def soft(dets, confidence=None, ax=None):
    thresh = cfg.STD_TH
    if cfg.STD.METHOD == 'stdiou' and thresh > .1:
        thresh = 0.01                      # 1. loose thresh clamped
    sigma = .5                             # 2. decay width hardcoded
    ...
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 3. legacy VOC +1 convention
    ious = np.zeros((N,N))                 # 4. precomputed, swapped only
    ...
    while i < N:
        maxpos = dets[i:N, 4].argmax()     # 5. first max in slot order
        maxpos += i
        dets[[maxpos,i]] = dets[[i,maxpos]]          # 6. pick swap (x4)
        confidence[[maxpos,i]] = ...
        ious[[maxpos,i]] = ...; ious[:,[maxpos,i]] = ...
        ovr_bbox = np.where((ious[i, i:N] > thresh))[0] + i  # 7. strict >
        if cfg.STD.METHOD == 'stdiou':
            p = np.exp(-(1-ious[i, ovr_bbox])**2/cfg.STD.IOU_SIGMA)
            dets[i,:4] = p.dot(dets[ovr_bbox,:4] / confidence[ovr_bbox]**2) \
                       / p.dot(1./confidence[ovr_bbox]**2)     # 8. fusion
        pos = i + 1
        while pos < N:
            if ious[i, pos] > 0:           # 9. ANY positive overlap
                ovr = ious[i, pos]
                dets[pos, 4] *= np.exp(-(ovr*ovr)/sigma)
                if dets[pos, 4] < 0.001:   # 10. tail-swap removal
                    dets[[pos,N-1]] = ...; confidence[[...]]; ious[[...]]
                    N -= 1; pos -= 1
            pos += 1
        i += 1
```

Notes: thresh clamp means any loose STD_TH behaves as 0.01 at inference; the decay
kernel width 0.5 is a different constant from STD.IOU_SIGMA; the +1 convention
applies to inter and areas but not the denominator; the greedy pick is the first
max in slot order and the pick swap mutates active-box relative order; voters use
strict `>` with self included; fusion is per-coordinate with
`coord_j = Σ_k p_k·v_kj/σ_kj² / Σ_k p_k/σ_kj²`, `p_k = exp(−(1−IoU)²/σ_IoU)`;
the suppression gate is ANY positive overlap with score ×= exp(−IoU²/0.5) and
tail-swap removal below 0.001.

## Key subtleties

1. **Slot-order semantics (deepest trap)**: on ties the pick is the first max in slot order,
   and both the pick swap and tail-swap removal rewrite that order. Random testing caught:
   8 boxes, boxes 1 and 5 tied at 0.217 with zero overlap — pick order [6,7,3,2,0,5,1,4]
   puts 5 before 1 solely because round-5's swap(4,6) reshuffled slots. Simulating by
   original-index order answers ties wrong; the oracle must fully simulate the swaps.
2. **Score never enters the vote weight**: the final paper formula (6) and the official
   code both omit any p_i^γ score term; score only acts through pick order and decay.
3. **Fused coords are one-shot outputs**: IoU is stale after fusion, but already-picked
   boxes never vote again and unpicked boxes keep original coords, so "stale IoU ×
   original coords" stays internally consistent — which is why the official code can
   skip recomputation.
4. Per-coordinate sigma makes the fusion 4 independent weighted averages; a single box
   fuses back to itself; two boxes with equal p reduce to a sigma-weighted average (p cancels).
5. No clipping: fused coords may leave the image.
6. Repeated-box chain decay: IoU=1 ⇒ each round ×exp(−2)≈0.135; starting at 0.5 the
   4th round crosses the 0.001 floor.

## Problem spec

- Inputs: dets (N×5), confs (N×4 sigmas), thresh, iou_sigma; fixed METHOD='stdiou',
  soft decay, σ_soft=0.5, floor=0.001.
- Ledger: ious (initial matrix), steps (idx = original index, box = fused coords,
  voters/weights recorded sorted by original index — the official accumulation runs in
  slot order, float diff ~1e-16, sorting is ledger normalization), final (processing
  order with decayed scores), removed (tail-swapped-out original indices in time order).
- Validation: confs same length as dets, 4 positive finite sigmas per row; dets 5 finite
  columns; iou_sigma>0; finite thresh. Empty input is legal (empty ledger).
- Oracle: independent slot simulation (order list + full pick/tail swap, direct original
  matrix lookup, score/box dicts keyed by original index) + 200 random cross-checks
  (seed 705, thresh ∈ {0.01,0.05,0.2,0.3,0.5}, iou_sigma ∈ {0.1,0.5,1.0}).

## Equivalent-mutation notes

- `sorted(voters)` ledger normalization vs the official slot-order accumulation is
  equivalent under rel_tol 1e-9 (addition commutativity/associativity).

## Mutation testing

14/14 caught: inter/area minus +1, disabled thresh clamp, reversed voter gate, unsquared
(1−iou), 1/σ²→1/σ, denominator missing p, soft σ 0.5→0.3, floor 0.001→0.01, decay gate
changed to thresh, global argmax, ties pick larger, fused-coord clipping, slot-index ledger.
