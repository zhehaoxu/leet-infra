# Vision Detection: Multi-Scale Anchor Generation and IoU Matching

## Problem

One-stage detectors (SSD, arXiv:1512.02325; RetinaNet, arXiv:1708.02002) tile a set of preset boxes (anchors) over every cell of a feature map, then decide each anchor's role — positive, negative or ignore — from its IoU with the ground-truth boxes. That step is called anchor matching and sits right before the detection loss: if it is wrong, no classification head can recover the boxes.

You must implement both halves yourself:

- **anchor generation**: for a list of feature-map sizes, strides, scales and aspect ratios, emit all anchors in a fixed order and clamp out-of-image coordinates back inside the image;
- **IoU threshold matching**: for every anchor pick the best-overlapping GT, label it positive / negative / ignore by two thresholds, and add the SSD-style **guarantee rule** — every GT keeps at least one positive anchor even when all anchors fall below the positive threshold.

The pitfalls are "order" and "degeneracy": the enumeration order (level → row → column → scale → ratio) fixes the anchor indices, and those indices decide which anchor the guarantee rule keeps; boxes are built first and clamped afterwards, so clipped boxes may degenerate yet still take part in IoU; with no GT you still have to return the full anchor list instead of short-circuiting to an empty result. CPU only, deterministic, no random numbers.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring, which is the sole judging contract.

- **Generation order** is fixed: level (input order) → row \(y\) (from 0) → column \(x\) (from 0) → scale \(s\) (input order) → aspect ratio \(r\) (input order). The center of cell \((x,y)\) in level \(l\) is \( \left((x+0.5)\cdot\mathrm{stride}_l,\ (y+0.5)\cdot\mathrm{stride}_l\right) \); it depends only on the stride, never on the feature-map size.
- Each \((s,r)\) pair yields width \( s\sqrt{r} \) and height \( s/\sqrt{r} \), centred on the cell and not rescaled per level. Build the full box first, then clamp each of the four coordinates into \( [0, W_{\text{img}}] \) and \( [0, H_{\text{img}}] \). A clamped box may have zero width or height; it is still kept (never dropped or deduplicated).
- The anchor count is \( A=\sum_l H_l W_l \cdot |\mathrm{scales}|\cdot|\mathrm{ratios}| \) in the order above; `anchors` is \( (A,4) \) with the same dtype as `gt_boxes` (float32 or float64).
- **IoU convention**: intersection uses the clamped coordinates and \( \mathrm{IoU}=\frac{\text{inter}}{S_a+S_g-\text{inter}} \); when both areas are zero the IoU is 0 (no 0/0).
- **Matching** per anchor:
  1. Take the GT with the largest IoU, giving `best_gt` and `best_iou`; an exact tie goes to the smallest GT index.
  2. `best_iou >= pos_thr` → positive, labelled with the matched GT index; `best_iou < neg_thr` → negative, labelled -1; otherwise (\( \mathrm{neg\_thr}\le\mathrm{best\_iou}<\mathrm{pos\_thr} \)) → ignore, labelled -2.
  3. The **guarantee rule runs after step 2**: for each GT \(g\), among the anchors whose `best_gt == g` take the one with the largest `best_iou` (ties → smallest anchor index) and force it positive with label \(g\), even if its IoU is below `pos_thr`. If no anchor points at that GT, skip it.
- When `gt_boxes` has 0 rows, the anchors are still generated in full, every label is -1, and the positive/ignore bins of the counts are 0.
- `counts` is an int64 tensor of length 3 holding `[positives, negatives, ignores]`, summing to \(A\).
- Invalid input raises `ValueError`: `gt_boxes` not a float32/float64 tensor, wrong shape (not \( (G,4) \)), empty or mismatched level lists, non-positive feature-map size or stride, empty `scales`/`ratios` or non-positive entries, `image_size` not two positive integers, and thresholds violating \( 0\le \mathrm{neg\_thr}\le \mathrm{pos\_thr} \).

## Example

One \(2\times2\) feature map, stride 8, scale 16, aspect ratio 1.0, image \(32\times32\); a single GT box \( [4,4,20,20] \):

```
anchor 0: cell(0,0) centre(4,4)    box(-4,-4,12,12) clamped (0,0,12,12)  IoU =  64/336 ≈ 0.190476
anchor 1: cell(1,0) centre(12,4)   box( 4,-4,20,12) clamped (4,0,20,12)  IoU = 128/320 =  0.4
anchor 2: cell(0,1) centre(4,12)   box(-4, 4,12,20) clamped (0,4,12,20)  IoU = 128/320 =  0.4
anchor 3: cell(1,1) centre(12,12)  box( 4, 4,20,20) clamped (4,4,20,20)  IoU = 256/256 =  1.0
```

Anchors 1 and 2 have an IoU **exactly equal** to `neg_thr=0.4`: the ignore band is \( \mathrm{neg\_thr}\le\mathrm{best\_iou}<\mathrm{pos\_thr} \), the equality belongs to ignore, so they become -2 rather than -1 (float32 evaluates these two as 0.40000000596, still inside the band). Anchor 0's 0.190476 is below 0.4 → -1, and anchor 3's 1.0 → 0:

```
anchors = [[0,0,12,12], [4,0,20,12], [0,4,12,20], [4,4,20,20]]
labels  = [-1, -2, -2, 0]
counts  = [1, 1, 2]
```

The guarantee rule shows up when no anchor reaches `pos_thr`. Replace the GT with a tiny box at the centre, `[(15.9,15.9,16.1,16.1)]`; the same anchors then have IoUs 0, 0, 0, 0.00015625 (the first three do not overlap it at all). All are below 0.5 and below 0.4, so thresholds alone would mark every anchor negative; the guarantee rule forces the best anchor 3 positive:

```
labels = [-1, -1, -1, 0]
counts = [1, 3, 0]
```

## Constraints

- Return `(anchors, labels, counts)`: `anchors` is \( (A,4) \) with the dtype of `gt_boxes`, `labels` is \( (A,) \) int64, `counts` is \( (3,) \) int64.
- Judging uses absolute tolerance \(10^{-5}\) (float32) / \(10^{-12}\) (float64) on `anchors` and `torch.equal` on `labels` and `counts`.
- The enumeration order, centre formula, clamping bounds, the zero-area IoU convention, the timing of the guarantee rule and its tie-breaking are all part of the contract; an "equivalent" but differently ordered enumeration will be rejected.
- CPU, deterministic, no random numbers; the no-GT branch and other degenerate inputs must follow the conventions above rather than raising.
