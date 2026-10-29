# Vision Detection: mAP Computation (11-point / All-point / COCO 101-point)

## Problem

mAP is the most quoted — and most frequently miscalculated — metric in detection. The skeleton is simple: per class, sort predictions by descending confidence, greedily match each against an unmatched GT of the same class by IoU, collect a TP/FP sequence, accumulate a precision-recall curve and integrate it. The "integrate" step, however, has **three different definitions** in the literature, and one PR curve yields three different numbers:

- **all-point** (VOC2010 onwards): a staircase integral over every recall change, \( \sum_i (r_i-r_{i-1})\,p_i \), where \(p_i\) is the right-hand envelope (max precision at or after that point). Closest to "area under the curve".
- **11-point** (VOC2007): sample \( p(r)=\max_{r_i\ge r}p_i \) at \(r=0,0.1,\dots,1.0\) and average. The grid is coarse and re-counts high-precision regions.
- **COCO 101-point**: the 11-point idea with a finer grid, \(r=0,0.01,\dots,1.0\) (101 points).

Implement the matching and all three integrations. Input is a flat "prediction table + GT table", with image ids and class indices deciding what may match. CPU only, deterministic, no random numbers.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring, which is the sole judging contract.

- Predictions and GTs are flat tables: boxes \( (N,4) \) in xyxy, scores \( (N,) \), classes \( (N,) \) and image ids \( (N,) \). Image ids only need to be **equal** to denote the same image; they need not be contiguous or start at 0.
- Matching for one class at one threshold:
  1. its predictions by **descending** score, ties by ascending original index;
  2. take each prediction in turn and, among GT of the same class in the same image that are not yet taken, pick the largest IoU (ties → smallest GT index); if that IoU \( \ge \) the threshold it is a TP and the GT is taken, otherwise an FP.
  3. IoU uses width/height clamped at zero; when both areas are 0 the IoU is 0.
- Matching is redone for **each** IoU threshold (GT occupancy resets between thresholds).
- Precision/recall accumulate from the TP/FP sequence: \( p_i=\frac{\mathrm{TP}_i}{\mathrm{TP}_i+\mathrm{FP}_i} \), \( r_i=\frac{\mathrm{TP}_i}{n_{\text{gt}}} \), where \(n_{\text{gt}}\) is the number of GT of that class (only same-class, same-image boxes can match).
- The three APs:
  - `"all"`: \( \mathrm{AP}=\sum_{i}(r_i-r_{i-1})\,\max_{j\ge i}p_j \) with \(r_{-1}=0\);
  - `"11point"`: \( \mathrm{AP}=\frac{1}{11}\sum_{k=0}^{10} p(r_k) \), \(r_k=k/10\), \(p(r)=\max_{i:\,r_i\ge r}p_i\) (0 if none);
  - `"coco"`: the same with \(r_k=k/100\), 101 points.
- The class set is the classes **present in the GT** (ascending); predicted classes without GT contribute neither columns nor averaging. A class with no predictions at that threshold scores AP 0 (e.g. everything missed).
- Return `ap` \( (T,C) \) and `mean_ap` \( (T,) \), both float64, where \(T\) is the number of thresholds and \(C\) the number of GT classes; with \(C=0\), `ap` is \( (T,0) \) and `mean_ap` is all zeros; with no predictions, `ap` is all zeros.
- Invalid input raises `ValueError`: boxes not float32/float64 or wrongly shaped, prediction scores with a different dtype than the boxes, classes/image ids not integer, mismatched lengths, prediction and GT box dtypes differing, an empty or out-of-range threshold list, or an unknown `method`.

## Example

One class, one image, 3 GT and 4 predictions. In confidence order they are: hit GT0, **miss**, hit GT1, hit GT2:

```
GT   : [0,0,10,10]  [20,20,30,30]  [40,40,50,50]
pred : 0.9 hits GT0 | 0.8 misses (far from every GT) | 0.7 hits GT1 | 0.6 hits GT2
TP/FP:   TP              FP                             TP              TP
prec :   1.0             0.5                            0.666667        0.75
recall:  0.333333        0.333333                       0.666667        1.0
```

The right-hand envelope is \( [1,\ 0.666667,\ 0.75,\ 0.75] \), so with `iou_thrs=[0.5, 0.75]` (both thresholds give the same numbers here, since the boxes either coincide or are disjoint):

```
all      : 1/3 × 1.0 + 1/3 × 0.75 + 1/3 × 0.75 = 0.8333333333
11point  : (4 × 1.0 + 7 × 0.75) / 11           = 0.8409090909
coco     : (34 × 1.0 + 67 × 0.75) / 101        = 0.8341584158
```

Why the three differ: the true curve only drops to \(0.75\) after recall \(1/3\), but the 11-point grid counts \(r=0.3\) as "high precision", giving 4/11 of the area weight (the truth is 1/3) — an **over**estimate. The 101-point grid gives 34/101, much closer to 1/3.

## Constraints

- Return `(ap, mean_ap)`: `ap` is \( (T,C) \) and `mean_ap` is \( (T,) \), both float64 regardless of the input box dtype (IoU and integration are done in float64).
- Judging uses absolute tolerance \(10^{-9}\) on `ap` and `mean_ap`.
- The equality direction of the match (\( \ge \)), GT occupancy, largest-IoU choice, tie-breaking for equal scores and equal IoU, resetting occupancy between thresholds, and how the class set is derived are all part of the contract.
- CPU, deterministic, no random numbers; no GT, no predictions and classes without predictions must return zeros per the conventions above rather than raising.
