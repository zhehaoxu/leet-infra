# Vision Detection: Weighted Box Fusion (WBF)

## Problem

When ensembling detectors, the same object is predicted several times — by different models, or by different scales/flips of one model — producing a pile of overlapping boxes with different scores. NMS keeps only the top-scoring one and discards the rest, throwing away the ensemble's evidence. WBF (Weighted Boxes Fusion, Solovyev et al., *Weighted boxes fusion: Ensembling boxes from different object detection models*, arXiv:1910.13302) takes another route: cluster overlapping same-class boxes and fuse each cluster into one box via a **confidence-weighted average**, then discount the confidence by how many models actually voted.

Implement the complete WBF post-processing:

- process boxes from highest to lowest confidence, put each one into the same-class cluster that has the largest IoU above the threshold with the cluster's current fused box, or start a new cluster;
- a cluster's fused box is the confidence-weighted average of its members (not the arithmetic mean, not the top-scoring box);
- a cluster's fused confidence is the arithmetic mean of member scores times \( \min(n, N_{\text{models}})/N_{\text{models}} \) (\(n\) = boxes in the cluster) — the fewer models that voted, the heavier the discount;
- output in descending fused confidence, ties broken by cluster creation order.

Three traps: merging uses **IoU strictly greater** than the threshold; candidate clusters are only **same-class**; the fused box is **recomputed immediately**, so later boxes are compared against the updated fused box. CPU only, deterministic, no random numbers.

## Rules

The full signature, parameter meanings and defaults live only in the starter docstring, which is the sole judging contract.

- Processing order: all boxes by `scores` **descending**, ties by ascending original index (stable sort).
- Clusters only form among boxes with the **same `labels`**; boxes of different classes stay in separate clusters even if geometrically identical.
- For the current box, walk the same-class clusters and compute the IoU against each cluster's **current fused box**; take the **largest** IoU and merge only if it is **strictly greater** than `iou_thr` (an IoU exactly equal to the threshold does **not** merge). Ties go to the earliest-created cluster. With no qualifying cluster, a new cluster is created; cluster order is first-appearance order.
- The fused box is recomputed right after each merge, so subsequent boxes are compared against the updated one.
- Fused box \(=\) confidence-weighted mean of members:
  \[
  \bar{x}=\frac{\sum_k s_k x_k}{\sum_k s_k},
  \]
  and likewise for the other three coordinates; when the cluster's scores sum to 0 it degrades to an **equal-weight** mean (no 0/0).
- Fused confidence:
  \[
  \bar{s}=\left(\frac{1}{n}\sum_{k=1}^{n} s_k\right)\cdot\frac{\min(n, N_{\text{models}})}{N_{\text{models}}},
  \]
  with \(N_{\text{models}}\) = `num_models` (an integer \( \ge 1 \)). No discount when \(n\ge N_{\text{models}}\), a proportional one when \(n<N_{\text{models}}\).
- Output order: fused confidence **descending**, ties by cluster creation order (stable sort). Boxes and scores follow the dtype of `boxes` (float32/float64); labels are always int64.
- Empty input returns three empty tensors of shapes \( (0,4) \), \( (0,) \), \( (0,) \).
- Invalid input raises `ValueError`: `boxes` not float32/float64 or not \( (N,4) \), `scores`/`labels` shape or dtype mismatch (`labels` must be integer), `iou_thr` outside \( [0,1] \), or `num_models` not an integer \( \ge1 \).

## Example

Four boxes, two classes, `iou_thr=0.55`, `num_models=1`:

```
A = [ 0, 0,10,10]  s=0.6  label 0
B = [ 1, 0,11,10]  s=0.6  label 0
C = [20,20,30,30]  s=0.9  label 0
D = [ 0, 0,10,10]  s=0.5  label 1
```

C (0.9) is processed first and opens cluster 0. Then A (0.6) has no overlap with C and opens cluster 1; B (0.6) has IoU \(90/110\approx0.818>0.55\) with cluster 1's fused box (just A), so it merges, and the fused box becomes the weighted average \((0.5,0,10.5,10)\) with score \(0.6\). D (0.5) carries label 1 and opens cluster 2.

```
fused_boxes  = [[20,20,30,30], [0.5,0,10.5,10], [0,0,10,10]]
fused_scores = [0.9, 0.6, 0.5]
fused_labels = [0, 0, 1]
```

Change `num_models` to 2: the singleton clusters are scaled by \( \min(1,2)/2=0.5 \) while the two-box cluster gets \( \min(2,2)/2=1 \), so C, A+B, D become \(0.45, 0.6, 0.25\) and the output order changes:

```
fused_boxes  = [[0.5,0,10.5,10], [20,20,30,30], [0,0,10,10]]
fused_scores = [0.6, 0.45, 0.25]
fused_labels = [0, 0, 1]
```

## Constraints

- Return `(fused_boxes, fused_scores, fused_labels)` with shapes \( (K,4) \), \( (K,) \), \( (K,) \); the first two follow the dtype of `boxes`, the last is int64.
- Judging uses absolute tolerance \(10^{-6}\) (float32) / \(10^{-12}\) (float64) on boxes and scores, and `torch.equal` on labels.
- The merge predicate (strict \(>\)), the best-IoU cluster choice, the immediate recomputation after a merge, the weighted mean and confidence discount, and both stable-sort conventions are part of the contract.
- CPU, deterministic, no random numbers; empty input, a single box and all-zero confidences must return per the conventions above rather than raising.
