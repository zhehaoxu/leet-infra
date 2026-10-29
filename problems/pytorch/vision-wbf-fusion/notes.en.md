# In Depth: Vision Detection: Weighted Box Fusion (WBF)

## Source

- The algorithm: Solovyev, Wang, Gabruseva, *Weighted boxes fusion: Ensembling boxes from different object detection models*, Image and Vision Computing 2021, arXiv:1910.13302. Algorithm 1 gives the clustering flow (process by descending confidence, merge into a cluster when the IoU with its fused box exceeds the threshold) and Algorithm 2 the coordinate/confidence updates.
- The two candidate confidence rescalings: Eq. (4) \( C=C\cdot\frac{\min(T,N)}{N} \) and Eq. (5) \( C=C\cdot\frac{T}{N} \), where \(T\) is the number of boxes in the cluster and \(N\) the number of models; the paper argues Eq. (4) is the one used in practice (since \(T>N\) should not keep boosting). This problem adopts Eq. (4): \( \bar{s}=\bar{s}_{\text{avg}}\cdot\frac{\min(n,N_{\text{models}})}{N_{\text{models}}} \).
- Fused coordinates: the paper uses the confidence-weighted mean \( X=\frac{\sum C_iX_i}{\sum C_i} \); implemented as stated, with an equal-weight fallback when the weights sum to 0.
- Engineering reference: ZFTurbo/Weighted-Boxes-Fusion `ensemble_boxes_wbf.py` is the widely used implementation of the same flow; this problem depends on none of it, the judge only has torch.
- CPU only and fully deterministic: sorting, IoU and weighted means are arithmetic only, no RNG.

## Full derivation and index convention

**Processing order and stable sorting.** Boxes are processed by descending score, ties by ascending original index. The reference uses `torch.argsort(-scores, stable=True)`, and stability preserves input order on equal scores. This order fixes cluster creation order and the tie direction of "which cluster to merge into", so it must be pinned explicitly.

**Cluster structure.** Each cluster stores `label`, its member indices and its **current fused box**. When visiting a box, only same-`label` clusters are considered:

\[
j^\*=\arg\max_{j:\ \text{label}_j=\text{label}}\ \mathrm{IoU}(b,\ \bar{B}_j),
\qquad
\text{merge into } j^\* \text{ if } \mathrm{IoU}(b,\bar{B}_{j^\*}) > \tau .
\]

Both the paper and the engineering implementation take the **largest-IoU** cluster rather than the "first cluster above threshold" — when several clusters qualify, merge into the most similar one. The reference initialises `best_iou` to \( \tau \) and only replaces it on a **strictly greater** value, so ties fall to the earliest-created cluster. Judging deliberately builds a box that hits one cluster at 0.4091 and another at 0.4762: only "largest" merges into the latter (measured fused \(x_1=10.1/1.3\approx7.76923\), score 0.65), while "first" gives a completely different result.

**The threshold is strict.** Two boxes of area 100 and 50 with intersection 50 make the IoU **exactly 0.5**: `iou_thr=0.5` does not merge (2 clusters, scores 0.9/0.8), while `iou_thr=0.49` merges into one cluster whose fused \(y_2=(10\times0.9+5\times0.8)/1.7=13/1.7\approx7.647059\) and score \(0.85\). An implementation using `>=` fails the first case.

**Recompute the fused box immediately on merge.** The third box is compared against the *updated* fused box, not against the cluster's first box. In the example the two comparisons for A and B are identical anyway (a single-member cluster's fused box is that member), but with repeated merges a delayed recomputation compares against a stale box and the chaining boundary drifts.

**Weighted vs arithmetic mean.** Each coordinate uses \( \frac{\sum s_kx_k}{\sum s_k} \). Boxes with \(s=0.75\) and \(s=0.25\) (\( [0,0,10,10] \) and \( [2,2,12,12] \)) separate the variants: weighted gives \( [0.5,0.5,10.5,10.5] \), equal weight \( [1,1,11,11] \), inverted weights \( [1.5,1.5,11.5,11.5] \). All-zero weights fall back to the equal-weight mean to avoid \(0/0\).

**Confidence rescaling, Eq. (4).** \( \bar{s}=\bar{s}_{\text{avg}}\cdot\min(n,N_{\text{models}})/N_{\text{models}} \) means "the fraction of models that cast a vote": ensembling 4 models, a 1-box cluster is scaled by 1/4 and a 4-box cluster is not discounted. Swapping \(n\) and \(N_{\text{models}}\) (\( \min(N,n)/n \)) inverts the semantics — singletons escape the discount while large clusters are penalised. With \(N_{\text{models}}=2\) the example's clusters score 0.45, 0.60, 0.25 and the output order changes from \( [0.9,0.6,0.5] \) to \( [0.6,0.45,0.25] \); with \(N_{\text{models}}=4\) they are \(0.3,0.225,0.125\).

**Both sorts must be stable.** Input order is descending score (ties by index) and output order is descending fused score (ties by creation order), both via stable sort. Judging includes a 0.5 tie between a two-box cluster averaging 0.75 and 0.25 and a 0.5 singleton; the two clusters must come out as `[0,1]` by creation order.

**Floating-point last bits break "mathematically equal" scores.** A measured, counter-intuitive effect: the float32 mean of 0.9 and 0.7 is `0.7999999761` while a singleton's 0.8 is `0.8000000119` — both are 0.8 mathematically but differ by 1 ulp, which flips the output order from `[0,1]` to `[1,0]`. That is exactly why the tie test uses 0.75/0.25 (binary-exact, mean exactly 0.5); it also warns implementers not to assert equality of floats.

**dtype.** Fusion accumulates in float64 and converts back to the input dtype to limit rounding in large clusters; boxes and scores follow `boxes`, labels are always int64.

## Key points and common mistakes

1. **Merge predicate is strict**: an IoU exactly equal to the threshold does not merge; `>=` loses one cluster on the IoU=0.5 case.
2. **Candidate clusters are same-class only**: identical coordinates with different labels must produce separate clusters — D vs A in the example.
3. **Merge into the largest-IoU cluster, not the first above threshold**: the two-cluster case separates the two implementations.
4. **Recompute the fused box immediately after a merge**: delaying changes the chaining boundary.
5. **Weighted mean, not arithmetic**: the weights are the confidences; equal-weight and inverted-weight versions are both caught by `check_weighted_fusion`.
6. **Zero weight sum falls back**: all-zero confidences use the equal-weight mean instead of producing NaN.
7. **Direction of the discount**: \( \min(n,N_{\text{models}})/N_{\text{models}} \) — boxes in the numerator, models in the denominator; swapping them rewards weak clusters.
8. **Two stable sorts**: input order determines cluster creation, output order determines the result; `argsort(..., stable=True)` is required.
9. **`num_models` must be an integer**: `2.5` raises `ValueError` rather than being truncated.
10. **Merged boxes disappear as clusters**: a box that joins a cluster is not also emitted alone; a "new cluster or merge" branch suffices, no deletion pass needed.

## Differences from neighboring problems

- **vs `vision-anchor-match` (same batch)**: anchor matching happens **before training**, turning anchors into positive/negative/ignore with IoU thresholds; WBF happens **after inference**, clustering scored boxes into fused ones and still outputting boxes with scores.
- **vs operator problems like `searchsorted-bincount`**: those are numeric primitives; this is a full post-processing pipeline of "sort + greedy clustering + weighted statistics".
- **vs `vision-map-coco` (same batch)**: mAP also does greedy IoU matching, but that matching is a one-to-one prediction→GT deduplication to count TP/FP; it neither fuses coordinates nor rescales confidences.
- **vs `vision-rotated-nms` (same batch)**: rotated NMS is also IoU threshold + greedy, but it **suppresses** (keep one, drop the rest) while WBF **fuses** (weighted average of all the evidence); and NMS here uses rotated IoU versus axis-aligned here.
- **vs `vision-fcos-centerness` (same batch)**: that problem labels grid locations and computes a soft score, with no clustering across boxes.

## Cross-check convention and complexity

- **Main oracle**: `_oracle` runs the same flow in pure Python — `sorted(range(N), key=lambda i: (-s, i))`, explicit `[label, members, fused_box]` records and per-term `sum(...)` weighted sums; the reference uses tensor sorting, dict-style clusters and float64 tensor weighted sums. Four configs are covered: 24 random boxes over 3 classes, 6 heavily overlapping jittered boxes, `num_models=3` and `num_models=2`.
- **Discriminating cases**: the measured two-class fusion example; `num_models ∈ {1,2,4}` (rescaling and order flips); the exact-0.5 threshold strictness; identical boxes with different labels; weighted / equal / inverted means; one box hitting two clusters (largest wins); score ties resolved by creation order; empty input and a single box; float64; ten invalid-input cases.
- **Non-obvious measured facts**: two boxes with \( \mathrm{IoU}=90/110\approx0.818 \) fuse into a box shifted by half a pixel on both x coordinates; with `iou_thr=0.49` the merged \(y_2=13/1.7\approx7.647059\) (not 7.5); in float32 the mean of 0.9 and 0.7 differs from a singleton 0.8 by 1 ulp, enough to flip the output order.
- **Complexity**: the naive implementation does \(O(N\cdot K)\) IoUs (\(K\) clusters, worst case \(O(N^2)\)) plus \(O(n)\) per merge for recomputation. Judge scale is \(N\le 24\), far below the limit. Production code prunes candidate clusters with grids/sorted indices; not required here.
