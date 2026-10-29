# In Depth: Vision Detection: mAP Computation (11-point / All-point / COCO 101-point)

## Source

- Dataset and evaluation protocol: Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014, arXiv:1405.0312. The official `pycocotools.COCOeval` averages precision over 101 recall points (`np.linspace(0, 1, 101)`); the `"coco"` branch here is exactly that convention.
- 11-point: the PASCAL VOC2007 devkit mAP. It averages interpolated precision at \( r\in\{0,0.1,\dots,1.0\} \), the standard metric before deep learning.
- All-point: the PASCAL VOC2010+ devkit convention, which switched to integrating the precision-recall curve (step integral over the right-hand envelope). Henderson & Ferrari, *End-to-end training of object class detectors for mean average precision*, ACCV 2016, also discusses why these discretisations produce different APs.
- This problem depends on none of pycocotools (the judge only has torch); matching and integration are rewritten with torch plus pure Python. CPU only, deterministic, no RNG.

## Full derivation and index convention

**Input layout.** Predictions are four equal-length flat tables (boxes \( (N,4) \) xyxy, scores \( (N,) \), classes \( (N,) \), image ids \( (N,) \)), ground truth three equal-length tables. Image ids are only ever compared for **equality**, so they need not start at 0 or be contiguous — the judge deliberately uses a non-zero id such as `1`. This removes the batch dimension and avoids the "sort by id then index-misalign" trap.

**Class set.** \( \mathcal{C}=\mathrm{unique}(\text{gt\_cls}) \) sorted ascending, taken **from the ground truth**, not from the predictions: a class that appears only among predictions has no GT, so recall is undefined and it is excluded from the mean. With no GT, \( C=0 \) and the output has shape \( (T,0) \) with `mean_ap` all zeros — not an error and not a special output shape beyond that.

**Greedy matching for one class and one threshold.** Predictions of that class are processed by ascending \((-\text{score}, \text{index})\); for each prediction,

\[
g^\*=\arg\max_{g\ \text{eligible}}\ \mathrm{IoU}(p,g),
\]

over same-image, same-class, unoccupied GTs, ties to the smallest \(g\). If \( \mathrm{IoU}(p,g^\*)\ge\tau \) it is a TP and \(g^\*\) becomes occupied, otherwise a FP. The comparison is \( \ge \): the judge uses prediction `[20,20,30,27.5]` against GT `[20,20,30,30]`, intersection \(10\times7.5=75\), union 100, IoU **exactly 0.75** — it must hit at `iou_thrs=[0.75]` (AP = 1) and miss at `[0.76]` (AP = 0).

**Occupancy resets between thresholds.** Each \( \tau \) reruns matching from scratch. Otherwise a GT consumed at a low threshold would stay unavailable at a high one, recall would stop being monotone in \( \tau \), and averaging over thresholds would be meaningless.

**Precision/recall sequences.** After each step \( p_i=\frac{\mathrm{TP}_i}{\mathrm{TP}_i+\mathrm{FP}_i} \), \( r_i=\frac{\mathrm{TP}_i}{n_{\text{gt}}} \), where \( n_{\text{gt}}=\#\{g:\text{cls}(g)=c\} \) is the class's total GT count across **all** images. So \( r_{\max} \) need not be 1: with missed GT it equals \( n_{\text{matched}}/n_{\text{gt}}<1 \).

**Three integration conventions.**

- `"all"`: right-hand envelope \( \tilde p_i=\max_{j\ge i}p_j \), then
  \[
  \mathrm{AP}=\sum_i (r_i-r_{i-1})\,\tilde p_i,\qquad r_{-1}=0 ,
  \]
  a left-endpoint step integral over the monotone PR curve.
- `"11point"`: \( \mathrm{AP}=\frac1{11}\sum_{k=0}^{10}\max_{i:\,r_i\ge k/10}p_i \), 0 when the set is empty.
- `"coco"`: the same with \( k=0,\dots,100 \), divided by 101.

**Why the three disagree.** The example's PR points are \( (1/3,1.0),(1/3,0.5),(2/3,0.6667),(1,0.75) \), envelope \( [1,0.75,0.75,0.75] \). The true area counts \( [0,1/3] \) at precision 1:

\[
\mathrm{AP}_{\text{all}}=\tfrac13\cdot1+\tfrac23\cdot0.75=0.8333333333 .
\]

The grids instead repeat a sample's precision across the whole interval: only 4 of 11 points satisfy \( r\le1/3 \) (0, 0.1, 0.2, 0.3), the other 7 take 0.75,

\[
\mathrm{AP}_{\text{11}}=\frac{4\times1+7\times0.75}{11}=0.8409090909 ,
\]

and 34 of 101 points satisfy \( r\le1/3 \) (0.00 through 0.33),

\[
\mathrm{AP}_{\text{coco}}=\frac{34\times1+67\times0.75}{101}=0.8341584158 .
\]

Here \( \text{all}<\text{coco}<\text{11point} \) because the curve drops only at \( r=1/3 \) and the 11-point grid counts \( r=0.3 \) as high precision, over-weighting that interval by \( 4/11-1/3\approx0.0303 \); the 101-point grid over-weights it by only \( 34/101-1/3\approx0.0033 \). This ordering is **not** universal — a different curve can invert it — so the judge only pins the values and ordering of this example.

**Build grid points by division.** The reference uses \( r=k/(\text{grid}-1) \) or `k/10.0`, `k/100.0`; never `r += step`, whose binary error makes the \( r_i\ge r \) boundary jitter (e.g. \( r_i=0.3 \) excluded by \( r=0.30000000000000004 \)). The 34-and-7 point counts in the tests are a direct consequence.

**Two semantic consequences of greedy matching** (each covered by a test):

1. *Each GT is used once*: two identical predictions — the first takes GT0, the second finds no intersection with the remaining GT1 — give TP, FP, precision \( 1,0.5 \), recall \( 0.5,0.5 \), hence \( \mathrm{AP}_{\text{all}}=0.5\cdot1.0+0\cdot0.5=0.5 \). Counting both duplicates as TP yields 1.0; silently dropping an unmatched prediction (instead of counting a FP) also yields 1.0.
2. *Take the largest IoU*: prediction `[0,0,10,5.5]` overlaps `[0,0,10,10]` with IoU \(55/100=0.55\) and `[0,0,10,6]` with IoU \(55/60=0.9167\), and must match the latter. Both exceed 0.5, so "largest" and "first GT by index" are both TPs consuming one GT and both give AP 0.5 — the rule is pinned by a semantic assertion (which GT got matched), while numeric agreement comes from the oracle.

**Image and class must both match.** Identical coordinates with a different image id (`[1,1]` vs `[0,0]`) cannot match, AP = 0; likewise for a different class. An implementation that matches on IoU alone passes every single-image single-class case and fails these.

**Empty inputs.** With no predictions the class still has \( n_{\text{gt}}>0 \) but no PR points, so AP is 0 (not NaN: `_ap_curve` returns 0 for an empty list). With no GT, \( C=0 \), `ap` is \( (T,0) \) and `mean_ap` is all zeros.

**dtype.** IoU and integration run in float64 and the outputs are float64 regardless of whether the input boxes are float32 or float64; the judge re-runs the same data cast to float64 and requires elementwise agreement to \(10^{-12}\). A float32 accumulation of 101 points drifts past the \(10^{-9}\) tolerance.

## Key points and common mistakes

1. **The predicate is \( \ge \), not \( > \)**: an IoU of exactly 0.75 must hit at `thr=0.75`; using `>` turns it into a FP and AP from 1 to 0.
2. **Reset GT occupancy per threshold**: reusing one occupancy set systematically lowers the high-threshold AP.
3. **The class set comes from GT**: using predictions adds classes with no GT (diluting the mean, changing the column count).
4. **Filter by image *and* class**: IoU-only matching pairs boxes from different images/classes.
5. **Take the max-IoU GT, ties to the smallest index**: not "the first GT above threshold"; with nested/overlapping GT the two differ.
6. **Duplicate predictions are FPs**: "no same-image same-class GT left" means FP, not "ignore"; the example's \(1,0.5,0.6667,0.75\) sequence depends on it.
7. **Recall denominator is the class's total GT count** across all images — not per image, not the number of matches.
8. **The envelope belongs to `all` only**: 11-/101-point take "max precision over \( r_i\ge r \)" per grid point, which is equivalent to sampling the envelope; "precision at the closest \( r_i \)" is a different and lower value (0.5 instead of 1.0 at \( r=0.3 \)).
9. **Do not accumulate grid points**: float error at 0.1 changes boundary counts; the 4+7 and 34+67 splits are the check.
10. **An empty PR sequence is 0**: with only FPs the sequence is non-empty (all FP), but with no predictions it is empty and must return 0 instead of dividing by zero.
11. **Implement all three methods**: only `all` is caught by `check_methods_differ`, and 11-point vs COCO (0.8409 vs 0.8342) differs far beyond \(10^{-9}\).
12. **Float64 throughout**: 101-point summation in float32 drifts past tolerance; output dtype is independent of input dtype.

## Differences from neighboring problems

- **vs `vision-anchor-match` (same batch)**: anchor matching is a **pre-training** assignment of anchors to positive/negative/ignore with two thresholds (`pos_thr`/`neg_thr`) and a per-GT guarantee; this problem is **evaluation-time** one-to-one greedy prediction↔GT matching with a single threshold plus equality direction, producing a scalar AP.
- **vs `vision-wbf-fusion` (same batch)**: WBF also sorts by score and thresholds IoU, but it **fuses** (weighted-average boxes and scores) and never enforces one-to-one occupancy; this problem only **counts** TP/FP and never modifies coordinates.
- **vs `vision-rotated-nms` (same batch)**: rotated NMS greedily **suppresses** boxes above an IoU threshold and has no GT and no recall; here greedy matching exists to align with GT and accumulate precision/recall.
- **vs `vision-fcos-centerness` (same batch)**: FCOS assigns labels and a soft score on the **training-target** side and aggregates nothing.
- **vs `topk-select` / `searchsorted-bincount`**: those are numeric primitives; this is a full evaluation pipeline (sort → match → curve → integrate) whose difficulty is in the conventions.

## Cross-check convention and complexity

- **Main oracle**: `_oracle` runs the flow in pure Python — `sorted(range(N), key=lambda i: (-s[i], i))`, a `set()` of consumed GTs, a scalar `_iou`, and `max(...)` per grid point; the reference uses a broadcast `_iou_matrix`, `argsort(stable=True)` and an envelope via `flip(cummax)`. Covered: 3 classes, 2 images, 10 GT, 18 predictions, thresholds \( [0.3,0.5,0.75] \), all three methods, `atol=1e-9`.
- **Discriminating cases**: the measured three-method example; `all<coco<11point` plus exact constants; the exactly-0.75 `>=` boundary (hit at 0.75, miss at 0.76); two identical predictions → AP 0.5; different image id → 0; different class → 0; nested GT picks max IoU; a class with no predictions → 0; no GT → \( (T,0) \); no predictions → 0; float32/float64 agreement; twelve invalid-input cases.
- **Non-obvious measured facts**: the high-precision weight is \(4/11\approx0.3636\) for 11-point and \(34/101\approx0.3366\) for 101-point against a true \(1/3\approx0.3333\) — the coarse grid **over**-estimates AP, and 101-point is an order of magnitude closer; the duplicate-prediction AP is exactly 0.5 (not 1 and not 0) because recall is stuck at 0.5 with only two precision samples.
- **Complexity**: matching costs \( O(T\cdot C\cdot N_{\text{pred}}\cdot N_{\text{gt}}) \) IoUs and integration \( O(T\cdot C\cdot(n_{\text{pred}}+n_{\text{grid}})) \). The reference vectorises the former into an \( (N_{\text{pred}},N_{\text{gt}}) \) IoU matrix; the judge scale is \( N_{\text{pred}}\le18 \), \(N_{\text{gt}}\le10 \), \( T\le3 \), far below the limit. Production code uses pycocotools' vectorised matching plus a 101-point `cummax`; only correctness is required here.
