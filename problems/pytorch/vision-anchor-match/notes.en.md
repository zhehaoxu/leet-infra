# In Depth: Vision Detection: Multi-Scale Anchor Generation and IoU Matching

## Source

- The anchor mechanism and the "tile boxes per scale/aspect ratio on multi-scale feature maps, then assign positives/negatives by IoU thresholds" convention: Liu et al., *SSD: Single Shot MultiBox Detector*, arXiv:1512.02325 (ECCV 2016); Sec. 2.2 "matching strategy" states the \( \mathrm{IoU}>0.5 \) positive rule and the best-IoU matching rule.
- The guarantee rule and the ignore band: Lin et al., *Focal Loss for Dense Object Detection* (RetinaNet), arXiv:1708.02002; Sec. 4 anchor assignment — positives at \( \mathrm{IoU}\ge 0.5 \), negatives at \( \mathrm{IoU}<0.4 \), \( [0.4,0.5) \) excluded from the loss (the ignore class here), plus "one anchor per GT is always assigned".
- The centre convention \( ((x+0.5)\cdot\mathrm{stride}, (y+0.5)\cdot\mathrm{stride}) \) and "build the box, then clamp" follow SSD/RetinaNet-family implementations (Detectron2 `anchor_generator.py`, torchvision `AnchorGenerator._generate_anchors`). This problem depends on none of them; the judge environment only has torch.
- CPU only and fully deterministic: anchor coordinates come from integer cell indices and float constants, IoU from arithmetic; no RNG, no images read.

## Full derivation and index convention

**Enumeration order fixes the indices.** For level \(l\) with feature map \(H_l\times W_l\) and stride \(s_l\), scale list \(S\) and aspect-ratio list \(R\), the global anchor order is

\[
l \;\to\; y=0..H_l-1 \;\to\; x=0..W_l-1 \;\to\; a\in S \;\to\; r\in R,
\]

i.e. the \( |S||R| \) anchors of one cell are contiguous and rows run before columns. The reference builds, per level, a \( (H_l,W_l,|S||R|,4) \) `combos` tensor (one \( (H_l,W_l,4) \) coordinate map per \((s,r)\), stacked at `dim=2` and then `reshape(-1,4)`), concatenating levels at the end. Indices are **cell-major**, not scale-major — which is exactly what decides the guarantee winner, and why `anchors[2]` is already at \(x=1\) in the `anchor_enumeration` case.

**Centre and size.** The centre of cell \((x,y)\) is

\[
c_x=(x+0.5)\,s_l,\qquad c_y=(y+0.5)\,s_l,
\]

independent of \(H_l,W_l\) (only the stride matters, the usual "one shared centre across levels" choice). For \((s,r)\):

\[
w=s\sqrt r,\qquad h=\frac{s}{\sqrt r},\qquad \text{area}=wh=s^2 .
\]

The area always equals \( s^2 \), regardless of the aspect ratio — that identity is where the later "tied IoU" comes from.

**Clamping.** Compute the full box first, then clamp each coordinate to \( [0,W_{\text{img}}] \) / \( [0,H_{\text{img}}] \). Cells near the border lose some width/height and may even degenerate to zero area; such anchors are **kept**, with smaller area and smaller IoU. In the example, cell \((0,0)\) builds \((-4,-4,12,12)\), clamped to \((0,0,12,12)\), shrinking the area from \(16\times16=256\) to \(12\times12=144\).

**IoU convention.** Intersection uses clamped coordinates:

\[
\text{inter}=\max(0,\min(x_2^a,x_2^g)-\max(x_1^a,x_1^g))\cdot
             \max(0,\min(y_2^a,y_2^g)-\max(y_1^a,y_1^g)),
\]
\[
\mathrm{IoU}=\frac{\text{inter}}{S_a+S_g-\text{inter}},\qquad 0 \text{ if } S_a+S_g=0 .
\]

For the example GT \((4,4,20,20)\) (area 256) the four measured IoUs are:

```
anchor 0 (0,0,12,12)    inter=64   S_a=144  union=336  IoU=0.19047619
anchor 1 (4,0,20,12)    inter=128  S_a=192  union=320  IoU=0.4 (float32 gives 0.40000000596)
anchor 2 (0,4,12,20)    inter=128  S_a=192  union=320  IoU=0.4
anchor 3 (4,4,20,20)    inter=256  S_a=256  union=256  IoU=1.0
```

**Three-way split and where the equality lands.** The rule is

\[
\text{label}=\begin{cases}
\arg\max_g \mathrm{IoU}, & \mathrm{best\_iou}\ge \mathrm{pos\_thr}\\
-2, & \mathrm{neg\_thr}\le \mathrm{best\_iou}<\mathrm{pos\_thr}\\
-1, & \mathrm{best\_iou}<\mathrm{neg\_thr}
\end{cases}
\]

The equality belongs to ignore. Anchors 1 and 2 in the example hit exactly 0.4 (0.40000000596 in float32), so \( \ge\mathrm{neg\_thr} \) and \( <\mathrm{pos\_thr} \) → -2. Lowering `neg_thr` to 0.2 keeps them at -2 while anchor 0's 0.190476 stays \( <0.2 \) → -1; lowering `pos_thr` to 0.4 flips them positive and yields `[-1,0,0,0]`, counts `[3,1,0]`. Those three measured threshold settings are the discriminating case for "which side owns the equality".

**Guarantee rule.** For each GT \(g\), among the anchors with \( \mathrm{best\_gt}=g \) force the one with the largest \( \mathrm{best\_iou} \) positive. It must run **after** the three-way split: compute `best_iou >= pos_thr`, then the ignore band, then overwrite — otherwise the ignore pass fights the overwrite. With the tiny example GT \((15.9,15.9,16.1,16.1)\) the four IoUs are \(0,0,0,0.00015625\) (the first three boxes do not intersect it at all); all are below 0.4, so the threshold rule gives `[-1,-1,-1,-1]` and the guarantee turns anchor 3 into 0, giving `[-1,-1,-1,0]` and counts `[1,3,0]`. Without the guarantee such small objects would receive no positive anchor at all and could never be learned — that is why the rule exists.

**Two levels of tie-breaking.** Ties across GTs go to the **smallest GT index** (the default `max(dim=1)` behaviour); ties during the guarantee go to the **smallest anchor index**. Both must be pinned down, because judging compares labels bit-for-bit with `torch.equal`.

**No GT.** With \(G=0\) there is no `best_gt` and no guarantee, but anchors must still be generated in full (downstream allocates heads of size \(A\)), all labels are -1 and counts are `[0,A,0]`. `check_no_gt` also asserts that the anchors from a no-GT call are bit-identical to those from a with-GT call, preventing implementations from hiding anchor generation inside the GT branch.

**dtype.** Anchors follow `gt_boxes` (float32/float64); labels and counts are always int64. In float64 both coordinates and IoU are computed in double precision; `check_float64_and_dtype` verifies with \(10^{-12}\) tolerance.

## Key points and common mistakes

1. **The enumeration is cell-major (level→row→column→scale→ratio)**, not scale-major. After `meshgrid` gives the \((H,W)\) coordinate maps, you must `stack(dim=2)` over \((s,r)\) and then `reshape(-1,4)` to get "the \( |S||R| \) anchors of one cell contiguous". Looping scales first keeps the same count but scrambles every index — and the guarantee then keeps a different anchor.
2. **The centre depends only on the stride**; never back-derive it from `W_img / W_l`. In the example `W_img=32, W_l=2, stride=8` happens to be consistent, so `check_anchor_enumeration` deliberately uses stride 16 with a 1×1 feature map and a 32×32 image to separate the two.
3. **Build the box first, then clamp**, and do not drop zero-area anchors; otherwise `counts` no longer sums to \(A\).
4. **The equality belongs to the ignore band**: `>= neg_thr`, not `> neg_thr`. The exactly-0.4 anchors are constructed on purpose; `[-1,-2,-2,0]` versus `[-1,-1,-1,0]` is a one-character difference that fails the judge.
5. **The guarantee runs after the split**, and only when some anchor already points at that GT (`cand` empty → `continue`, never `argmax` an empty tensor).
6. **GT ties and anchor ties both take the smallest index**: `torch.max(dim=1)` and `argmax` return the first maximum by default; if you use `sort`/`topk` you must supply stability yourself.
7. **No GT still generates anchors**, identical to the with-GT path; no early empty return.
8. **Zero-area IoU is 0**, never NaN; both branches of `torch.where` are evaluated, so the division may well produce NaN/Inf and must be selected away by `where`.
9. **Threshold validation is \(0\le \mathrm{neg\_thr}\le \mathrm{pos\_thr}\)**; `neg_thr > pos_thr` must raise `ValueError`, since the ignore band would be empty and the semantics ill-defined.
10. **The guarantee pick under exact ties is float-noise sensitive**: in the \((3,4)\)-grid config with \(s\in\{8,12\}\), \(r\in\{0.5,1,2\}\), anchors 69 (\(s=12,r=0.5\)) and 70 (\(s=12,r=1\)) have mathematically identical IoU \(72/472\) against GT \((20,20,40,40)\) (area is always \(s^2=144\) and both intersections are 72), so float32 and float64 last-bit noise crown different winners. The problem only promises "exactly one positive per GT", not which tied anchor wins; see the cross-check convention below.
11. **counts must agree with labels**: the three bins sum to \(A\) and equal the counts of `>=0`, `==-1`, `==-2`. Count them from labels; do not recompute separately.

## Differences from neighboring problems

- **vs `vision-fcos-centerness` (same batch)**: that problem computes the FCOS center-ness soft label \( \sqrt{\min(l,r)/\max(l,r)\cdot\min(t,b)/\max(t,b)} \) on positions already known to be foreground, taking a "position → GT" assignment as input; this problem produces exactly that assignment (hard labels plus the guarantee) and outputs discrete labels rather than a continuous quality score.
- **vs `vision-fpn-levels` (same batch)**: FPN level selection maps a GT to one level by its log-scale size (\(k=\lfloor k_0+\log_2\sqrt{wh/224}\rfloor\) and friends) deciding only "which level is responsible", with no box generation and no IoU; this problem tiles boxes on every level and uses IoU for positive/negative/ignore.
- **vs `vision-rotated-nms` (same batch)**: that one is post-processing over scored candidates with rotated IoU suppression; this one is pre-processing deciding which anchors take part in training.
- **vs `vision-rle-mask-iou` (same batch)**: that is RLE encoding of segmentation masks and mask IoU (pixel-set operations); this is rectangle IoU.
- **vs `iou-variant-losses` (existing)**: GIoU/DIoU/CIoU only evaluate two **given** boxes; they involve no anchor grid and no label assignment. The IoU here is the plain axis-aligned one, and the focus is enumeration order and threshold semantics.

## Cross-check convention and complexity

- **Main oracle**: `_oracle` is a pure-Python quadruple loop (level→row→column→scale→ratio) that emits anchors one by one, then loops anchors × GTs through `_iou_loop` and scans for the guarantee; structurally different from the reference's "`meshgrid` coordinate maps + `_iou_matrix` broadcast matrix + `argmax`". Four configs are covered: single level/single GT, multiple GTs with multi-scale multi-ratio border cases, two levels with different strides, and heavy clamping on an 8×8 image.
- **Tie tolerance**: `check_pointer_oracle` first computes the oracle IoU matrix and each GT's maximum \(g_{\max}\), treating anchors with \( \mathrm{IoU}\ge g_{\max}-10^{-6} \) as "guarantee tie candidates"; it fails only when one side is positive and the other negative/ignore **and the position is not a tie candidate**. It also asserts "at least one positive per GT", "every positive's IoU is either \(g_{\max}\) or \( \ge \mathrm{pos\_thr}\)", "the positive total matches the oracle", and "counts agree with labels". This pins the semantics while not tripping over last-bit float differences.
- **Discriminating cases**: the measured `EXAMPLE_*` constants; the exactly-0.4 equality under three threshold settings (`[-1,-2,-2,0]` / `[-1,-2,-2,0]` / `[-1,0,0,0]`); the tiny-GT guarantee (`[-1,-1,-1,0]`); two GTs each keeping a positive; duplicate GTs taking the smallest index; identical anchors with and without GT; clamping on an 8×8 image; float64 dtype; ten invalid-input cases.
- **Non-obvious measured facts**: anchor area is always \(s^2\), so different \((s,r)\) pairs can produce mathematically equal IoU on a given GT (measured \(72/472\)), making the guarantee pick float-noise dependent; float32 evaluates 0.4 as 0.40000000596, so the equality-to-ignore decision relies on `>= neg_thr` rather than float equality; clamping squeezes a 256-area box to 144 and drops its IoU from 1.0 to 0.190476.
- **Complexity**: generation \(O(A)\); the IoU matrix is \(O(AG)\) time and \(O(AG)\) memory (broadcast); the guarantee is \(O(A+AG)\) of `nonzero`/`argmax`. Judge scale is \(A\le 72\), \(G\le 2\), far below the limit. Production code runs \(A\) in the hundreds of thousands and chunks the IoU; not required here.
