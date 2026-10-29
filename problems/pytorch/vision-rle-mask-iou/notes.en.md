# In Depth: Detection: RLE Codec and Mask IoU

## Source

- Dataset and mask format: Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014, arXiv:1405.0312. COCO stores each instance's `segmentation` as `{"counts": [...], "size": [h, w]}`, exactly the RLE form used here.
- Official implementation: `cocodataset/cocoapi`, `common/maskApi.c`. `rleEncode` / `rleDecode` convert between dense masks and runs, and `rleIou` merges two `counts` arrays directly to produce the intersection — this problem rebuilds those three pieces; `maskUtils.iou(dt, gt, iscrowd)` is the upper-layer usage (the mask IoU matrix).
- The **column-major** convention is written as "Fortran order" in the official API: `maskUtils.encode(np.asfortranarray(mask))`, the same tradition as MATLAB's column-major storage.
- The judge has no pycocotools; the reference encodes with `torch.unique_consecutive` and merges with two pointers. CPU only, deterministic, no RNG.

## Full derivation and index convention

**Linearisation.** For an \( (H,W) \) mask define

\[
\mathrm{lin}(x,y)=x\cdot H+y,\qquad x\in[0,W),\ y\in[0,H),
\]

i.e. the \(H\) pixels of column \(x=0\) come first, then column \(x=1\). Reading the 0/1 values in ascending \( \mathrm{lin} \) gives the sequence underlying the RLE. In torch that is `mask.permute(1, 0).reshape(-1)` (`permute` yields a non-contiguous view, and `reshape` copies in the new order). **Row-major** would be \( \mathrm{lin}=y\cdot W+x \), a different array on the same mask.

**Canonical encoding.** Let the underlying sequence be \( s_0,\dots,s_{HW-1} \) and compress it into maximal equal-value runs \( (v_0,n_0),\dots,(v_{R-1},n_{R-1}) \); adjacent runs necessarily differ. Runs alternate starting from background, so

\[
\text{counts}=
\begin{cases}
[0,\ n_0,\ n_1,\ \dots] & v_0=1\ (\text{prepend } 0 \text{ when the sequence starts with foreground}),\\
[n_0,\ n_1,\ \dots] & v_0=0 .
\end{cases}
\]

That yields three checkable properties: the first entry may be 0, every other entry is positive, and the entries always sum to \(HW\). Note the trailing background run is included automatically because maximal runs cover the whole sequence; an implementation that stores only foreground runs and treats background as a separator loses the tail length, so `sum(counts) < HW` — which the sum check rejects.

**Intersection by merging.** Both counts partition the same linear interval, just at different cut points. A two-pointer walk does:

\[
t=\min(\text{rem}_a,\text{rem}_b),\qquad
\mathrm{inter}\mathrel{+}=t \ \text{ if both current run values are } 1 .
\]

When one side's remaining length hits 0, advance it and flip its run value (equivalently, index parity: even = background, odd = foreground). Total \( O(R_a+R_b) \), never expanding counts into \(HW\) elements. The reference uses `va = ia & 1`, reading the value from the parity instead of maintaining a flip variable.

**Areas and union.** \( |A| \) can be summed from the mask directly (the mask is dense input anyway), while \( |B| \) is the sum of the **odd-indexed** counts — even indices are background. Use

\[
\mathrm{union}=|A|+|B|-\mathrm{inter}
\]

instead of a second merge for the union: one less state, and no chance of getting the direction wrong. With two all-background masks \( |A|=|B|=\mathrm{inter}=0 \), so `union=0` and the IoU is defined as 0: \( 0/0 \) must never reach floating point.

**IoU in float64.** Intersection and union are integers and their ratio is exact in float64 at this scale; even when built from `inter / union` in Python, the 0-dim tensor must be created with `dtype=torch.float64`, otherwise it follows the global default dtype.

**Validation.** Each `counts` rule maps to a real error: negative (corrupt encoding), interior zero (unmerged adjacent runs), sum not \(HW\) (wrong size or dropped trailing background), empty (impossible when \(HW\ge1\)). `size` must equal `mask.shape` exactly because COCO stores `rle["size"]` separately from counts and the official code silently produces wrong results on mismatch; this problem raises instead. The mask may be bool or uint8, but a uint8 value of 2 must raise, otherwise the "0/1 mask" semantics is broken. `counts` may be int32 or int64, but must be cast to int64 before summing: `int32.sum()` overflows on large masks.

## Key points and common mistakes

1. **Column-major, not row-major**: the discriminating case is the mask `[[1,1,0],[0,0,0]]` — column-major encodes as `[0,1,1,1,3]`, row-major as `[0,2,4]`; feeding `[0,2,4]` back as column-major drops the IoU from 1 to \(1/3\).
2. **A leading 0 is legal and must be kept**: when the first column-major pixel is foreground, counts start with 0; writing `[n_0, ...]` shifts everything by one run and inverts every subsequent value.
3. **Do not drop the trailing background run**: foreground-only encodings lose it and break `sum(counts) == HW`; `[0,2,2]` versus `[0,2]` is exactly the two background pixels of the second column.
4. **Merge adjacent equal runs**: `[0,1,0,3]` is non-canonical and must raise `ValueError`, not "happen to decode".
5. **Union from areas, not from OR**: `union = |A| + |B| - inter`; `+ inter` or forgetting to subtract is a classic slip.
6. **`|B|` sums odd indices only**: even indices are background runs.
7. **Two empty masks give IoU 0**: `union=0` must short-circuit, otherwise `0/0` is NaN and `abs(iou-0)<tol` fails.
8. **Cast `counts` to int64 before summing**: an int32 sum overflows and a valid input looks like a sum mismatch.
9. **uint8 accepts only 0/1**: uint8 is often treated as "nonzero means foreground", but here 2 or 255 is invalid input.
10. **`size` must match the mask exactly**: this mismatch really exists in COCO (size and counts are stored separately); do not silently overwrite with `mask.shape`.
11. **No mutation**: `permute` + `reshape` copies, and `counts_a` is a new tensor.
12. **Scalar outputs are 0-dim tensors**: `iou`/`intersection`/`union` are `torch.tensor(v)`, neither Python scalars nor length-1 vectors.

## Differences from neighboring problems

- **vs `vision-map-coco` (same batch)**: mAP matches with **box** IoU (rectangular areas), no masks and no RLE; this problem is **pixel-level** mask IoU, whose difficulty lies in the column-major run format and the merge.
- **vs `vision-anchor-match` / `vision-fcos-centerness` (same batch)**: those IoUs act between anchors/predictions and GT boxes for training-target assignment; this problem has no boxes at all.
- **vs `vision-rotated-nms` (same batch)**: rotated IoU is a **polygon** intersection (Sutherland–Hodgman clipping) over continuous vertices; this is a **discrete pixel-set** intersection over compressed integer runs, so the intersection is an integer.
- **vs operator problems such as `where-select` / `masked-*`**: those operate elementwise on dense tensors with \(O(HW)\) memory; the point here is precisely **not** expanding counts — one pass over the mask plus a merge of two run lists.
- **vs `searchsorted-bincount`**: that is a value→bucket histogram primitive; run merging has nothing to do with histograms.

## Cross-check convention and complexity

- **Main oracle**: `_oracle` expands counts back into `HW` 0/1 values (`flat_b.extend([cur]*n)`), zips them against the column-major flattening of the mask, and recomputes the encoding with an independent linear `_rle` scan; the reference encodes with `unique_consecutive` and merges two pointers — a completely different structure. Sizes cover 1×1, 1×7, 7×1, 5×4 and 6×6, including all-background, all-foreground, checkerboards and arithmetic patterns.
- **Discriminating cases**: the example's `counts_a=[0,1,4,1]`, `counts=[0,1,1,2,1,1]`, inter 2 / union 4 / IoU 0.5; the column-major vs row-major `[0,1,1,1,3]` vs `[0,2,4]` (wrong order gives IoU 1/3); six canonical encodings (all-background `[HW]`, all-foreground `[0,HW]`, leading 0, trailing background, alternating runs, no leading 0); self-comparison identity; subset / disjoint / empty-vs-full / both-empty; bool vs uint8, int32 vs int64 agreement and input immutability; eighteen invalid-input cases.
- **Non-obvious measured facts**: in the example A's row-major and column-major flattenings **coincide**, so the example itself cannot separate the orders — the discriminating check is `check_column_major`; interpreting `[0,2,4]` as column-major leaves exactly 1 intersection pixel, an IoU of exactly \(1/3\), which is the fingerprint of a transposed implementation.
- **Complexity**: encoding must read the mask once, \( O(HW) \) time plus one flattening copy; the merge is \( O(R_a+R_b) \). Judge scale is at most 6×6, far below the limit. Production code (COCO `rleIou`) has the same structure: `rleArea` for areas and a merge for the intersection, never a dense \(HW\) mask.
