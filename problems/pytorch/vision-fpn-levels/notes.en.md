# In Depth: Detection: FPN Top-Down Fusion and Level Accounting

## Source

- The algorithm: Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017, arXiv:1612.03144. Section 3 gives the top-down pathway: a 1×1 lateral per level unifies channels (256 in the paper), the previous level is 2× nearest-upsampled and added elementwise, then a 3×3 convolution removes the aliasing of upsampling.
- This problem **omits** that final 3×3 output convolution: it changes no size, stride or anchor count and is not part of the ledger; keeping it would only add another weight tensor. The lateral parameter count here is therefore just the 1×1 convolution (\( \text{fpn\_dim}\cdot C_i \) plus bias).
- The P6 construction is not standardised: RetinaNet (arXiv:1708.02002) attaches a 3×3 stride-2 convolution to P5, while Detectron2's `LastLevelMaxPool` uses a 1×1-kernel stride-2 max pool. This problem fixes `kernel_size=2, stride=2, padding=0` max pooling — a self-contained, hand-checkable third convention whose numbers differ from both; the judge follows this one.
- The upsampling index matches `torch.nn.functional.interpolate(..., size=(H_t,W_t), mode="nearest")` (`scale = S_up / S_t`, floor). `mode="nearest-exact"` is a different rounding: measured on 3×3→5×5 it yields source indices \( [0,0,1,2,2] \), unlike the \( [0,0,1,1,2] \) required here.
- The default stride \(2^{2+i}\) (4/8/16/32) assumes the list starts at C2; a list starting at C3 must pass `[8, 16, ...]` explicitly — implementations never infer strides from \(H_i\).
- CPU only, deterministic, no RNG; the 1×1 convolution runs through `F.conv2d` (or an equivalent channel matmul) and upsampling through `index_select` with integer indices.

## Full derivation and index convention

**Lateral (1×1 convolution).** For level \(i\) with features \( F\in\mathbb{R}^{N\times C_i\times H_i\times W_i} \), weight \( W_i\in\mathbb{R}^{\text{fpn}\times C_i} \) and optional bias \( b_i\in\mathbb{R}^{\text{fpn}} \),

\[
\text{lateral}_i[n,o,y,x]=\sum_{c=0}^{C_i-1}W_i[o,c]\,F[n,c,y,x]+b_i[o],
\]

i.e. a per-pixel matrix product \( W_i\,F[n,:,y,x] \). Parameter count \( \text{fpn}\cdot C_i+(\text{fpn}\ \text{if a bias exists}) \). Reshaping the 2-D weight to a \( (\text{fpn},C_i,1,1) \) kernel and calling `F.conv2d` is equivalent.

**Integer nearest upsampling.** With coarser size \( (H_u,W_u) \) and target \( (H_t,W_t) \),

\[
\text{src}(d)=\left\lfloor \frac{d\cdot H_u}{H_t}\right\rfloor,\qquad d=0,\dots,H_t-1 .
\]

Two properties remove the need to clamp: \( d\le H_t-1 \Rightarrow d\cdot H_u/H_t<H_u \), so \( \text{src}<H_u \), and \( \text{src}\ge0 \) since \( H_u\ge1 \). An exact 2× reduces to \( \lfloor d/2\rfloor \), duplicating every row; a non-integer ratio (\( H_u=3,H_t=5 \)) gives \( [0,0,1,1,2] \) — source rows repeated twice, twice, once, with no interpolation. Measured `F.interpolate(size=(5,5), mode="nearest")` follows exactly this mapping; a float `scale = H_u/H_t` with rounding gives \( [0,1,1,2,2] \) and `nearest-exact` gives \( [0,0,1,2,2] \), both different.

**Alignment on add.** The target size comes from this level's lateral output \( (H_i,W_i) \), so the upsampled tensor and `lateral[i]` already agree; no cropping or broadcasting is needed. Adding first and upsampling later (or adding the coarser level without upsampling) does not even produce matching shapes.

**P6.** Pooling the fused P5 gives

\[
\text{P6}[y,x]=\max_{dy,dx\in\{0,1\}}\text{P5}[2y+dy,\,2x+dx],\qquad
y< \lfloor H_5/2\rfloor,\ x<\lfloor W_5/2\rfloor .
\]

Windows tile without overlap at `stride=2`, so odd sides drop their last row/column (`ceil_mode=False` behaviour). A \(3\times3\) P5 covers only its top-left \(2\times2\), so the maximum cannot include row/column 3. P6 has no lateral convolution, hence `lateral_params` has length \(L_{\text{in}}\) rather than \(L\).

**Ledger.**

- `sizes`: each fused \( (H,W) \), including P6;
- `strides`: default \(2^{2+i}\), recorded verbatim when supplied; P6 \( =2\cdot\text{strides}[-1]\);
- `lateral_params`: per level \( \text{fpn}\cdot C_i+(b_i\ ?\ \text{fpn}:0) \), only up to \(L_{\text{in}}\);
- `num_upsample`: \(L_{\text{in}}-1\); P6 pooling is not an "upsample + add";
- `anchors_per_level`: \(A\cdot H_lW_l\), including P6; the sum is `num_anchors`.

The most insidious "self-consistently wrong" entry is the stride: inferring it as \( \lceil H_{\text{img}}/H_i\rceil \) produces values other than 4/8/16 on non-power-of-two sizes, while FPN strides are structural constants. The contract therefore says "record supplied values verbatim", and a dedicated case pairs \(16\times16\) features with `strides=[3,7,13,29]`.

**dtype/device.** Fused outputs follow the input features; the 1×1 convolution needs weights and features of the same dtype (the judge uses float32/float64 only), and upsampling plus addition are same-dtype operations. A None bias means no addition: numerically equivalent to a zero bias, but it is **not** counted as a parameter.

## Key points and common mistakes

1. **The weight is \( (\text{fpn},C_i) \), not \( (C_i,\text{fpn}) \)**: a transposed weight fails loudly when channels differ, but silently computes the wrong thing when \(C_i=\text{fpn}\).
2. **Upsample before adding**: `fused[i] = lateral[i] + up(fused[i+1])`; adding first produces a different (and shape-invalid) result.
3. **Integer indices, floor**: computing a float `scale` and rounding gives \( [0,1,1,2,2] \) on 3→5 instead of \( [0,0,1,1,2] \).
4. **Non-integer ratios must work**: adjacent FPN levels are usually 2×, but 3→5 is allowed here; "take every other row" implementations go out of bounds.
5. **P6 floors**: odd sides drop their last row/column; `ceil_mode=True` adds one and corrupts P6's size and anchor count.
6. **P6 precondition**: \(H_5\) or \(W_5\) of 1 leaves no room for the window, so `ValueError` rather than an empty tensor.
7. **P6 contributes no lateral parameters**: `lateral_params` has length \(L_{\text{in}}\) and `num_lateral_params` excludes it, while P6 does enter `sizes`/`strides`/`anchors_per_level`.
8. **Strides are declared**: never inferred from \(H_i\); P6's stride is twice the coarsest level's.
9. **`num_upsample = L_in - 1`**: 0 for a single level; P6 pooling is not counted.
10. **Bias is per-level optional**: None and a zero bias agree numerically but not in parameter count; `lateral_bias` length must equal the number of levels.
11. **`fpn_dim` is shared**: every weight's first dimension must equal the same `fpn_dim`, otherwise the addition cannot work.
12. **Sizes come from outputs, not inputs**: `sizes` is recorded from the fused tensors, and with P6 there is one more level than input.
13. **No mutation**: convolution, `index_select` and addition are out-of-place; the `features` list must not have P6 appended to it (the result is a new list).
14. **dtype follows the input**: float64 input must yield float64 output and must not be demoted by Python scalars or a global default dtype.

## Differences from neighboring problems

- **vs `upsample-nearest`**: that problem resizes a single tensor and tests the `scale`/`size` parameterisations; here resizing is one step inside multi-level fusion, and the focus is channel unification, per-level addition and the whole ledger.
- **vs `vision-anchor-match` (same batch)**: that problem **generates anchor boxes** from feature sizes and strides and matches them by IoU; this one produces only the ledger (sizes, strides, per-level anchor counts) that such code consumes, with no box coordinates.
- **vs `vision-fcos-centerness` (same batch)**: FCOS uses `regress_ranges` to assign GT of different scales to different levels, a downstream use of this ledger; no label assignment here.
- **vs `vision-deformable-sample` (same batch)**: deformable sampling interpolates bilinearly at continuous offsets; the upsampling here is integer nearest-neighbour with no sub-pixel positions.
- **vs operator problems like `unfold-dilated-conv` / `unfold-fold-roll`**: those focus on single-tensor unfolding; the 1×1 convolution here is just a channel matmul and not the difficulty.

## Cross-check convention and complexity

- **Main oracle**: `_oracle` runs a five-deep Python loop for the 1×1 convolution (summing channels per \(n,o,y,x\)), copies with integer indices for upsampling, takes explicit `max` for P6, and accumulates the ledger in Python; the reference uses `F.conv2d`, `index_select`, `F.max_pool2d` and vectorised arithmetic. Four configs are covered: 3 levels with N=2 and non-square sizes (6→3, 3→2), 2 non-square levels with P6 and custom strides (2→5), a single level, and 4 levels with P6.
- **Discriminating cases**: the example's elementwise values and full ledger; the 3→5 and 2→4 mappings; P6 windows on 3×3 (max 5, excluding 7/8/9) and 4×4 (`[[6,8],[14,16]]`); parameters \( [25,15,15,5] \) summing to 60; anchors \( [768,192,48,12,3] \) summing to 1023; custom strides `[3,7,13,29]` with P6 at 58; `num_upsample=0` for a single level; float64 input not demoted; nineteen invalid-input cases.
- **Non-obvious measured facts**: in the example `lateral1`'s second channel is \( 6+7+1=14 \), not 13 (the bias participates), which makes P4's second channel `[[24,34],[44,54]]`; P6 of a \(3\times3\) P5 drops the last row/column, so the maximum is 5 rather than 9; the default strides (stride 4 on a 2×2 feature) are deliberately inconsistent with the example's sizes — that is the demonstration that strides cannot be inferred from sizes.
- **Complexity**: laterals cost \( O(N\cdot\text{fpn}\cdot C_i\cdot H_iW_i) \) (matmul), upsampling \( O(\text{fpn}\cdot H_tW_t) \), P6 \( O(\text{fpn}\cdot H_5W_5) \). The judge scale tops out at \(16\times16\times5\), far below the limit. Production FPN uses cuDNN convolutions and `upsample_nearest2d` on GPU and computes the ledger once at build time.
