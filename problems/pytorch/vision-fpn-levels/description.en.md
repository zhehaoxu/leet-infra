# Detection: FPN Top-Down Fusion and Level Accounting

## Problem

FPN (Feature Pyramid Network, Lin et al., CVPR 2017) exists because detectors need features at many scales while a backbone provides only a bottom-up trunk — deep layers are semantic but coarse, shallow layers are fine but weak. FPN adds a **top-down** pathway that carries deep semantics back to shallow levels: each level first applies a 1×1 lateral convolution to unify the channel count to `fpn_dim`, then adds the previous (coarser) fused map upsampled with nearest-neighbour to the current resolution.

What is actually easy to get wrong in production is not the convolution or the addition but a **ledger**: how large each level is, what its stride is, how many parameters the laterals hold, how many anchors sit at each location and in total. That ledger feeds anchor generation, loss assignment and inference decoding; one wrong entry trains without errors and quietly costs mAP. In particular:

- strides are **declared** configuration, never inferred from feature-map sizes (sizes need not be powers of two);
- the optional P6 comes from pooling P5, so it has **no lateral convolution** but still needs its own stride (twice the previous level) and anchor count;
- non-integer upsampling (e.g. 3→5) must follow the integer index rule; interpolation would produce different values.

The task is to get "1×1 lateral → top-down nearest-neighbour add → optional P6 pool" and the whole ledger right in one pass. CPU only, deterministic, no RNG.

## Rules

The complete signature, parameter meanings and defaults live only in the starter docstring, which is the sole judging contract.

- The input is a list of backbone feature maps ordered from high to low resolution, each \( (N,C_i,H_i,W_i) \), with a shared \(N\), free \(C_i\), and \(H_i,W_i\ge1\).
- Lateral: level \(i\) applies a 1×1 convolution with a \( (\text{fpn\_dim},C_i) \) 2-D weight, producing \( (N,\text{fpn\_dim},H_i,W_i) \); a bias may be supplied per level or be None, and when present it is a 1-D tensor of shape \( (\text{fpn\_dim},) \). `fpn_dim` must be the same for every level.
- Top-down: the coarsest fused map is just its lateral output; for \( i=L_{\text{in}}-2,\dots,0 \), upsample \( \text{fused}[i+1] \) to \( (H_i,W_i) \) with the **integer nearest-neighbour** rule and add it to \( \text{lateral}[i] \):
  \[
  \text{src\_y}=\left\lfloor \frac{\text{dst\_y}\cdot H_{\text{up}}}{H_i}\right\rfloor,\qquad
  \text{src\_x}=\left\lfloor \frac{\text{dst\_x}\cdot W_{\text{up}}}{W_i}\right\rfloor .
  \]
  No interpolation, no clamping; for a non-integer ratio (3→5) this maps source rows/columns as \( [0,0,1,1,2] \). The target size comes from this level's lateral output, so the addition aligns by construction.
- Optional P6: only when requested, max-pool the fused P5 with `kernel_size=2, stride=2, padding=0` to \( (\lfloor H_5/2\rfloor,\lfloor W_5/2\rfloor) \) — odd sides drop their last row/column and no ceil padding is applied; \(H_5,W_5\ge2\) is required, otherwise `ValueError`. P6 has no lateral convolution.
- Ledger (a dict with fixed keys):
  - the fused feature list, length \(L=L_{\text{in}}+1\) with P6 or \(L_{\text{in}}\) without;
  - per-level sizes as int64 \( (L,2) \);
  - per-level strides \( (L,) \): by default level \(i\) uses \(2^{2+i}\) (4/8/16/32 starting from C2); when supplied explicitly they are recorded **verbatim** and never inferred from sizes; P6's stride is twice the coarsest level's;
  - per-level lateral parameter counts \( (L_{\text{in}},) \): \( \text{fpn\_dim}\cdot C_i + (\text{fpn\_dim}\ \text{if that level has a bias}) \), **excluding P6**, plus their sum;
  - the number of top-down "upsample + add" steps, \( L_{\text{in}}-1 \) (0 for a single level; P6 pooling is not counted);
  - per-level anchor counts \( (L,) \) = anchors per location \( \times H_l\times W_l \), plus their sum.
- Invalid input raises `ValueError`: an empty feature list or a non-4-D / non-tensor / \(H,W<1\) entry; weight, bias or stride list length not equal to the level count; a weight that is not a \( (\text{fpn\_dim},C_i) \) 2-D tensor (`fpn_dim<1`, channel mismatch, or inconsistent `fpn_dim` across levels); a bias that is neither None nor \( (\text{fpn\_dim},) \); a non-positive stride; anchors per location \(<1\); P6 requested while P5 has \(H\) or \(W<2\). No input tensor is modified.

## Example

Two levels, `fpn_dim=2`. The first is \(2\times2\) with two channels, the second \(1\times1\) with three channels:

```
f0 (ch0) = [[1,  2],      f0 (ch1) = [[10, 20],
            [3,  4]]                  [30, 40]]
f1 = [5, 6, 7]^T   (1x1, three channels)
```

Lateral weights: the first level uses the identity (channel pass-through, no bias), the second uses `[[1,1,0],[0,1,1]]` with bias `[0,1]`:

```
lateral0 = f0
lateral1 : ch0 = 5+6+0 (+0) = 11
           ch1 = 0+6+7 (+1) = 14
```

Top-down: upsample `lateral1` to \(2\times2\) and add elementwise,

```
P5 = [[[11]], [[14]]]                       (the coarsest level is its lateral output)
up = each P5 cell replicated into 2x2 -> ch0 all 11, ch1 all 14
P4 = lateral0 + up:
     ch0 = [[1+11, 2+11], [3+11, 4+11]] = [[12, 13], [14, 15]]
     ch1 = [[10+14, 20+14], [30+14, 40+14]] = [[24, 34], [44, 54]]
```

Ledger:

```
sizes             = [[2,2], [1,1]]
strides           = [4, 8]                (default 2^(2+i))
lateral_params    = [2*2, 2*3+2] = [4, 8] (the second level has a bias)
num_lateral_params= 12
num_upsample      = 1
anchors_per_level = [3*4, 3*1] = [12, 3]
num_anchors       = 15
```

Note that strides 4/8 are not consistent with \(2\times2\) feature maps — that is the point: strides are a **declared** ledger entry (starting from C2 by default) and are never inferred from \(H\).

## Constraints

- The returned dict has exactly the keys `features`, `sizes`, `strides`, `lateral_params`, `num_lateral_params`, `num_upsample`, `anchors_per_level`, `num_anchors`; fused features keep the input dtype/device.
- The judge uses absolute tolerance \( 10^{-5} \) for fused features (float32 input) or \( 10^{-12} \) (float64 input); `torch.equal` for `sizes`/`strides`/`lateral_params`/`anchors_per_level`; exact equality for the three scalars.
- The upsampling indices, P6's floor, cross-level `fpn_dim` consistency, strides not being inferred from sizes, P6 not contributing lateral parameters, and `num_upsample = L_in - 1` are all part of the contract.
- CPU only, deterministic, no RNG; a single level, non-square features, non-integer upsampling and odd-sided P6 must all produce the above results rather than raising.
