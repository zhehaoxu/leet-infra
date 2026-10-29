# Notes: Video Patches in Merge-Friendly Order

## Source and scope

The layout transformation is based on `patchify`, lines 275–313 of [`models/qwen2_vl/video_processing_qwen2_vl.py`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py#L275-L313) in `transformers==5.20.2`.

The original batched split shape is:

```text
[B, gt, tau, C, gh/m, m, p, gw/m, m, p]
```

Its permutation is `permute(0,1,4,7,5,8,3,2,6,9)`. This task extracts last-frame padding, spatiotemporal patch packing, and merge-friendly ordering for one video, and adds coordinates. Resizing, normalization, Conv3d, vision encoding, and actual patch merging are out of scope.

The ordinary single-image raster flattening in `mm-patchify` is insufficient here; `mm-patch-merger` has no temporal packing dimension. `N/m²` is only the later merged LLM slot count. This task still returns `N` tokens.

## Reference approach

1. Check `H % (p*m)` and `W % (p*m)` first, so an underlying reshape error does not replace the required `ValueError`.
2. Compute the padding count as `(-T) % tau`, and concatenate repetitions of the last frame. The same rule handles `T=1` with `tau>1`.
3. Reshape the padded video to:

   `[gt, tau, C, gh/m, m, p, gw/m, m, p]`

4. Apply `permute(0,3,6,4,7,2,1,5,8)` to obtain:

   `[gt, gh/m, gw/m, m, m, C, tau, p, p]`

5. Flatten the first five axes into the token axis and the last four into the feature axis. A grid over the same first five axes yields coordinates `(t, block_h*m+inner_h, block_w*m+inner_w)`.

The reference uses `reshape`, not a `view` that assumes contiguous input. Reshaping after permutation may copy data, but must not mutate the input.

## Why it works

For token-axis indices `(t,bh,bw,ih,iw)`, the flattened indices are:

```text
n = ((((t*(gh/m)+bh)*(gw/m)+bw)*m+ih)*m+iw)
h = bh*m+ih
w = bw*m+iw
k = ((c*tau+dt)*p+ph)*p+pw
```

The permutation places exactly the five token axes first and `C,tau,p_h,p_w` last. Consequently, `patches[n,k]` equals

`video[min(t*tau+dt,T-1), c, h*p+ph, w*p+pw]`.

Padded frames only repeat the last frame, and every original pixel is retained. Each `(t,h,w)` occurs exactly once. The coordinates therefore allow elementwise reconstruction of the padded video, confirming that this is reversible rearrangement rather than convolution or downsampling.

## Complexity

Let `T' = ceil(T/tau)*tau`, `P = T'*C*H*W`, and `N = gt*gh*gw`. Time and space including outputs are both `O(P+N)`. Padding and rearrangement may create temporary tensors of the same asymptotic size.

## Tests and pitfalls

There are **13 `check_*` functions**, with `check_example.example = True`. The independent oracle reads individual elements using strictly nested loops over five token axes and four feature axes; it does not reuse the reference reshape/permute. Inverse reconstruction visits tokens in reverse order, writes elements according to returned coordinates, and checks original and padded frames.

Coverage includes odd `T` and last-frame duplication, `T=1`, exact divisibility, `m=1/2/3/4`, non-raster order, rectangles, multi-channel feature order, noncontiguous offset views, all three dtypes, maximum parameters, randomized comparisons, invalid spatial dimensions, input/backing-storage preservation, explicit-padding equivalence, and value-shift invariance.

Common mistakes include zero-padding, placing `tau` before `C`, using global raster order, returning the merged token count, and confusing grid coordinates with pixels or original frame indices.

Code and independent tests passed submission-mode self-testing before the example was executed and the documentation written. Result: **13/13 AC**.

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-video-patch-order --mode submit
```
