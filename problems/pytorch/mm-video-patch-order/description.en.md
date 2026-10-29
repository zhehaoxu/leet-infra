# Multimodal: Video Patches in Merge-Friendly Order

Split a video into spatiotemporal patches, arrange the tokens for subsequent spatial merging, and return each token's grid coordinates.

This is **pure processor-side data rearrangement, not Conv3d**. There are no weights, sums, averages, projections, or actual token merges. Unlike `mm-patchify`, which uses ordinary raster order for a single image, this task includes temporal grouping and merge-friendly ordering. `mm-patch-merger` does not include this temporal packing dimension.

## Inputs and bounds

| Input | Meaning and bounds |
| --- | --- |
| `video` | CPU tensor `[T,C,H,W]`, dtype `int64`, `float32`, or `float64`; `1 <= T <= 9`, `1 <= C <= 4`, `1 <= H,W <= 32` |
| `temporal_patch_size` | Temporal patch size `tau`, an integer in `1..4` |
| `patch_size` | Spatial patch side length `p`, an integer in `1..4` |
| `merge_size` | Spatial merge-group side length `m`, an integer in `1..4` |

Both `H` and `W` must be divisible by `p*m`; raise `ValueError` if either is not. All other inputs are valid. The input may be noncontiguous and must not be mutated.

## Ordering rules

1. If `T` is not divisible by `tau`, repeat the last frame to reach the next multiple. Do not zero-pad or discard trailing frames.
2. Let `gt = ceil(T/tau)`, `gh = H/p`, `gw = W/p`, and `N = gt*gh*gw`.
3. Emit tokens in lexicographic order over these axes, with the last axis varying fastest:
   - Temporal group `t`: `0 <= t < gt`;
   - Spatial block row `block_h`: `0 <= block_h < gh/m`;
   - Spatial block column `block_w`: `0 <= block_w < gw/m`;
   - Within-block row `inner_h`: `0 <= inner_h < m`;
   - Within-block column `inner_w`: `0 <= inner_w < m`.
4. The token's patch-grid coordinate is `(t,h,w)`, where `h = block_h*m + inner_h` and `w = block_w*m + inner_w`. These are not pixel coordinates, and `t` is not an original frame index.
5. Flatten each token's contents in `c, dt, ph, pw` order, last axis fastest. The axes range over `0..C-1`, `0..tau-1`, `0..p-1`, and `0..p-1`, respectively. Each element comes from:

   `video[min(t*tau+dt, T-1), c, h*p+ph, w*p+pw]`

Thus all temporal slices of one channel precede the next channel. When `m>1`, each consecutive group of `m*m` tokens belongs to one spatial block, rather than necessarily following ordinary raster order.

## Return values

Return a tuple `(patches, grid_thw, coords)`:

| Item | Requirement |
| --- | --- |
| `patches` | CPU tensor `[N,C*tau*p*p]`, with the input dtype |
| `grid_thw` | Python integer tuple `(gt,gh,gw)`, not a tensor |
| `coords` | CPU `int64` tensor `[N,3]`; row `n` is token `n`'s `(t,h,w)` |

Return all `N` patch tokens. Later, each group of `m*m` tokens may correspond to one LLM slot, giving `N/m²` slots. Do not perform that merge or return the slot count.

## Example

The input `video` has dtype `int64` and shape `[3,1,2,4]`:

```text
video = [
  [[[ 0,  1,  2,  3], [ 4,  5,  6,  7]]],
  [[[ 8,  9, 10, 11], [12, 13, 14, 15]]],
  [[[16, 17, 18, 19], [20, 21, 22, 23]]]
]
temporal_patch_size = 2
patch_size = 1
merge_size = 2
```

Output:

```text
patches = [
  [ 0,  8], [ 1,  9], [ 4, 12], [ 5, 13],
  [ 2, 10], [ 3, 11], [ 6, 14], [ 7, 15],
  [16, 16], [17, 17], [20, 20], [21, 21],
  [18, 18], [19, 19], [22, 22], [23, 23]
]
grid_thw = (2, 2, 4)
coords = [
  [0,0,0], [0,0,1], [0,1,0], [0,1,1],
  [0,0,2], [0,0,3], [0,1,2], [0,1,3],
  [1,0,0], [1,0,1], [1,1,0], [1,1,1],
  [1,0,2], [1,0,3], [1,1,2], [1,1,3]
]
```

The third frame is duplicated once, so the final eight tokens have identical values at their two temporal positions. The third token comes from grid coordinate `(0,1,0)`, not raster coordinate `(0,0,2)`. This example is implemented and execution-checked by `check_example`.
