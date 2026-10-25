# RoI Pooling

## Problem Description

RoI Pooling is a key operator in Faster R-CNN: it maps the rectangle of a candidate box (Region of Interest, ROI) in the **original image**
coordinate system onto the feature map via `spatial_scale`, then **splits this rectangle (quantized to integer grid points) evenly into
`pooled_size × pooled_size` bins and takes the max within each bin's window**, thereby normalizing ROIs of arbitrary size into
a fixed-size feature. Its "quantization rounding" introduces position error, exactly what RoI Align later eliminates.

## Rules

- Implement `solve(features, rois, pooled_size, spatial_scale)`.
  - `features` has shape `(N, C, H, W)`.
  - `rois` has shape `(K, 5)`, each row `[batch_index, x1, y1, x2, y2]`, coordinates at the **original image** scale (float).
  - `pooled_size` is an integer $P$ (output is $P\times P$, square); `spatial_scale` is the original-image→feature-map scale factor.
- For each ROI:
  1. Map and quantize to the feature map: $f_{x1}=\lfloor x_1 s\rfloor,\ f_{y1}=\lfloor y_1 s\rfloor,\ f_{x2}=\lfloor x_2 s\rfloor,\ f_{y2}=\lfloor y_2 s\rfloor$.
  2. Split the integer rectangle $[f_{x1},f_{x2}]\times[f_{y1},f_{y2}]$ into $P\times P$ bins using **integer division**;
     the row range of bin $(i,j)$ is $[\,f_{y1}+(f_{y2}-f_{y1})\lfloor i/P\rfloor,\ \dots)$, i.e.
     $$ y_{lo}=f_{y1}+(f_{y2}-f_{y1})\cdot i\ /\ P\ (\text{integer division}),\quad y_{hi}=f_{y1}+(f_{y2}-f_{y1})\cdot(i+1)\ /\ P $$
     Same for $x$.
  3. If a bin is empty ($y_{hi}\le y_{lo}$ or $x_{hi}\le x_{lo}$), its output is 0; otherwise **clip** the range to
     $[0,H]\times[0,W]$ and take the **max** within the window for each channel.
- Output shape `(K, C, P, P)`.

> torchvision is not available in this environment; correctness is judged against an **independent numpy reference implementation** (the judge verifies with a separate numpy implementation).
> Implement strictly per the quantization / integer-division / clipping / max rules above.

## Example

```
Input: features = arange(16).reshape(1,1,4,4)
       [[ 0, 1, 2, 3]
        [ 4, 5, 6, 7]
        [ 8, 9,10,11]
        [12,13,14,15]]
     rois = [[0, 0, 0, 3, 3]],  pooled_size=2, spatial_scale=1.0
   fx1=0,fy1=0,fx2=3,fy2=3; integer division splits it into 2×2:
     bin(0,0)=rows[0,1)cols[0,1)={0}          max=0
     bin(0,1)=rows[0,1)cols[1,3)={1,2}        max=2
     bin(1,0)=rows[1,3)cols[0,1)={4,8}        max=8
     bin(1,1)=rows[1,3)cols[1,3)={5,6,9,10}   max=10
Output: [[[[0, 2], [8, 10]]]]   shape (1,1,2,2)
```

## Constraints

- float32 input; `batch_index` is an integer index (cast with `int`)
- `pooled_size >= 1`; `spatial_scale > 0`; bin boundaries use **integer floor division**
- Out-of-bounds ROI ranges are clipped to the feature-map boundaries; empty bins output 0
- Do not use `torchvision.ops.roi_pool`; hand-write the quantized binning + window max
