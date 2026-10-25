# Intersection over Union (IoU) in Detail

## Geometry and formula

The intersection of two axis-aligned rectangles $[x_1,y_1,x_2,y_2]$ is still a rectangle (or empty). The left and right edges of the intersection take "the larger left bound,
the smaller right bound":
$$ \text{ iw}=\max(0,\min(x_{2a},x_{2b})-\max(x_{1a},x_{1b})),\quad \text{ih likewise} $$
$\text{iw}\cdot\text{ih}$ is the intersection area, and `clamp_min(0)` handles non-overlapping boxes (otherwise the width would be negative). The union uses inclusion–exclusion:
$\text{area}_a+\text{area}_b-\text{inter}$. IoU is their quotient, in $[0,1]$.

## Coordinate formats: xyxy vs xywh

This problem uses $[x_1,y_1,x_2,y_2]$ (corners). Another common format is $[x_c,y_c,w,h]$ (center + width/height) or
$[x,y,w,h]$ (top-left + width/height, e.g. COCO raw annotations). Mixing formats is one of the most insidious bugs in detection code—
confirm the convention before computing, and convert if necessary.

## IoU variants

- **GIoU** (Generalized): adds a penalty for "the part of the smallest enclosing box not covered", so that non-overlapping boxes still have a gradient direction;
  used for detection regression losses.
- **DIoU / CIoU**: further penalize center distance / aspect-ratio consistency, converge faster, and are the default box losses in YOLOv4+.
- **IoU thresholds**: in detection, IoU ≥ 0.5 is often judged TP; mAP@0.5:0.95 averages over multiple thresholds.

## Relation to NMS

The "NMS" problem in this collection turns the pairwise IoU here into an **IoU matrix**, used to decide which highly overlapping candidates should be suppressed.
`_iou_matrix` uses broadcasting to turn $(N,4)$ into $(N,N)$, equivalent to calling this problem's `solve` on every box pair.
