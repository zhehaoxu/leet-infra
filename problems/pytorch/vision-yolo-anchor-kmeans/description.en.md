# YOLOv2 Anchor k-means (1-IoU Distance) Ledger

## Background

YOLOv2 (1612.08242 §2.2) replaces hand-picked priors with k-means over GT box widths/heights, using `1 − IoU` as the distance (avoiding large boxes dominating Euclidean error). The authoritative transcription source is darknet `scripts/kmeansiou.c` (AlexeyAB's `detector.c calc_anchors` shares this lineage).

## Exact Ledger

1. **IoU is wh-only, center-aligned**: `inter = min(w1,w2)·min(h1,h2)`, `union = w1h1 + w2h2 − inter`; non-positive dims → IoU 0. `dist = 1 − IoU`.
2. **closest_center**: strict `<` update — ties go to the lower index.
3. **Lloyd loop** (assignments initialized to all 0):
   - `k == 1` special case: run maximization once first (otherwise the all-0 assignments make the first expectation falsely converge, leaving the centroid at its init value);
   - `while not expectation(): maximization()` — converged only when all assignments unchanged; no M-step in the converging round.
4. **Empty-cluster official quirk**: maximization zeroes centers before accumulating, so a count-0 cluster's centroid stays **(0,0)** (no resampling, no deletion); next round its IoU with any box is 0 (dist=1).
5. **WCSS = mean(1 − dist) = mean IoU** (official name says WCSS but the code computes avg IoU).
6. Init: `centers[i] = boxes[init_indices[i]]` (indices as explicit inputs).

## Constraints

1. boxes non-empty, each [w,h] positive finite floats; k ≥ 1; init_indices length k with valid indices.
2. No input mutation; no max_iter — the only convergence condition is the official "assignments unchanged".

## Data Scale

- n ≤ 10⁴, k ≤ 20.

## Acceptance Anchors (excerpt)

- `boxes=[[1,1],[1,2],[10,10]]`, k=2, init=[0,2] → anchors `[[1,1.5],[10,10]]`, assignments `[0,0,1]`, avg_iou=(2/3+3/4+1)/3.
- k=1, `[[2,2],[4,6]]`, init=[0] → after the special-case M step, anchors `[[3,4]]`, avg_iou=(1/3+1/2)/2.
- k=3, `[[1,1],[10,10]]`, init=[0,1,1] → M step leaves cluster2 empty → anchors `[[1,1],[10,10],[0,0]]`, avg_iou=1.
- k=2, `[[1,1],[2,2]]`, init=[0,0] → first E converges, quirk not reached, anchors stay at init.
