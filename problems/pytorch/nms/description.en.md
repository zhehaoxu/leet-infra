# Non-Maximum Suppression (NMS)

## Problem Description

An object detector emits many heavily overlapping candidate boxes for the same object. NMS is responsible for
**deduplication**: repeatedly keep the box with the highest current score and suppress every box whose IoU with it exceeds the threshold, until the candidates run out.
It is the last stage of almost every detection/segmentation/pose pipeline
(Faster R-CNN, YOLO and DETR post-processing all rely on it).

## Rules

- Implement `solve(boxes, scores, iou_thresh)`, where `boxes` has shape `(N,4)` (`[x1,y1,x2,y2]`) and
  `scores` has shape `(N,)`.
- The greedy procedure: (1) sort by score descending; (2) take the box at the head of the queue into `keep`; (3) remove from the remaining candidates every box
  whose IoU with that box is `> iou_thresh`; (4) repeat until the candidates are empty.
- Return the **original indices** of the kept boxes (a `long` tensor), ordered from the highest score to the lowest.
- IoU is computed as in the "Intersection over Union (IoU)" problem (corner format, `clamp_min(0)`, a union of 0 counts as 0).

## Example

```
Input: boxes = [[0,0,10,10],   score 0.9
              [1,1,11,11],   score 0.8   (heavily overlaps box 0)
              [50,50,60,60], score 0.7
              [51,51,61,61]] score 0.6   (heavily overlaps box 2)
     iou_thresh = 0.5
  take box 0 → suppress box 1 (IoU is large); left [2,3]; take box 2 → suppress box 3; nothing left
Output: keep = [0, 2]
```

## Constraints

- float32 input; `iou_thresh ∈ (0,1)`
- The returned indices are unique and sorted by descending score; `scores` has no ties, so the result is deterministic
