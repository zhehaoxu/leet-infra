# Non-Maximum Suppression (NMS) in Detail

## The Essence of Greedy Deduplication

NMS is a greedy "take the maximum and suppress the neighbourhood": after sorting the candidates by confidence, the highest-scoring box is certainly trusted and kept,
and every box whose overlap (IoU) with it exceeds the threshold is deleted as a "duplicate detection of the same object"; the rest then repeat the process. It compresses
"a pile of ambiguous candidates" into "one representative box per object".

## Complexity and Implementation Pitfalls

A naive implementation computes the IoU of the current box against all remaining boxes every round, which is $O(N^2)$ in the worst case; this solution computes the $(N,N)$ IoU matrix in one pass,
then filters with a list - the logic is clear, but the matrix costs $O(N^2)$ memory. Engineering optimizations: `torchvision.ops.nms` uses C++/CUDA,
area prefix sums and bucketing by coordinate to avoid the full matrix. When $N$ is large (such as DETR's 300 queries), $O(N^2)$ is still acceptable.

## Thresholds and "Wrongly Deleting Adjacent Objects"

Lowering the threshold suppresses more aggressively (it may collapse two **distinct** objects that sit very close into one), while raising it leaves more duplicate boxes.
In dense small-object scenes this is a pain point, which led to improvements such as **Batched NMS** (multiple classes together, with coordinates offset per class) and
Soft-NMS (instead of hard deletion, **lower** the scores of suppressed boxes, penalizing them harder the more they overlap).

## Class-Agnostic and Multi-Class

Standard NMS is usually done "per class" (boxes of different classes do not suppress each other even when they overlap). This solution processes all boxes together, which suits the single-class case
or the case where boxes have already been grouped by class; with several classes you must first split by class and run each group separately, or add a `class*big` offset to the coordinates.

## Frontier: Learnable / NMS-Free

DETR, YOLOv10 and others try to **remove NMS** - through set prediction or dual label assignment the detector itself emits only one
high-scoring box per object, so post-processing is internalized into the model: end-to-end, with no threshold hyperparameter. Understanding NMS is exactly the prerequisite for understanding the motivation of this work.
