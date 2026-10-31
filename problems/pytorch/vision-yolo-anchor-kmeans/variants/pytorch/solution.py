"""YOLOv2 anchor k-means（1−IoU 距离）账（solution）。

转写 darknet scripts/kmeansiou.c。
"""

import math


def iou_wh(b, c):
    w1, h1 = b
    w2, h2 = c
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union


def _closest_center(box, centers):
    best = 0
    best_dist = 1.0 - iou_wh(box, centers[0])
    for j in range(1, len(centers)):
        d = 1.0 - iou_wh(box, centers[j])
        if d < best_dist:          # 严格小于：并列取小下标
            best_dist = d
            best = j
    return best


def solve(boxes, k, init_indices):
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("boxes must be a non-empty list")
    for b in boxes:
        if not isinstance(b, list) or len(b) != 2 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0
            for v in b
        ):
            raise ValueError("each box must be [w, h] positive finite floats")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be int >= 1")
    if not isinstance(init_indices, list) or len(init_indices) != k or not all(
        isinstance(i, int) and 0 <= i < len(boxes) for i in init_indices
    ):
        raise ValueError("init_indices must be k valid box indices")

    n = len(boxes)
    cols = 2
    centers = [[float(boxes[i][0]), float(boxes[i][1])]
               for i in init_indices]
    assignments = [0] * n

    def expectation():
        converged = True
        for i in range(n):
            closest = _closest_center(boxes[i], centers)
            if closest != assignments[i]:
                converged = False
            assignments[i] = closest
        return converged

    def maximization():
        nonlocal centers
        counts = [0] * k
        sums = [[0.0] * cols for _ in range(k)]   # zero_matrix quirk
        for i in range(n):
            counts[assignments[i]] += 1
            sums[assignments[i]][0] += boxes[i][0]
            sums[assignments[i]][1] += boxes[i][1]
        centers = [[sums[j][0] / counts[j], sums[j][1] / counts[j]]
                   if counts[j] else [0.0, 0.0]
                   for j in range(k)]

    if k == 1:
        maximization()             # 官方特判
    while not expectation():
        maximization()

    ious = [iou_wh(boxes[i], centers[assignments[i]]) for i in range(n)]
    avg_iou = sum(ious) / n        # WCSS = mean(1 − dist) = mean iou
    return {"anchors": [[c[0], c[1]] for c in centers],
            "assignments": list(assignments), "avg_iou": avg_iou}
