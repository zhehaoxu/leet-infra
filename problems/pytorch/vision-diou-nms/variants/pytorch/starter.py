"""DIoU-NMS 抑制判据账（starter）。

口径钉死（1911.08287 AAAI 2021，DIoU-NMS；官方 repo 已 404，按论文公式）：

- 框格式 [x1, y1, x2, y2]；中心 ((x1+x2)/2, (y1+y2)/2)。
- `iou(a, b)`：标准交并比（inter/union，退化并集 → 0）。
- `diou(a, b) = iou(a, b) − ρ²/c²`：
  * ρ² = 中心点欧氏距离平方；
  * c² = 最小外接框的 w² + h²（对角线平方）。
- NMS 流程：分数**降序稳定**排序（并列取小下标）→ 贪心保留当前最高分
  → 抑制与已保留框 DIoU > thresh 的剩余框（严格大于）→ 重复。
- 输出 {"keep": [原始下标按保留顺序], "suppressed_by": {被抑制下标:
  抑制它的保留下标}}。
- 校验：boxes 非空、坐标 x2 ≥ x1、y2 ≥ y1、分数有限、thresh ∈ [0, 1]。
"""

import math


def diou(a, b):
    raise NotImplementedError


def solve(boxes, scores, thresh):
    raise NotImplementedError
