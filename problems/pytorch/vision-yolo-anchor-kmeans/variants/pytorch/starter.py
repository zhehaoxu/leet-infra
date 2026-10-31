"""YOLOv2 anchor k-means（1−IoU 距离）账（starter）。

口径钉死（darknet scripts/kmeansiou.c，AlexeyAB 版）：

- `iou_wh(b, c)`：仅宽高、中心对齐——`inter = min(w1,w2)·min(h1,h2)`，
  `union = w1·h1 + w2·h2 − inter`；任一维 ≤ 0 时 IoU = 0。
- `dist = 1 − iou`。
- `closest_center`：遍历 centers，`new_dist < best_dist`（严格）才更新
  ——并列取**小下标**。
- Lloyd 循环（assignments 初始全 0）：
  ```
  if k == 1: maximization()        # 官方特判
  while not expectation(): maximization()
  ```
  expectation 重算全部 assignment，与旧值全同 → 收敛退出（不再 M 步）；
  maximization 清零累加求均值，**counts=0 的簇质心保持 (0,0)**（官方
  zero_matrix quirk，不重采样）。
- 初始化：`centers[i] = boxes[init_indices[i]]`（下标显式入参，替代
  smart_centers 的 D² 随机采样；该采样逻辑不入题）。
- 输出 {"anchors": [[w,h]...], "assignments": [...],
  "avg_iou": mean(1−dist)=WCSS}。
- 校验：boxes 非空、每项 [w,h] 为正有限浮点、k ≥ 1、init_indices 长
  度 = k 且下标合法。
"""

import math


def iou_wh(b, c):
    raise NotImplementedError


def solve(boxes, k, init_indices):
    raise NotImplementedError
