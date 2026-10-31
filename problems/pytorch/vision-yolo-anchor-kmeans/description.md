# YOLOv2 anchor k-means（1−IoU 距离）账

## 背景与位置

YOLOv2（1612.08242 §2.2）不用手工先验框，而在训练集 GT 宽高上跑 k-means，距离用 `1 − IoU` 替代欧氏距离（避免大框主导误差）。官方转写源是 darknet `scripts/kmeansiou.c`（AlexeyAB 版 `detector.c calc_anchors` 同源）。

## 精确账目

1. **IoU 仅宽高、中心对齐**：`inter = min(w1,w2)·min(h1,h2)`、`union = w1h1 + w2h2 − inter`；非正维 → IoU 0。`dist = 1 − IoU`。
2. **closest_center**：遍历 centers，`new_dist < best_dist`（严格）才更新——**并列取小下标**。
3. **Lloyd 循环**（assignments 初始全 0）：
   - `k == 1` 特判：先跑一次 maximization（否则全 0 assignments 使首个 expectation 假收敛，质心停留初始值）；
   - `while not expectation(): maximization()`——expectation 全部不变才收敛，收敛轮不再 M 步。
4. **空簇官方 quirk**：maximization 先清零累加，counts=0 的簇**质心保持 (0,0)**（不重采样、不删除）；下一轮它与任何 box 的 IoU=0（dist=1）。
5. **WCSS = mean(1 − dist) = mean IoU**（官方命名 WCSS 但实际是平均 IoU）。
6. 初始化：`centers[i] = boxes[init_indices[i]]`（下标显式入参）。

## 约束

1. boxes 非空、每项 [w,h] 正有限浮点；k ≥ 1；init_indices 长度 = k 且下标合法。
2. 禁改入参；无 max_iter——以官方「分配不变」为唯一收敛条件。

## 数据规模

- n ≤ 10⁴、k ≤ 20（官方 `1 - IoU` 距离每次 O(k)）。

## 验收锚点（节选）

- `boxes=[[1,1],[1,2],[10,10]]`、k=2、init=[0,2] → anchors `[[1,1.5],[10,10]]`、assignments `[0,0,1]`、avg_iou=(2/3+3/4+1)/3。
- k=1、`[[2,2],[4,6]]`、init=[0] → 特判 M 步后 anchors `[[3,4]]`、avg_iou=(1/3+1/2)/2。
- k=3、`[[1,1],[10,10]]`、init=[0,1,1] → M 步 cluster2 空 → anchors `[[1,1],[10,10],[0,0]]`、avg_iou=1。
- k=2、`[[1,1],[2,2]]`、init=[0,0] → assignments 全 0 首轮即收敛，quirk 不触发、anchors 停初始值。
