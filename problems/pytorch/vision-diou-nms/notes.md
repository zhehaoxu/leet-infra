# 出题笔记：vision-diou-nms

## 回源

- 论文 1911.08287（AAA1 2021, Zheng et al.）式 (3)：`DIoU = IoU − ρ²(b, b_gt)/c²`；DIoU-NMS 即 Algorithm 1 中判据替换。
- 官方仓 Zzhiter/DIoU-NMS 已 404（与 abhi2610/Ohem-caffe 同命运），以论文公式为唯一口径；mmcv/ultralytics 均无独立 DIoU-NMS 公开实现（torchvision nms 不支持），notes 声明转写依据。

## 关键口径判断

- **c² 是最小外接框的对角线平方**，不是 IoU 分母的 union，也不是两框各自对角线——变异 `enclose-swap`（min/max 反转）与 `c-not-squared` 钉住。
- **ρ² 用中心点**，不是角点——`rho-center-off` 钉住。
- **方向性设计**：分离框 DIoU ≤ 0、正阈值永不抑制——这是判据换元的直接推论，测试单独钉（`check_disjoint_never_suppressed`）。
- 本题真正的命题点在 `check_diou_catches_iou_miss`：同一对框在 IoU-NMS 下抑制、DIoU-NMS 下不抑制（0.0345 > 0.03 vs −0.139 < 0.03）——「中心惩罚项压过重叠项」的数值证据。
- NMS 流程与 `nms`(66) 完全同构（降序稳定、贪心、严格大于），变异 `keep-all`（不删除已处理）钉流程正确性。

## 边界划定

- 与 `nms`(66)：判据 IoU；`soft-nms`(308)：分数衰减不删除；`vision-rotated-nms`(379)：旋转框。本题判据公式换元、流程照旧。
- 不做 DIoU/CIoU loss（与损失侧划界，可作后续候选）；不做 rotated DIoU。

## 测试设计

- oracle `_py` 独立重放（弹首个、逐对比较，无共享工具函数）。
- 9 个 check：DIoU 数值锚点（2/63 恰等边界）、ρ=0 退化、分离不抑制、重合抑制、**换判据差异对照**、分数并列稳定、三框链、随机 200 组、非法输入 6 档。
- 变异 9/9：惩罚项移除/符号反转、外接框反转、c² 变 c、中心错位、≥、keep-all、抑制自映射、面积公式改和。
