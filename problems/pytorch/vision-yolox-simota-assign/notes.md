# YOLOX SimOTA 动态标签分配账 — 实现笔记

## 回源

- 官方仓：Megvii-BaseDetection/YOLOX。
- 路径：`yolox/models/yolo_head.py` 的 `get_assignments`（调
  `get_geometry_constraint` 算几何候选 → 组 cost → 调 `simota_matching`）+
  `yolox/utils/boxes.py::bboxes_iou`（`xyxy=False` 路径，GT 与 pred 都是 cxcywh）。
- 论文：YOLOX (2107.08430)；SimOTA 思想源自 OTA (2108.07755) 的简化版。

## 官方代码逐行账

```python
# get_geometry_constraint
x_centers = ((x_shifts + 0.5) * expanded_strides)      # anchor 中心
center_radius = 1.5
center_dist = expanded_strides * center_radius          # 半径随 stride 缩放！
c_l = x_centers - (gx - center_dist)                    # 四向 delta
...
is_in_centers = center_deltas.min(dim=-1).values > 0.0  # 严格 >
anchor_filter = is_in_centers.sum(dim=0) > 0            # 任一 GT 邻域内

# get_assignments
pair_wise_ious = bboxes_iou(gt, pred, False)            # cxcywh
pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
cls_preds_ = (cls_preds.sigmoid_() * obj_preds.sigmoid_()).sqrt()
pair_wise_cls_loss = F.binary_cross_entropy(...).sum(-1)  # 逐类求和
cost = cls_loss + 3.0 * ious_loss + 1e6 * (~geometry_relation)

# simota_matching
n_candidate_k = min(10, n_anchor)
topk_ious = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # int 截断
for gt: matching_matrix[gt][topk(cost[gt], k, largest=False)] = 1
anchor_matching_gt = matching_matrix.sum(0)
if (anchor_matching_gt > 1).any():                      # 冲突仲裁
    cost_argmin = torch.min(cost[:, mask], dim=0)       # 留 cost 最小 GT
    matching_matrix[:, mask] *= 0
    matching_matrix[cost_argmin, mask] = 1
```

## 关键账点

1. **邻域半径随 anchor 自己的 stride 缩放**：`center_dist = 1.5·stride`。
   同一 GT，stride 8 的候选窗口 ±12px、stride 32 的 ±48px。与 FCOS 固定
   radius、TAL 无几何门都不同。中心 = `(shift+0.5)·stride`（格点右半格）。
2. **几何门严格 `>0`**：anchor 恰在半径边界（delta=0）不算命中。
3. **cxcywh IoU 的 en 语义**：`en = (tl<br).prod(2)`，任一维不严格小于
   → IoU 直接 0；相切（tl==br）时 area_i = 0，结果与 max(0,·) 风格同值
   （`<=` 变异经分析等价：area_i=0·x=0，分母无 +1 无 eps）。面积
   `prod(wh)` 无 +1。
4. **cls 分数组合**：`sqrt(σ(cls)·σ(obj))` 逐类；BCE 对 one_hot 逐类算
   再 `sum(-1)`（C 项全加，负类也贡献 log(1−p)）。
5. **cost 三项**：cls + 3.0·(−log(iou+1e-8)) + 1e6·(~geometry)。
   1e6 只作用于候选集内的"几何不命中"anchor（geometry_relation 是
   G×A_cand，anchor_filter 已筛过一轮——理论上候选都命中至少一个 GT，
   但对单个 GT 行 ~geometry 可为真）。
6. **dynamic_k**：`clamp(int(Σ top-10 IoU), min=1)`。int 截断：4.585→4
   （round 会给 5）。top-10 不足时全取（n_cand<10）。topk 含 0 值填充。
7. **per-GT 选框**：cost 最小的 k 个（largest=False）。平分时取小下标
   （纯 Python sorted 稳定语义；torch.topk 平分顺序不保证——出题锚点
   避开 cost 平分）。
8. **冲突仲裁**：anchor 被 ≥2 个 GT 选中 → 全清，只留该 anchor cost
   最小的 GT（`torch.min(dim=0)` 平分取首遇索引，题面定义小 GT 下标）。
9. **出列制**：anchor_filter 不过的 anchor 不进候选，matching_matrix
   只有 n_cand 列；fg_mask 回写 `fg_mask[fg_mask.clone()] = fg_mask_inboxes`，
   即最终正样本 = 几何候选 ∩ 被某 GT 选中。

## 出题口径

- 输入：gt_bboxes（cxcywh 像素）、gt_classes、expanded_strides/x_shifts/
  y_shifts（A）、cls_preds（A×C raw logits）、obj_preds（A raw）、
  bboxes_preds（A×4 cxcywh）、num_classes。
- 校验：G≥1、A≥1、行长 4 且有限、wh>0、class∈[0,C)、stride≥1、
  cls 行长 == num_classes；空 GT/anchor、0/负 wh、inf 抛 ValueError。
- 输出 ledger：anchor_filter/is_in_centers、ious、cost、dynamic_ks、
  matching_matrix（G×n_cand，仲裁后）、fg_mask（A）、num_fg、
  matched_gt_inds/gt_matched_classes（A，未匹配 −1）、
  pred_ious_this_matching（A，未匹配 0.0）。
- oracle：矩阵式直算（无 topk/argmin 抽象、显式排序）+ 200 组随机对拍
  （seed 706，G∈1..3、A∈4..15、C∈2..4、stride∈{8,16,32}）。

## 等价变异说明

- **IoU 相切 `<=` 变异等价**：一维相切时 area_i = 0·h = 0（或 h·0），两维
  同时相切也是 0——`<`（en=0 提前 return 0.0）与 `<=`（en=1 但 area_i=0）
  的 iou 恒同值（分母两分支同为 area_a+area_b−0）。记 notes 不收。

## 变异测试

15 抓 + 1 等价：半径 1.5→2.5、半径不乘 stride、min>0→>=0、IoU +1、
cls 不 sqrt、obj 不 sigmoid、iou_loss 去 eps、权重 3.0→1.0、惩罚 1e6→1e5、
int→round（边界锚点 top-10 和 4.585）、下限 1→0、top-10→top-5（top-5 和
3.1→k 3）、仲裁取大、不仲裁、per-GT 取最大。
