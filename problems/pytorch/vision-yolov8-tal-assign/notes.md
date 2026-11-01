# YOLOv8 TAL 任务对齐分配账 — 实现笔记

## 回源

- 官方仓：ultralytics/ultralytics 主分支 `ultralytics/utils/tal.py`
  `TaskAlignedAssigner`（单图、全 GT 有效、topk2==topk 关闭次级过滤）。
- 论文：TOOD (2108.07755) 提出 task alignment；YOLOv8 (2301.05495) 采用。
- 现版本与旧版差异：新增 stride_val 小边扩展与 topk2；**forward 无
  topk=min(topk,na) clamp**（na<topk 时 torch.topk 直接 RuntimeError），
  count>1 清零主要服务无效 GT 行。

## 官方代码逐行账

```python
# select_candidates_in_gts：小边扩展 + eps 严格四向
wh_mask = gt_xywh[..., 2:] < self.stride_val        # < 严格
gt_xywh[..., 2:] = where(wh_mask & mask_gt, stride_val, wh)   # 关于中心
mask = (xc - lt_x > eps) & (yc - lt_y > eps) & (rb_x - xc > eps) & (rb_y - yc > eps)

# get_box_metrics
bbox_scores = pd_scores[b, anchor, gt_class]        # GT 类位分数
overlaps = CIoU(gt, pd).clamp_(0)                   # 负值归零
align_values = scores^alpha * overlaps^beta         # 1.0 / 6.0

# select_topk_candidates
topk(metrics, self.topk, largest=True)              # 无 clamp
count_tensor.scatter_add_(...); count_tensor[count>1] = 0

# select_highest_overlaps
max_overlaps_idx = overlaps.max(1).indices          # 按 overlap 仲裁！
is_max_overlaps.scatter_(1, idx, 1)                 # 平分取首个最大
mask_pos = where(mask_multi_gts, is_max_overlaps, mask_pos)

# _forward 归一化
align_metric *= mask_pos
pos_align_metrics = align_metric.amax(-1, keepdim=True)   # 每 GT
pos_overlaps = overlaps.amax(-1, keepdim=True)            # 每 GT
align_metric.mul_(pos_overlaps).div_(pos_align_metrics + eps)
norm_align_metric = align_metric.amax(-2).unsqueeze(-1)   # 每 anchor 跨 GT
target_scores = target_scores * norm_align_metric         # 软标签
```

## 关键账点

1. **小边扩展**：`wh < stride_val`（严格 <，w==stride_val 时扩边恒等
   ——`<=` 变异等价）按**中心对称**扩到 stride_val（[cx−s/2, cx+s/2]，
   非贴边）。中心 = 原框中心。
2. **eps 严格四向**：`delta > eps`，恰等不算。x1=0 时 anchor x=1e-9
   与 eps=1e-9 浮点恰等可构造锚点。
3. **bbox_scores 是 GT 类位分数**：`pd_scores[anchor, gt_class]`——
   不是最大类分。输入 pd_scores 为 sigmoid 后概率（v8DetectionLoss
   传 `.sigmoid()` 后值），题面直接收概率并校验 [0,1]。
4. **CIoU clamp 0**：CIoU 可为负（−ρ²/c² 项）→ clamp 归 0 → align=0。
   完全重合框 iou=1、v=0 → alpha=0/0 → 官方 torch 出 NaN（clamp 不清
   NaN）；题面加 v=0 保护（alpha=0 → CIoU=1），notes 记差异。
5. **beta=6 幂次**：overlap 0.5 → 0.0156；align 动态范围极大。
   **eps 淹没效应**：pos_align < eps（1e-9）时 norm = align·po/(pa+eps)
   ≈ align·po/eps ≪ po——软标签被压半甚至更低，eps 不是装饰。
   官方 beta=6 下 align~1e-10 常态，此效应真实存在。
6. **top-k 与 in_gts 的 AND**：topk 先选（可在框外——metrics 只在
   in_gts 内非零，但 topk 会补零位），`mask_pos = topk · in_gts · mask_gt`
   把框外的 topk 位清掉。账面 topk 与 pos 分开记。
7. **仲裁按 overlap 非 align**：`overlaps.max(1)`——align 大但 IoU 小的
   GT 输。平分取首个最大（小 GT 下标；torch.max dim 语义 CPU 保证首遇）。
8. **norm 链顺序**：align·=mask_pos（仲裁后）→ pos_align/pos_overlap
   （每 GT 跨 anchor max）→ align·po/(pa+eps) → 逐 anchor 跨 GT max →
   写入 one_hot。无分配 anchor 的 norm=0、scores 全 0、idx 记 −1。
9. **get_targets 垃圾标签**：未分配 anchor 的 target_labels 取 gt 0
   （flatten 索引 0）但 scores 乘 fg_mask 清 0——题面 fg=False 记 −1。

## 出题口径

- 输入：pd_scores（A×C 概率）、pd_bboxes（A×4 xyxy）、anc_points（A×2）、
  gt_labels（G）、gt_bboxes（G×4 xyxy）、num_classes、topk=13、
  alpha=1.0、beta=6.0、stride_val=16.0、eps=1e-9。
- 校验：G≥1、A≥1、**A≥topk**（官方 na<topk 会 RuntimeError）、行长/
  概率 [0,1]/xyxy 有序/class 界内/stride_val>0/eps>0。
- 输出 ledger：expanded_boxes、mask_in_gts、bbox_scores、overlaps、
  align_metric、mask_topk、mask_pos（仲裁前）、mask_pos_arb、fg_mask、
  target_gt_idx（−1 未匹配）、pos_align_metrics、pos_overlaps、
  norm_align_metric、target_scores（A×C 软标签）。
- oracle：显式枚举 + 200 组随机对拍（seed 708，topk∈{3,5,8}、
  stride_val∈{8,16}）。

## 等价变异说明

- **扩边阈值 `<` → `<=` 等价**：w==stride_val 时扩边是恒等变换
  （同中心同宽），两分支结果相同。记 notes 不收。

## 变异测试

15 抓 + 1 等价：扩边贴边/含等号、中心门 >=（eps 恰等锚点 x1=0,
ax=1e-9）、CIoU 不 clamp、align 线性、类分取 max、topk 反向、不并
in_gts、仲裁按 align/平分取大/不仲裁、norm 无 eps/分子错/多除 po/
软标签写死 1、未匹配记 gt0。
