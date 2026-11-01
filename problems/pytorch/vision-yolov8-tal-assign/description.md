# YOLOv8 TAL 任务对齐分配账

## 题面

YOLOv8 抛弃 anchor 与 IoU 阈值分配，用 TAL：对齐分
`align = 分类分^α × CIoU^β` 同时衡量"分类准"与"定位准"，每 GT 取
align 最高的 top-k 个中心在框内的 anchor，多 GT 冲突按 overlap 仲裁，
最后把 align 归一化成**软标签**写进目标分类分数。给定预测与 GT，复现
官方 `TaskAlignedAssigner` 的完整账（单图、全 GT 有效）。

给定：
- `pd_scores`：A×C 预测概率（sigmoid 后）
- `pd_bboxes`：A×4 xyxy
- `anc_points`：A×2 anchor 中心
- `gt_labels`：G 个 int；`gt_bboxes`：G×4 xyxy
- `num_classes`、`topk`（默认 13）、`alpha`（1.0）、`beta`（6.0）、
  `stride_val`（16.0）、`eps`（1e-9）

## 处理链

1. **中心候选**：小于 `stride_val` 的 GT 边**按中心对称**扩到
   stride_val；anchor 中心须四向 delta **> eps**（恰等不算）。
2. **对齐分**：
   - `bbox_scores[k] = pd_scores[k][gt_class]`（GT 类位，非最大类分）；
   - `overlap = CIoU(gt, pd)` **clamp 0**（负值归零）；
   - `align = score^alpha · overlap^beta`（beta=6 → overlap 衰减极快）。
3. **top-k**：每 GT 取 align 最大的 topk 个（平分取小 anchor 下标），
   `mask_pos = topk ∧ 中心候选`。
4. **多 GT 仲裁**：anchor 被 ≥2 个 GT 认领 → 留 **overlap 最大**的 GT
   （非 align 最大；平分取小 GT 下标）。
5. **软标签归一**：
   - `pos_align[g] = max_k align[g][k]`（分配内）、
     `pos_overlap[g] = max_k overlap[g][k]`；
   - `norm[k] = max_g align[g][k]·pos_overlap[g]/(pos_align[g]+eps)`
     （仅对分配给 g 的 anchor；**eps 会淹没 align<eps 的行**）；
   - `target_scores[k][gt_class] = norm[k]`（其余类位 0；未分配 anchor
     全 0）。

## 输出 ledger

```json
{
  "expanded_boxes":    [[x1,y1,x2,y2]×G],   // 扩边后
  "mask_in_gts":       [[bool×A]×G],
  "bbox_scores":       [[f×A]×G],
  "overlaps":          [[f×A]×G],
  "align_metric":      [[f×A]×G],
  "mask_topk":         [[0|1×A]×G],
  "mask_pos":          [[0|1×A]×G],         // 仲裁前
  "mask_pos_arb":      [[0|1×A]×G],
  "fg_mask":           [bool×A],
  "target_gt_idx":     [int×A],             // -1 未匹配
  "pos_align_metrics": [f×G],
  "pos_overlaps":      [f×G],
  "norm_align_metric": [f×A],
  "target_scores":     [[f×C]×A]
}
```

## 输入约束

- G≥1、A≥1、**A≥topk**（官方 na<topk 直接报错）；pd_scores 行长 C 且
  值∈[0,1]；bbox 行 4 有限值且 x1<x2、y1<y2；class∈[0,C)；
  anc_points 行 2 有限值；alpha/beta≥0；stride_val/eps>0；
  违规抛 `ValueError`。

## 样例

GT `[10,10,40,40]`、anchor27 中心 (32.5,32.5) 预测框与 GT 重合、
类分 0.9：CIoU=1、align=0.9 → topk 首位；pos_align=0.9、pos_overlap=1
→ norm=0.9·1/(0.9+1e-9)≈1 → `target_scores[27][0]≈1`。
GT `[10,10,12,60]`（w=2<16）扩边为中心对称 `[3,10,19,60]`。
