# YOLOX SimOTA 动态标签分配账

## 题面

YOLOX 抛弃 anchor set 与固定 label assignment，用 SimOTA 在训练中动态
决定"每个 GT 该教哪些预测格"。分配由三段组成：几何粗筛（中心邻域）、
代价矩阵（cls + IoU）、按 dynamic-k 取最小 cost + 冲突仲裁。给定 GT、
anchor 网格与预测，复现官方 `get_assignments` 的完整账。

给定：
- `gt_bboxes`：G×4 cxcywh（像素坐标）
- `gt_classes`：G 个 int（0..num_classes−1）
- `expanded_strides / x_shifts / y_shifts`：各 A 个（anchor 格点）
- `cls_preds`：A×C raw logits（sigmoid 前）
- `obj_preds`：A 个 raw logits
- `bboxes_preds`：A×4 cxcywh（像素）
- `num_classes`：int

## 处理链

1. **几何约束**（get_geometry_constraint）：
   - anchor 中心 `= (shift + 0.5)·stride`（每 anchor 用自己的 stride）；
   - 邻域 = GT 中心 ± `1.5·该 anchor 的 stride`（**半径随 stride 缩放**，
     非固定像素）；
   - 命中 = 四向 delta 的 min **> 0**（严格，压线不算）；
   - `anchor_filter` = 在任一 GT 邻域内的 anchor；只有它们进候选。
2. **候选 IoU**（bboxes_iou xyxy=False，cxcywh）：
   `tl = max(c−w/2)`、`br = min(c+w/2)`、面积 = w·h（无 +1 无 eps）；
   **任一维 `tl<br` 不严格成立 → IoU = 0**（相切不算重叠）。
3. **cost**（G×n_cand）：
   - cls 分数 `p = √(σ(cls_logit)·σ(obj_logit))` 逐类；
   - cls 项 = Σ_c BCE(p_c, one_hot(gt_class))（C 项全加）；
   - IoU 项 = 3.0·(−log(iou + 1e-8))；
   - 几何项 = 1e6（该 anchor 不在该 GT 邻域时，否则 0）。
4. **dynamic-k**：每 GT 取其 top-10（不足全取）IoU 之和
   `k = max(int(和), 1)`（**int 截断非 round**），选该 GT cost 最小的
   k 个候选（平分取小下标）。
5. **冲突仲裁**：候选被 ≥2 个 GT 选中 → 该列全清，只留 cost 最小的 GT
   （平分取小 GT 下标）。
6. **输出**：最终正样本 = 被某 GT 选中的候选；已选 anchor 的 matched
   gt/class/IoU 逐 anchor 输出（未匹配 −1/−1/0.0）。

## 输出 ledger

```json
{
  "anchor_filter": [bool×A],
  "is_in_centers": [[bool×A]×G],
  "ious":  [[f64×n_cand]×G],          // 候选序
  "cost":  [[f64×n_cand]×G],
  "dynamic_ks": [int×G],
  "matching_matrix": [[0|1×n_cand]×G], // 仲裁后
  "fg_mask": [bool×A],
  "num_fg": int,
  "matched_gt_inds": [int×A],          // -1 = 未匹配
  "gt_matched_classes": [int×A],
  "pred_ious_this_matching": [f64×A]   // 0.0 = 未匹配
}
```

## 输入约束

- G≥1、A≥1；bbox 行 4 个有限值且 w/h>0；class∈[0,C)；stride≥1；
  cls 行长 == num_classes；各列表长度一致；违规抛 `ValueError`。

## 样例

GT `(100,100,20,20)`，anchor `(stride=8, x_shift=12, y_shift=12)`：
中心 (100,100)，dist=12 → 命中；同心 pred `(100,100,14.142,14.142)`
→ IoU=0.5（pred ⊂ gt，iou = p²/400）。若它是唯一候选：dynamic_k=
max(int(0.5),1)=1，cost = Σ_c BCE + 3·(−log(0.5+1e-8)) → 该 anchor
被选，`num_fg=1`、`matched_gt_inds=[0]`。
