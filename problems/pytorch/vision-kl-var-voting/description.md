# KL-Loss variance voting 融合账

## 题面

KL-Loss（CVPR'19）让检测头为每个坐标额外预测一个定位方差 σ。推理时这些
σ 有两个用处：NMS 衰减之后做 **variance voting**——把每个保留下来的框
用"方差小者话语权大"的方式与邻居融合。官方实现藏在
`py_cpu_nms.soft` 的 `stdiou` 分支里，外层是 soft-NMS 贪心循环。
给定检测列表、逐坐标 σ 与参数，请复现官方的完整账。

给定：
- `dets`：N×5 `[x1, y1, x2, y2, score]`（float）
- `confs`：N×4 逐坐标 σ（x1,y1,x2,y2 各一个，>0；越小越确信）
- `thresh`：STD_TH（stdiou 下 >0.1 会被钳到 0.01）
- `iou_sigma`：投票高斯核宽 σ_IoU

## 处理链

1. **成对 IoU（legacy VOC +1 约定）**：
   `inter 宽 = max(0, min(x2)−max(x1)+1)`、面积同 +1；
   `iou = inter / (area_i + area_j − inter)`。全量预计算，
   之后**只交换行列、永不重算**。
2. **贪心主循环（槽位视角）**：在剩余槽位 `[i, n)` 里取分数
   **槽位序首个最大**者（numpy argmax 语义），与槽位 i 交换
   （dets/confs/原索引/IoU 行列四件套同换）。
   ⚠️ pick swap 会改写活跃框的相对顺序——**平分时 pick 依赖槽位序**，
   按原索引序模拟会答错。
3. **variance voting（stdiou 分支）**：
   - 邻居：`iou(pick, k) > thresh`（严格大于；thresh>0.1 先钳 0.01；
     自身必在，iou(i,i)=1）；
   - 逐坐标融合（4 个坐标独立）：
     `coord_j = Σ_k p_k·v_kj/σ_kj² / Σ_k p_k/σ_kj²`，
     其中 `p_k = exp(−(1−iou_k)²/iou_sigma)`；
   - 权重**不含分数**（score 只通过 pick 顺序与衰减起作用）；
   - 融合坐标**不 clip**、IoU **不重算**（已 pick 的框出列后不再参与
     投票，未 pick 的框坐标未变，旧 IoU × 原坐标自洽）。
4. **soft 衰减**：对每个后续槽位框，只要 `iou(pick, k) > 0`
   （**任何正重叠**，与 thresh 无关）：
   `score_k ×= exp(−iou²/0.5)`（σ=0.5 硬编码）。
5. **破线移除**：`score < 0.001` → 与槽位 n−1 交换并收缩 n
   （tail-swap；换过来的框在同位置继续检查）。
6. 处理序即输出序：每轮 pick 融合后的坐标与**当时**（衰减后）分数
   依次输出。

## 输出 ledger

```json
{
  "ious":    [[...]],            // 初始 N×N IoU 矩阵
  "steps":   [{"idx": <原索引>,  // 本轮 pick 的原始索引
               "box": [x1,y1,x2,y2],        // 融合后坐标
               "voters": [...],             // 投票者原索引（升序记账）
               "weights": [[w_x1,w_y1,w_x2,w_y2], ...]}],  // p/σ² 逐坐标
  "final":   [[x1,y1,x2,y2,score], ...],   // 处理序，分数为衰减后
  "removed": [...]               // 破 0.001 被 tail-swap 移除的原索引（时序）
}
```

## 输入约束

- `confs` 与 `dets` 等长，每行恰 4 个**正**有限 σ；
- `dets` 每行恰 5 个有限值；
- `iou_sigma > 0`；`thresh` 有限；违规抛 `ValueError`；
- 空输入合法（返回全空 ledger）。

## 样例

`dets=[[0,0,10,10,0.9],[2,2,12,12,0.7]]`，`confs=[[1,1,1,1],[2,2,2,2]]`，
`thresh=0.3, iou_sigma=0.5`：
IoU=81/161≈0.503（+1 约定）；pick 框 0，voters=[0,1]，
p=exp(−(1−81/161)²/0.5)；融合 x1=(0·1+2·p/4)/(1+p/4)、
x2=(10+12·p/4)/(1+p/4)（p 在单对框时可消去一半 → 坐标偏向 σ 小者）；
框 1 衰减 `0.7·exp(−(81/161)²/0.5)`。第二轮 pick 框 1 时 voters 仅 [1]
（框 0 已出列）→ 坐标不变。
