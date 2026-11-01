# KL-Loss variance voting 融合账 — 实现笔记

## 回源

- 官方仓：ethanhe42/KL-Loss（yihui-he/KL-Loss 重定向至此），Caffe2 Detectron 系。
  论文：*Bounding Box Regression with Uncertainty for Accurate Object Detection*（CVPR'19, arXiv 1809.08545）。
- 路径：`detectron/utils/py_cpu_nms.py::soft`，`cfg.STD.METHOD == 'stdiou'` 分支。
  调用链：`detectron/core/test.py::box_results_with_nms_and_limit` → `cfg.STD_NMS` 时
  `py_nms_utils.soft(dets_j, confidence=confs_j)`，其中 `confs_j = confs[inds, j*4:(j+1)*4]`
  是 **每框每类 4 个 σ**（KL-Loss 头逐坐标预测的定位方差）。
- 注意：`detectron/utils/boxes.py::box_voting` 是 Detectron 原版投票（score 加权平均、
  无 σ），**不是**本题口径；variance voting 只在 `py_cpu_nms.soft` 的 stdiou 分支。

## 官方代码逐行账（stdiou 分支）

```python
def soft(dets, confidence=None, ax=None):
    thresh = cfg.STD_TH
    if cfg.STD.METHOD == 'stdiou' and thresh > .1:
        thresh = 0.01                      # ① 松 thresh 钳到 0.01
    sigma = .5                             # ② soft-NMS 衰减核宽硬编码 0.5
    ...
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # ③ legacy VOC +1 像素约定
    ious = np.zeros((N,N))                 # ④ 全量预计算，之后只交换不重算
    ...
    while i < N:
        maxpos = dets[i:N, 4].argmax()     # ⑤ 槽位序首个最大（numpy argmax）
        maxpos += i
        dets[[maxpos,i]] = dets[[i,maxpos]]        # ⑥ pick swap：四件套同换
        confidence[[maxpos,i]] = ...
        ious[[maxpos,i]] = ...; ious[:,[maxpos,i]] = ...
        ovr_bbox = np.where((ious[i, i:N] > thresh))[0] + i   # ⑦ 严格 >，含自身
        if cfg.STD.METHOD == 'stdiou':
            p = np.exp(-(1-ious[i, ovr_bbox])**2/cfg.STD.IOU_SIGMA)
            dets[i,:4] = p.dot(dets[ovr_bbox,:4] / confidence[ovr_bbox]**2) \
                       / p.dot(1./confidence[ovr_bbox]**2)     # ⑧ 逐坐标融合
        pos = i + 1
        while pos < N:
            if ious[i, pos] > 0:           # ⑨ 衰减门是任何正重叠
                ovr = ious[i, pos]
                dets[pos, 4] *= np.exp(-(ovr*ovr)/sigma)
                if dets[pos, 4] < 0.001:   # ⑩ 破线 tail-swap 移除
                    dets[[pos,N-1]] = ...; confidence[[...]]; ious[[...]]
                    N -= 1; pos -= 1
            pos += 1
        i += 1
```

## 关键账点

1. **+1 像素约定**：inter 宽高与面积都 +1（VOC 时代惯例）。`iou = inter / (area_i + area_j − inter)`，
   分母无 +1。这使 IoU>0 的门在"恰好贴边"时也成立（w = max(0, ·) 截断后为 0 → 不重叠）。
2. **thresh 钳制**：`stdiou 且 thresh > 0.1 → 0.01`。即官方推理时 STD_TH 无论设 0.3/0.5
   都等效 0.01——几乎一切正 IoU 邻居都投票。
3. **槽位序语义**（本题最深陷阱）：
   - argmax 是 **槽位序** 的首个最大（numpy argmax 平分取先）。
   - pick swap（maxpos↔i）会把原槽位 i 的框换到 maxpos 槽位——**活跃框的相对顺序被改写**。
   - tail-swap 移除（pos↔N−1）同样改写。
   - 随机对拍曾抓出：8 框、1 与 5 分数同为 0.217 且不重叠，pick 序 `[6,7,3,2,0,5,1,4]` 里
     5 先于 1，正是因为第 5 轮 swap(4,6) 把 4 换到了槽位 6…… 简言之**按原索引序近似槽位序
     会答错平分 pick**。oracle 必须完整模拟 swap。
4. **逐坐标融合**：`coord_j = Σ_k p_k·v_kj/σ_kj² / Σ_k p_k/σ_kj²`，σ 是 (N,4)。
   - 自身必在 voters（iou(i,i)=1 > 0.01）；单框融合还原自身。
   - p 相同的成对框融合时 p 可消去（→ 坐标偏向 σ 小者），多框则不能。
   - 权重**不含分数**：论文最终公式 (6) 与官方实现都没有 p_i^γ 项（score 只通过
     pick 顺序与衰减间接起作用）。早期博客版有 score 项，勿混。
5. **融合后不回写 IoU**：ious 全量预计算后只交换不重算。融合改变 pick 坐标，但
   - 后续轮的 voters 候选从 i+1 槽位起，已 pick 的框**出列后不再参与任何投票**；
   - 尚未 pick 的框坐标还是原坐标，所以"旧 IoU × 原坐标"没有实际不一致——
     融合坐标是一次性输出。这解释了为何官方可以放心不重算。
6. **衰减门 IoU>0**：与投票门 thresh 无关。IoU∈(0,thresh) 的框不投票但照样衰减。
   衰减核 `exp(−iou²/0.5)`，σ=0.5 硬编码（与 STD.IOU_SIGMA 是两个参数）。
7. **破线移除**：score<0.001 tail-swap 出列。重复框链式衰减：iou=1 时每轮 ×exp(−2)≈0.135，
   0.5 起步第 4 轮破线。
8. **无 clip**：融合坐标可越界（σ 大的框被 σ 小的邻居拉出图像也不裁剪）。

## 出题口径

- 输入：dets (N×5)、confs (N×4 σ)、thresh、iou_sigma；固定 METHOD='stdiou'、soft 衰减、
  σ_soft=0.5、floor=0.001。
- 输出 ledger：ious（初始矩阵）、steps（idx=原索引、box=融合坐标、voters/weights
  按**原索引升序**规范化记账——官方按槽位序累加，浮点差 ~1e-16，记账排序便于断言）、
  final（处理序 [x1,y1,x2,y2,衰减后分数]）、removed（tail-swap 移除的原索引时序）。
- 校验：confs 与 dets 等长且每行 4 个正有限 σ、dets 行 5 列有限、iou_sigma>0、
  thresh 有限。空输入合法（空 ledger）。
- oracle：独立槽位模拟（order 列表 + pick swap + tail-swap 完整模拟、原矩阵直查、
  score/box 用 dict 按原索引存）+ 200 组随机对拍（seed 705，含 thresh 0.01/0.05/0.2/0.3/0.5
  与 iou_sigma 0.1/0.5/1.0）。

## 等价变异/写法说明

- `_close(a, a)` 自反占位断言无判别力（如曾写在 stale_iou 用例），已改为实质断言。
- `sorted(voters)` 记账排序与官方槽位序累加等价（加法交换/结合在 rel_tol 1e-9 下无观测差）。

## 变异测试

14/14 全抓：inter/area 去 +1、thresh 钳制失效、邻居门反向、(1−iou) 不平方、
1/σ²→1/σ、分母漏 p、soft σ 0.5→0.3、floor 0.001→0.01、衰减门改 thresh、
argmax 全局取、平分取更大者、融合坐标 clip、记账用槽位号。
