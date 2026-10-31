# OHEM 难例选择掩码账

## 背景与位置

Online Hard Example Mining（1604.03540，CVPR 2016）用「按 loss 挑难例」替换均匀采样，是检测/分割训练的标配组件。原始 caffe 实现已散佚（abhi2610/Ohem-caffe 404），现代生态的权威转写是 **mmseg `OHEMPixelSampler`**（像素级双分支）与 **mmdet `OHEMSampler`**（RoI 级 topk）。本题转写 mmseg 像素级双分支账本。

## 精确账目（两分支）

1. **batch_kept = min_kept × batch_size**。
2. **loss 分支**（无阈值）：losses 降序稳定排序（并列取靠前下标），前 `batch_kept` 个 mask=1；配额 > n 时全保留。
3. **threshold 分支**：输入是每样本**正确类概率** probs（不是 loss）：
   - 升序排序，`min_threshold = sort_prob[min(batch_kept, n−1)]`（n=0 时 0.0）；
   - **threshold = max(min_threshold, t)**——rank 兜底保证至少保留 batch_kept 个，t 保证不放过低于 t 的更难样本；两者取大；
   - `mask = probs < threshold`（严格小于，等值不保留）。
4. 输出 mask + 最终 threshold（loss 分支为 None）。

## 约束

1. min_kept ≥ 1、batch_size ≥ 1、t ∈ [0,1]、probs ∈ [0,1]、losses ≥ 0 有限。
2. 两分支互斥：thresh 给定必须传 probs（losses=None），否则反之。
3. 禁改入参；纯排序与比较，无浮点累积。

## 数据规模

- n ≤ 10⁵（排序 O(n log n)）；min_kept ≤ 10⁵；batch ≤ 64。

## 验收锚点（节选）

- probs `[0.04,0.01,0.05,0.02,0.03]`、t=0.03、min_kept=3 → threshold=0.04、mask `[0,1,0,1,1]`。
- probs `[0.1..0.5]`、t=0.25、min_kept=3 → **rank 兜底 0.4 胜出** threshold=0.4、mask `[1,1,1,0,0]`。
- losses `[1,3,2,3,0.5]`、配额 3 → mask `[0,1,1,1,0]`（两个 3.0 都进，2.0 充数）。
