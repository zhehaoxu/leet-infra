# 出题笔记：vision-ohem-selection

## 回源

- 原始仓库 abhi2610/Ohem-caffe 已 404（github code search 需登录），改用 **mmseg `mmseg/structures/sampler/ohem_pixel_sampler.py`**（main 分支，85 行全文核对）为唯一口径源。
- RoI 级参照 **mmdet `task_modules/samplers/ohem_sampler.py`**：`hard_mining` = per-sample loss `topk(num_expected)`，候选 ≤ 配额时全保留——语义入 notes 不入题（与像素级两分支混出一题会乱）。

## 关键口径判断

- **threshold 是 max 不是 min**：`max(第 batch_kept 难的概率, t)`。第一直觉「先按 t 过滤再 top-N」是错的——官方语义是 rank 兜底与 t 各自的保证取交集，两个下界取大。
- **比较是严格小于**：prob == threshold 的样本不保留。手算锚点两次踩坑：0.5 < 0.5 为 False、兜底值 0.4 > t=0.25 时 threshold 是 0.4 不是 0.25。solution 与 oracle 一致后仍要手验锚点方向。
- **probs 是「正确类概率」**：升序排（越低越难），与 loss 分支的降序方向相反——两个分支方向相反是命题点。
- batch 维度：`batch_kept = min_kept × batch_size`（官方 `self.min_kept * seg_label.size(0)`）。

## 边界划定

- 不做真实 loss 计算（CE/focal 输入即 per-sample loss 标量）；不做 RoI 级 sampler（mmdet 语义记 notes）；不做 GHM/focal 等其它难例加权。
- 与 `soft-nms`(308) 划界：那是推理期压分数，这是训练期选样本。

## 测试设计

- oracle `_py` 独立重放（分支内 for 重写，不复用 solution 的 sorted/key）。
- 9 个 check：双锚点、rank 兜底胜出、batch 缩放、loss 并列稳定、配额超量、随机 200 组、非法输入 7 档。
- 变异 9/9：rank ±1、max→min、严格→非严格、batch 忽略、降→升、并列反转、分支方向、mask 反转、配额 +1。`tie-unstable`（去掉 tie-break 的 i）为等价变异（Python sort 稳定），换 `tie-reversed` 后全抓。
