# Overthinking 指标：token 效率与准确率-长度分箱账

## 题意

推理模型的「overthinking」评估关心一件事：答错（或答对）的样本各自花了多少推理 token、按长度分箱后准确率怎么衰减。本题把这套指标做成确定性账本：给定 N 条样本的 `correct`/`tokens` 与显式分箱边 `bin_edges`，输出总准确率、每箱 count/accuracy/mean_tokens、答对/答错的平均 token 数与效率比、最优箱号。

## 求解

口径全部题面钉死：

- 分箱：`bin_edges` 严格递增共 K+1 条 → K 个箱；箱 j（0 ≤ j < K−1）为 `[edges[j], edges[j+1])`，末箱 `[edges[K-1], +∞)`；低于 `edges[0]` 的样本落入末箱（无更小的箱可去，题面钉死）。
- 空箱 `accuracy`/`mean_tokens` 为 None；`best_bin` 只在非空箱里选，并列取最小箱号（实现用严格 `>` 扫描）。
- `len_correct`/`len_wrong` 分别为答对/答错样本的 token 均值（各自缺失为 None）；`efficiency_ratio = len_wrong / len_correct`（任一缺失或 len_correct=0 为 None）。
- `best_bin` 语义：准确率最高的箱（全错时仍取首个非空箱，0.0 > 初始 −1）。

关键点：accuracy 是 n_correct / N（全样本）；箱内 accuracy 分母是箱内 count；边界值左闭右开、末箱右端含。

## 数据规模

- 1 ≤ N ≤ 64，tokens ≥ 0，2 ≤ len(bin_edges) ≤ 8。
- 长度不齐、空记录、边不严格递增、bool/负数/非整混入 → `ValueError`。
