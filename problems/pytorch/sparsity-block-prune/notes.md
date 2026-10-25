# 详解：块结构剪枝（按块范数）

## 出处
块 / 结构化剪枝（block pruning，cf. Luo et al. *ThiNet*, arXiv:1708.01852；及 NVIDIA 对结构化稀疏的动机）：以规整小块为丢弃单位，用聚合范数衡量块重要性，兼顾压缩率与硬件可加速性（连续内存、可向量化的稠密子块）。

## 核心与易错点
1. **剪除单位是整块**：一个块要么全留、要么全零，不能像非结构化只剪块里几个——这是与逐元素题的本质区别。
2. **块重要性=块内 \(|w|\) 之和（L1）**：聚合后再全局比较，单个大元素未必保住它所在的块。
3. **全局竞争预算**：所有块（跨行）一起按范数排序，剪最小的 \(\mathrm{round}(\text{sparsity}\cdot B)\) 个，\(B\) 为块总数。
4. **reshape 切块要对齐最后一维**：`reshape(R, C//block, block)`，`C%block==0` 前提；mask 用 `view(R,nb,1)` 广播回每块。
5. 并列按 (行,块) 展平索引升序、`stable=True` 保确定性。

## 对拍口径
参考解 `reshape→abs.sum(dim=2)→argsort(stable)→mask.view(R,nb,1)` 广播。判题 `_oracle` 逐块 Python 双循环累 L1、`sorted` 取最小 k 块、其余逐元素拷回（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「被剪块整块为 0」「保留块元素逐一不变」「sparsity=0 原样 / =1 全零」断言。

## 复杂度
\(O(N + B\log B)\)。
