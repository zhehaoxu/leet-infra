# 详解：全局幅值剪枝

## 出处
Han et al., *Deep Compression*（arXiv:1510.00149）的全局幅值阈值剪枝：跨整个张量丢弃绝对值最小的一批权重。是最基础、面试最高频的非结构化剪枝判据。

## 核心与易错点
1. **全局而非逐行/逐通道**：所有元素竞争同一稀疏预算，阈值是第 \(k\) 小的 \(|w|\)，不同行会被剪得不均。
2. **剪的是最小的 \(k\) 个、不是小于某固定阈值**：\(k=\mathrm{round}(\text{sparsity}\cdot N)\)，与稀疏率精确对应。
3. **看绝对值不看符号**：\(-9\) 比 \(1\) 更重要；只按 \(|w|\) 排序。
4. **并列确定性**：相同 \(|w|\) 谁先剪要固定（本题按展平索引升序），用 `stable=True` argsort，否则不同实现结果不一致。

## 对拍口径
参考解 `argsort(|w|.flatten(), stable=True)[:k]` 展平置零再 reshape。判题 `_oracle` 用纯 Python `sorted(range(N), key=(abs,index))` 取前 \(k\) 置零、逐元素重算（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「sparsity=0 原样 / sparsity=1 全零」「剪掉的恰是全局最小 \(k\) 个」「未剪元素值符号不变」断言。

## 复杂度
排序 \(O(N\log N)\)（或 `kthvalue` 阈值可 \(O(N)\)）。
