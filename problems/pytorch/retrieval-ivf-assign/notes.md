# 详解：IVF 粗量化器分配

## 出处
Jégou et al., *Product Quantization for Nearest Neighbor Search*（IEEE TPAMI 2011，arXiv 版 inria-00514462）中的 **coarse quantizer**；Faiss `IndexIVF`/`faiss::search_preassigned` 的分配阶段。IVF 把向量分派到最近粗中心以裁剪搜索范围，是「先粗筛再精排」ANN 范式的代表。

## 核心与易错点
1. **距离是平方欧氏**：\(\|x_i-c_j\|^2=\sum_t(x_{i,t}-c_{j,t})^2\)。开方单调不改 argmin，直接用平方更稳更快。
2. **展开式陷阱**：\(\|x\|^2+\|c\|^2-2x\cdot c\) 在 float32 下当 x≈c 会因灾难性抵消放大误差；逐差平方和更可靠。
3. **并列取小下标**：`argmin` 天然返回首个最小（下标最小），与手写「严格小于才更新」等价。
4. 每行独立，无跨行耦合。

## 对拍口径
参考解用 `torch.cdist(x, centroids)` 再 `argmin(dim=1)`。判题 `_oracle` 三重纯 Python 循环显式累加逐维平方差、严格 `<` 才更新（结构不同、数值等价）。另设「与逐中心暴力取最小一致」「k=1 时恒为 0」「结果是 {0..k-1} 的整数」断言，并用一个与某中心完全重合的向量验证命中该中心。

## 复杂度
\(O(n\,k\,D)\)。
