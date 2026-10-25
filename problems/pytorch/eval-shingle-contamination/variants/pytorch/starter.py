import torch


def solve(eval_docs, train_docs, n, threshold):
    """训练集污染审计。eval_docs / train_docs: list[list[int]]（token id 序列，已切好，不用分词器）；
    n: int，shingle 长度（连续 id 窗口）；threshold: float，命中阈值。

    每篇文档取它所有**连续 n-gram**（重叠窗口）构成的**集合**（重复的 n-gram 只算一次）；
    长度 < n 的文档得到空集。

    对 (评测文档 i, 训练文档 j) 这一对：
      包含率 contam[i,j] = |A_i ∩ B_j| / |A_i|（分母是**评测侧**的 shingle 数，非对称）
      Jaccard jac[i,j]   = |A_i ∩ B_j| / |A_i ∪ B_j|
    空集一律记 0.0（|A_i| = 0 时不出现 0/0）。

    flagged：升序的评测文档下标列表，条件是 |A_i| > 0 且训练集非空 且
    max_j contam[i,j] >= threshold（**阈值边界算命中**）。

    返回 (contam, jac, flagged)：两个 (M_eval, M_train) 的 float32 张量 + 一个 int64 张量。"""
    # TODO
    raise NotImplementedError
