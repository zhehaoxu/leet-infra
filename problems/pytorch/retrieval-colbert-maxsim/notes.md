# 详解：ColBERT MaxSim 与 MaxPooled 聚合

## 出处

- MaxSim / late interaction：Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, SIGIR 2020, arXiv:2004.12832。每个 query token 对文档 token 取最大余弦、再对 query 求和，是论文式 (1) 的 scoring 定义。
- 压缩与索引侧：Santhanam, Khattab, Saad-Falcon, Potts, Zaharia, *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*, NAACL 2022, arXiv:2112.01488（残差压缩 + denoised supervision）。
- chunk 级「先块内 max、再跨块 max」的 MaxPooled 聚合是长文档工程的去重口径（同一 query 词的贡献只计一次），不是上述论文命名的独立算子；具体工业实现若需逐行追溯，**待回源**。本题按 starter docstring 的机制定义判题。
- 相邻背景：`retrieval-mmr` 是重排，`retrieval-rag-chunk` 只产出滑窗下标矩阵，`retrieval-pq-codes` 是向量压缩编码；三者都不做 MaxSim 打分。

## 完整推导与下标约定

**归一化**：对行 \(x\) 取 \(n=\sqrt{\sum_k x_k^2}\)，输出 \(x/\max(n,1)\) 的写法在 \(n=0\) 时会把零向量留成全 0；参考实现用 `torch.where(norm > 0, norm, 1)` 做安全分母。零向量与任何向量的余弦按 0 处理，因此不会出现 `0/0` 的 NaN。注意归一化是**逐行**的，不是对整个矩阵除以 Frobenius 范数。

**基础分数**：归一化后 \(C=\hat Q\hat D^\top\in\mathbb R^{L_q\times L_d}\)，\(S=\sum_i\max_j C_{ij}\)。\(L_d=0\) 时 \(C\) 形状 \((L_q,0)\)，每个 query 词的 max 约定为 0，于是 \(S=0\)、`base_idx` 全 `-1`；\(L_q=0\) 时求和为空，\(S=0\)。

**并列 max 取小下标**：`torch.max` 只给值，`torch.argmax` 在重复最大值上取首次出现；参考实现先取每行最大值 `max_val`，再用 `(C == max_val).argmax(dim=1)`，布尔矩阵的第一个 `True` 必然是最小下标，不依赖后端 argmax 的并列策略。

**切块与尾块**：块数 \(C=\lceil L_d/\text{chunk\_size}\rceil\)，第 \(c\) 块是半开区间 \([c\cdot s,\min((c+1)s,L_d))\)。尾块长度可以小于 \(s\)，但**不能丢弃**；`chunk_size` 大于等于 \(L_d\) 时退化为 \(C=1\)。块与块不重叠（与 `retrieval-rag-chunk` 的 overlap 滑窗不同）。

**MaxPooled 恒等于基础分数**：对固定 query 词 \(i\)，
\[
\max_{c}\ \max_{j\in \text{chunk }c} C_{ij}
=\max_{j\in[0,L_d)} C_{ij},
\]
因为各块是对文档下标的划分。于是先块内 max、再跨块 max、最后对 \(i\) 求和，和直接对全文档 max 求和是**同一个值**。实测示例两者都是 `2.0`，`torch.equal(maxpooled, base)` 为 `True`。所以在这道题里「MaxPooled 与基础分数有差」是错误预期；真正有差的是重复计分口径。

**重复计分口径**：`repeated = Σ_i Σ_c m_{i,c}`，其中 \(m_{i,c}=\max_{j\in c}C_{ij}\)。它把每个 query 词在每个块的最大值都加了一遍。实测：
- 示例（2 个 query 词、3 块）`repeated = 3.4`、`maxpooled = base = 2.0`，差 1.4，是基础分数的 1.7 倍；
- 极端构造：query 只有 1 个词，文档 4 个 token 全是同一个最佳向量，`chunk_size=1` 时 4 个块的最大值都为 1，`repeated = 4.0`、`base = 1.0`，正好 \(C\) 倍；
- 同一构造改成 `chunk_size=4`（单块），`repeated = base = 1.0`，这就是「单 chunk 退化」。

余弦非负时必有 `repeated ≥ base`；但余弦可以为负，若每个块的最大值都是负数，逐块求和可能**小于**基础分数（例如两个单 token 块的最大值都是 \(-1\)，`base=-1`、`repeated=-2`）。所以 starter 的措辞是「一般更大」而不是「必然更大」。

**用零填充做块对齐的陷阱**：若把文档 pad 到 \(C\cdot s\) 后 reshape 再 max，pad 位置的余弦是 0；当真实块内余弦全为负时，0 会错误地成为最大值。正确做法是按真实块边界切片（参考实现逐块切片），或把 pad 位置填 `-inf` 再 max。

**空输入与 dtype**：\(L_d=0\) 时逐块矩阵形状 \((L_q,0)\)，三个标量都是 0；\(L_q=0\) 时矩阵形状 \((0,C)\)、下标为空；双空为 \((0,0)\)。所有浮点输出跟随输入 dtype，`base_idx` 恒 int64。判题对 float32 用 \(10^{-5}\)、float64 用 \(10^{-12}\)，下标用 `torch.equal`。

**错误面**：非二维、特征维不一致、dtype 不一致或非浮点、`chunk_size<1` 都是 `ValueError`；空 query/doc 不是错误。

## 核心与易错点

1. **逐行归一化、零行安全**：零向量必须保持 0 并把余弦记 0；直接 `x / x.norm()` 会得到 NaN 并污染 max。
2. **max 的方向**：每个 query 词对**所有文档词**取 max，再对 query 词求和；反过来（先对 query 取 max 再对文档求和）是另一种（错误的）打分。
3. **MaxPooled 不是新分数**：块内 max 再跨块 max 恒等于全局 max，本题中 `maxpooled_score == base_score`；不要为了「让它有差」而改成先对块内 query 求和。
4. **重复计分才会膨胀**：`repeated` 是全部块内 max 之和，块越多越大；单块时必须回落到基础分数。
5. **尾块保留、块数向上取整**：\(C=\lceil L_d/s\rceil\)，最后一块允许短；漏掉尾块会同时改变矩阵形状与 `repeated`。
6. **并列取小下标**：`base_idx` 必须稳定取最小下标；用布尔矩阵首个 `True` 或升序严格 `>` 更新都可。
7. **不要把 pad 的 0 当成余弦**：余弦有负值时 0 会抢占 max；按边界切片或用 `-inf` mask。
8. **空文档不是错误**：\(L_d=0\) 时 max 按 0、下标 `-1`、矩阵 \((L_q,0)\)；\(L_q=0\) 时矩阵 \((0,C)\)。
9. **dtype 一致**：float64 输入不能悄悄落回 float32（判题用 \(10^{-12}\)）；非浮点输入抛 `ValueError`。

## 与邻近题的区别

- **vs `retrieval-mmr`**：MMR 是**重排**（在候选文档集合上贪心选序，惩罚与已选集合的冗余），输入是文档级向量；本题是**打分**（query token × doc token 的 late interaction），不选序。
- **vs `retrieval-rag-chunk`**：那题只输出滑窗切块的 **token 下标矩阵**（`-1` 补尾），不涉及向量、mask 或分数；本题的块是**不重叠硬切**，且每块要参与余弦 max 与聚合。
- **vs `retrieval-pq-codes`**：PQ 把向量压成子码本下标，输出 \((n,m)\) 的整数编码；本题不做量化，保留浮点余弦并输出分数与矩阵。
- **vs `cosine-similarity`**：那题是一对向量的单个余弦；本题是 \(L_q\times L_d\) 的余弦矩阵 + 双重 max + 聚合。
- **vs `retrieval-ivfpq-adc`**（若该题已落地）：ADC 用查表近似距离并取 top-k；本题是精确浮点 MaxSim，且额外暴露「重复计分 vs 只计一次」的口径差。
- **vs `retrieval-bm25-score`**：BM25 是词袋稀疏打分；本题是稠密 token 向量的 late interaction。

## 对拍口径与复杂度

- **主 oracle**：`_oracle` 用纯 Python 逐 query 词 × 逐文档词双重循环重算余弦（独立做范数、除法、零范数返回 0），再逐块 `while` 循环取 max，最后显式计算三种聚合；与参考实现的矩阵乘 + 逐块切片路径结构不同。覆盖：示例、负余弦、零 query 行、零文档行、尾块、单块、空 query、空文档、双空、float64。
- **显式判别用例**：并列 max 取小下标（重复行、全负、零 query）、零向量不产生 NaN、尾块形状 \((L_q,3)\)、单 chunk 时 `repeated == base`、4 个单 token 块时 `repeated = 2` 而 `base = 1`、空输入形状、float64 的 \(10^{-12}\) 对拍、7 类 `ValueError`。
- **复杂度**：余弦矩阵 \(O(L_qL_dD)\) 时间、\(O(L_qL_d)\) 内存；逐块 max 为 \(O(L_qL_d)\)；输出矩阵 \(O(L_qC)\)。参考实现的块循环只做切片 max，不复制整篇文档。判题规模很小（\(L_q,L_d\le 9\)），容差远不是瓶颈。
