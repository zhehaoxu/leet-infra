# 详解：最大边际相关 MMR 重排

## 出处
Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*（SIGIR 1998）。原文定义 \(P_1(d|q)\) 为到查询的相似、\(P_2(d|s)\) 为到已选文档的相似，\(\lambda\) 调节相关-多样折中；\(\lambda=1\) 纯相关、\(\lambda=0\) 纯多样。LangChain `MMRRetrieval`、文本摘要选句均用之。

## 核心与易错点
1. **惩罚项是「与已选集合的最大相似度」**：\(\max_{s\in S}\text{sim}(d,s)\)，不是求和、不是与全体文档。冗余度取最坏者。
2. **第一步无已选集合**：\(\max\) 对空集应视作 0，于是得分退化为 \(\lambda\cdot\text{sim}(d,q)\)，argmax 即纯相关性最高者（\(\lambda>0\)）。
3. **sim 用余弦**：\(\frac{a\cdot b}{\|a\|\|b\|}\)，零范数须特判为 0（否则 0/0=NaN 污染 argmax）。
4. **每轮更新 \(S\) 并从候选移除**：贪心序列化，选中的不再回来。
5. **并列取小下标**：浮点相等判定需容差；升序遍历候选、严格 `>` 才更新即等价。

## 对拍口径
参考解先把 docs/query 归一化，`rel=Du@q`、`sim=Du@Duᵀ`，再对 \(k\) 轮在未选集合上取 `lam*rel[i]-(1-lam)*max(sim[i,S])` 的 argmax（升序、容差、并列取小）。判题 `_oracle` 用逐对纯 Python 余弦函数（独立算范数、除、零范数返回 0.0），双循环重算每轮得分——结构与矩阵化实现不同、数值等价。另设「λ=1 等于相关性 top-k」「λ=0 首选恒为下标 0 且后续纯多样」「输出长度 k、下标互异」断言。

## 复杂度
\(O(nD + k\cdot n^2)\)（相似度可预计算，选择 \(k\) 轮各扫 \(O(n\cdot|S|)\)）。
