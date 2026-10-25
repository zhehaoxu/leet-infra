# 详解：倒数排名融合 RRF

## 出处
Cormack, Clarke, Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*（SIGIR 2009）。生产实现见 Elasticsearch/OpenSearch 的 `rrf` retriever、LangChain `EnsembleRetriever`（默认权重版）。核心式 \(\text{score}(d)=\sum_{r}\frac{1}{k+r(d)}\)，\(k=60\) near-optimal 且不敏感。

## 核心与易错点
1. **用名次不用原分数**：RRF 的价值就是免标定——绝不要把各路 BM25/点积分数混进来。
2. **1-based 名次 + `0` 哨兵**：分母是 \(k+\text{rank}\)，rank 从 1 起；`0` 表示缺席，须跳过（否则会算成 \(1/k\) 的幽灵贡献）。
3. **平滑常数 \(k\)** 是给定的标量入参（口径 60），别写成名次偏移。
4. **并列下标升序**：`torch.argsort(-score, stable=True)` 保持原列序（下标升序），与手写 `-score 再 index` 排序一致。
5. 缺席全 0 的文档 score=0，排在最后。

## 对拍口径
参考解 `contrib = where(ranks>0, 1/(k+ranks), 0).sum(0)` 得 score，再 `argsort(-score, stable=True)`。判题 `_oracle` 双循环纯 Python 累加 `1/(k+r)`（r>0 才加），`sorted(range(N), key=lambda d:(-sc[d], d))`（结构不同、数值等价）。另设「单路即该路名次序」「全缺席文档排最后」「输出是 0..N-1 的排列」断言。

## 复杂度
\(O(L\cdot N)\) 打分 + \(O(N\log N)\) 排序。
