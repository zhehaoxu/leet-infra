# 检索：最大边际相关 MMR 重排

## 题目描述

纯按相关性排序会返回一堆近乎重复的文档（检索与摘要里的「冗余」问题）。**最大边际相关**（MMR, Maximal Marginal Relevance，Carbonell & Goldstein, SIGIR 1998）在每一步贪心地选「既相关、又与已选集合尽量不同」的文档：

\[ \text{MMR}=\arg\max_{d\in R\setminus S}\Big[\lambda\,\cdot\,\text{sim}(d, q)\;-\;(1-\lambda)\,\cdot\,\max_{s\in S}\text{sim}(d, s)\Big] \]

其中 \(S\) 是已选集合、\(q\) 是查询。它是 LangChain `MMRRetrieval`、各类「多样重排 / 结果去冗」的底层算法。本题实现 MMR 的贪心选择序列。

## 规则

实现 `solve(query, docs, lam, k)`。输入 `query`（float32，\((D,)\)）、`docs`（float32，\((n, D)\)）、`lam`（\(\lambda\in[0,1]\)）、`k`（选取个数，\(k\le n\)）。\(\text{sim}\) 用**余弦相似度**（零范数向量与任何向量的相似度记 0）。逐个选出共 \(k\) 个：
- 第一个：选 \(\text{sim}(d,q)\) 最大的 \(d\)（即纯相关性）；
- 其后每个：在未选集合上最大化 \(\lambda\cdot\text{sim}(d,q)-(1-\lambda)\cdot\max_{s\in S}\text{sim}(d,s)\)。

返回 int64 张量，形状 \((k,)\)，为**被选文档下标按选取顺序**。并列取文档下标最小者。

## 示例

\(q=[1,1,0]\)，\(docs=[[1,0,0],[0.9,0.1,0],[0,1,0]]\)，\(\lambda=0.5, k=3\)（余弦：doc0=.707、doc1=.781、doc2=.707）：

```
选1: 相关性最高 -> doc1
选2: doc2 与 doc1 的 sim(=.110) 远小于 doc0 与 doc1 的 sim(=.994)
     -> doc2 得分 0.5*0.707-0.5*0.110=0.298 > doc0 的 -0.144 -> 选 doc2（更边际）
选3: 只剩 doc0
out = [1, 2, 0]
对比：若 λ=1（纯相关）则输出 [1, 0, 2]——doc0 因与 doc1 冗余而排到 doc2 之后
```

## 约束

- CPU float32；\(\lambda\in[0,1]\)、\(1\le k\le n\)。
- \(\lambda=1\) 退化为纯相关性 top-k；\(\lambda=0\) 纯追求多样。
- 相似度一律余弦；零范数向量相似度取 0。
- 并列取文档下标最小者，保证确定性。
