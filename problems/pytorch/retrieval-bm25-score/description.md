# 检索：BM25 完整打分

## 题目描述

BM25 是稀疏检索的基线打分函数（Robertson & Walker, SIGIR 1994；Lucene `BM25Similarity`）：它把「词在文档里出现得越多越相关」（词频饱和）与「文档越长越吃亏」（长度归一化）合成一个可加性的分数，再用逆文档频率 \( \mathrm{idf}(t) \) 压低到处都出现的常见词。真实系统里它是 hybrid search 的关键词一路，也是 RRF 等融合器的上游。

本题要求从**纯 token id 语料**出发，现算文档频次 \( \mathrm{df}(t) \) 与平均文档长度 \( \mathrm{avgdl} \)，对 query 的每一个 token 做完整 BM25 累加，返回全库分数与 top-k 下标。难点有三处：

- 两种 idf 口径：Lucene 的 \( \log(1+\cdot) \) 恒正，Robertson 的原式在 \( \mathrm{df}(t)>N/2 \) 时**为负**，必须按原式保留，不许截断成 0；
- query 中重复出现的 token 按出现次数**重复累加**（不做去重），这来自 BM25 对 query 词的可加性；
- 并列分数必须按文档下标升序打破，空 query、语料中不存在的 token、\( \mathrm{avgdl}=0 \) 都有明确退化约定。

CPU、全程确定性、无随机数调用。

## 规则

完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。

- 语料是 \((N,L)\) 的 int64 张量，第 \(d\) 行前 `doc_lens[d]` 个位置是文档 \(d\) 的 token id 序列，其余位置是 padding（内容任意，可为 -1），**一律忽略**；长度 \(L\) 可以为 0，但 \(N\ge 1\)。
- 文档频次 \( \mathrm{df}(t) \) 是「有效前缀中含 token \(t\) 的文档数」，由语料现算；\( \mathrm{avgdl}=\frac{1}{N}\sum_d |d| \)，其中 \( |d|=\text{doc\_lens}[d] \)。
- 两套 idf 由 `idf_mode` 选择：`"lucene"` 用 \( \mathrm{idf}(t)=\log\!\left(1+\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}\right) \)；`"robertson"` 用 \( \mathrm{idf}(t)=\log\!\left(\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}\right) \)。后者在 \( \mathrm{df}(t)>N/2 \) 时为负，分数中允许出现负值，**不得**取 \( \max(0,\mathrm{idf}) \)。
- 文档 \(d\) 对 query 中一次出现的 token \(t\) 的贡献为
  \( \mathrm{idf}(t)\cdot\frac{\mathrm{tf}(t,d)\,(k_1+1)}{\mathrm{tf}(t,d)+k_1\left(1-b+b\,|d|/\mathrm{avgdl}\right)} \)，
  其中 \( \mathrm{tf}(t,d) \) 是 \(t\) 在文档 \(d\) 有效前缀内的出现次数。文档分数是 query 每一次出现的贡献之和；同一个 token 在 query 里出现两次就加两遍。
- \( \mathrm{avgdl}=0 \)（所有文档都为空）时长度比值约定取 0，避免 \(0/0\)；此时所有分数为 0。
- query token 不在语料中时 \( \mathrm{df}=0 \)、所有 \( \mathrm{tf}=0 \)，贡献为 0，但 idf 仍按公式有定义。
- 返回二元组：全库分数 `scores` 是 \((N,)\) 的 **float64**；`topk` 是 \(( \min(k,N), )\) 的 **int64**，按（分数降序，文档下标升序）排列。`k=0` 返回空，`k>N` 返回全部 \(N\) 个。
- 空 query 时所有分数为 0，`topk` 就是 \( [0,1,\dots,\min(k,N)-1] \)。
- 非法输入一律抛 `ValueError`：形状/dtype 不符、`doc_lens` 越界、有效 token 或 query token 为负、\(N=0\)、`k<0`、`k1<=0`、`b` 不在 \([0,1]\)、`idf_mode` 未知。

## 示例

语料 \(N=3\)，\(L=4\)，`doc_lens = [4,3,2]`，query 为 `[2,2,4]`（token 2 出现两次），取 \(k_1=1.2\)、\(b=0.75\)：

```
d0: [1, 2, 2, 3]   |d0|=4  tf(2)=2
d1: [2, 3, 4, 0]   |d1|=3  tf(2)=1, tf(4)=1
d2: [1, 4, -1, -1] |d2|=2  tf(4)=1      (-1 是 padding，被忽略)
avgdl = (4+3+2)/3 = 3
df(2) = 2 (d0,d1)，df(4) = 2 (d1,d2)
```

lucene：\( \mathrm{idf}(2)=\mathrm{idf}(4)=\log(1+1.5/2.5)=\log 1.6\approx 0.4700036292 \)。

```
score(d0) = 2 × [0.4700036 × 2×2.2/(2+1.5)]                 = 1.1817234107
score(d1) = 2 × [0.4700036 × 1×2.2/(1+1.2)] + 0.4700036×1   = 1.4100108877
score(d2) = 0.4700036 × 1×2.2/(1+1.2×(0.25+0.75×2/3))       = 0.5442147286
topk (k=3) = [1, 0, 2]
```

robertson：\( \mathrm{df}=2>N/2=1.5 \)，于是 \( \mathrm{idf}=\log(0.6)\approx-0.5108256238<0 \)，分数整体变负、顺序反转：

```
scores = [-1.2843615683, -1.5324768713, -0.5914823012]
topk   = [2, 0, 1]
```

若把 query 的重复 token 去重成 `[2,4]`，分数会变成 \( [0.5908617,\ 0.9400073,\ 0.5442147] \)——本例 top-1 仍是 d1，但数值完全不同，足以让「去重」实现被判错。

## 约束

- 返回 `(scores, topk)`：`scores` 为 \((N,)\) float64（含 0 与负值），`topk` 为 int64 且长度为 \( \min(k,N) \)。
- 判题对 `scores` 用绝对容差 \(10^{-9}\)（float64），对 `topk` 用 `torch.equal`；实现路径不限，但必须全程 float64 累加。
- 并列严格按文档下标升序；空 query、OOV token、\( \mathrm{avgdl}=0 \)、`k=0`、`k>N` 都必须给出上述退化结果而不是抛错。
- CPU、确定性、不调用随机数；`doc_lens` 与 padding 的约定是判题的一部分，不能把 padding 计入 `df` 或 `tf`。
