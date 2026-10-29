# 检索：倒排 postings 与 Block-Max WAND

## 题目描述

倒排索引把「文档 → 词」翻成「词 → 文档」，是 BM25 稀疏检索的存储底座。查询时有两件互补的事：

- **布尔 AND**：求同时包含所有 query term 的文档。每条 postings 的 docid 升序，块尾 docid 可作 **skip 指针**，跳跃式求交，避免逐条扫描长表；
- **排序 top-k**：WAND 系列算法用「每个 term 对当前候选文档的分数上界之和」与当前 top-k 阈值比较，整块整块地剪掉不可能进榜的文档；Block-Max WAND（Ding & Suel, SIGIR 2011）把上界从「全表最大值」收紧为「当前块内最大值」，剪枝更狠，但**上界必须仍是真上界**，剪枝不能改变最终 top-k。

本题把两者串起来：输入是若干条已经建好的 postings（每条是升序的 `(docid, tf)` 行），按 `block_size` 自动切块（每块最后一条的 docid 就是 skip 目标），要求同时返回 skip-AND 的交集候选、Block-Max WAND 的 top-k，以及**被整块剪掉的块数**。打分用 BM25 + Lucene idf，公式与参数在 starter docstring 里钉死。

## 规则

完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。

- `postings` 是 list/tuple，长度等于 query 的 term 出现次数（空 list 即空 query）。每个元素是一条 \((n_i,2)\) 的 int64 倒排表，行是 `(docid, tf)`；`docid` 必须**严格递增**且落在 \([0,N)\)，`tf>=1`。形状 \((0,2)\) 表示该 term 不在语料；同一 term 出现多次就放多份表，分数按出现次数重复累加。
- 语料规模 \(N\) 由 `doc_lens`（长度 \(N\ge1\) 的 int64 向量）给出，\(|d|=\text{doc\_lens}[d]\)，\( \mathrm{avgdl}=\frac1N\sum_d |d| \)；\( \mathrm{avgdl}=0 \) 时长度比值约定取 0。
- 打分是 BM25（Lucene idf）：第 \(i\) 条 postings 的 \( \mathrm{df}_i=n_i \)，\( \mathrm{idf}_i=\log\!\left(1+\frac{N-\mathrm{df}_i+0.5}{\mathrm{df}_i+0.5}\right) \)；文档 \(d\) 的分数为所有包含 \(d\) 的 term 的
  \( \mathrm{idf}_i\cdot\frac{\mathrm{tf}_{i,d}(k_1+1)}{\mathrm{tf}_{i,d}+k_1(1-b+b|d|/\mathrm{avgdl})} \)
  之和。没有出现在任何 postings 的文档分数记 0，但不进入 top-k。
- **skip-AND**：返回同时出现在所有**非空** postings 里的 docid，升序。空 query 或任一条 postings 为空时交集为空。块由连续 `block_size` 行切出（最后一块可不满），块尾 docid 是 skip 目标；指针只能沿 docid 单调前进。
- **Block-Max WAND**：每条 term 的每个块以「块内所有行的最大 BM25 贡献」作上界。用上界之和与当前 top-k 阈值（第 \(k\) 好的分数；不足 \(k\) 个时为 0）比较，整块跳过的块计入 `pruned_blocks`。被部分扫描过的块不计入。剪枝后 topk 必须与「暴力全打分再排序」完全一致。
- top-k 的候选域是**至少出现在一条 postings 里的文档**（并集），共 \(U\) 个；`topk` 长度 \( \min(k,U) \)，按（分数降序，docid 升序）排列。`k=0` 为空，`k>U` 返回全部 \(U\) 个。
- 返回三元组：`candidates` 为 \((C,)\) int64，`topk` 为 \(( \min(k,U), )\) int64，`pruned_blocks` 为非负 Python `int`。空 query 或 `k=0` 时 `pruned_blocks=0`。
- 非法输入一律抛 `ValueError`：`postings` 不是 list/tuple、元素不是 \((n,2)\) 整型张量、docid 非严格递增或越界、`tf<1`、`doc_lens` 形状/取值非法、\(N=0\)、`k<0`、`k1<=0`、`b` 不在 \([0,1]\)、`block_size<1`。

## 示例

\(N=6\)，`doc_lens = [4,4,4,4,4,4]`（\( \mathrm{avgdl}=4 \)），`block_size=2`，三条 postings：

```
term0: (0,2) (1,1) (2,1) (3,1) (4,1) (5,2)   块: [0,1] [2,3] [4,5]
term1: (0,1) (5,3)                            块: [0,5]
term2: (0,2) (5,2)                            块: [0,5]
```

\( \mathrm{df}_0=6 \)，\( \mathrm{idf}_0=\log(1+0.5/6.5)\approx0.0741080 \)；\( \mathrm{df}_1=\mathrm{df}_2=2 \)，\( \mathrm{idf}_1=\mathrm{idf}_2=\log(1+4.5/2.5)=\log2.8\approx1.0296194 \)。

```
score(d0) = 0.1018985 + 1.0296194 + 1.4157267 = 2.5472445775
score(d1..d4) = 0.0741080                      (只有 term0, tf=1)
score(d5) = 0.1018985 + 1.6179734 + 1.4157267 = 3.1355985302
```

- `candidates = [0, 5]`（三条表都含 0 和 5）。
- 取 `k=2`：`topk = [5, 0]`，`pruned_blocks = 1`。先给 doc0 打分（2.5472），再给 doc1 打分（0.0741080），阈值变成 0.0741080；term0 的块 `[2,3]` 的块内最大贡献恰好也是 0.0741080，**严格大于**阈值才可能进榜，于是整块跳过。doc2、doc3 的分数虽与 doc1 并列，但 docid 更大，并列时输给 doc1，跳过是安全的。
- 取 `k=3` 时 `pruned_blocks = 0`：此时 doc2 在阈值填满前已被打分，块 `[2,3]` 只被部分扫描，按定义不算「整块剪掉」。

## 约束

- `candidates`、`topk` 均为 int64 一维张量，`pruned_blocks` 为 Python `int`；判题对两个张量用 `torch.equal`，对块数用整数相等。
- 剪枝正确性是硬约束：topk 必须等于对全部并集文档暴力打分后的结果；上界只允许偏松，不允许偏紧。
- docid 单调、skip 跳跃、块上界、阈值并列（严格 `>`）与 `pruned_blocks` 的「整块」口径都是判题的一部分。
- CPU、确定性、不调用随机数；`block_size=1` 与最后一块不满都合法。
