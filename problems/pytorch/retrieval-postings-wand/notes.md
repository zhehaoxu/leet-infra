# 详解：检索：倒排 postings 与 Block-Max WAND

## 出处

- WAND 的阈值上界思想：Broder, Carmel, Herscovici, Soffer, Zien, *Efficient query evaluation using a two-level retrieval process*, CIKM 2003；块级上界版本：Ding & Suel, *Faster top-k document retrieval using block-max indexes*, SIGIR 2011。本题的 pivot / 块上界 / 整块跳过属于这一族算法，具体伪代码按本题 docstring 的口径实现（不同论文的 pivot 细节略有差异，未逐行照抄）。
- skip 指针：Manning, Raghavan, Schütze, *Introduction to Information Retrieval*, 2008, 第 2 章（布尔检索的 skip pointers）。本题把「每块最后一条 docid」作为 skip 目标，块大小由 `block_size` 自动推出。
- 打分：BM25 + Lucene `BM25Similarity` 的 idf（Lucene 官方 Javadoc），与 `retrieval-bm25-score` 同一公式；本题的 df 直接取 postings 行数。
- CPU、纯确定性：所有推进都由 docid 比较与固定块边界驱动，没有随机数。

## 完整推导与下标约定

**postings 与块**：第 \(i\) 条表是升序的 \( (d_{i,0},f_{i,0}),\dots,(d_{i,n_i-1},f_{i,n_i-1}) \)，docid 严格递增。按 `block_size = B` 切成行区间

\[
[bB,\ \min((b+1)B,n_i)-1],\qquad b=0,1,\dots
\]

最后一块可不满。块尾 docid \(d_{i,\min((b+1)B,n_i)-1}\) 就是 skip 目标；块内最大贡献

\[
M_{i,b}=\max_{r\in \text{block } b} c_i(d_{i,r})
\]

其中 \(c_i(d)\) 是第 \(i\) 条 term 对文档 \(d\) 的 BM25 贡献。注意这里取的是**贡献**的最大值而不是 tf 的最大值：长度归一化随文档长度变化，只有逐行算好贡献再取 max 才是严格上界。

**skip-AND 求交**：维护每条表的指针 \(p_i\)，当前 docid \(c_i=d_{i,p_i}\)。每轮取候选

\[
\text{cand}=\max_i c_i,
\]

把所有 \(c_i<\text{cand}\) 的指针推进到第一条 docid \(\ge \text{cand}\) 的行；若某条表耗尽则交集为空；若所有 \(c_i=\text{cand}\) 则输出 cand 并全体前移一格，否则以新的 max 继续。正确性：任何公共 docid 必须 \(\ge\) 所有当前 docid，因此 \(\ge\text{cand}\)；被跳过的行都 \(<\text{cand}\)，不可能成为交集。跳跃实现为：若当前指针在块首且块尾 docid \(<\) 目标，则整块跳到下一块首，否则逐行前进——指针只增不减，总推进次数不超过 \(\sum_i n_i\)。

**BM25 贡献**：\(N\) 为语料文档数，\( \mathrm{df}_i=n_i \)，\( \mathrm{idf}_i=\log(1+(N-n_i+0.5)/(n_i+0.5)) \)，\( \mathrm{avgdl}=\frac1N\sum_d |d| \)，则

\[
c_i(d)=\mathrm{idf}_i\cdot
\frac{f_{i,d}(k_1+1)}
     {f_{i,d}+k_1\left(1-b+b\,|d|/\mathrm{avgdl}\right)}.
\]

示例中 \(N=6\)、\( |d|=4 \)，于是 \(b|d|/\mathrm{avgdl}=0.75\)、分母的 \(k_1\) 项恒为 1.2。实测 \( \mathrm{idf}_0=\log(1+0.5/6.5)=0.07410797215372183 \)，\( \mathrm{idf}_1=\mathrm{idf}_2=\log 2.8=1.0296194171811581 \)。三条表下

```
score(d0) = 2.5472445775
score(d1..d4) = 0.0741080
score(d5) = 3.1355985302
```

**Block-Max WAND 的上界与 pivot**：设当前各表最小 docid 为 \( \mathrm{lo}=\min_i c_i \)，当前块的块尾最小值

\[
\mathrm{hi}=\min_i \text{end}_i.
\]

对任意 \(d\in[\mathrm{lo},\mathrm{hi}]\) 和任意 term \(i\)：若 \(c_i\le d\)，则 \(d\) 落在第 \(i\) 条表当前块内（因为 \(c_i\le d\le \mathrm{hi}\le \text{end}_i\)，且 \(d\ge c_i\ge\) 块首），贡献 \( \le M_{i,b_i} \)；若 \(c_i>d\)，该 term 在 \(d\) 上没有 posting，贡献为 0。因此

\[
\mathrm{score}(d)\le \sum_{i:\,c_i\le d} M_{i,b_i}.
\]

把各表按 \(c_i\) 升序排列，累加上界，第一个使前缀和 \(>\theta\)（当前 top-k 阈值）的 \(c_i\) 就是 pivot。任何 \(d<\text{pivot}\) 的上界和不超过阈值，可以安全跳过；若直到 \( \mathrm{hi} \) 都没有前缀和超过阈值，整个 \([\mathrm{lo},\mathrm{hi}]\) 都可以跳过。这正是参考实现的两个分支：

- `pivot == lo`：完整打分 lo（只有 \(c_i=\mathrm{lo}\) 的表有贡献），压入/替换堆，所有 \(c_i=\mathrm{lo}\) 的指针前移；
- `pivot > lo`：把所有 \(c_i<\text{pivot}\) 的指针推进到 \(\ge\text{pivot}\)（用 skip），跳过的整块计入 `pruned_blocks`；
- 无 pivot：把所有 \(c_i\le\mathrm{hi}\) 的指针推到 \(>\mathrm{hi}\)，整段跳过，整块计入 `pruned_blocks`。

**为什么上界只看「当前块」是安全的**：关键在于同时把候选区间截到 \( \mathrm{hi}=\min \text{end}_i \)。若不加这个截断，某个 term 的当前块可能早在 \(d\) 之前结束，块最大值就不再覆盖 \(d\)，上界会偏紧、剪掉真结果。参考实现用 `hi = min(block_end)` 保证了「区间内每个 term 的当前块都覆盖整个区间」。

**阈值与并列**：堆里维护 \(k\) 个最好的文档，键为 `(score, -docid)`，堆顶是「分数最小、同分时 docid 最大」的最差项；阈值 \(\theta\) 是第 \(k\) 好的分数（不足 \(k\) 个时为 0）。只有 `acc > theta` 才会形成 pivot，因此分数**恰好等于**阈值的文档不会进榜。这一口径安全的前提是文档按 docid 升序被打分：所有已入堆的文档 docid 都小于当前 lo，同分的后来者本来就该输给更早的下标。实测：

- 示例 `k=2` 时 `pruned_blocks=1`。doc0（2.5472）与 doc1（0.0741080）入堆后阈值是 0.0741080，term0 的块 `[2,3]` 上界恰为 0.0741080，不能严格超过阈值，整块跳过；doc2、doc3 即使同分也因下标更大而落榜。
- 同一示例 `k=3` 时 `pruned_blocks=0`。这时 doc2 在堆填满前已经被完整打分，块 `[2,3]` 只被部分扫描，按「整块跳过」的定义不计入；随后 `[4,5]` 块里 doc4 的低分上界超过阈值，pivot 命中 doc4，也没有整块可剪。
- 21 篇文档的长尾用例（term0 在 doc 0..19 上 tf=1、doc20 上 tf=50；term1 只在 doc0/doc20 有 posting）实测 `k=1` 与 `k=2` 都剪掉 **9** 块，`k=3` 剪掉 **8** 块——阈值随堆内容变化，剪枝块数不是常数。

**`pruned_blocks` 的计数口径**：`_advance` 只在「指针位于块首且该块块尾 docid \(<\) 目标」时整块跳到下一块首并计 1；如果指针已经在块中间，则逐行走到块尾，这一块算「部分扫描」，不计。这样同一个块不会被重复计数，且「剪掉」严格对应「块内没有任何文档被完整打分」。

**AND 与 top-k 的关系**：`candidates` 是布尔 AND 的交集；`topk` 的候选域是各 postings 的**并集**（WAND 是弱 AND：文档只需含部分 term 也能靠累加分进榜）。空 query 时两者都为空；任一 postings 为空时 AND 交集为空，但 top-k 仍可对剩下的表打分——这是「query 词不在语料」与「AND 无解」两种语义并存的直接结果。

**空文档与 avgdl=0**：所有 `doc_lens` 为 0 时长度比值约定为 0，分母变成 \(f+k_1(1-b)\)，贡献仍为正（tf 与 idf 都为正），不会出现 NaN。

## 核心与易错点

1. **docid 必须严格递增**：重复或倒序直接抛 `ValueError`；实现里用指针推进而不是建 \(N\) 长数组，postings 稀疏时这是本质区别。
2. **skip 目标是块尾 docid，不是块首**：只有块尾 \(<\) 目标才能整块跳过；块首就跳会漏掉块内可能命中的行。
3. **块上界必须覆盖整个候选区间**：用 \( \mathrm{hi}=\min \text{end}_i \) 截断；否则当前块结束后，块最大值不再是后续文档的上界，会剪掉正确答案。
4. **上界取「贡献」的 max，不是「tf」的 max**：BM25 的长度归一化让同样 tf 的不同文档贡献不同，只有逐行算贡献再取 max 才严格。
5. **阈值并列用严格 `>`**：等于阈值不能进榜；这依赖「按 docid 升序打分 + 同分下标小者优先」的稳定口径。实测 `k=2` 与 `k=3` 的剪枝块数不同（1 与 0），正是并列与部分扫描共同作用的结果。
6. **整块计数不能重复**：指针在块中间时逐行走完，不计；只有从块首整块跳过才计 1。
7. **topk 候选域是并集不是全集**：没出现在任何 postings 的文档分数记 0，但不占 top-k 名额；`k>U` 返回 \(U\) 个而不是 \(N\) 个。
8. **空表的两种语义**：空 query 让 `candidates`、`topk` 都为空；某条 term 为空只让 AND 交集为空，top-k 仍对其它 term 的并集打分。
9. **重复 term 按出现次数累加**：同一张表放两份，分数翻倍、候选不变；不要擅自去重。
10. **`k=0` 短路**：不建堆、不打分，`pruned_blocks=0`，但 `candidates` 仍要正常求。
11. **`block_size=1` 合法**：每行一块，skip 与块上界退化为逐行；最后一块不满也合法。
12. **剪枝不能改变结果**：这是硬约束，判题对每组都跑暴力全打分；任何「上界偏紧」的实现都会在某个用例上让 topk 与暴力不一致。

## 与邻近题的区别

- **vs `retrieval-rrf-fusion`（同合集）**：RRF 只做名次融合，输入是排好的列表，完全没有 postings、tf、块与上界；本题要建/消费倒排表，用 BM25 实数分数和块上界做 top-k 剪枝。
- **vs `retrieval-rag-chunk`（同合集）**：RAG 分块只输出「token 下标矩阵」，是纯切分，不打分也没有候选剪枝；本题的输出是 docid、top-k 与剪枝计数。
- **vs `retrieval-ivf-assign`（同合集）**：IVF 是稠密向量的欧氏距离 argmin，候选域是全库；本题是稀疏倒排、docid 指针与阈值剪枝，二者数据结构与算法都不同。
- **vs `retrieval-bm25-score`（同批次）**：那道题从稠密 token 语料现算 df/avgdl，返回**全库**分数；本题输入已经建好的 postings，只对并集做 top-k，并额外要求 AND 交集与 `pruned_blocks`。同一 BM25 公式，一个是打分本体，一个是倒排查询执行。
- **vs `searchsorted-bincount`**：那是「值 → 桶号 + 计数」的直方图原语，没有文档、term、docid 单调性，也没有上界剪枝。

## 对拍口径与复杂度

- **主 oracle**：`_oracle` 是纯 Python 暴力——逐文档、逐 term 扫完整 postings 累加 BM25，再 `sorted(union, key=(-score, docid))`；AND 交集则对每个 docid 逐表线性查找。它与参考解的「块上界 + pivot + skip 指针」结构完全不同，专门抓上界偏紧、skip 跳错和并列口径错误。
- **覆盖**：示例（`candidates/topk/pruned` 全部实测回填）、5 组混合语料（含零长文档、`k=100`、`block_size=1/2/3`、`b=0.2/1.0`、空 posting）、skip-AND 的长表（\(N=25\)，`block_size=1/2/4/64`，交集 `[12,24]`）、空 query、缺失 term、三篇同分、`k=0`、`k>U`、单 term、重复 term、14 组非法输入。每组都用 `torch.equal` 对拍 `candidates` 与 `topk`，并断言 `pruned_blocks` 为非负整数。
- **剪枝真实发生**：示例 `k=2` 实测 `pruned_blocks=1`；21 篇长尾用例 `k=1/2` 实测 9、`k=3` 实测 8。另有 300 组随机小 postings（\(N\le12\)、随机 block_size）本地对拍，topk 与暴力全打分无差异，单例最大剪枝 13 块。
- **复杂度**：建块 \(O(\sum_i n_i)\)；skip-AND 最坏 \(O(\sum_i n_i)\)（指针单调）；WAND 每次迭代对活跃表排序 \(O(T\log T)\)，每次完整打分 \(O(T)\)，迭代次数为「被打分文档数 + 整块跳过数」，最坏 \(O((U+B)T\log T)\)，其中 \(U\) 为并集大小、\(B\) 为块数；堆操作 \(O(U\log k)\)。空间 \(O(\sum_i n_i + k)\)。判题规模很小，远低于时限。
