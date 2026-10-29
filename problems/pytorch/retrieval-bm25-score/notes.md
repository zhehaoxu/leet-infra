# 详解：检索：BM25 完整打分

## 出处

- BM25 的原始出处：Robertson & Walker, *Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval*, SIGIR 1994（TREC-3 论文）。它给出词频饱和与长度归一化的概率检索形式，即本题的 \( \mathrm{tf}(k_1+1)/(\mathrm{tf}+k_1(1-b+b|d|/\mathrm{avgdl})) \)。
- 两套 idf 的工程口径：Lucene `BM25Similarity` 官方文档（Javadoc）明确使用 \( \log(1+(N-\mathrm{df}+0.5)/(\mathrm{df}+0.5)) \)；Robertson 原式 \( \log((N-\mathrm{df}+0.5)/(\mathrm{df}+0.5)) \) 在 \( \mathrm{df}>N/2 \) 时为负，Lucene 的 `+1` 正是为了避免这一负值。本题两种都实现，但 robertson 模式**不截断**，保留原式。
- 应用背景：Elasticsearch/Lucene 的默认相似度、稀疏检索基线、hybrid search 的关键词一路（融合见 `retrieval-rrf-fusion`）。本题只做「给定语料与 query 的完整打分」，不涉及索引构建与压缩。
- CPU、纯确定性：df/tf 只依赖计数，idf 只依赖 `math.log`，没有随机数。

## 完整推导与下标约定

**语料与有效前缀**：`docs` 是 \((N,L)\)，第 \(d\) 行的有效区间是前 \( |d|=\text{doc\_lens}[d] \) 个位置。padding 的数值完全不参与统计，可以是任意值（示例里用 -1）。因此

\[
\mathrm{df}(t)=\sum_{d=0}^{N-1}\mathbf{1}\!\left[t\in \text{docs}[d,:|d|]\right],
\qquad
\mathrm{avgdl}=\frac{1}{N}\sum_{d=0}^{N-1}|d|.
\]

参考实现用广播掩码 `positions = arange(L) < doc_lens[:,None]`，再对每个 query token 做 `(positions & (docs == t)).sum(dim=1)` 得 tf，`(tf > 0).sum()` 得 df——不建词表、不调 `bincount`，所以 token id 不需要落在某个固定 vocab 内。

**BM25 贡献**：对 query 中一次出现的 token \(t\)，

\[
c_t(d)=\mathrm{idf}(t)\cdot
\frac{\mathrm{tf}(t,d)(k_1+1)}
     {\mathrm{tf}(t,d)+k_1\left(1-b+b\,|d|/\mathrm{avgdl}\right)}.
\]

分母里的 \(k_1(1-b+b|d|/\mathrm{avgdl})\) 是长度惩罚：\(b=1\) 时完全按相对长度缩放，\(b=0\) 时退化为纯饱和。\(k_1\) 控制饱和速度，\(k_1\to0\) 时 \(c_t(d)\to \mathrm{idf}(t)\cdot \mathbf{1}[\mathrm{tf}>0]\)（只看出现与否）。

**两套 idf 与负值**：lucene 的 \( \log(1+x) \) 对任意 \(x>0\) 恒正；robertson 的 \( \log x \) 在

\[
\frac{N-\mathrm{df}+0.5}{\mathrm{df}+0.5}<1
\iff \mathrm{df}>\frac N2
\]

时为负。示例 \(N=3,\mathrm{df}(2)=\mathrm{df}(4)=2>1.5\)，实测

\[
\mathrm{idf}_{\text{lucene}}=\log 1.6=0.47000362924573563,\qquad
\mathrm{idf}_{\text{robertson}}=\log 0.6=-0.5108256237659907,
\]

两者比值实测为 \( -1.0868546368158187 \)。因此 robertson 下三篇文档的分数是 lucene 分数的同一负倍数：

```
lucene     : [1.1817234107, 1.4100108877, 0.5442147286]  -> topk [1,0,2]
robertson  : [-1.2843615683, -1.5324768713, -0.5914823012] -> topk [2,0,1]
```

排序整体反转（负倍数保持单调性但方向翻转）。若实现里写了 `max(0, idf)`，robertson 会退化成全 0 分数，`topk` 变成 `[0,1,2]`，与期望的 `[2,0,1]` 完全不同——这是本题最锋利的判别点。

**重复 query token 的线性性**：BM25 对 query 词可加，所以 query 中同一个 token 出现 \(m\) 次时总分是单次的 \(m\) 倍：

\[
\mathrm{score}(q=[t,t])=\sum_d 2c_t(d)=2\,\mathrm{score}(q=[t]).
\]

实测示例中 `score([2,2])` 与 `2*score([2])` 逐元素相等（`1e-12` 内），`score([2,2,4]) = 2*score([2]) + score([4])`。示例三篇文档的分解是

```
score([2]) = [0.5908617, 0.4700036, 0.0]
score([4]) = [0.0,       0.4700036, 0.5442147]
score([2,2,4]) = [1.1817234, 1.4100109, 0.5442147]
```

d1 之所以超过 d0，是因为它在两次 token 2 上各拿一份、token 4 上再拿一份，而 d0 的 token 2 只拿两份。若把 query 去重成 `[2,4]`，分数变成 `[0.5908617, 0.9400073, 0.5442147]`；虽然本例 top-1 仍是 d1，数值已完全不同。

**avgdl=0 的退化**：所有文档为空时 \( |d|/\mathrm{avgdl} \) 是 \(0/0\)。题面把它约定为 0，于是分母是 \( \mathrm{tf}+k_1(1-b) \)，而所有 tf 本来就是 0，分数恒 0。参考实现用 `torch.where(tf > 0, ..., 0)` 同时挡掉 `tf=0` 的除法，实测不会产生 NaN。注意 \(b=1\) 且 \( |d|=0 \) 时单个文档的 norm 也可能为 0，但该文档 tf 必为 0，仍被掩码挡住。

**空 query 与 OOV**：空 query 没有累加项，`scores` 全 0，`topk` 是稳定排序后的前 \( \min(k,N) \) 个下标，即 `[0,1,...]`。OOV token 的 \( \mathrm{df}=0 \)，lucene idf 为 \( \log(1+(N+0.5)/0.5) \)、robertson 为 \( \log((N+0.5)/0.5) \)，但所有 tf 为 0，贡献仍是 0；**不能因为 df=0 就抛错或跳过 idf 计算**（否则实现会与「OOV 分数为 0」的约定混为一谈）。

**top-k 的并列口径**：题目要求（分数降序，下标升序）。参考实现用 `torch.argsort(-scores, stable=True)`：稳定排序保留相等元素的原始下标顺序，正好是下标升序；等价于 Python 的 `sorted(range(N), key=lambda d: (-score[d], d))`。`k` 截断在排序之后，因此 `k>N` 自然返回全部 \(N\) 个。

**dtype**：分数全程 float64（`doc_lens` 转 float64、tf 转 float64、idf 是 Python float），输出 scores 为 float64；`topk` 为 int64。若用 float32 累加，robertson 负值相消和多 token 累加会在 \(10^{-9}\) 容差下被抓到。

## 核心与易错点

1. **df/avgdl 必须现算且只统计有效前缀**：把 padding 计入 df/tf 会让示例里 d2 的 -1 变成一个真实 token，分数与 topk 都错。
2. **robertson 负 idf 不截断**：`df > N/2` 时 idf 为负是定义的一部分；`max(0, idf)`、`abs(idf)`、把分数截到 0 都是错的。示例的 `[2,0,1]` 专门抓这个。
3. **重复 query token 按次数累加**：去重是另一种口径，判题会用 `score([2,2]) == 2*score([2])` 直接探针。
4. **avgdl=0 的 0/0**：必须约定长度比值为 0，并用 tf 掩码避免 NaN；`torch.where` 的两支都会求值，所以分母即使为 0 也不能让 NaN 泄漏到输出。
5. **并列按文档下标升序**：`argsort(descending=True)` 默认不保证稳定，必须显式 `stable=True` 或用 `argsort(-scores, stable=True)`；三篇完全同分的用例会抓。
6. **k 的三种退化**：`k=0` 返回空而不是报错；`k>N` 返回全部 \(N\) 个而不是补 0；空 query 时 topk 是下标前缀而不是空。
7. **OOV 的 idf 仍有定义但贡献为 0**：不要把 df=0 当成非法输入。
8. **有效 token 与 query token 非负**：padding 可以为 -1，但有效前缀里的 -1 必须抛 `ValueError`；两者的区别靠 `doc_lens` 掩码判断。
9. **浮点精度**：float64 是硬要求；\(10^{-9}\) 绝对容差下 float32 不够。
10. **b 的边界**：\(b=0\) 与 \(b=1\) 都合法；\(b=1\) 且空文档时 norm=0，必须靠 tf 掩码兜底。
11. **N=0 非法**：avgdl 无定义，题面统一抛 `ValueError`，不要返回空张量。

## 与邻近题的区别

- **vs `retrieval-rrf-fusion`（同合集）**：RRF **只用名次**，把多路结果按 \(1/(k+\text{rank})\) 融合，完全不看 term 统计、tf 与 idf；本题是稀疏打分的**本体**，输出的是可供 RRF 消费的分数与名次。RRF 的输入是排好的列表，本题的输入是原始 token 语料。
- **vs `retrieval-rag-chunk`（同合集）**：RAG 分块只输出「哪些 token 下标进哪一块」的整数矩阵，不涉及任何打分；本题消费 token id 并产生实数分数，二者的数据角色相反。
- **vs `retrieval-ivf-assign`（同合集）**：IVF 是稠密向量的欧氏距离 argmin（粗量化分配），没有 term、df、倒排表；本题是稀疏词项统计与 BM25。
- **vs `retrieval-postings-wand`（同批次）**：那道题输入已经建好的 postings，只对并集做 top-k 并用块上界剪枝，不返回稠密分数；本题输入稠密 token 语料，返回全库分数，df 必须从语料现算。两者共享 BM25 公式，但数据布局、输出与算法目标不同。
- **vs `searchsorted-bincount`**：那是「值 → 桶号 + 计数」的直方图原语，与检索打分无关；本题没有边界与桶的概念。

## 对拍口径与复杂度

- **主 oracle**：`_oracle_bm25` 是纯 Python 双层循环——先逐文档扫有效 token 求 df，再逐文档 `list.count` 求 tf、逐 query 出现累加分数；与参考解的「广播比较 + 张量求和」结构完全不同。覆盖 5 组语料：重复 query 词、零长文档、`k1=0.9/b=0.3` 的 robertson、`k1=2.0/b=0.0`、单文档；分数用 `atol=1e-9, rtol=0`，topk 用 `torch.equal`。
- **判别性用例**：示例的 lucene/robertson 双口径（常量实测回填）、`score([2,2]) == 2*score([2])` 与 `score([2,2,4]) == score([2,2]) + score([4])`、robertson 分数全负且与 lucene 成固定负比、空 query、OOV 双 idf、三篇完全同分的并列升序、全空文档（无 NaN）、`k=0/1/N/100`、13 组非法输入。
- **实测非显然事实**：示例中 d1 的 1.4100108877 高于 d0 的 1.1817234107，原因是重复的 token 2 在 d1 上贡献两次；robertson 的负 idf 让排序从 `[1,0,2]` 反转为 `[2,0,1]`；全空文档在 `avgdl=0` 下必须给 0 而不是 NaN。
- **复杂度**：参考实现对每个 query 出现做一次 \(O(NL)\) 的掩码比较与求和，总计 \(O(QNL)\) 时间、\(O(NL)\) 临时布尔矩阵；排序 \(O(N\log N)\)。判题规模 \(N\le 6, L\le 8, Q\le 4\)，远低于时限。若要做大语料，可先建 token→posting 的倒排再用 `index_add_`，本题不要求。
