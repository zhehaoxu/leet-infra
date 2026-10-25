# 详解：nDCG@k

## 出处
- Kalervo Järvelin & Jaana Kekäläinen, *Cumulated Gain-based Evaluation of IR Techniques*, ACM TOIS 20(4), 2002。原文的动机是：precision/recall 只认"前 \(k\) 里命中了几个"，把命中第 1 名和第 10 名视为等价，也无法利用分级标注（graded relevance）。于是定义
  \[ \text{CG}_k = \sum_{i=1}^{k} r_i,\qquad
     \text{DCG}_k = \sum_{i=1}^{k} \frac{r_i}{\log_2(i+1)} \;(\text{原文对高相关文档要求更高代价时用的形式}) \]
  折损形式 \(1/\log_2(i+1)\) 是他们用**人工满意度实验**筛出来的：比线性折损、\(1/i\) 折损更贴近用户对名次的感知。
- 归一化项 IDCG 来自同一篇：把同一批标注按理想方式排序后算出的 DCG 上界，\(0 \le \text{nDCG} \le 1\) 才能跨 query、跨候选池大小比较。原文还给出 \(r_i\) 直接取标注值（**线性/算术增益**）和取 \(2^{r_i}-1\)（**指数增益**）两种用法，后者强调"高档标注"，是现代实现的默认。
- 现代实现：`sklearn.metrics.ndcg_score`（参数 `norm` 取 `exponential`/`arithmic`，折损同样是 \(\log_2(\text{rank}+1)\)、rank 从 1 起）、`torchmetrics.RetrievalNormalizedDCG`（`gain_function` 可选）、RankLib、pytrec_eval（BEIR/MTEB 榜单走它，但它的增益形式与 sklearn 默认不同）。各家"分数对不上"的第一步排查永远是**增益形式**和 **rank 起点**，判题环境里 sklearn 也不存在，所以必须按定义手写。

## 完整推导与下标约定
设候选文档 \(d_0,\dots,d_{N-1}\)，模型打分 \(s_i\)，人工标注 \(r_i \in \mathbb{Z}_{\ge 0}\)。

1. **名次**。取一个 \(s\) 的降序排列 \(\pi\)，要求"分数相等时按下标升序"。这个 tie-break 不是可选项：它是唯一能让实现确定化的约定（稳定排序、或按 \((-s_i, i)\) 排序，两种写法等价）。
2. **截断**。\(m = \min(k, N)\)。DCG 只累加前 \(m\) 个名次，\(k > N\) 与 \(k = N\) 完全等价——因为 \(m\) 之后没有文档可取，而不是用 0 补齐。
3. **增益**。
  \[ g(r) = \begin{cases} 2^{r}-1, & \text{指数（默认）}\\ r, & \text{线性}\end{cases} \]
  \(g(0) = 0\) 在两种口径下都成立，所以"无关文档排在前面"不贡献分子、却**挤掉了**后面能贡献的名次，这正是惩罚。
4. **折损**。名次 \(i\) 从 1 开始，\(c_i = 1/\log_2(i+1)\)：\(c_1 = 1\)（首位不打折）、\(c_2 \approx 0.6309\)、\(c_3 = 0.5\)、\(c_5 \approx 0.4307\)。写成 \(1/\log_2 i\) 会在 \(i=1\) 处除零得到 \(\infty/\text{NaN}\)；写成 \(1/\log_2(i+2)\)（0-based 下标常见误写）则整体把每位都多折一次，DCG 单调偏低，但**注意 IDCG 若同样错则 nDCG 部分抵消**，只看比值可能发现不了——所以题目要求同时返回 `dcg` 与 `idcg`。
5. **累积与理想值**。
  \[ \text{DCG}@m = \sum_{i=1}^{m} g(r_{\pi(i)})\,c_i,\qquad
     \text{IDCG}@m = \sum_{i=1}^{m} g(r^{\downarrow}_i)\,c_i,\quad r^{\downarrow} = \text{sort}(\text{rels}, \text{desc}) \]
  IDCG 只依赖标注的多重集合，与打分无关（这点常被误以为要"用理想模型的分数排序"）。
6. **归一化与哨兵**。
  \[ \text{nDCG}@k = \begin{cases} \text{DCG}/\text{IDCG}, & \text{IDCG} > 0\\ 0.0, & \text{IDCG} = 0\end{cases} \]
  \(\text{IDCG} = 0 \iff\) 前 \(m\) 个理想标注的增益全为 0 \(\iff\) 所有 \(r_i = 0\)（或 \(m = 0\)）。此时"排序好坏"无从谈起，主流做法是记 0 而不是 NaN；把它记成 1 会让"全库都无关"的检索系统拿满分。

## 核心与易错点
1. **折损下标差一**（上面第 4 条），最常见的两个形态：\(i\) 用 0-based 却代 \(1/\log_2(i+1)\)（首位变成 \(\log_2 1 = 0\) ⇒ 除零），或分子分母用了不同的 \(m\)。
2. **IDCG 忘了截断到同一个 \(m\)**：用整份列表的 IDCG 去除 \(\text{DCG}@k\)，\(k\) 越小分数越被压低，且没法解释"为什么我的 nDCG@10 比别人的低 3 个点"。
3. **tie-break 不稳定**：`np.argsort` 默认快排、不保证稳定，torch 的 `argsort` 要显式 `stable=True`（或先按分数降序、再按原始下标升序做二次键）。平局在真实系统里很常见（同一个 embedding 命中多篇、reranker 输出被截到 float16 后大面积同分），一旦 tie-break 不固定，同一个 checkpoint 两次评测会得到不同的 nDCG，属于"评测不可复现"的经典事故。本题的平局样例（两个 0.85）就是专门用来抓这个的：写错时 \(0.874331 \to 0.718479\)。
4. **排序时把 `scores` 排了、`rels` 忘了同步**：正确做法是拿 argsort 的**索引**去 gather 标注；`zip` 后分别排序、或者 `sorted(rels)` 当理想排序，都会静默出错。
5. **\(2^{r}-1\) 的溢出与 dtype**：标注到 60 时 `float32` 精度已经不够，本题用 float64 中间量再降回 Python float；另外 \(r\) 必须是整数，若标注被上游当成 float 传入，\(\lfloor r \rceil\) 的口径要和标注文档对齐。
6. **除零/NaN 传染**：IDCG 为 0 时如果不走哨兵分支，得到 NaN，再 `mean` 一次整批评测就是 NaN。批量评测里更稳的做法是逐 query 返回 nDCG（含 0）再对 query 求平均，即 **macro** 口径；这与 `eval-token-accuracy` 的 micro/macro 之分同源。
7. **\(k > N\) 时不要越界**：`min(k, N)` 必须在索引之前算好。

## 与邻近指标/算法的区别
- **vs `retrieval-rrf-fusion`**：RRF 的输入是**多个系统各自的名次**、按 \(1/(k+i)\) 型权重相加，没有任何人工标注，产出仍是"一个排序"；nDCG 的输入是**一个排序 + 分级标注**，产出是"一个分数"。RRF 里没有"增益"概念，nDCG 里没有"融合"概念。
- **vs `retrieval-mmr`**：MMR 是**查询时的重排策略**（相关性 × 多样性换权），nDCG 是**评测事后打分**；MMR 的多样性收益只有配上带分级标注的 nDCG 才量化得出来。
- **vs precision@k / recall@k / hit-rate**：那些是二元标注（命中/未命中）、名次内部无差别；把 nDCG 的 `rels` 全部换成 0/1 并用线性增益，它退化成"带名次折损的 precision 型"指标，但仍不等价于 P@k。
- **vs `eval-bleu-score` / `eval-rouge-score` / `eval-token-accuracy`**：那一组评的是**生成内容本身**（n-gram、LCS、逐位对齐），nDCG 评的是**候选集合的顺序**，输入长度、参考形式完全不同。
- **vs `eval-shingle-contamination`**：contamination 是数据审计（判定评测集有没有污染训练集），nDCG 是排序质量；两者都会出现在同一份 RAG 评测报告里，但没有任何公式重叠。

## 对拍口径
参考解：`torch.tensor(float64)` + `torch.argsort(descending=True, stable=True)` 拿名次 + 一次 `torch.sort` 得理想序列 + `torch.pow/log2/arange` 向量化求和。判题 `_oracle` 完全另起一路：**手写选择排序**每轮在未排队里找最大且只在"严格大于"时替换（由此天然得到下标升序的平局），增益用**循环倍增**而不是 `2**r`，折损用 `math.log(i+2, 2)`（0-based 循环 + 1），理想序列用**插入排序**，全程纯 Python 标量累加，不用任何张量。
除样例外还有：理想排序 ⇒ 1.0 且 DCG == IDCG、\(k > N\) 与整份列表一致且 DCG/IDCG 随 \(m\) 单调不减、全 0 标注 ⇒ `(0.0, 0.0, 0.0)`、空输入与 \(k = 0\)、同分平局的 DCG 逐项手算、单文档、打分互异时的排列不变性、指数增益 < 线性增益的具体数值，另加 8 组随机（含负打分、\(k > N\)、两种增益）全量对拍，统一 `isclose(atol=1e-5, rel_tol=1e-5)`。

## 复杂度
排序 \(O(N \log N)\)（稳定排序）；若只要前 \(k\) 个名次，可用选择/堆做到 \(O(Nk)\) 或 \(O(N + k\log N)\)。DCG/IDCG 各 \(O(k)\) 次乘加，空间 \(O(N)\)（标注与名次数组）。批量评测时逐 query 独立，总代价 \(\sum_q O(N_q \log N_q)\)。

## 实践提醒
榜单上 nDCG@10 与 nDCG@5 差 1~2 个点很常见，报数必须写清 \(k\)；标注尺度（0–3 vs 0–4 vs 0–10）换了以后 \(2^r-1\) 的量级会指数级变化，跨数据集比较等于换了指标。用 LLM-as-judge 生成分级标注时，务必固定 prompt 与档位定义，否则 IDCG 的分母本身就在漂。
