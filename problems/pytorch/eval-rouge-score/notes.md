# 详解：ROUGE-1 / ROUGE-2 / ROUGE-L

## 出处
- Chin-Yew Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*（ACL 2004 Text Summarization Branches Workshop）与 Lin & Hovy, *Automatic Evaluation of Summaries Using N-gram Co-Occurrence Statistics*（NAACL 2003）；完整定义见 Lin 的 CMU 博士论文（2004）。
- ROUGE 家族：\(R_{n}/P_{n}/F_{n}\)（ROUGE-N）、ROUGE-L（最长公共子序列）、ROUGE-W（加权 LCS）、ROUGE-S（跳字 bigram / skip bigram）。本题实现前两个最常用的：ROUGE-1/2 与 ROUGE-L。
- Lin 的原文把精确率与召回组合成带权调和平均 \(F_{meas} = \frac{(1+\alpha)RP}{R+\alpha P}\)（\(\alpha\) 用来强调召回）；\(\alpha = 1\) 就是本题用的普通 F1，也是各家开源包的默认输出。
- 常见实现：官方 Perl 包 `ROUGE-1.5.5.pl`、Python 的 `pltrdy/rouge-score`、`torchmetrics.text.ROUGEScore`、`nltk.metrics`（只有简易版）。判题环境里一个都没有，所以必须按定义手写。

## 逐参考最大 vs 参考并集：口径分歧
多参考时各家不一致：官方 Perl 包对 ROUGE-N 会把同一条假设的**所有参考的 n-gram 计数并成一个"模型"**（于是召回分母是全部参考 n-gram 之和）；`rouge-score`/`torchmetrics` 走**逐条参考各算一遍、取最大**。两者在参考之间差异大时会给出明显不同的数。本题为了让样例可手算、结论确定，统一采用**逐参考取最大 F1**，再对假设求平均——这条约定写在题面里，实现时别偷偷换成"参考并集"或"参考平均"。

## 数学定义与递推
记 \(\text{count}_x(g)\) 为串 \(x\) 中 n-gram \(g\) 的出现次数，\(|\cdot|\) 为长度。

**ROUGE-N（单对）**
\[ O_n=\sum_{g\in h}\min(\text{count}_h(g),\text{count}_r(g)),\quad
R_n=\frac{O_n}{\sum_g \text{count}_r(g)},\quad
P_n=\frac{O_n}{\sum_g \text{count}_h(g)},\quad
F_n=\frac{2P_nR_n}{P_n+R_n} \]
分母 \(L-n+1\)（不足 \(n\) 则为 0 项，此时 F1 定义为 0）。裁剪的作用是给"重复刷同一 n-gram"封顶：这与 `eval-bleu-score` 的裁剪**同源但用法不同**——BLEU 的 max 是跨同一句的多条参考取最大频次后再裁剪且不关心召回，ROUGE 是在单条参考内裁剪并同时给出召回。

**ROUGE-L**：设 \(l(i,j)\) 为 \(h_{1..i}\) 与 \(r_{1..j}\) 的 LCS 长度，标准递推
\[ l(i,j)=\begin{cases} l(i-1,j-1)+1, & h_i = r_j\\ \max\big(l(i-1,j),\ l(i,j-1)\big), & h_i \ne r_j \end{cases},\qquad l(0,\cdot)=l(\cdot,0)=0 \]
然后 \(R = l(|h|,|r|)/|r|\)、\(P = l(|h|,|r|)/|h|\)、F1 同上。LCS 允许**跳着匹配**，所以它衡量的是"顺序正确的主干有多长"，与"连续窗口"（longest common **substring**）是两回事——把 `dp[i-1][j-1]+1` 之外的一支写成 0，就变成了 substring 版本，指标会莫名偏低。

## 核心与易错点
1. **裁剪（clipped overlap）不是"命中种类数"**：\(O_n\) 对每个不同 gram 取 \(\min(\text{count}_h, \text{count}_r)\)，所以参考里只出现 1 次的词，假设里写 5 次也只能算 1 次。
2. **分母是 n-gram 个数**：\(|r|-n+1\) 而不是"不同 n-gram 数"；\(|h|<n\) 时假设侧一个都没有 ⇒ F1 必须显式记 0，而不是 \(0/0\)。
3. **\(P+R=0\) 的除零**：完全不相交的集合会让两者都是 0，此时 F1 定义为 0（不是 NaN，也不该报错）。
4. **LCS 的下标**：DP 表要开 \((|h|+1)\times(|r|+1)\) 并把第 0 行/列置 0，直接按 0-based 写会漏掉边界；空间可压成两行，但本题规模用完整表更清楚。
5. **平均是在假设上做的**：空假设也是一条假设（贡献 0），别把它从分母里剔除，否则"模型输出空串"这种致命错误会被平均掉——这与 `eval-token-accuracy` 里"空样本不进 macro 分母"正好相反，因为那里统计的是**有效位**，这里统计的是**样本**。
6. **ROUGE-L 与 ROUGE-2 不可互相替代**：换序但用词全对时，ROUGE-1 仍是 1，ROUGE-2 塌方，ROUGE-L 居中；本题样例假设 0 就是 1 / 0.3333 / 0.75。三者一起报才是摘要评测的惯例。
7. **不要在这里做词干化/停用词**：官方包支持 `--stem` 和停用词表，但那依赖外部资源；判题环境没有，所以输入直接是 id 序列，比较只看相等。

## 与邻近指标的区别
- **vs `eval-bleu-score`**：BLEU = 纯精度 + 简洁惩罚，无召回；ROUGE = 精度与召回合成 F1。同一句过短的译文，BLEU 可能因惩罚而低、ROUGE-1 召回也低，但原因不同。两者都必须语料/假设级聚合。
- **vs `eval-token-accuracy`**：token acc 是**逐位一对一**的比较，对顺序完全敏感、且要求等长对齐；ROUGE-L 允许跳格匹配，因此能评长度不同的摘要。
- **vs `eval-shingle-contamination`**：那里 Jaccard/包含系数是**对称/无方向地衡量集合重叠**（用于跨语料审计），这里的 \(R,P\) 是**有方向**的（谁充当"真值"），别把分母写反。
- **vs `retrieval-rrf-fusion`**：RRF 是名次融合，没有"内容重合"概念；ROUGE 完全基于内容重叠。

## 对拍口径
参考解：`Counter` 建 n-gram 频次表 + 完整 \((|h|+1)\times(|r|+1)\) DP 矩阵求 LCS + 显式除零分支。判题 `_oracle` 完全另起一路：手工切片建 n-gram 列表、用 `list.count` 数频次（无 Counter、无哈希表）、LCS 改用**记忆化递归**（自顶向下，不是自底向上填表），F1 由独立的 `f1()` 计算。除样例外还断言：完全相同 → 1、换序 → unigram 不掉而 bigram/LCS 掉、空假设、完全不相交 → 0、单词假设 → ROUGE-2 为 0、多参考取最大、参考复制不变、空候选集 → 全 0，另有随机语料全量对拍，统一 `allclose(atol=1e-5, rtol=1e-5)`。

## 复杂度
设总 token 数 \(T\)、参考总长 \(S\)：ROUGE-N 为 \(O(T + S)\) 的 n-gram 建表与查询（元组构造 \(O(n)\)）；ROUGE-L 每对 \(O(|h||r|)\) 时间、同阶空间（可优化到 \(O(\min(|h|,|r|))\) 空间）。

## 实践提醒
ROUGE 对长摘要偏宽松（多写就多命中），这也是它常被用作 rule-based reward 时被模型"灌水"利用的原因；评测里最好同时报 \(P, R, F\)，并固定参考数量与聚合口径，否则跨论文的 0.02 差异毫无意义。
