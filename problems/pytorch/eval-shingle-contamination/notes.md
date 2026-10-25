# 详解：shingle 污染审计

## 出处

- **shingling / containment 的原始出处**：Andrei Broder, *On the Resemblance and Containment of Documents*, SEQUENCES'97（Compression and Complexity of Sequences, IEEE, 1997）。这篇文档近似重复检测的奠基工作里同时定义了两个重叠度——resemblance \(|A\cap B|/|A\cup B|\)（即今天说的 Jaccard 相似度）与 **containment** \(|A\cap B|/|A|\)（分母只在一侧、天然非对称），并提出用"压缩指纹"（shingle + 哈希）在超大规模语料上估计它们。本题的两个矩阵正是这两个量的逐对版本。
- **MinHash 近重复检测**：同一时期 Broder–Glassman–Manasse–Zweig 的 MinHash 系列（min-wise 独立置换 sketch，1997；arXiv/技术报告号 待回源）把 \(|A\cap B|/|A\cup B|\) 与 containment 的估计压到常数级哈希，是今天所有近重复去重（LSH 分桶）的算法底座。
- **LLM 语料去重**：Lee et al., *Deduplicating Training Data Makes Language Models Better*（ICML 2022；arXiv 号 待回源）把 MinHash-LSH shingling 当作大规模训练语料去重的标准件——注意它做的是**同库去重**（决定删掉谁），与本题的**跨集合审计**是两码事（见"与邻近指标/算法的区别"）。
- **n-gram containment 作为污染/泄漏判据**：训后评测里用连续 n-gram 包含率检查"评测集有没有出现在训练数据里"已是主流基准的常规卫生步骤；面向此的代表工作如 Roberts et al., *Is My Data in Your Language Model?*（2023；arXiv 号 待回源）——用 n-gram/三 gram 的 containment 而非 Jaccard 判训练数据重叠，动机与本详解第 2 节一致。
- 判题环境没有任何去重/相似库（`datasketch`、`text-dedup`、`nltk` 均禁止且不存在），上述指纹/LSH/估计技巧都不可用，必须按定义手写集合与交并。

## 完整推导与下标约定

记评测文档 \(i\) 的 shingle 集为 \(A_i\)、训练文档 \(j\) 的为 \(B_j\)，矩阵下标 \(i\) 走评测侧、\(j\) 走训练侧，全程固定这个方向。

1. **窗口与集合大小**。长度 \(L\) 的文档在 \(L \ge n\) 时给出 \(L-n+1\) 个重叠窗口、\(L<n\) 时给出 0 个。但 shingle 集是**去重后的集合**，所以 \(|S| = \) 不同 \(n\)-gram 的**种类数**，\(\le L-n+1\)。把"个数"当成 \(L-n+1\) 是第一步就把口径定歪了（见第 3 节）。
2. **交集**。\( |A_i \cap B_j| \) 是"既在这篇评测文档、又在那篇训练文档里出现的不同 shingle 数"——两侧都先去重，交集自然是集合交。
3. **包含率钉在评测侧**。\( \text{contam}[i,j] = |A_i\cap B_j| / |A_i| \)。因为 \(A_i\cap B_j \subseteq A_i\)，其值域是 \([0,1]\) 恒成立。它是**非对称**的：交换 \(i,j\)（或换分母到 \(|B_j|\)）得到的是不同的量，本题不存在反方向的矩阵。语义是"评测文档 \(i\) 被训练文档 \(j\) 覆盖了多少比例"——这正是污染审计要问的问题。
4. **Jaccard 钉在并集上**。\( \text{jac}[i,j] = |A_i\cap B_j| / |A_i\cup B_j| \)，用容斥 \(|A_i\cup B_j| = |A_i|+|B_j|-|A_i\cap B_j|\) 求并。它对称、值域 \([0,1]\)。由于 \(A_i \subseteq A_i\cup B_j\)，有 \(|A_i\cup B_j| \ge |A_i|\)，故**逐元素** \(\text{jac}[i,j] \le \text{contam}[i,j]\)（分子相同、Jaccard 分母更大），这就是判题里那条 `jac.max() <= contam.max()` 不变量的来源。
5. **哨兵**。\( |A_i| = 0 \) 时 contam 整行记 0（回避 \(0/0\)）；\(|A_i\cup B_j| = 0\)（两侧皆空）时 jac 记 0。任一侧语料为空则矩阵相应维度为 0——\((M_{\text{eval}},0)\)、\((0,M_{\text{train}})\)、\((0,0)\)——`flagged` 为空。
6. **flagged**。对每个非空 \(A_i\) 取 \(\max_j \text{contam}[i,j]\)，与 `threshold` 比 `>=`（含等号）。要求训练侧非空才能取 max（对空集合求 max 会崩）。`threshold > 1` 必空（上界 1）；`threshold = 0` 时凡 \(A_i\neq\varnothing\) 者都命中（max \(\ge 0\) 恒成立）。

## 核心与易错点

1. **为什么用包含率而不是 Jaccard（量化）**。取评测文档 `[1,2,3,6,4]`（3 个 shingle），配一篇训练文档：它把这三个 shingle 全包含，另外还有 25 个自己的 shingle（\(|B| = 28\)）。containment \(= 3/3 = \) **1.0**（逐字命中），Jaccard \(= 3/(3+28-3) = 3/28 \approx\) **0.107**——被训练侧多出的 27 个 shingle 稀释到几乎为 0。真实语料单篇动辄上万 token，用 Jaccard 判污染几乎永远过不了阈；而污染信号本只该由"评测文档被覆盖了多少"决定，不该被训练文档的长度左右。这正是 `check_containment_beats_jaccard_on_long_docs` 守住的不变量。
2. **分母口径钉死在评测侧**。写成 \(/|B_j|\) 或 \(/(|A_i|+|B_j|)\) 就丢掉"覆盖比例"语义、并且把长训练文档变成减分项。非对称是特性不是 bug：contam[i,j] 与 contam[j,i] 语义不同（"评测被训练覆盖"vs"训练被评测覆盖"），本题只要前者。
3. **集合口径 vs 频次口径**。本题 shingle 是**集合**，重复的 n-gram 只算一次。`[1,1,1]` 在 \(n=2\) 下有两个窗口 (1,1)、(1,1)，但集合只有 \(\{(1,1)\}\)、\(|A|=1\)，配训练 `[1,1]`（集合 \(\{(1,1)\}\)）得 contam = jac = 1。若改用**频次/多重集**口径、分子数 2 个命中窗口而分母按去重集算成 1，包含率变成 \(2/1 = 2 > 1\)，直接越界——"比例"必须 \(\le 1\) 是集合口径的硬约束（`check_repeated_shingles_collapse` 就断言 `contam.max() <= 1`）。凡窗口高度重叠、有连续重复 token 的文档，频次口径都会把分子/分母算歪。
4. **空集与哨兵别记成 NaN 或 1**。\(|A_i|=0\)（评测文档长度 < \(n\)）⇒ 该行全 0 且**永不进 flagged**（哪怕 `threshold=0`）：空集谈不上"被覆盖比例"，\(0/0\) 必须回避。任一侧语料为空 ⇒ 形状退化、flagged 空；别在空训练侧上求 max。把空行记成 1 会让"全空审计"拿满分，与 `eval-ndcg-at-k` 里 IDCG=0 记 0 的哲学一致。
5. **阈值边界含等号**。判定用 `>=`：包含率恰等于 `threshold` 也算命中。`check_boundary_threshold_inclusive` 里 max 恰为 0.5，`threshold=0.5` 命中、`threshold=0.5+1e-6` 不命中。写成 `>` 会漏掉"刚好踩线"的评测文档。
6. **flagged 是升序下标、不是命中次数**。它是"哪些评测文档被污染"的列表，每个符合的下标出现一次、升序。`check_duplicate_train_columns` 说明训练侧复制一篇只是多出一列相同值，flagged 不变——别把"命中了几篇训练文档"塞进 flagged。
7. **dtype 与形状双核对**。contam/jac 是 float32、flagged 是 int64；判题同时检查 `list(flagged.shape)==[len(flagged)]` 与 `flagged.dtype==int64`。float32 存 \(1/7\)、\(1/3\) 有舍入，判题用 `atol=rtol=1e-5`。

## 与邻近指标/算法的区别（分工）

- **vs `data-tokenization`（或拟新增 `data-pipeline`）里规划中的库内去重题**（本仓库 todo #66③：exact hash + MinHash + LSH 近重复；尚未落地，落地前勿臆造 slug）。那类题做的是**单一语料内部**的重复消除——exact hash 抓完全相同、MinHash+LSH 抓近重复，输出是"保留/删除哪些文档"（keep-set 或去重后的语料）。本题是**两个集合之间**的审计：不删任何东西、也不判"训练集内部谁和谁像"，输出的是 eval-doc × train-doc 的**重叠矩阵**（contam/jac）加一份"哪些评测文档被训练语料覆盖"的 flagged 列表。方向、粒度、产物都不同——一个清洗训练集本身，一个给评测集做卫生检查。
- **vs `retrieval-*` 合集**（`retrieval-ivf-assign` / `retrieval-pq-codes` / `retrieval-hnsw-greedy` / `retrieval-mmr` / `retrieval-rrf-fusion` / `retrieval-rag-chunk`）：检索那组是**给 query 找近邻文档并打分/排序**（向量距离、召回、融合），关心"哪个文档最相关"。本题是**数据卫生审计**：输入是文档切成的离散 n-gram 集合、做的是集合交并，全程没有 embedding、距离或 top-k。即便 `retrieval-rag-chunk` 也切文档、也出矩阵，那是 chunk×query 的相似度检索，与这里的 eval×train shingle 重叠不是一回事。
- **vs 其余 `eval-*`**：`eval-ndcg-at-k` / `eval-pass-at-k` / `eval-perplexity-logits` / `eval-token-accuracy` 评的是**模型输出**（排序质量、通过率、语言建模概率、逐 token 正确率），`eval-bleu-score` / `eval-rouge-score` 评的是**生成文本与参考的 n-gram 重叠**。本题评的是**数据集本身**——训练/评测语料的卫生度，模型还没跑就先审计。与 BLEU/ROUGE 关系最微妙：它们也用连续 n-gram，但 BLEU 把 n-gram 当**带裁剪/带多重集**的打分原料、ROUGE 用**召回/ LCS**，比较"假设 vs 参考"；本题的 n-gram 是**纯集合去重**、比较"评测 shingle 是否出现在训练 shingle 里"，并额外给出非对称的包含率。同是 n-gram，口径与目的都不同。

## 对拍口径

参考解把 shingle 直接做成 `set[tuple[int]]`（手工窗口 + set 去重），逐对用 `len(a & b)` 数交集、容斥求并集，交集向量一次成行、`torch.where(union > 0, ...)` 兜哨兵；空 \(A_i\) 与空侧语料各有分支。判题 `_oracle`（见 `check_matches_oracle`）完全另起一路、不用任何集合代数：把每个 shingle 编码成 `"1|2|3"` 这样的字符串键存进**普通 list** 保序，靠一个 `seen` dict 手工判重，交集大小用**显式双重循环**逐键 `if x in bset` 数出来，并集用 \(|A|+|B|-|A\cap B|\)。除样例外还覆盖：\(n=1\) 的 unigram 集、\(n=4\) 的更大窗口、`[1,1,1]` 去重塌缩、含空文档/空侧语料、评测集==训练集的对角全 1、词表不相交、`threshold ∈ {0,0.25,0.75,1.0,1.5}` 的边界、以及 6 组随机语料（`torch.Generator().manual_seed(300)`）× 多阈值的全量对拍，统一 `allclose(atol=1e-5, rtol=1e-5)`；flagged 另按"非空行最大值 ≥ th"（空行永不参与）单独核对。

## 复杂度与大规模做法

设评测/训练文档数 \(M_e, M_t\)，第 \(i/j\) 篇去重后的 shingle 数为 \(|A_i|, |B_j|\)，shingle 长 \(n\)。构造集合是逐文档滑窗 \(O(\sum (|d|-n+1)\cdot n)\)（元组构造 \(O(n)\)）加哈希去重。暴力逐对：每个 \((i,j)\) 求 \(|A_i\cap B_j|\) 取较小集合遍历、平均 \(O(\min(|A_i|,|B_j|))\)，总计 \(O(M_e M_t \bar{|S|})\)；而 \(\text{contam}/\text{jac}\) 是 \(M_e\times M_t\) 的稠密矩阵，\(O(M_eM_t)\) 空间无法回避——输出本身就是这个量级。**本题为何用暴力集合**：判题规模是几十篇、每篇几到几十个 token 的 toy 语料，暴力集合最确定、无浮点近似，且要求**精确**矩阵而非估计。一旦上到真实规模，标准做法是把每篇的 shingle 集换成 **MinHash sketch**（\(O(|S|)\) 建、常数级比较，把 resemblance/containment 都变成对 sketch 的估计），再用 **LSH 分桶**只对候选近重复对精算；或用**倒排索引**（shingle → 含它的训练文档列表）把每个 eval-shingle 的命中累加到对应 \(|A_i\cap B_j|\) 分子上、跳过显然为 0 的对，把逐对比较压到与稀疏共现量成正比。判题环境里这些库都不存在、又需要精确值，所以手写集合反而是正解。

## 实践提醒

- **\(n\) 的选择是把双刃剑**：\(n\) 太小（1、2）shingle 遍地都是，跨文档偶然共现就多，假阳性爆表（"评测里的 `the cat` 训练里当然也有"）；\(n\) 太大则要求近乎逐字命中，稍有改写/同义替换就漏检，且长度 \(<n\) 的短评测文档直接变空集漏审。LLM 泄漏审计常用词级 5~13 n-gram（三 gram/五 gram 都见过，视语言与改写强度而定），本题以 token id 输入把这个旋钮交给调用者。
- **flagged 用 \(\max_j\) 是"存在性"口径**：只要有**任何一篇**训练文档把该评测文档覆盖到足够比例就算污染——污染是存在命题，不需要对训练文档求和或平均。若要更保守（多篇各覆盖一部分合成总污染），得另设口径，本题不做。
- **训练侧复制/近重复会抬高"命中列数"但不改 flagged**（见 `check_duplicate_train_columns`）：真实审计通常先对训练集去重再跑，否则同一污染被计多次、矩阵变宽而判定不变。
- **报数要带 \(n\) 与 `threshold`**：同一对语料，换 \(n\) 或换阈值结论就翻盘；一份审计报告只写"污染率 X%"而不写 shingle 长度与阈值，等于没写。
