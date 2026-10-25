# 详解：Contrastive Search 候选退化惩罚选择

## 出处

- **Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, Nigel Collier**,
  *A Contrastive Framework for Neural Text Generation*, **arXiv:2202.06417**（提交日期 2022-02-13，v2 于
  2022-05-30 更新）。Contrastive search 是该文 §3.2 提出的解码方法（论文里 CNTRM 是它的代号），与 SimCTG
  训练目标配套。已回源核对到的原文要点：① 每步"从模型预测的最可能候选集合里选"，候选集合记作 top-\(k\)，
  原文 \(k\) 常取 \(3\sim 10\)；② 选择式（原文 Eq. 5）的第一项是**模型置信度** \(\log p(v\mid x_{<t})\)，第二项
  是**退化惩罚**，定义为"候选表示与上下文 \(x_{<t}\) 中**所有**已生成 token 表示之间的**最大余弦相似度**"
  （相似度函数即原文 Eq. 3，\(\cos\) 值域 \([-1,1]\)）；③ \(\lambda\) 折权两项，\(\lambda=0\) 时"退化为贪心搜索"。
  这三点与本题的接口形状、\(\lambda=0\) 退化性质完全吻合。
- **Yixuan Su, Nigel Collier**, *Contrastive Search Is What You Need For Neural Text Generation*,
  **arXiv:2210.14140**（2022-10-25）。后续研究：跨 16 种语言评估后认为 SimCTG 的"各向异性"结论只在
  GPT-2-small/medium 成立，而 **contrastive search 这个解码方法本身不需要额外训练**就普遍有效——这正是本题
  可以脱离语言模型、只把相似度矩阵当入参来出题的现实依据。
- **本题与原文的两处口径差异（以题面 / `starter.py` 为准）**：
  1. 原文惩罚项直接用**原始余弦相似度** \(s\in[-1,1]\)；本题先线性映射成 \(\deg=\frac{s+1}{2}\) 再夹到 \([0,1]\)。
     由于 \(\max_i\frac{s_i+1}{2}=\frac{\max_i s_i+1}{2}\)（映射单调），两者的差别只是
     "\(\lambda\) 的刻度差 2 倍"加上一个**与候选无关的公共平移** \(-\lambda/2\)：平移不改变 argmax，
     所以本题的 \(\lambda\) 对应原文口径的 \(\lambda/2\)。夹到 \([0,1]\) 也不改变次序（夹取单调）。
     原文并未给出 \((1+s)/2\) 这种写法，该归一化是**本题的工程约定**（让惩罚落在 \([0,1]\)、与"相似度 1 = 最差"
     的直觉对齐）；若要去比对某个具体上游实现的归一化写法，**待回源**。
  2. 原文的候选隐态要把"前缀 + 候选"再喂进模型跑一次前向才能拿到；判题环境没有语言模型，所以本题把
     \(sim\in\mathbb{R}^{K\times T}\) 直接作为显式入参，并规定**行序 = 概率降序候选**。
- **明确不采信的说法**：本合集条目把出处写成 "Chan et al., arXiv 2202.10479, *Language Models with Contrastive
  Search*"。2026-10-25 实测：arXiv:2202.10479 的元数据是 *A flavor-inspired radiative neutrino mass model*
  （中微子质量模型，与文本生成无关），且 arXiv 的标题检索 `ti:"Language Models with Contrastive Search"` 与
  `au:"Chan" AND ti:"contrastive search"` 均返回 0 条结果。因此**不引用该编号**，条目的这一句记为 **待回源**。
  另外 `solution.py` 头注释里的"Wu et al. 2022"与 arXiv:2202.06417 的实际作者（Su 等）不符，属注释笔误，
  编号本身可用。
- **判题环境里没有任何现成实现可借**：`.venv-eval` 的 `transformers` 为 5.20.2，包内只剩
  `GenerationMode.CONTRASTIVE_SEARCH` 这个模式枚举与一条社区模型 URL，`generation/logits_process.py` 里
  `grep` 不到 contrastive / `penalty_weight` 相关处理器；llama.cpp 也不存在。所以打分与选择都必须手写。

## 完整推导与下标约定

设 \(z\in\mathbb{R}^{V}\)，\(p=\operatorname{softmax}(z)\)，\(\ell=\log\operatorname{softmax}(z)\)，
候选数 \(K=\min(k,V)\)，历史隐态数 \(T\)。

1. **候选序**。把下标按二元键 \((-\,p_i,\ i)\) 升序排，取前 \(K\) 个，得到 \(\text{cand}_0,\dots,\text{cand}_{K-1}\)。
   softmax 是单调的，所以"按概率降序"与"按 logit 降序"给出同一批 token；但**并列时必须用索引升序**，
   否则同一份输入两次运行可能换人。本题里"候选内秩 \(j\)"与"词表索引 \(\text{cand}_j\)"是两套下标，
   `chosen` 返回后者，`deg` / `scores` 的第 \(j\) 个元素属于前者——把它们混用是最典型的越界来源。
2. **退化惩罚**。\(sim_{j,i}=\cos\big(h_{\text{cand}_j},\,h^{\text{hist}}_i\big)\)。
   \[
   \deg_j=\operatorname{clamp}_{[0,1]}\Big(\max_{1\le i\le T}\tfrac{sim_{j,i}+1}{2}\Big),\qquad
   \deg_j\equiv 0\ \ (T=0).
   \]
   \(\max\) 而非 mean：只要历史里有一个隐态跟候选几乎同向，就判它"退化"。\(i\) 的下标顺序不参与运算，
   因此 \(sim\) 的**列**（历史）可以任意排列，**行**（候选）不可以。
   \(T=0\) 是合法输入（第一个 token 之前没有历史），此时 \(sim\) 传 \(K\times 0\) 的空张量，
   分数退化成 \((1-\lambda)\ell\)，\(\lambda\) 完全不起作用。
3. **打分**。
   \[
   \text{score}_j=(1-\lambda)\,\ell_{\text{cand}_j}-\lambda\,\deg_j .
   \]
   \(\ell\) 用全词表 log-softmax：\(\sum_{i}e^{\ell_i}\) 不为 1 也没关系，**不要**把 \(\top_k\) 核内归一化后再取
   log——那会把两项的相对刻度改掉，等价于偷偷换了 \(\lambda\)。
4. **argmax 与并列**。\(j^{*}=\arg\max_j \text{score}_j\)，并列取最小 \(j\)。实现上等价于一次"严格大于才更新"
   的线性扫描，也与 CPU 上 `torch.argmax` 的首次命中一致；写成 `>=` 会翻到并列里最靠后的候选。
5. **翻盘点（示例数据可验证）**。两个候选 \(a,b\) 谁胜，取决于
   \((1-\lambda)(\ell_a-\ell_b)\gtrless \lambda(\deg_a-\deg_b)\)，即阈值
   \(\lambda^{*}=\frac{\ell_a-\ell_b}{(\ell_a-\ell_b)+(\deg_a-\deg_b)}\)。示例中 \(a=0,b=1\)：
   \(\ell_0-\ell_1=0.3\)、\(\deg_0-\deg_1=0.775\)，故 \(\lambda^{*}=0.3/1.075=0.279070\)。
   实测 \(\lambda=0.279069\) 时 scores \([-0.837489,\,-0.837490,\,-1.453769]\) 选 \(0\)；
   \(\lambda=0.279071\) 时 \([-0.837489,\,-0.837488,\,-1.453767]\) 选 \(1\)。
6. **跨步语义（不在本题计算范围内）**。真实解码中，被选 token 的隐态进入下一步历史，于是
   与它同向的后续候选 \(\deg\) 会抬到接近 \(1\)、被压下去；这就是"重复越多、惩罚越重"的自反馈。
   本题只交出**单步**的四元组，历史由入参 \(sim\) 扮演。

## 核心与易错点

1. **没有随机数**。这是本合集里唯一一个"确定性解码"的采样题：不需要 \(u\)，也不许自己造 \(u\)。
   任何 `torch.rand` 都会让同一输入在不同运行间给出不同 `chosen`。
2. **\(sim\) 的行序**：第 \(j\) 行属于**第 \(j\) 个降序候选**。如果按原词表顺序取前 \(K\) 行，
   只要候选不是恰好 \(0..K-1\) 就整体错位（示例里候选正好是 \([0,1,2]\)，抓不住这个 bug；
   随机对拍用例里候选序是乱的，能抓住）。
3. **\(\log\) 不做核内归一**：用全词表 \(\log\operatorname{softmax}\)。这与 `top-k-sampling`、
   `top-p-sampling` 里"保留集合归一化后再用"的写法相反，照抄会错。
4. **\(\max\) 之后再夹**：\(\operatorname{clamp}(\max)=\max(\operatorname{clamp})\)（夹取单调），两种写法等价；
   但若上游把 \(sim\) 夹到 \([0,1]\) 而不是把 \(\deg\) 夹到 \([0,1]\)（即对**负的**相似度也做 0 截断），
   惩罚会被系统性地抬高、结果变贪心。
5. **并列取更靠前者**：`score` 完全相等时（示例里 \(k>V\)、\(sim=0\)、两个 1.0 logit 那一行）选秩 0。
   用 `argmax` 之外的手写比较时，条件必须是严格 `>`。
6. **\(k>V\) 要夹**：\(K=\min(k,V)\) 之后 \(sim\) 的行数才确定；不夹就索引越界。
7. **两个退化性质是硬性检查点**：\(\lambda=0\) ⇒ 结果 = 全局 argmax 且 `scores` = \(\ell[\text{cand}]\)；
   \(T=0\) ⇒ \(\deg\equiv 0\)、\(\lambda\) 无影响（此时任意 \(\lambda\) 都选同一个 token）。
   \(\lambda=1\) ⇒ `scores` \(=-\deg\)、选 \(\arg\min\deg\)，**候选集合不变**。
8. **dtype 与顺序**：`cand_idx` 必须 `int64`；`deg`/`scores` 与输入同 dtype（float32），
   输出顺序必须与候选顺序逐项对齐，判题按数组等值比较。

## 与邻近题的区别

- **vs `top-k-sampling`**：两者**同样**先取 \(\top_k\) 候选，但那里的 \(k\) 就是"保留集合"、集合内做
  核内归一化后用显式 \(u\) 抽签，唯一准则是概率本身；这里的 \(\top_k\) 只是**打分范围**，选择分数由
  相似度退化惩罚主导，且是 argmax、不抽签。同一批候选会选出不同 token——示例里候选 \([0,1,2]\) 在
  \(\lambda=0.6\) 下选出 1，而 top-k 采样的期望质量仍集中在 0。
- **vs `penalty-sampling`**：那是 **token-count 驱动**的重复抑制——按 `history` 窗口里的出现次数对
  logits 做 repeat 乘/除、再减 \(\text{count}\cdot\text{freq}+\text{present}\)，作用于**整个词表**、发生在
  softmax **之前**，而且它**不选 token**（返回惩罚后的 logits）。本题是**嵌入相似度**驱动、只作用于 \(K\) 个候选、
  在候选集合内 argmax 并直接给出 `chosen`。语义级 vs 字面级的差别最明显：换个说法的复读在 penalty 下没事，
  在本题 \(\deg\) 会很高。
- **vs `dry-sampling`**：DRY 用反向 Z-algorithm 找**精确 n-gram 重复**再按 \(base^{R-\text{allowed}}\) 罚 logit，
  同样是字符串级、且不选 token；本题不做子串匹配。
- **vs `mirostat-v2-sampling`**：Mirostat v2 维护一个跨步的 \(\mu\)（目标困惑度）状态，按 \(-\log_2 p\le\mu\)
  截断前缀、抽签，并把 \(\mu_{\text{new}}=\mu-\eta(b-\tau)\) 交回调用方——它是**反馈控制回路**。
  本题没有状态、没有反馈、不抽签，\(\lambda\) 是固定超参。两者都想"随置信度自适应地避免退化"，
  但 \(\mu\) 走的是**熵/困惑度**通道，本题走的是**表示相似度**通道；把本题的 \(\deg\) 换成熵就得到另一族方法。
- **vs `top-n-sigma-sampling` / `min-p-sampling` / `top-p-sampling` / `typical-sampling`**：这些都是
  与已生成内容**无关**的概率/统计量过滤（阈值来自本步分布自身形状），本题的分数显式依赖历史隐态；
  过滤型方法输出的是 \(filtered\) 与抽签结果，本题输出候选、惩罚、打分与 argmax 选择。
- **vs `xtc-sampling`**（同合集）：XTC 砍的是**概率头部**（过阈值的连续前缀），完全不看历史；
  本题一个 token 都不砍，只在候选内重新排序选择，看的全是历史。

## 对拍口径

参考解用 `softmax` + `argsort(descending=True, stable=True)` 拿候选、`(sim+1)/2` 后 `max(dim=-1)` 再
`clamp(0,1)` 拿 \(\deg\)、一次显式的"严格大于"线性扫描做 argmax。判题 `_oracle` 另起一路：纯 Python
`math.exp` + `math.fsum` 手算 softmax 与 `x - max - log(sum)` 形式的全词表 log-softmax、
用 `sorted(range(V), key=lambda i: (-p[i], i))` 拿候选、逐候选逐历史列手工取最大并夹取、
打分用列表推导、选择同样用严格 `>` 扫描；用例覆盖 60 组随机 \((V\in[3,13),\,k\in[1,V+2),\,T\in[0,5),\,
\lambda\in\{0,0.35,0.6,1\})\) 全量对拍，另有 \(\lambda=0\) 贪心、\(T=0\) 零惩罚、\(\lambda=1\) 选最不退化、
\(k>V\) 夹取与打分并列四条专项。

## 复杂度

一次 softmax 与一次 log-softmax 各 \(O(V)\)；取 top-\(K\) 为 \(O(V\log V)\)（`torch.topk` 可降到
\(O(V+K\log K)\)）；\(\deg\) 为 \(O(KT)\)，打分与 argmax 为 \(O(K)\)。空间 \(O(V)\)（概率与 log 概率）+
\(O(KT)\)（相似度）。真实解码里每步要为每个候选各跑一次前向拿隐态，所以 \(O(K)\) 次前向才是它的实际瓶颈；
本题把这部分折叠进 \(sim\)，剩下的都是廉价张量运算。

## 实践提醒

\(\lambda\) 与 \(sim\) 的刻度绑定：本题把 \(\deg\) 落在 \([0,1]\)、原文落在 \([-1,1]\)，同一个 \(\lambda\)
数值在两处口径下的强度差 2 倍（\((1+s)/2\) 的公共平移不影响选择）。原文把候选数 \(k\) 建议为 \(3\sim10\)，
并在附录 B 专门对 \(k\) 与 \(\lambda\) 做消融（**具体的推荐取值区间 待回源**，本批只核对到正文的
"top-\(k\)，\(k\) 常取 \(3\sim10\)"与"该附录存在"两件事）；\(\lambda\) 过大会把生成推向"语义无关但很不像"
的尾部候选，表现是 gen-ppl 变差、多样性虚高。工程上还要注意：\(sim\) 需要**候选接上前缀之后**的隐态，
用"前缀最后一步的隐态"近似会让惩罚整体偏小、看起来更像贪心。
