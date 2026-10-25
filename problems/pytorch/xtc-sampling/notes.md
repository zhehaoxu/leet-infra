# 详解：XTC 中间概率剔除（Exclude Top Choices）

## 出处

- **没有论文**。XTC 不是一篇学术工作提出的算法，而是推理引擎里"把截断刀口反过来用"的实现技巧。
  本题口径直接对齐 **ggerganov/llama.cpp** 的 `src/llama-sampler.cpp`：采样器结构体 `llama_sampler_xtc`
  持有 `probability / threshold / min_keep / seed` 四个字段，核心逻辑在 `llama_sample_xtc_apply`，
  由 `llama_sampler_init_xtc(p, t, min_keep, seed)` 注册；参数从 `common/sampling.cpp` 里的
  `xtc_probability / xtc_threshold / min_keep` 装配进来（该文件把 `llama_sampler_init_xtc` 排进采样链）。
  以上函数名、字段名与判断顺序本批**已逐行核对到 master 分支源码**，不是转述。原文语义：
  ① `probability <= 0.0f || threshold > 0.5f || cur_p->size < 2` 三者任一命中就 `return`；
  ② 抽一个 `uniform_real(0,1)` 的 `chance`，`chance > probability` 则 `return`；
  ③ 先对候选数组做 softmax 并降序整理，再 `for i: if p[i] >= threshold → pos_last = i else break`
  （`pos_last` 初值 0）；④ `if (size - pos_last >= min_keep && pos_last > 0) { data += pos_last; size -= pos_last; }`。
- 另一个常被引用的实现是 **oobabooga/text-generation-webui** 的 logit filter（`xtc_filter`，同样带
  `xtc_threshold` / `xtc_min_tokens` 两个参数）。**其当前文件路径与逐行判据本批未能从判题机取到 upstream
  源码（`raw.githubusercontent.com` 该仓库连续超时），列为 待回源**；本题不引用它的任何细节。
- **与题名、与合集条目的范围差异（必须明说）**。本合集条目把 XTC 描述成"只在 \(p>10^{-35}\) 的 token 上算
  \(\mu,\sigma\)，再剔除概率落在 \(\mu/\sigma\) 构成的**闭区间带**内的 token"。本批核对到的 llama.cpp 实现
  **全程不使用任何 \(\mu/\sigma\) 统计量**，也没有 \(10^{-35}\) 这样的概率下限（它比较的是一个固定的绝对概率
  阈值 \(t\)）；那种"μ/σ 中段带剔除"的写法在 arXiv 与 upstream 源码里都找不到出处，**待回源**，且与本题实现
  无关——它更接近 `top-n-sigma-sampling` 那族统计截断的**反向**用法。题名里的"中间概率剔除"因此应读作：
  剔除过阈值的**头部连续前缀**（保留前缀最后一个）之后，概率质量由**中段与尾部**候选重新占据；
  而被真正删掉的是**高概率头部**，这正是上游名字 *Exclude Top Choices* 的字面含义。判题以
  `starter.py` / `solution.py` 的口径为唯一准绳。
- 判题环境里同样无可借之物：`.venv-eval` 的 `transformers` 5.20.2 没有任何 XTC 相关的
  `LogitsProcessor`/sampler 类，llama.cpp 不存在，必须手写前缀计数与散射掩码。

## 完整推导与下标约定

一行输入 \(z\in\mathbb{R}^{V}\)，\(p=\operatorname{softmax}(z)\)，参数 \(\pi\)（触发概率）、\(t\)（阈值）、
\(m\)（`min_keep`），显式随机数 \(u_{\text{gate}},u\in[0,1)\)。**"秩"与"索引"两套下标必须分清**：
秩是降序数组里的位置，索引是原词表里的位置。

1. **早退**是全局开关：\(\pi\le 0\ \vee\ t>0.5\ \vee\ V<2\) ⇒ 本行 `keep ≡ 1`、`applied = False`。
   \(t>0.5\) 这条不是随便写的：概率和为 1，两个候选同时 \(\ge t>0.5\) 不可能，于是达标集至多一个元素、
   剔除集必为空——上游索性整块关掉。\(V<2\) 时"保留前缀最后一个、剔除其余"没有意义。
2. **降序秩**。按 \((-\,p_i,\ i)\) 升序排列得到 `order`，秩 \(r\) 的原索引是 \(\text{order}_r\)。
   并列必须用索引升序固定下来，否则 `keep` 掩码在平局处不确定。
3. **达标前缀**。令 \(a_r=p_{\text{order}_r}\)，则 \(a_0\ge a_1\ge\dots\ge a_{V-1}\)，于是
   \[
   \{r: a_r\ge t\}\ \text{是一个从 }0\text{ 开始的连续前缀},\qquad
   cnt=\#\{r: a_r\ge t\},\quad \text{pos\_last}=cnt-1 .
   \]
   连续性来自单调性（第一个 \(a_r<t\) 之后所有 \(a\) 都更小），这正是"数前缀长度"与上游
   "遇到第一个 \(a_r<t\) 就 break"等价的理由。反过来，若把判据写成 logit 域（如 `top-n-sigma`），
   单调性同样成立，但阈值含义完全不同——本题的 \(t\) 是**概率**。
   边界值：\(cnt=0\Rightarrow\text{pos\_last}=-1\)（连 \(a_0\) 都不达标）；\(cnt=1\Rightarrow\text{pos\_last}=0\)。
4. **触发与剔除**。剔除集是秩区间 \([0,\text{pos\_last})\)（**左闭右开**，`pos_last` 本身保留），
   因此被删掉的秩数恰为 \(\max(\text{pos\_last},0)\)，存活数 \(=V-\text{pos\_last}\)。三个条件
   \[
   u_{\text{gate}}\le\pi,\qquad \text{pos\_last}>0,\qquad V-\text{pos\_last}\ge m
   \]
   全成立才触发，`applied` 记 `True`。注意 \(\text{pos\_last}>0\) 这条：它把"只有一个头部达标"排除掉，
   使得 XTC **永远不会把 argmax 单独留下**（示例里 \(z=[9,-1,-2,-3]\)、\(t=0.5\) 就卡在这条上）。
   实现上 `keep = 1 − [(rank < pos_last) ∧ fired]`，用 `scatter` 把 `order` 位置的掩码写回原索引序。
5. **归一化**。
   \[
   \text{filtered}_i=\frac{p_i\cdot\text{keep}_i}{\sum_j p_j\,\text{keep}_j},
   \]
   分母是**存活概率之和**（示例里 \(0.216040\)），顺序保持原词表序。因为触发时 \(\text{pos\_last}\le V-1\)
   至少留 1 个候选，分母恒正。
6. **逆变换采样**。
   \[
   \text{chosen}=\min\{i:\ \text{CDF}_i\ge u\},\quad \text{CDF}_i=\sum_{j\le i}\text{filtered}_j ,
   \]
   CDF 用**原顺序**（不是降序）——这一点与同合集其它采样题一致。若 \(\forall i,\ \text{CDF}_i<u\)
   （\(u\) 逼近 1 的浮点边角）取最后一个下标。\(u=0\) 恒返回 \(0\)，哪怕 \(\text{filtered}_0=0\)：
   这是本题固定的约定，实测示例行 \(u=0\) 给出 `chosen = 0`（一个已被剔除的 token）。
7. **数值自检**（示例行，全部由参考解/环境算出）。\(p=[0.487983,0.295977,0.179519,0.024295,0.008938,0.003288]\)，
   其降序 CDF 为 \([0.487983,0.783960,0.963479,0.987774,0.996712,1.000000]\)；
   \(t=0.05\) 时达标数 3、\(\text{pos\_last}=2\)、剔除秩 0/1、存活质量 \(0.216040\)、
   `filtered` 存活部分 \([0.830953,0.112457,0.041371,0.015219]\)。顺带可见 top-p 式截断
   （核 \(=0.95\)）保留的秩集合 \(\{0,1,2\}\) 与本题剔除的秩集合 \(\{0,1\}\) 只差一个边界候选——
   XTC 与核采样用的是**同一个前缀概念，方向相反**。

## 核心与易错点

1. **方向**：这是本题真正的考点。绝大多数截断采样砍**尾部**，XTC 砍**头部**；`check_cuts_head_not_tail`
   就是专门抓"把剔除写成保留"的实现——断言 `filtered[0] == 0` 且尾部的 `filtered[3]`、`filtered[5]` 仍为正。
   写成 `rank <= pos_last`（把边界候选也删掉）或 `rank < pos_last` 用在**保留**掩码上，都会在这一条上崩。
2. **右开区间 / 保留边界者**。上游搬指针是 `data += pos_last`，即秩 \(\text{pos\_last}\) **留在**候选里。
   所以存活数是 \(V-\text{pos\_last}\) 而不是 \(V-\text{pos\_last}-1\)；`min_keep` 检查的是**剔除之后**的
   存活数，不是剔除之前的候选数。
3. **`pos_last > 0` 与 `cnt` 的初值**。上游 `pos_last` 从 0 起（第一个元素不达标时仍为 0，不砍），
   本题从 \(cnt-1\) 起（该情形给 \(-1\)，同样不砍）。两种写法结果一致，但**不能**把"只有一个达标"
   （\(\text{pos\_last}=0\)）当成"砍掉 0 个、但 applied=True"——`applied` 必须为 `False`。
4. **\(\pi\) 与 \(t\) 的角色别混**。\(t\) 决定砍到哪，\(\pi\) 决定这一步砍不砍。两个显式随机数各管各的：
   \(u_{\text{gate}}\) 只进门、绝不复用作采样 \(u\)（否则同一行会强相关，破坏"多数时候不触发"的语义）。
   \(\pi=0\) 是**完全关闭**（早退），\(\pi=1\) 是**必然过门**（此时只剩 \(pos\_last\) 与 `min_keep` 两条）。
5. **门不开/早退时 `filtered` 也必须重归一化后的结果 = 原 softmax**。因为 keep 全 1、分母 = \(\sum p=1\)。
   写成"早退就直接返回未 softmax 的 logits"是常见错误；`chosen` 在早退时**照样要出**（用同一套 CDF 逆变换）。
6. **平局要稳定**。\(V=2\)、\(t=0.5\)、\(p=[0.5,0.5]\) 这一行两个候选都达标 ⇒ \(\text{pos\_last}=1\) ⇒
   剔除秩 0 ⇒ `filtered=[0,1]`、`chosen=1`：**argmax 被砍掉且这一步变成确定性选择**。
   若并列破法与参考解不同，这类全等概率行会给出不同的秩区间。
7. **不要混用 logit 与概率**。\(t\) 比的是 \(p\)（softmax 之后）。把判据写成 `logit >= t` 或
   `log_softmax >= t` 都会在高温/低温下跑出完全不同的剔除集。
8. **批量与 reshape**。`applied` / `chosen` 的形状是 `(...,)`，把 `u_gate` 展平后要用 `reshape(shape)`
   送回行维度；`(..., V)` 里 `V` 只是最后一维，早退条件 \(V<2\) 沿该维判断，与 batch 维无关。

## 与邻近题的区别

- **vs `top-n-sigma-sampling`**（最容易被弄混的一对，也是本题的核心对照）。同一族"自适应截断"的思路，
  但**判据与方向都相反**：`top-n-sigma` 在 **logit 域**用整行的 \(\mu,\sigma\) 造出上侧带
  \([\max-n\sigma,\ +\infty)\)，**选留**统计上靠上侧的候选、把长尾清掉（argmax 恒在保留集内）；
  本题**不用任何 \(\mu/\sigma\)**，在**概率域**用固定阈值 \(t\) 找出过阈值的最头几个候选并**剔除**它们
  （保留边界那一个），头部被清空、尾部整条留下。一句话：`top-n-sigma` 是"留上侧带"，XTC 是"砍上侧带"。
  另注：合集条目把 XTC 写成"μ/σ 闭区间带剔除中段"，本批已在 出处 一节说明它不对应任何核对到的实现。
- **vs `min-p-sampling` / `top-p-sampling`**：那两者都以最高概率为参照**砍尾**——`min-p` 保留
  \(p\ge p\cdot p_{\max}\) 的地板以上部分，`top-p` 保留累积质量到 \(p\) 的前缀、之后全删；
  它们的共同点是**头部一定活着**（`min-p` 连 argmax 自己都在保留集内）。本题恰好相反：头部是**唯一**
  被动刀的对象，尾部一个都不删。同一份示例数据下，`top-p(0.95)` 留秩 \(\{0,1,2\}\)、本题砍秩 \(\{0,1\}\) 留 \(\{2,3,4,5\}\)。
- **vs `top-k-sampling`**：那里的 \(k\) 是"保留几个"（名次口径），且保留集内归一化后抽签；
  本题的 \(t\) 是"概率高到多少算离谱"（绝对概率口径），且**触发本身是随机的**（\(\pi\) 门），
  还会因 `min_keep` 整步放弃——`top-k` 没有这两个开关。
- **vs `typical-sampling`**：典型性采样按 \(|-\log p-H|\) 从**中心**向两侧扩张地保留"不太意外也不太典型"的候选，
  可能丢掉 argmax，但丢掉的是**意外度极小**的一批；XTC 丢掉的是**概率极大**的一批，与熵无关。
- **vs `penalty-sampling` / `dry-sampling` / `contrastive-sampling`**：那一族靠**历史**抑制重复
  （n-gram 计数 / Z-algorithm / 嵌入相似度），XTC 与历史**完全无关**，只看当前这一步的概率形状；
  `contrastive-sampling` 甚至不删任何候选。同为"反重复"，XTC 是靠"别老让最确定的那几个赢"来实现的。
- **vs 组合流水线（如 `llama-sampling-pipeline`）**：上游默认链路把 xtc 排在 `top-k / top-p / top-n-sigma /
  min-p` 之后、`typical-p / temperature` 之前（`common/sampling.cpp` 的装配 switch 即按这一枚举顺序列出）；
  而 `llama_sample_xtc_apply` 内部**自己**先做一次 `llama_sampler_softmax_impl(cur_p, true)`（算概率并整理成降序）
  再扫前缀，且它**不重新归一化**（归一留给链路后面的采样器）、也**不选 token**。本题把这三件事合并成一步：
  显式 softmax、存活集归一化后按原词表顺序返回 `filtered`、并用显式 \(u\) 直接给出 `chosen`，
  同时对 \(V\)、\(t\)、\(\pi\)、`min_keep` 的独立影响逐条断言。

## 对拍口径

参考解：`softmax` + `argsort(descending=True, stable=True)` + `(sorted_p >= t).sum(-1)` 得 \(cnt\)，
`scatter_` 写回 `keep` 掩码，向量化逆变换用 `cumsum` 与 `bool.argmax(-1)`（外加 `any` 兜底取末位）。
判题 `_oracle` 完全另起一路：逐行把概率转成 Python list、用 `sorted(range(V), key=lambda i: (-p[i], i))`
拿降序、**for-else break** 手扫连续前缀计数、按索引置零 `keep`、手工累加求和归一、再用
`acc += x; if acc >= uv: break` 的循环做逆变换（并把 `u_gate`/`u` 的元素数不足时按末位复用）。
用例覆盖 30 组 \(\text{logits}\in\mathbb{R}^{5\times12}\) × \(\pi\in\{0,0.5,1\}\) ×
\(t\in\{0.01,0.05,0.3\}\) × \(m\in\{0,3,8\}\) 的三输出全量对拍，另有门不开恒等、参数关闭（含 \(V=1\)）、
单头部达标不砍、`min_keep` 卡住、砍头不砍尾五条专项。

## 复杂度

每行一次 softmax \(O(V)\)、一次降序排序 \(O(V\log V)\)（只需前缀时可用选择法降到 \(O(V\cdot cnt)\)）、
前缀计数 \(O(cnt)\le O(V)\)、掩码与归一化 \(O(V)\)、逆变换 CDF \(O(V)\)。空间 \(O(V)\) 存 `order`/`keep`/`cdf`。
上游 `min_keep` 与 `probability` 都是常数时间判断，因此 XTC 在整个采样链里属于很便宜的一环。

## 实践提醒

\(t\) 的刻度很敏感：常用值在 \(0.05\sim0.2\)。\(t\) 太接近 0.5 时几乎只有"两个头部并列极高"的罕见步才触发；
\(t\) 太小（如 \(10^{-3}\)）则达标前缀会涵盖几十个秩、剔除过头，输出明显变"降一档"。
\(\pi\) 是实际的强度旋钮——\(\pi=1\) 会让每一步都动刀，重复抑制变成稳定的"次优偏好"；
\(\pi\approx0.3\) 更接近上游的用法。`min_keep` 建议显式给到 \(\ge 2\)：本批实测的 \(V=2,t=0.5\) 那一行说明
`min_keep=0` 允许 XTC 把分布压成单点，从而在低温小词表场景里意外杀死随机性。
最后：XTC 与 `top-p`/`min-p` 搭配时顺序很关键（上游是先截尾、再 XTC 砍头），本题把它拆成孤立算子，
所以"存活数" \(V-\text{pos\_last}\) 用的是**全词表**行数而不是截断后的候选数。
