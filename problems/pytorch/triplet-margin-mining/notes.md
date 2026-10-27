# 详解：Triplet Margin 损失与在线难例挖掘

## 出处

- triplet margin loss：FaceNet，Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*（CVPR 2015, arXiv:1503.03832）。原文给出 batch_all / batch_hard / semihard 三种挖掘与"选一个难负例在线训练"的方案。
- batch-hard 的系统论证：Hermans, Baeza-Yates, Leibe, *In Defense of the Triplet Loss for Person Re-Identification*（arXiv:1703.07737），说明 batch_hard 与随机三元组在期望上等价、但方差小得多。
- `swap` 是 PyTorch 的扩展位（论文里没有）：本题公式 **不采信文档措辞**，改为对 `F.triplet_margin_loss` 逐三元组实测（p∈{1,2} × 16 条随机三元组 × swap 两态 + 一个判别性单三元组），定稿为 \(\operatorname{relu}(\text{margin}+d_{ap}-\min(d_{an},d_{pn}))\)。
- 本题与 softmax 对比损失（`mm-clip-infonce` 一类）无关：那是"一正多负 + logit 归一化"，本题是显式三元组 + hinge。

## 完整推导与下标约定

**下标约定**：anchor \(a\)、正例 \(p_a\)、负例 \(n_a\) 都是**同一个 batch 内的样本下标**；\(d_{ap}=D_{a,p_a}\)、\(d_{an}=D_{a,n_a}\)、\(d_{pn}=D_{p_a,n_a}\)。距离矩阵 \(D\in\mathbb R^{N\times N}\) 一次算好，逐对复用。

**\(\ell=\operatorname{relu}(m+d_{ap}-d_{an})\) 的单调性是一切挖掘策略的根**：它对 \(d_{ap}\) 单调不减、对 \(d_{an}\) 单调不减地"反向"（即随 \(d_{an}\) 增大而减小）。因此固定 anchor 时，
\[\max_{p\in\mathcal P_a,\,n\in\mathcal N_a}\ \ell(a,p,n)=\operatorname{relu}(m+\max_p d_{ap}-\min_n d_{an}),\]
**最远正例 + 最近负例**同时取到最大值——这正是 batch_hard 的定义：它不需要枚举三元组，只用两次 argmax/argmin 就拿到该 anchor 的 loss 上确界，成本 \(O(N)\)。

**semihard 的开区间就是"损失严格介于 0 与 margin 之间"的负例集合**：
\[d_{ap}<d_{an}<d_{ap}+m\iff 0<m+d_{ap}-d_{an}<m\iff 0<\ell<m .\]
左边界之外（\(d_{an}\le d_{ap}\)）是 batch_hard 会挑的"极难负例"——在带噪标注下它常常是误标（人脸里是同一人的极端姿态），梯度把它拽远反而损害收敛；右边界之外（\(d_{an}\ge d_{ap}+m\)）损失恒 0、无梯度。于是 semihard 只在"活着但不过火"的带内取最近者；带内为空时本题**规定回退到全局最难**（FaceNet 原文未规定这一支，是题面口径，必须照做才能对上判题）。

**两个策略在 margin=0 时重合**：区间 \((d_{ap},d_{ap})\) 恒空 → semihard 每步都走回退 → 输出与 batch_hard 逐 anchor 相同。这是可从式子直接读出的性质（判题的 margin=0 用例正靠它兜底）。

**swap 的几何意义与"等价于取较大者"**：把 \(-\min(d_{an},d_{pn})\) 写成 \(\max(-d_{an},-d_{pn})\)，并用 relu 单调不减，得
\[\operatorname{relu}(m+d_{ap}-\min(d_{an},d_{pn}))=\max\big(\underbrace{\operatorname{relu}(m+d_{ap}-d_{an})}_{\text{原三元组}},\ \underbrace{\operatorname{relu}(m+d_{ap}-d_{pn})}_{\text{把 }a\leftrightarrow p\text{ 交换后的三元组}}\big).\]
第二项正是"交换 anchor 与 positive"后的损失（新 anchor 是 \(p\)，\(d_{ap}'=d_{pa}=d_{ap}\)、\(d_{an}'=d_{pn}\)）。所以 swap **恒 ≥ 非 swap**：它要求 anchor 和它的正例**都**离负例足够远，只要有一个被贴住就构成违约。判别性用例 \(a=(0,0),p=(3,0),n=(3,1)\)：\(d_{ap}=3,\ d_{an}=\sqrt{10}=3.1623,\ d_{pn}=1\) → 非 swap 得 \(\operatorname{relu}(1+3-3.1623)=0.8377\)（已接近满足），swap 得 \(\operatorname{relu}(1+3-1)=3\)（正例被负例贴死）。若把 min 误加在 \(d_{ap}\) 一侧会得到 0，与实测的 3 相差 3——判题用一条 `>1.0` 的断言把这个方向错误钉死。

**并列取最小下标的实现**：把非候选位置填 \(-\infty\)（argmax 侧）或 \(+\infty\)（argmin 侧）后，torch 的 argmax/argmin 在并列时返回**首个**极值位置，即最小下标。纯 Python oracle 用"从左到右扫描 + 严格比较"实现同一语义；判题另有 `tie_last=True` 的反向 oracle，专门断言"取错下标会得到明显不同的损失"——三组用例（负例并列、正例并列、semihard 区间内并列）各自把 \(d_{pn}\) 设计成不同，使 swap 分支也能暴露差异。

**跳过与分母**：正例池空（该类只有自己）、负例池空（全部同标签）时该 anchor **不产生项也不进分母**，均值是「有效 anchor 数」而非 N。margin=0 用例 \(x=[[0],[10],[2]],\ y=[0,0,1]\)：a0 → \(\operatorname{relu}(0+10-2)=8\)，a1 → \(\operatorname{relu}(0+10-8)=2\)，a2 正例池空被跳过 → 均值 \((8+2)/2=5\)，若错误地除以 3 得 3.33。

## 核心与易错点

1. **swap 的 min 在负例一侧**：`min(d_an, d_pn)`，不是 `min(d_ap, d_pn)`；方向写反结果可以是 0 而非 3。
2. **semihard 区间是开区间**：\(d_{an}=d_{ap}\) 与 \(d_{an}=d_{ap}+m\) 都不入选；带内取**最小**距离（不是最大）。
3. **带内为空要回退**，回退到全局最难（与 batch_hard 同取法），不是"该 anchor 跳过"。
4. **正例取最远、负例取最近**：两个方向都是"最难"，写成相反即 batch_easy，判题立刻红。
5. **分母是有效 anchor 数**，跳过的不进分母；N=0/全同类/全异类/N=1 都返回 0。
6. **\(p\) 范数按定义算**：\(\sum|\Delta|^p\) 再 \(1/p\) 次方；\(p=1\) 时这条路径精确，走欧氏专用算子会把曼哈顿距离算错。
7. **自身永远不是正例**：正例池显式排除 \(j=a\)；负例池靠"标签不同"自动排除自身，若标签数组有 nan/浮点标签则不成立（本题 labels 是 long）。
8. **返回 0 维标量张量**：`torch.stack(losses).mean()`，不能返回 python float 或 1 维张量；无有效 anchor 时要用与输入同 dtype 的 `zeros(())`。
9. **错误面**：strategy 非 batch_hard/semihard（含 "random"/"batch_all"）抛 ValueError。

## 与邻近题的区别

- **vs `embedding-margin-losses`（本批次）**：`nn.HingeEmbeddingLoss`/`nn.CosineEmbeddingLoss` 是**成对**损失（给定 pair + 1/−1 标签），不需要挖掘、没有 anchor 概念；本题是**三元组**损失且挖掘规则就是题目本身。
- **vs `mm-clip-infonce` / 各类对比损失**：InfoNCE 是"1 正 \(K\) 负 + softmax over logits"，负例全部进分母；triplet 只挑**一个**负例进 hinge。二者的"难度"概念也不同：InfoNCE 的难例由 softmax 自动加权，triplet 需要显式挖掘规则。
- **vs `euclidean-distance` / `pairwise-distance` 类**：那类只算距离；本题的距离只是中间量，判题重点是挖掘选择与 hinge。
- **vs autograd 合集（todos #65）**：这里只给标量损失值，不求 \(\partial\ell/\partial x\)。

## 对拍口径与复杂度

- oracle `_oracle_solve` 是**双重 Python 循环**：逐 anchor 构造 pos/neg 列表、逐维 `abs(u-v)**p` 累加再 `**(1/p)`、显式比较取极值——与解法的张量矩阵 + argmax/argmin 完全不同的实现路径；`tie_last` 参数让同一份 oracle 能产出"错误 tie-break"的对照组。
- 与原生库的交叉校验只在 **swap 公式**这一处使用（`F.triplet_margin_loss` 的 4 组配置 × 16 条三元组 = 64 个 per-triplet 值，外加一个端到端三样本 batch），因为挖掘、跳过、回退三处 `F` 里没有对应物；这种"只在可对拍处对拍"的划分避免了用库函数替代题面语义。
- 判别性用例：semihard 触发/回退对照（同一 batch 下 `sh < bh` 且带内无解时 `sh == bh`）、三组 tie-break 探针（各配一组"取错下标必须不同"的断言）、margin=0 的非零与全零两侧、四类退化输入、非法 strategy。
- 整数网格随机数据（`randint(0,5)`）专用于避免浮点近并列导致 tie-break 假性不稳定；另有两组连续 `randn` 数据覆盖浮点路径。容差 \(10^{-5}\)，示例常量实测回填（可手工复核：\(\sqrt8,\sqrt{13},\sqrt5\) 三个距离推出 0.7928932309 与 0.5986121893）。
- 复杂度：距离矩阵 \(O(N^2D)\) 时间、\(O(N^2)\) 内存；挖掘循环 \(O(N^2)\)（每 anchor 两次带掩蔽的 arg）。batch_all 是 \(O(N^3)\) 级枚举，本题两种策略把它降到 \(O(N^2)\)——这正是它们存在的理由。
