# 详解：Cosine / Hinge 两种嵌入 margin 损失

## 出处

- 判题口径的唯一来源是 starter docstring，以及 `F.cosine_embedding_loss` / `F.hinge_embedding_loss` 的**实测行为**（本题对拍断言 atol \(10^{-6}\)）。
- 学术渊源：cosine embedding loss 常追到 Jia et al., *Large-scale Video Classification with Convolutional Neural Networks*（CVPR 2014，为处理"缺标/多标"而用夹角而非内积）；hinge embedding 的 \(\max(0,m-x)\) 即软间隔 SVM 的 hinge（Cortes & Vapnik, *Support-Vector Networks*, Machine Learning 1995——期刊文献，无 arXiv 编号）在孪生网络上的成对形式。
- **两处必须回源、不能信文档措辞的地方**（均已实测，torch 2.14.1）：
  1. `nn.CosineEmbeddingLoss` 的官方 docstring 把 target=−1 支写成 \(\max(0,\text{margin}-\cos)\)，实现却是 \(\max(0,\cos-\text{margin})\)。实测 x 与自身（\(\cos=1\)）、target=−1：margin=0 → **1.0**、margin=0.5 → **0.5**、margin=2 → **0.0**，只有后者能解释这组数。本题按实现口径出题。
  2. `F.cosine_similarity` 与 `F.cosine_embedding_loss` 的 eps 兜底**彼此并不一致**（见下文），所以题面把规则写死成"乘积整体 `clamp_min(1e-8)`"，判题只在正常量级数据上与 F 对拍。

## 完整推导与下标约定

**下标约定**：\(i\) 是第 \(i\) 个样本对；\(\cos_i\) 是它的余弦；\(\text{target}_i\in\{1,-1\}\)；两分支都是**逐元素**的，reduction 只在最后做一次。

**cosine 的两支语义**：target=1 支 \(1-\cos\) 在 \(\cos=1\) 处为 0、\(\cos=-1\) 处为 **2**（上界 2，不是 1）；target=−1 支 \(\max(0,\cos-m)\) 是"允许 \(\cos\) 涨到 \(m\) 才开始罚"的单侧 hinge。判题的反向平行用例把这点摆在一起：三对 \(\cos=[-1,1,-1]\)、target \(=[1,-1,-1]\)、\(m=0\) → \([2,1,0]\)：第一对是"标成同类却反向"，吃到最大惩罚 2；第三对"标成异类且反向"，损失恰为 0（已经满足）。这组数同时验证了"target=1 用 \(1-\cos\)、target=−1 用 \(\cos-m\)"两个方向都没写反。

**为什么用夹角不用内积**：内积 \(\langle x_1,x2\rangle=\lVert x_1\rVert\lVert x_2\rVert\cos\theta\) 会把"向量长度"也当成相似度的一部分；cosine embedding 把长度除掉，损失只由方向决定。孪生网络里这带来一个真实收益——同一对样本不会因为特征整体被放大而误判为"更像"。（代价是梯度里多了一个正交分量，本题不涉及反向。）

**分母 clamp 的三种写法与可观测差异**：令 \(w_i=\lVert x_i\rVert_2\)。
① 乘积整体 clamp（本题口径）：\(\cos=\text{dot}/\max(w_1w_2,10^{-8})\)；
② 模长各自 clamp：\(\cos=\text{dot}/(\max(w_1,10^{-8})\max(w_2,10^{-8}))\)；
③ 严格大于 0 才除、否则记 0。
取 \(w_1=w_2=10^{-5}\)（两向量同向，dot \(=10^{-10}\)）：② 的 \(10^{-5}>10^{-8}\) 不触发 → \(\cos=1.0\)；① 的分母被兜到 \(10^{-8}\) → \(\cos=0.01\)。实测 `F.cosine_similarity` 给 **1.0**（即写法②），而 `F.cosine_embedding_loss` 给的角度等价 **0.9901**——连原生两个函数在这一区间都不同。判题只考 \(O(1)\) 量级数据 + 零向量：零向量下 ①②③ 都归约到"\(\cos=0\)"（分子为 0、分母有限），所以本题的兜底规则**只在零向量处可观察**，且必须给出 \(\ell(\text{target}=1)=1\)、双零向量 target=−1 → 0（`isfinite` 断言）。

**hinge 两支的形状**：target=1 支就是 \(x\) 本身，**不套 relu**，所以损失可以负、且随 \(x\) 增大无下界——判题用 `torch.equal(got, scores)` 精确钉死，并用实测确认 `F.hinge_embedding_loss(x=[-1], target=[1]) = -1.0`。这不是笔误而是 `nn.HingeEmbeddingLoss` 的定义：正对希望分数**越大越好**，所以直接把分数当损失（最小化即最大化分数）。target=−1 支 \(\max(0,m-x)\) 是标准 hinge：分数低于 \(m\) 即满足，越低不再罚。
两支在 \(x=m\) 处连续吗？target=−1 支在 \(x\ge m\) 时为 0，而 target=1 支在该点值为 \(m>0\)——两支作用在**互斥的样本子集**上（按 target 取值分派），不存在同一个样本跨越的问题，因此没有"必须衔接"的约束。

**margin 默认值分派**：`margin=None` 在两个 mode 下落到不同常数（0.0 / 1.0），这**不是**"默认值统一为 None→0"，必须按 mode 分支各取默认；与原生签名 `cosine_embedding_loss(margin=0)` / `hinge_embedding_loss(margin=1)` 一致。

**空集**：原生对空张量求 mean 得 NaN（实测 `F.hinge_embedding_loss(zeros(0), zeros(0)) = nan`），本题约定返回标量 0；sum 与 none 本来就直接是 0 / (0,)。这是"题面覆盖原生"的一处显式差异，必须写进 notes 以免用 F 对拍时把 NaN 当 bug。

## 核心与易错点

1. **target=−1 的 cosine 支方向**：\(\max(0,\cos-m)\)，不是 \(\max(0,m-\cos)\)；判题的 margin 序列 (0, 0.5, 2) 对同一 \(\cos=1\) 给 (1, 0.5, 0)，写反则恒为 0。
2. **hinge 的正对支是恒等**，允许负损失；写成 `relu(x)` 或 `1-x` 立刻挂在 `torch.equal` 探针上。
3. **clamp 作用在乘积上**（本题唯一口径）；先各自 clamp 再相乘在极端量级下差两个数量级。
4. **零向量不是错误**：不抛异常、不返回 NaN，target=1 时损失恰为 1。
5. **两个 mode 的 margin 默认值不同**；用同一个默认（例如都 1.0）会让 cosine 分支在 margin 上对不上原生。
6. **hinge 忽略 x2**：x2 传什么都不得报错（"忽略 ≠ 校验"），但 cosine 下 x2=None 必须 ValueError——两个 mode 对同一个参数一个是"忽略"一个是"必填"。
7. **空集 mean 返回 0 而不是 NaN**，且必须是 0 维张量。
8. **reduction 三态都要实现**：`"sum"` 在本题是合法值（与 `iou-variant-losses` 那题相反，那边 "sum" 要抛错）；别把两批的约定串味。
9. **错误面**：mode ∉ {cosine,hinge}、reduction ∉ {mean,sum,none}、cosine 且 x2=None → ValueError。

## 与邻近题的区别

- **vs `cosine-similarity`**：那题只算 \(F\).cosine_similarity 原语（沿任意 dim、eps 语义）；本题在它之上做**带 target 分派的两支损失**与归约，且分母规则独立成题面口径。
- **vs `triplet-margin-mining`（本批次）**：三元组损失需要"从 batch 里挖负例"，输入是 embeddings+labels 且要处理池空/回退/并列；本题输入已经**是给定的样本对**，没有挖掘、没有 anchor 概念，target 只有 \(\pm1\)。
- **vs `contrastive-sampling`**：那题考的是"如何从标签采样成对/三元组数据"（数据侧），本题是损失本身（算子侧）。
- **vs `mm-clip-infonce`**：InfoNCE 用 softmax 把多负例归一化进分母，是"多对一"结构；cosine embedding 是逐对独立的两支 hinge，无归一化竞争。

## 对拍口径与复杂度

- 主对拍是**原生库交叉校验**：`F.cosine_embedding_loss` 覆盖 margin ∈ {0.0,0.2,0.7} × reduction ∈ {none,mean,sum}（16×7 随机向量），`F.hinge_embedding_loss` 覆盖 margin ∈ {1.0,0.5} × 三种 reduction（20 维随机分数）。这两个式子在 \(O(1)\) 量级数据上与本题口径逐元素相同，因此不需要另写 python oracle——但**空集 mean** 与**极小分母**两处不能对拍（前者原生给 NaN、后者原生自身不一致），改由题面约定 + 显式断言覆盖。
- 判别性用例：反向平行三样本（同时钉住两支方向与 target=1 的上界 2）、target=1 且 cos<0 → 损失 >1、margin>0 的截断（cos=1、m=0.5 → 0.5）、全 −1 / 全 1 的 hinge 分支（后者用 `torch.equal` 断言输出与输入**逐位相同**）、零向量与双零向量的有限性、\(N=0\) 三态、四类 ValueError。
- 示例常量实测回填且可纸面复核：两对 cosine 的 \(\cos\) 都恰为 \(1/\sqrt{10}\)（点积均为 1、模长积均为 \(\sqrt{10}\)），故两支损失相加为 1.0——这是刻意构造的自检关系。
- 复杂度：两分支都是 \(O(ND)\)（点积与两次范数各一遍）／\(O(N)\)（hinge），归约 \(O(N)\)；内存除中间张量外无额外开销。判题不涉及批量维度的平方级开销，是纯粹的逐元素算子。
