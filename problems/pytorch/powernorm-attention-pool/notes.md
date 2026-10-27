# 详解：PowerNorm 与加性注意力池化读出

## 出处

- PowerNorm：Shen et al., *PowerNorm: Rethinking Batch Normalization in Transformers*（ICML 2020, arXiv:2003.07845）。原文的系统研究把 Transformer 里 BN 的失效归因于小 batch/长序列下方差估计的剧烈抖动，并用 Taylor 展开估计中心矩；本题取其**最基础的无中心化二阶矩口径**（纯 \(x/\sqrt{\mathrm{mean}(x^2)}\)），与 starter docstring 完全一致，不涉及 Taylor 中心化变体。
- 加性注意力读出：Lee et al., *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*（arXiv:1810.00825）的 PMA（Pooling by Multi-head Attention）用 \(k\) 个 seed 向量做多路读出；本题是其**单 seed 退化**：单头、单查询、点积打分除以 \(\sqrt D\)、softmax 后对值加权求和。
- RMS≈1 性质、非中心化探针与常数输入行为以 starter docstring / test_cases.py 的判别用例为唯一口径。

## 完整推导与下标约定

**下标约定**：powernorm 输入展平为 (N,C,M)，位置 \((n,m)\) 上 \(S(n,m)=\frac1C\sum_c x[n,c,m]^2\)、\(y[n,c,m]=x[n,c,m]/\sqrt{S(n,m)+\varepsilon}\)；4-D (N,C,H,W) 时 \(M=H\cdot W\)，空间维整体摊平、**通道维是唯一统计轴**。attn-pool 输入 (B,L,D)，权重沿 L（dim=-1）归一化，输出 (B,D)。

**为什么"不减均值"是可判别的**：BatchNorm/LayerNorm 的分母是中心化后的方差 \(\mathrm{var}=\overline{x^2}-\bar x^2\)。只要通道均值 \(\bar x\neq0\)，\(\sqrt{\mathrm{var}+\varepsilon}\) 就与本题的 \(\sqrt{\overline{x^2}+\varepsilon}\) 分道扬镳。极端对照：所有通道等于同一个常数 \(c\) 时，BN 分母 \(\to\sqrt\varepsilon\)（方差为 0，输出爆成 \(\pm c/\sqrt\varepsilon\) 的巨数），而本题 \(S=c^2\)、\(y=c/|c|=\mathrm{sign}(c)\) 恰为 \(\pm1\)。判别探针抓两个可观察量：① 输出与 BatchNorm 式独立重算不 allclose；② 输出的通道均值明显非零——中心化归一化的输出通道均值**必为 0**，未中心化的**必非 0**，这是"有没有减均值"最直接的物证。

**RMS≈1 的精确刻画**：把输出代回定义，\(\mathrm{mean}_c(y^2)=\dfrac{S}{S+\varepsilon}\)，故每个 (n,m) 位置的通道 RMS \(=\sqrt{S/(S+\varepsilon)}\)。\(\varepsilon\ll S\) 时 RMS≈1（归一化生效）；\(\varepsilon\) 与 \(S\) 同量级时明显小于 1——探针取 \(S\approx4.67/1.17\) 的输入、\(\varepsilon=0.25\)，逐点对齐理论值 \(\sqrt{S/(S+\varepsilon)}\approx0.974/0.907\)。这说明 PowerNorm 的"单位尺度"是**由 eps 相对大小控制的近似性质**，与中心化归一化"严格零均值单位方差"不同。

**常数输入与零输入**：\(x\equiv c\)（\(|c|\gg\sqrt\varepsilon\)）→ \(y\equiv c/|c|=\mathrm{sign}(c)\)；\(x\equiv0\) → \(S=0\)、\(y=0/\sqrt\varepsilon=0\)。除数恒正，逐元素符号不变（sign 探针）。这一族行为都是"除以正数"的直接推论，也是 eps 只进分母的证据——eps 若加到别处，零输入就不再是 0。

**attn-pool 的两个恒等式**：

1. **均匀权重**：softmax 平移不变——logits 全等时每个权重恰为 \(1/L\)，公共打分值在减最大值时被消去，\(\mathrm{out}=\frac1L\sum_l x_l=\mathrm{mean}\)。探针构造 \(q=e_0\)、所有 \(x_{l,0}=1\)（内积恒 1）验证 out == 逐元素均值。这也说明打分只通过 softmax 的**相对差**起作用：整体平移 logits 不改变结果。
2. **L=1**：单元素 softmax 恒为 1（减最大值后 exp(0)/exp(0)=1），out 与该行**逐位相等**（torch.equal 而非 allclose）——exp 与除法都不引入任何数值误差。

**打分为什么要除 \(\sqrt D\)**：logits \(=xq/\sqrt D\) 与点积注意力的标准缩放同源——\(q\) 固定时 \(xq\) 是 \(D\) 个乘积之和，量级随 \(\sqrt D\) 增长，不缩放会把 softmax 推向饱和 one-hot。本题没有温度参数，\(1/\sqrt D\) 是唯一缩放（温度 softmax 原语另见邻近题）。

## 核心与易错点

1. **别顺手减均值**：写惯了 LN/BN 的人容易把 \(S\) 写成方差 \(\overline{x^2}-\bar x^2\)——非中心化探针专抓这个（输出通道均值非零是硬指标）。
2. **eps 只进分母**：不是加在输入上、也不是除完再加；零输入靠它兜底成 0 而不是 NaN。
3. **4-D 展平**：空间维 H·W 摊进 M，统计只沿通道维 C 算；把空间位置也算进统计轴就错了。
4. **softmax 沿 dim=-1（L 维）**：权重归一化在序列维上，不在 D 维；加权和前注意 (B,1,L)×(B,L,D) 的对齐。
5. **softmax 减最大值**：不减虽然数学等价，但大 logits 会溢出；确定性契约要求稳定式。
6. **查询 dtype 转换**：查询向量允许与输入 dtype 不同，先 cast 到输入 dtype 再点积；返回 dtype 恒同输入。
7. **"忽略"不是"校验"**：powernom 下查询向量哪怕不合法也必须被无视、attn-pool 下 eps 必须被无视——把忽略实现成校验会把合法调用判死。
8. **错误面**：非法 mode、非 float dtype、维度数不符、attn-pool 缺查询/查询非一维/长度 ≠D、eps≤0，全部 ValueError。

## 与邻近题的区别

- **vs `batchnorm-forward` / `layernorm-forward` / `rmsnorm`**：中心化归一系。BN 减 batch 统计的均值、LN 减特征维均值；RMSNorm 与本题同为"只除均方根"，但沿特征/通道维归一化且带缩放参数，本题在**每个空间位置 (n,m)** 上独立沿通道统计、无缩放参数、eps 语义只防零。
- **vs `global-pooling`**：global-pooling 是纯统计量读出（mean/max，无权重）；本题的池化权重是数据依赖的（softmax 打分）。均匀权重探针恰好给出退化关系：打分全等时 attn-pool 塌回全局平均池化。
- **vs `attention-softmax-temp`**：那边是解码 logits 上的温度 softmax 原语（温度显式可调）；本题 softmax 作用于 \(xq/\sqrt D\) 打分，无温度参数。
- **vs `multi-head-attention`**：MHA 是 token 间自注意力（Q 来自输入、输出保持序列形状）；本题是集合→单点的读出，查询是外部 seed、输出长度 1。

## 对拍口径与复杂度

- 独立重算 `_oracle_pn`：纯 Python 逐 (n,m) 位置显式累加通道平方、显式除法，全程 float64 后 cast 回——与解法的向量化 mean/pow 路径完全不同；另有分开实现的 eps 可调版，避免共享出错面。attn-pool 的 `_oracle_attn` 手写减最大值 softmax + 显式加权和，覆盖 float32 三组与 float64 一组（1e-12）。
- 判别探针：非中心化（与 BN 式显著不同且输出通道均值非零）、RMS=\(\sqrt{S/(S+\varepsilon)}\) 理论对齐、常数 →sign(c)、零 →0、保号、均匀权重 →mean、L=1 逐位相等，外加 10 个 ValueError 错误面。
- 复杂度：powernorm 时间 \(O(N\cdot C\cdot M)\)（平方求和与除法各一遍）、中间张量同量级；attn-pool 时间 \(O(B\cdot L\cdot D)\)（打分点积 + softmax + 加权和各一遍）、额外空间 \(O(B\cdot L)\)（权重）。
