# 详解：Squared-ReLU 注意力的行归一前向

## 出处

- Shen et al., 2023, arXiv **2309.08586**。该文把 softmax 注意力改写成点式注意力族 \(\phi=L^{-\alpha}h\)，\(h\in\{\mathrm{relu},\mathrm{relu}^2,\dots\}\)、\(\alpha\in[0,1]\)，并指出 **squared ReLU** 在视觉 Transformer 上最接近 softmax 的表现。本题取其中 \(h=\mathrm{relu}^2\) 的一支。
- QK-Norm 的原始出处是 Liu et al., *Understanding the Difficulty of Training Transformers*, 2020, arXiv **2010.04245**：在点积之前对 q、k 在 head 维上做归一化。
- 与论文的差别（有意为之）：论文用长度常数 \(L^{-\alpha}\) 做缩放，本题改用**逐行求和归一** \(\text{weights}=a/\max(\sum_j a_j,\texttt{eps})\)，这样权重与输入数据相关、行和可控，也就不存在"需要选 \(\alpha\)"的问题。归一化方式不同，但"先激活再归一"的骨架一致。
- 库内分工：`logsumexp`/`softmax` 是纯归约题，`nu-online-softmax-rescale` 讲的是 softmax 的在线数值稳定，`attention-softcap` 改的是 logits 的帽函数，`layernorm-attention-scale` 讲的是 LayerNorm 与 \(1/\sqrt{h_d}\) 的耦合。只有本题把 **softmax 整个拆掉**、换成逐元素激活 + 行归一，并叠加 QK-Norm。

## 核心与易错点

1. **先 relu 再平方，顺序不能反**。要求的是 \(\mathrm{relu}(\text{logit})^2\)；写成 \(\mathrm{relu}(\text{logit}^2)\) 等于 \(\text{logit}^2\)，负 logit 也会拿到正贡献，`check_activation_zeroing` 与 `check_not_shift_invariant` 都会失败。
2. **行归一 ≠ 论文的 \(L^{-\alpha}\)**。论文的缩放因子只依赖序列长度，是全局常数；本题的分母是数据相关的行和。别把两者混着实现（比如既做行归一又乘 \(1/L\)）。
3. **二次齐次让温度精确失效**。relu² 是二次齐次：整体放大 \(c\) 倍 logits，分子分母同乘 \(c^2\)，权重逐位不变。因此 \(\sqrt{h_d}\) 这个温度写不写结果都一样，把 q 或 k 乘任何正数也不改变权重。`check_scale_invariance` 用 7、0.05、1e3 三个因子验证这一点——这是该注意力族的**性质**，不是实现自由度。
4. **但 `eps` 会打破第 3 条**。分母用了 \(\max(\text{denom},\texttt{eps})\)，当整行 logits 极小（例如 q、k 都在 \(10^{-4}\) 量级）时真实分母小于 `eps`，下界接管，行和被等比缩小到 \(<1\)。`check_zero_and_tiny_denominator` 用一个 \(10^{-4}\) 的例子钉死这条语义。
5. **零 logit 拿严格 0 权重**。softmax 会给零 logit 分到 \(1/n\)，relu² 给的是 0；这正是该激活"稀疏、可切块"的来源。整行非正时权重全 0、输出全 0，不能写成 0/0 出现 NaN。
6. **失去平移不变性**。给所有 logit 同加一个常数，softmax 不变，relu² 会大变；`check_not_shift_invariant` 构造了"平移后原本为负的 logit 转正"的例子，并先断言该改动确实是同加常数、softmax 确实不变，再断言本题权重必须变。
7. **qk_norm 真正改变的是什么**。它按 head 维做**无权** RMSNorm \(x/\sqrt{\mathrm{mean}(x^2)+\texttt{eps}}\)，只保留方向、丢掉范数：模长 100 倍的 key 不再独占权重，方向相同的 key 拿到完全相同的分数，且归一化后 \(|\text{logit}|\le\sqrt{h_d}\)。注意 eps 在 RMSNorm 里也会起作用，所以"逐 key 缩放不变"只是近似成立（检查里用了 1e-4 的容差，而不是逐位相等）。
8. **mask 的作用时机**。True = 保留，作用在**激活之后**、求和之前：被屏蔽位置置 0，权重恒 0；整行被屏蔽等价于分母 0 → 该行全 0。mask 只需能广播到分数矩阵，不要求与 q/k 的前置维同形。
9. **前置维按广播规则对齐**，例如 q `(2, 3, L, h_d)` 配 k、v `(2, 1, S, ·)`，输出形状由广播后的前置维决定；head 维必须 q、k 相同，k 与 v 的序列维与前置维必须一致。
10. **精度口径**：内部 float64 计算，返回 float32；输入 float32/float64 都要能跑。

## 对拍口径

- `_oracle` 用 numpy 独立重写：`maximum(logits, 0) ** 2`、`np.broadcast_to` 后的 `where` 掩码、`maximum(denom, eps)` 与最后一维矩阵乘，和 torch 向量化实现没有共享代码。
- 11 个 `check_*`：样例、激活置零与行和性质、非平移不变、平方增长（1:2:3 → 1:4:9）、缩放不变、QK-Norm、零/微分母、mask 与整行屏蔽、广播批次、随机对拍（6 组配置含 mask、qk_norm、\(10^{-4}\) 小尺度，另加 float32 输入）、21 种非法输入。
- 变异测试（先平方后 relu、忽略 mask、去掉分母 eps 下界、qk_norm 漏做、把行归一换成 softmax 等）均被至少一个检查抓住；唯一存活的是"去掉 \(1/\sqrt{h_d}\) 温度"，因为二次齐次 + 行归一让它成为精确等价的变换，属于设计性质而非测试漏洞。
- 非法输入只捕获 `ValueError`：非张量、整型/布尔张量、`ndim < 2`、NaN/±inf、head 维为 0、head 维不一致、k/v 序列维不一致、前置维不可广播、mask 类型或形状错误、`qk_norm` 非 bool、`eps` 为 bool/0/负数/NaN/字符串。

## 复杂度

- 时间 \(O(\text{batch}\cdot\text{heads}\cdot L\cdot S\cdot h_d)\)：logits 一次矩阵乘、激活与行归约 \(O(L\cdot S)\)、再乘 v 一次矩阵乘。
- 空间除输入输出外，权重矩阵占 \(O(\text{batch}\cdot\text{heads}\cdot L\cdot S)\)；若按块流式归一，可降到 \(O(L\cdot h_d + S\cdot v_d)\) 量级，这正是该注意力族相对 softmax 的工程卖点之一。
