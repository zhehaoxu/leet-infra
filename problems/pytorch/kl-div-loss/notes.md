# 详解：kl_div 损失（log-prob 输入与 batchmean 口径）

## 出处

`F.kl_div` 的接口形状直接来自蒸馏流水线：Hinton 等 2015（1506.04420）里学生端算的是 `log_softmax`，教师端给概率，若函数收 $q$ 而不是 $\log q$，就必须在内部对极小的 $q$ 再取一次对数（float32 下 $q<10^{-45}$ 已经是 0）。把 $\log q$ 作为入参正好绕开这一步，因此 PyTorch 与 Keras 都选了「第一个参数是对数概率」这个别扭但合理的约定。`batchmean` 是后来才补的档位，专门用来让「按 batch 平均的逐样本 KL」与数学定义一致；默认值仍是 `"mean"`（历史原因不能改），这就是本题考点。RLHF 侧 `to-kl-controller` / PPO 的 KL 惩罚、VAE 的 β-正则都吃这个口径。

## 逐元素项的符号与退化

$$\ell_c = p_c(\log p_c - \log q_c)$$

三种边界各自要有确定的取值：

| 条件 | $\ell_c$ | 实现要点 |
| --- | --- | --- |
| $p_c > 0,\; q_c > 0$ | $p_c(\log p_c - \log q_c)$，可正可负 | 正常式 |
| $p_c > 0,\; q_c = 0$ | $+\infty$ | $\log q = -\infty$，相减自然给 $+\infty$ |
| $p_c = 0$ | $0$（约定 $0\log 0 := 0$） | 必须显式掩码，否则 $0\cdot(-\infty) = \mathrm{NaN}$ |

参考答案用一次 `where` 处理第三行：`torch.where(t > 0, t * (t.log() - q), zeros)`。`log_target=True` 时 $p=0$ 表现为 `target = -inf`，判据换成 `isinf(t) & (t < 0)`。题面明确**不复制** `F.kl_div` 在 $p=0,q=0$ 处给 NaN 的实现细节——判题在这一格按 0 断言。

## batchmean 与蒸馏权重

设输出形状 $(B, C)$（广播后），则

$$\text{batchmean} = \frac{1}{B}\sum_{b}\sum_{c}\ell_{b,c}, \qquad \text{mean} = \frac{1}{BC}\sum_{b,c}\ell_{b,c} = \frac{\text{batchmean}}{C}$$

蒸馏里 `loss = (1-α)·CE + α·β²·KL` 这类公式的系数是按「每样本 KL」推的，误用 `mean` 会把 KL 项缩小 $C$ 倍（$C=150k$ 的词表下等于悄悄关掉正则）。判题用 $C=6$ 的小规模直接断言 `batchmean == mean × 6`，把这个数量级差异钉死。另外 KL 的非负性只在**约化之后**成立（Gibbs 不等式），逐元素项可负——所以题面用「同一批分布」构造断言 `batchmean >= 0`，而不是对 `none` 输出做非负检查。

## 核心与易错点

1. **参数顺序**：`solve(log_pred, target)`，写反了会在稠密随机用例上立刻挂（$p$ 与 $\log q$ 数量级完全不同）。
2. **`batchmean` 除的是输出的第 0 维**：`target` 广播成 $(5,4)$ 时 $B=5$，不是入参 `target` 的长度 1；参考答案取 `elem.shape[0]`。
3. **`batchmean` 要求至少一维**：0 维输入没有「第 0 维长度」，抛 ValueError。
4. **one-hot 退化**：$p$ 是 one-hot 时 $\sum_c \ell = -\log q_{c^*}$ 恰是交叉熵，判题用这个恒等式独立校验（与 `cross-entropy-loss` 相衔接）。
5. **float64**：$\log p$ 在 $p\sim10^{-8}$ 时对 float32 精度很敏感，容差 atol=1e-6 需要 float64 中间量。

## 与邻近题的区别

- `kl-divergence`：给定两个概率向量算标量 $\mathrm{KL}(P\|Q)$ 与 $\mathrm{KL}(Q\|P)$，考数学定义与方向性。
- `jsd-loss`：对称化的 Jensen–Shannon 散度，内部要各自对 $M=(P+Q)/2$ 求 KL。
- `cross-entropy-loss`：$p$ 为 one-hot 时的特例，本题的硬标签分支正好复现它。
- `to-kl-controller` / `to-grpo-advantage`：RLHF 里把 KL 当惩罚项的**控制器与系数**，不考逐元素口径。

## 对拍口径与复杂度

- oracle 是 `F.kl_div`（稠密用例逐点对拍；$p=0,q=0$ 那格按题面约定单独断言），另含 one-hot 解析校验、$q=0$ 的 $+\infty$、广播 target、`batchmean`/`mean` 差 6 倍、`log_target=True` 往返与 4 个非法 `reduction`。
- 复杂度 $O(N)$，$N$ 为广播后的元素数；两次逐元素遍历（`where` 与 `sum`）。
