# Gumbel-Softmax 详解

## Gumbel-max：把分类采样变成取 max

若 $G_i\sim\text{Gumbel}(\log p_i,1)$ 相互独立，则 $\arg\max_i G_i$ 的分布正好是
$p$。这是「Gumbel-max trick」：加噪后取 max 等价于按 $p$ 采样。而
$$ G_i = \log p_i + \tilde g_i,\quad \tilde g_i=-\log(-\log u_i),\ u_i\sim U(0,1) $$
所以从 `logits`（即 $\log p$ 的未归一化形式）出发，加标准 Gumbel 噪声再 argmax 就是采样。

## 从「硬」到「软」：换成 softmax 才能反传

argmax 不可导。Gumbel-Softmax（Jang et al. 2016; Maddison et al. 2016）把 argmax 松弛成
温度 softmax：$y=\text{softmax}((\log\text{it}+g)/\tau)$。$\tau\to0$ 时 $y$ 趋近采到的
one-hot，有限 $\tau$ 时是可导的软近似，梯度可流过——这正是离散 VAE / NAS 架构编码 /
硬注意力能端到端训练的关键。

## 重参数化 = 随机性外移

核心思想：把「随机」集中到外部噪声 $u$（与参数无关），使 $y$ 成为参数的确定函数。
本题把 $u$ 作为显式输入正是贯彻这一点，也顺带让判题可复现。类似技巧有 VAE 的
$z=\mu+\sigma\epsilon$（把噪声挪到 $\epsilon$）。

## 温度的作用与调度的坑

- $\tau$ 越小越接近真实离散样本、但梯度方差越大、越易饱和；越大越平滑但偏「糊」。
- 训练常**退火** $\tau$（从大到小）以平衡探索与逼近硬样本。
-  Straight-through（ST）变体：前向用 hard one-hot、反向用 soft 梯度，兼顾无偏与可导。
