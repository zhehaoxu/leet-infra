# 二元交叉熵损失 详解

## 定义

伯努利分布的负对数似然。真值 $y\in\{0,1\}$、预测概率 $p\in(0,1)$：

$$ \mathcal L = -\big[y\log p + (1-y)\log(1-p)\big] $$

$y=1$ 退化为 $-\log p$（希望 $p\to1$），$y=0$ 退化为 $-\log(1-p)$（希望 $p\to0$）。

## 梯度

$$ \frac{\partial\mathcal L}{\partial p}=\frac{p-y}{p(1-p)} $$

配上 sigmoid 输出时链式一约，$\partial\mathcal L/\partial z = p - y$（$z$ 为 logits）——和 softmax+CE 一样干净。这正是**推荐用 `BCEWithLogitsLoss`**（见下）的原因。

## 概率版 vs logits 版（关键区别）

| | 输入 | 数值稳定做法 | PyTorch API |
|--|------|--------------|-------------|
| **BCE**（本题） | 概率 $p\in(0,1)$（已过 sigmoid） | $\log$ 前把 $p$ clamp 到 $[\epsilon, 1-\epsilon]$ | `F.binary_cross_entropy` |
| **BCEWithLogits** | logits $z\in\mathbb R$（未过 sigmoid） | 合并 sigmoid 与 log 成 softplus 形式 | `F.binary_cross_entropy_with_logits` |

生产代码首选 **WithLogits**：把 `sigmoid` 与 `log` 融在一起，从根上避免 $p$ 饱和到 0/1 造成 $\log 0=-\infty$。本题练的是概率版，坑就在**必须截断**：`p.clamp(min=eps, max=1-eps)` 后再 log，浮点末位差异在容忍范围内。

## 口径

- `mean` 除以全部元素数 `numel`（batch × 类），适配多标签；`sum`、`none` 同 MSE。
- `pos_weight`（类别不平衡加权）是 BCE 的常见扩展，本题不实现。
- `target` 可为 soft 标签（$[0,1]$ 连续值），公式仍成立——这是 label smoothing 在多标签上的写法。

## 关联

sigmoid 提供概率输出；cross-entropy 是多分类版（独热 vs 多伯努利）；GAN 判别器、多标签分类、mask 预测都吃 BCE。
