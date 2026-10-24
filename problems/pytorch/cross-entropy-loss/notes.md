# 交叉熵损失 详解

## 定义

对单个样本，真实分布是独热向量、预测是 softmax 概率 $p$：

$$ H(y,p)=-\sum_{c} y_c\log p_c = -\log p_{\text{true}} $$

$$ \operatorname{CE}=-\log\frac{e^{z_{\text{true}}}}{\sum_j e^{z_j}} = -z_{\text{true}}+\log\sum_j e^{z_j} $$

`reduction="mean"` 对 $N$ 个样本再平均。

## 为什么合并 softmax 与 log（数值稳定）

朴素 `log(softmax(z))`：$z$ 大时 $e^{z}$ 上溢成 inf、softmax 归一到 0/1，再取 log 得 $-\infty$/NaN。稳定技巧是先减每行最大值 $m$：

$$ \log\text{softmax}(z)=\underbrace{(z-m)}_{\text{shifted}}-\log\sum_j e^{z_j-m} $$

减 $m$ 不改变 softmax（分子分母同乘 $e^{-m}$），但保证 $e^{z_j-m}\le1$ 不上溢、且至少有一项 $=e^0=1$ 使分母 $\ge1$ 不 $\log0$。这就是本题参考解做的。

## 实现三段

1. 稳定 log-softmax（减最大值 + `logsumexp`）。
2. `gather(-1, target)` 取出真实类那一列的 log-prob。
3. 取负、按 reduction 归约。

坑：`target` 需 `long` 且形状 `(N,1)` 才能 gather；类别维在最后。

## 与 label smoothing / ignore_index

- **label smoothing**：把独热换成 $(1-\epsilon)$ + 均匀 $\epsilon/C$，CE 变凸组合，本可不实现，是常见扩展。
- **ignore_index**（如 -100）：`F.cross_entropy` 会跳过这些样本、mean 也只对有效样本取平均。本题不涉及，`target` 全是有效类。

## 梯度彩蛋

softmax+CE 的梯度干净得惊人：$\partial\mathcal L/\partial z = p - y$（预测概率减独热）。这是它成为分类标配的深层原因。

## 关联

log-softmax 题提供稳定内核、softmax 题是其归一化形式、one-hot 是标签的稠密表示、`nll_loss` = 已喂 log-prob 版的本题。
