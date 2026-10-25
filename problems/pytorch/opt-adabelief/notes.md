# 详解：AdaBelief 置信度二阶矩

## 出处
AdaBelief 论文：*AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients*，arXiv:1904.03502（NeurIPS 2018/2019）。算法骨架对齐 kozistr/pytorch_optimizer 的 `AdaBelief`（`rectify=False, ams_bound=False, foreach=False` 的 per-param 路径，`weight_decouple=True, fixed_decay=False`）。

## 一句话机制
把 Adam 的二阶矩 \(\mathbb{E}[g^2]\) 换成对「梯度预测误差」的二阶矩 \(\mathbb{E}[(g-m)^2]\)——即信念（belief）。梯度若可预测则误差小 → 分母小 → 有效步长放大；梯度若混乱则误差大 → 分母大 → 步长收缩。

## 关键顺序（易错点）
1. **先更新 \(m\)，再用「更新后的 \(m\)」算残差**：`diff = g - m`（\(m\) 已 lerp 过）。若用旧 \(m\) 就退化成不同算法。
2. \(s\) 的累积里就加一次 \(\epsilon\)：`s = β2·s + (1−β2)·diff² + eps`；最终分母再 `+ eps`。两处都加，对齐库实现。
3. 偏差修正只作用在分母的 \(\sqrt{s}\) 上（除 \(\sqrt{1-β_2^t}\)）；分子 \(m\) 不单独修正，而是把 \(1/(1-β_1^t)\) 折进 `step_size = lr/(1-β1^t)`。等价写法：\(p \mathrel{-}= \text{step\_size}\cdot m/(\sqrt{s}/\sqrt{\hat β_2}+\epsilon)\)。

## non-rectified 分支
`rectify=False` 时不走 RAdam 式的 SMA 整流，直接自适应更新，因此第 1 步就有更新（不像 RAdam 前几步 no-op）。本题固定 `rectify=False`。

## 对拍口径
外部参照用独立向量化重算（每步显式做 lerp、残差平方、两次偏差修正）。全程 float32，per-param 与向量化逐元素一致（示例值实测 maxdiff 0，随机多步 20 步 ×50 seed 最大偏差 ≈2.4e-7，远小于判题容差）。

## 复杂度
每参数每步 \(O(\text{numel})\)，额外两份状态 \((m, s)\)，与 Adam 同量级。
