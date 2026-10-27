# 结构化丢弃：DropPath 与 Dropout2d/3d

## 题目描述

普通 dropout 逐元素独立掷硬币，对空间强相关的卷积特征几乎没有正则效果（Tompson 2015 的观察：i.i.d. 丢弃退化为等效学习率下降）；结构化丢弃改成**成块丢弃**——整样本（DropPath / stochastic depth：Huang et al., ECCV 2016 在训练时以概率随机跳过整条残差支路；timm 的 `DropPath` 是其标准实现）或整通道（Dropout2d/3d 的 spatial dropout：整张特征图置零）。本题实现三者统一的前向 \(y=x\cdot mask_{\text{广播}}/(1-p)\)（inverted scaling），关键差别在**丢弃位置不由 RNG 生成**：mask 是显式布尔 / 0-1 张量入参（1=保留、0=丢弃），CPU 下完全确定——与用 seed 造 mask 的 `dropout` 题刻意分工。三种 mask 粒度分别对应样本级（timm DropPath 结构语义）与通道级（`F.dropout2d`/`F.dropout3d` 在 training 态的结构语义，mask 显式给定而非采样）。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。入参：x 为 float 张量（CPU，float32/float64）；p 为丢弃概率；mask 为显式布尔或 0/1 张量；mode ∈ `"droppath"` / `"dropout2d"` / `"dropout3d"` 三选一，其余抛 `ValueError`。
- **统一语义**：\(y=(x\cdot mask_{\text{广播}})/(1-p)\)——inverted scaling（与 `F.dropout` training=True 一致）；被丢的样本/通道整体**严格为 0**，存活元素 \(=x/(1-p)\)。
- **p 校验**：\(0\le p<1\)；\(p\ge1\)、\(p<0\) 或非有限值（inf/nan）抛 `ValueError`。
- **三种 mask 粒度**（其余形状一律抛 `ValueError`；mask 广播后的形状必须与 x 完全一致，不许广播出额外的 batch 维）：
  - droppath：x 秩 \(\ge1\)（\((B,\dots)\) 任意尾部形状），整样本丢弃；mask \((B,)\) 或 \((B,1,\dots,1)\)（1 的个数 = x 维数 − 1）。秩不符抛 `ValueError`。
  - dropout2d：x 恰 4 维 \((N,C,H,W)\)，整通道丢弃（整张 \(H\times W\) 特征图置零）；mask \((N,C)\) 或 \((N,C,1,1)\)。秩不符抛 `ValueError`。
  - dropout3d：x 恰 5 维 \((N,C,D,H,W)\)，含深度维整通道丢弃；mask \((N,C)\) 或 \((N,C,1,1,1)\)。
- **退化与恒等**：p==0 且 mask 全 1 时严格恒等（`torch.equal(y,x)`）；p==0 且 mask 含 0 时 \(y=x\cdot mask\) 逐元素精确相等。
- mask 为 bool 时先转成 x 的 dtype 再相乘；返回与 x 同形状、同 dtype 的**新张量**，不修改入参；全程无随机数调用。

## 示例

主示例（数值取自 test_cases 的 `EXAMPLE_*` 常量）：x \((2,3,2,2)\)、p=0.5、mask=\([1,0]\)、mode=droppath——样本 0 存活、整体 \(\times2\)（\(=x/(1-0.5)\)），样本 1 被整丢、严格全 0：

\[
\text{样本 0}=\begin{bmatrix}1&-2\\4&0\end{bmatrix},\begin{bmatrix}7&3\\-5&8\end{bmatrix},\begin{bmatrix}0.5&-1.5\\2.5&5.5\end{bmatrix};
\qquad \text{样本 1}=\text{全 } 0
\]

探针组：mask=\([1,0,1,0]\)、p=0.25 → 存活样本恰为 \(x/0.75\)、被丢样本 `all(==0)`。dropout2d：mask 全 1 \((N,C)\)、p=0.4 → \(y=x/0.6\)（无一通道被丢）；mask 某位为 0 → 对应 \((n,c)\) 的整张特征图全 0。dropout3d：mask 全 1、p=0.5 → \(y=2x\)。mask 的 \((B,)\) 与 \((B,1,1,1)\) 两种形状、bool 与 0/1 浮点两种类型，结果 `torch.equal`。

## 约束

- mask 形状白名单按 mode 二选一，白名单外（如 droppath 的 3 维 x 收到 \((B,C)\)、\((B,1,1,1,1)\) 或 \((B,1)\)；dropout2d 收到 \((N,)\)、\((N,C,1)\)、\((N,C,1,1,1)\) 或形状值不配的 \((N,C',1,1)\)）抛 `ValueError`；x 秩与 mode 不符抛 `ValueError`。
- 对拍：独立纯 python oracle 手工广播 + \(1/(1-p)\) 缩放，与张量化解法分路径，`allclose` 1e-6（float64 1e-12），形状与 dtype 逐一断言；确定性伪随机张量与 bool mask 由黄金比例哈希生成（无 RNG）。
- 判别口径评测直接断言：被丢块严格全 0（`==0` 而非近似）、存活块恰为 \(x/(1-p)\)、两种 mask 形状等价、bool ≡ 0/1 浮点、p=0 恒等（`torch.equal`）、p 与 mask 形状与 mode 的错误分类。
