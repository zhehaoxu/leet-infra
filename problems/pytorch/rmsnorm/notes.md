# RMSNorm 详解

## 定义

LayerNorm 的简化版：**不减均值**，只按均方根缩放：

$$ \hat{x}_i=\frac{x_i}{\sqrt{\frac1d\sum_{j}x_j^2+\epsilon}},\qquad y_i=\gamma_i\hat{x}_i $$

没有 $\beta$（无中心化的偏置项）。

## 为什么用它

- 少一次均值归约与减法，更快、更省，实验效果与 LN 相当甚至略好。
- LLaMA、Gemma 等主流大模型 Transformer 全部用 RMSNorm 替代 LN——理解它 = 读懂现代 LLM 归一化的入口。

## 实现要点

- 均方根 `rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)`；对 `(B,...,C)` 沿末维。
- **只除以 rms，不做 `x - mean`**，这是与 LN 唯一的结构差异，写成减均值就错。
- $\gamma$（weight）形状 `(C,)`，广播到末维；可无 bias。
- eps 在开方**内**。

## 尺度不变性

$y$ 对输入整体缩放不敏感：$x\to cx$ 时 rms 同乘 $c$，比值不变（这正是 QK-Norm 里用它稳定注意力的原因，见「QK-Norm」关）。

## 对拍参考解

无内置 `F.rms_norm`（旧版本），手工参考：`x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps) * weight`。`torch.rsqrt` 比 `1/sqrt` 略稳且快。
