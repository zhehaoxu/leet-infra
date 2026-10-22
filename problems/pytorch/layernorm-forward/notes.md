# LayerNorm（前向）详解

## 定义

对**每个样本自己**在特征维上归一化，与 batch 无关：

$$ \hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y=\gamma\hat{x}+\beta $$

$\mu,\sigma$ 在归一化的那一维（Transformer 里通常是隐层维 `d_model`）上按**逐样本**计算。

## 与 BatchNorm 的本质区别

- BN 沿 batch 统计、按通道共享、需要 running stats、依赖 batch 大小；LN **逐样本**沿特征统计，batch 大小无关、训练推理同一套、无 running stats。
- 所以 RNN/Transformer（变长序列、单样本推理）用 LN 而非 BN。

## 维度坑（本关最易错）

- 对 $(B,T,C)$ 归一化最后维 `dim=-1`：$\mu,\sigma$ 形状 $(B,T,1)$，$\gamma,\beta$ 形状 $(C,)$ 广播到末维。
- 归约维度要对准 `normalized_shape`，多模态里有时归一多个连续维（如最后两维）——按题给维度。
- $\epsilon$ 在开方**内**（同 BN），默认 `1e-5`。

## RMSNorm 的关系

LN 减均值是它的「中心化」步骤；RMSNorm 去掉减均值、只除以均方根，更省。理解 LN 后 RMSNorm 是它的简化版。

## 对拍参考解

`F.layer_norm(x, normalized_shape, weight, bias, eps)`，`normalized_shape` 是末维（或末几维）的形状元组。
