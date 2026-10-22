# BatchNorm（前向）详解

## 定义

沿 **batch 维**（通常 dim=0）对每个特征通道独立归一化，再仿射：

$$ \hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat{x}+\beta $$

$\mu_B,\sigma_B^2$ 是当前批的均值/方差，$\gamma$（weight）、$\beta$（bias）是按通道的可学习参数。

## 训练 vs 推理：本关最大的坑

两种口径**不同**，别混：

- **训练（train）**：用**当前 batch** 的 $\mu_B,\sigma_B^2$（有偏估计即可，torch 默认 unbiased=False 做前向统计）。
- **推理（eval）**：用训练期间累积的**移动平均** running_mean/running_var，与当前 batch 大小无关（哪怕单样本也能算）。

判题若给的是 `training=True`，就对当前批求统计；给 running_* 且 `training=False`，就用它们。选错口径数值全错。

## 维度与归约

- 对 $(N,C,H,W)$：$\mu,\sigma$ 形状 $(C,)$，归约要**对 N、H、W 求均值**（dim=(0,2,3)），再把 $(C,)$ reshape 成 $(1,C,1,1)$ 广播。只对 batch 求、忘记 H/W 是常见错误。
- 卷积的 BN 常写作 "sync BN"：多卡时 $\mu,\sigma$ 跨设备聚合，本关单卡不涉及。

## eps 的位置

$\epsilon$ 在**开方之内**：$\sqrt{\sigma^2+\epsilon}$，防止零方差除零；默认 $10^{-5}$。放根号外是不同公式。

## 对拍参考解

用 `F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)`；注意传入的 running_* 在 `training=True` 时仍会被更新（但我们只取前向输出）。
