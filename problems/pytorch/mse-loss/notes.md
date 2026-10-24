# MSE 损失 详解

## 定义

$$ \mathcal L_{\text{MSE}}=\frac1n\sum_{i=1}^{n}(p_i-t_i)^2 \quad(\text{mean});\qquad \sum_i(p_i-t_i)^2\quad(\text{sum}) $$

逐元素平方误差，再按 `reduction` 归约。梯度 $\partial\mathcal L/\partial p_i = \frac{2}{n}(p_i-t_i)$（mean）。

## 为什么平方

- **平滑可导**：处处有梯度，利于优化；原点附近罚得轻、远处罚得重。
- **对应高斯噪声极大似然**：若目标 $t=p+\varepsilon,\ \varepsilon\sim\mathcal N(0,\sigma^2)$，最小化 MSE 等价于 MLE。
- **对离群点敏感**是双刃剑：一个坏标注会显著抬高损失——需要鲁棒时改用 Huber/L1（见 huber-loss）。

## 口径坑

- **没有 $\frac12$ 系数**。有些教材写 $\frac12(p-t)^2$（让梯度干净），但 PyTorch `F.mse_loss` 不加 $\frac12$。判题对齐 `F.mse_loss`，别自己补 $\frac12$。
- `mean` 除以的是**元素总数** `numel`（含 batch 与特征维），不是样本数；`reduction="none"` 保留全形状。
- 广播：`target` 可为标量或可广播形状，`(pred-target)` 先广播再平方。

## reduction 家族口径

所有损失题（MSE/BCE/CE/Huber）共用同一 `reduction∈{mean,sum,none}` 约定，与 `torch.nn.functional.*` 的 `reduction` 参数语义一致；未知值抛 `ValueError`。

## 关联

- L1 损失 $=\frac1n\sum|p-t|$（对离群点更稳）；
- Huber 在小残差走 MSE、大残差走 L1，兼取两者之长；
- MAE / MSLE 等都是本式的变体（换幂次或先取对数）。
