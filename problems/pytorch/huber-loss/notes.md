# Huber 损失 详解

## 定义

设残差 $z=|p-t|$、阈值 $\delta$：

$$ \mathcal L_\delta(z)=\begin{cases}\tfrac12 z^2 & z\le\delta\\ \delta\,(z-\tfrac12\delta) & z>\delta\end{cases} $$

小误差走 MSE、大误差走线性。梯度对 $p$：$z\le\delta$ 时 $=(p-t)$，否则 $=\delta\,\mathrm{sign}(p-t)$（被截到 $\pm\delta$）。

## 为什么这样设计

- MSE 梯度随残差线性放大：一个离群点（$z$ 巨大）能把整批梯度带偏。
- MAE 梯度恒为 $\pm1$：鲁棒但在 0 附近不可导、且小残区也罚满、收敛慢。
- Huber 取中：内区二次（近最优点有二阶信息、收敛快），外区线性且梯度饱和到 $\delta$（离群点不再放大）。$\delta$ 是「你愿意容忍多大残差再当离群点」的旋钮。

## $\delta$ 的含义与口径坑

- $\delta$ 大 → 更像 MSE；$\delta$ 小 → 更像 MAE（再小到 0 即 L1 尺度）。
- **Huber ≠ Smooth L1**：`F.smooth_l1_loss` 是 Huber 的 $\delta=1$ 特例，但其线性段写作 $z-0.5$（即 $\delta(z-\tfrac12\delta)$ 在 $\delta=1$）。二者 $\delta=1$ 时相等，$\delta\ne1$ 时只有 Huber 能调，别把两个 API 的默认阈值混用。
- 分段边界用 $z\le\delta$ 走二次（与 `F.huber_loss` 一致）；边界两侧函数与导数都连续，等号放哪边都不影响数值。

## 实现

`torch.where(z<=delta, 0.5*z*z, delta*(z-0.5*delta))` 一次向量化完成，再按 reduction 归约。

## 家族

Fair / Cauchy / LogCosh / Charbonnier 都是「内二次外亚/超二次」的鲁棒损失变体；目标检测回归框、物理量回归、含噪回归常用 Huber。

## 关联

mse-loss 是其内区；L1/MAE 是其外区近似；reduction 口径与全部损失题一致。
