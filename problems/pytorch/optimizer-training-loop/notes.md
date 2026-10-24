# SGD 训练循环与数值复现 详解

## 目标函数与超参

标量参数 $w$ 从 $2.0$ 起步，最小化 $\text{loss}=w^2$；`SGD(lr=0.1, momentum=0.9)` 跑 20 步，记录每步**更新前**的 loss。

## 五步固定顺序（判题命脉）

每一步严格按：

1. `opt.zero_grad()` —— 清掉上一步残留梯度（`.grad` 是累加的，不清会叠加）。
2. `loss = (w**2).sum()` —— 前向构图。
3. `losses.append(loss.item())` —— **在 backward/step 之前**记录，所以第 $k$ 个记录是第 $k$ 次更新前、已应用了前 $k-1$ 步后的 loss。
4. `loss.backward()` —— 反传填 `w.grad`。
5. `opt.step()` —— 按梯度更新 `w`。

顺序错位（比如 step 后才记录、或 backward 在 zero_grad 前）会让整条 loss 曲线偏移，判题逐元素比对必然失败。`sum()` 是为让非标量 $w^2$ 变标量以直接 backward。

## momentum 的数学

动量法维护速度缓冲区 $v$：

$$ v_t=\mu\,v_{t-1}+\eta\,g_t,\qquad w_{t+1}=w_t-v_t $$

$\mu=0.9,\eta=0.1$。它给下降方向加惯性，越过窄谷震荡、加速收敛。本题 $g_t=2w_t$ 解析可推，故 20 步序列是**完全确定**的浮点值——这正是能逐位对拍的前提（无随机数据、无 dropout）。

## 为什么能精确复现

无数据随机性、无浮点归约顺序差异（单标量），PyTorch 的 SGD+momentum 更新式确定。因此把每一步 loss 原样 dump 进题面即可作为期望值，不必设种子。

## 关联

autograd-basics 提供 backward 机制；nn-module-block 提供被优化的参数容器；dataset-dataloader 是「有批量数据」版本的本循环。
