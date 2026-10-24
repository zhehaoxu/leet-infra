# SGD 动量更新 详解

## 更新式（对照 PyTorch）

`torch.optim.SGD`（默认 `dampening=0, nesterov=False`）每步做：

$$ g_t \leftarrow \nabla f(p_t) + \lambda\, p_t \quad(\lambda=\text{weight\_decay}) $$
$$ b_t = \mu\, b_{t-1} + g_t \quad(\mu=\text{momentum}) $$
$$ p_{t+1} = p_t - \eta\, b_t $$

momentum=0 时无缓冲，$p_{t+1}=p_t-\eta g_t$。三个顺序不能颠倒：**先**把 weight_decay 并进梯度、**再**更新 buffer、**最后**用 buffer 更新参数。

## 直觉

- **momentum**：把梯度看成加速度、$b$ 看成速度，$\mu$ 越接近 1 惯性越强。沿一致方向时速度累积（比纯 SGD 快），来回震荡方向时正负抵消（抑制震荡）。有效学习率被放大成约 $\eta/(1-\mu)$。
- **weight_decay vs L2**：这里 WD 在梯度上加 $\lambda p$，是「解耦权重衰减」的原始形式（对 SGD 而言 weight_decay ≡ L2，但对 Adam 不等价，见 adam 笔记的坑）。

## nesterov（本题不启用）

Nesterov 版是 $b_t=\mu b_{t-1}+g_t,\ p_{t+1}=p_t-\eta(g_t+\mu b_t)$，即「先探一步再修正」。判题口径是 nesterov=False，别加这一项。

## 口径坑

- **buffer 初值必须是 0**：PyTorch 首步 `buf = grad`（因为 `0.9*0 + grad`）。若误用 None 或不清零会整体偏移。
- 返回 `(new_params, new_buffers)` 两项都要给，测试比 buffer 是否 carry-over 正确。
- 不要原地改传入张量（clone），否则多次调用互相污染。

## 与 optimizer-training-loop 的分工

那题是「训练循环骨架 + 逐步 loss 记录」，这题聚焦**单个更新算子的数学**——把优化器从黑盒拆成三步代数。

## 关联

adam-step 是带二阶矩 + 偏差修正的更复杂更新；weight-norm / batchnorm 等训练题的背后都是这类 step 函数在驱动。
