# ELU 详解

## 定义

$$ \text{elu}(x)=\begin{cases} x, & x>0\\ \alpha(e^{x}-1), & x\le 0\end{cases}\qquad(\alpha \text{ 默认 }1) $$

## 设计动机

- 负区间指数饱和到一个有限渐近线 $-\alpha$（而非无限制），使**均值更接近 0**，加速收敛；同时负梯度非零，缓解 ReLU 的「死神经元」。
- 在 $x=0$ 处连续且可导（左导 $=\alpha e^0=\alpha$，右导 $=1$，$\alpha=1$ 时 $C^1$ 光滑）。

## 实现与数值要点

- `torch.where(x>0, x, alpha*(torch.exp(x)-1))`，或先 `exp` 再按符号选择。
- $x$ 很负时 $e^x\to0$，输出 $\to-\alpha$，不会下溢成问题；但 $e^x-1$ 在 $x$ 接近 0 时用 `torch.expm1(x)` 更准（避免灾难性抵消）。
- 反向：$x\le0$ 时梯度 $=\alpha e^x$（趋于 0 但非恒 0）。

## 谱系

Leaky-ReLU 是「负区间乘个小斜率」的线性版；ELU 是「负区间指数饱和」的平滑版；SELU = 自归一化网络里带固定 $\lambda,\alpha$ 缩放的 ELU 变体。
