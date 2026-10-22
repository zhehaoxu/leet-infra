# Softplus 详解

## 定义

ReLU 的光滑版本，是对数-sum-exp（LSE）在两项的特例：

$$ \text{softplus}(x)=\ln(1+e^{x}) $$

## 关键性质

- 是 $\max(0,x)$ 的光滑逼近：$x\gg0$ 时 $\approx x$，$x\ll0$ 时 $\approx 0$。
- 导数恰是 sigmoid：$\text{softplus}'(x)=\sigma(x)$——它是 sigmoid 的原函数。
- 恒正，常用来把无约束参数映射成尺度/方差等必须为正的量的参数化：$\text{scale}=\text{softplus}(\rho)$。

## 数值稳定：本关真正的坑

朴素 `log(1+exp(x))` 在 $x$ 很大时 `exp` 上溢成 inf，log(inf)=inf，丢掉 $\approx x$ 的真值。稳定等价式：

$$ \text{softplus}(x)=\max(x,0)+\ln\!\big(1+e^{-|x|}\big) $$

因为 $e^{-|x|}\le 1$ 永不溢出。PyTorch 里 `F.softplus`、或 `torch.logaddexp(0, x)` 都已内建这条保护。

`beta` 缩放版：$\tfrac1\beta\ln(1+e^{\beta x})$，$\beta\to\infty$ 收敛回 ReLU。
