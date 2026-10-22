# Mish 详解

## 定义

$$ \text{mish}(x) = x\cdot\tanh\!\big(\text{softplus}(x)\big),\qquad \text{softplus}(x)=\ln(1+e^{x}) $$

## 设计动机

和 Swish 同属「$x$ × 光滑门」，但门用 $\tanh(\text{softplus}(x))$：

- 处处光滑**且一阶导也连续**（Swish 的更高阶导不连续），据称更利于优化。
- 负区间有一个浅谷（约 $-0.31$ 于 $x\approx-1.19$），保留负信号但不饱和。

## 数值稳定要点

- softplus 直接 `torch.log1p(torch.exp(x))` 在 $x$ 大时会 `exp` 溢出。稳定写法：
  $$ \text{softplus}(x)=\max(x,0)+\ln\!\big(1+e^{-|x|}\big) $$
  torch 有 `F.softplus`，或判题约定用 `torch.logaddexp(0, x)`。
- 大正 $x$：softplus$\to x$，$\tanh(x)\to1$，mish$\to x$；大负 $x$：softplus$\to0$，$\tanh\to0$，mish$\to0$。两端行为符合直觉。

## 与相邻激活对比

| 激活 | 门 | 负区间 |
|---|---|---|
| ReLU | 阶跃 | 恒 0 |
| Swish/SiLU | $\sigma(x)$ | 浅谷 |
| GELU | $\Phi(x)$ | 浅谷 |
| Mish | $\tanh(\text{softplus}(x))$ | 稍深谷、二阶更光滑 |
