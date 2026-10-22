# Tanh 详解

## 定义

$$ \tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}=2\sigma(2x)-1 $$

把实数压到 $(-1,1)$，**零中心、奇函数**。

## 与 sigmoid 的关系

同一族，差一个仿射：$\tanh(x)=2\sigma(2x)-1$。所以 tanh 输出均值更接近 0，梯度比 sigmoid 强（$2\times$ 斜率），这是 RNN 早期偏爱 tanh 的原因。

## 实现与数值要点

- `torch.tanh` 已数值稳定：$x\to\pm\infty$ 分别饱和到 $\pm1$，不会溢出。
- 若手写 $(e^{2x}-1)/(e^{2x}+1)$，$x$ 大时 $e^{2x}$ 上溢 → `inf/inf=NaN`。要用 `torch.exp` 前先分流正负，或直接用库函数。

## 梯度

$$ \tanh'(x)=1-\tanh^2(x) $$

在 $x=0$ 处梯度 1（最大），两端趋 0 —— 与 sigmoid 同为饱和激活，深层/两端会引发梯度消失；LSTM/GRU 的门用 sigmoid、状态候选用 tanh，正是取长补短。
