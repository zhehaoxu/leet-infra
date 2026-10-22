# Swish / SiLU 详解

## 定义

$$ \text{swish}(x) = x\cdot\sigma(\beta x) $$

$\beta=1$ 时即 **SiLU**（本合集实现），也是 LLaMA 类模型激活。它是自门控（self-gating）激活：用自身经 sigmoid 当门。

## 为什么用它

- **平滑非单调**：在 $x<0$ 有小负谷（约 $-0.278$ 于 $x\approx-1.28$），比 ReLU 保留少量负信号、比 sigmoid 不饱和消失；处处可导，梯度更稳定。
- 实验上在深层网络优于 ReLU，代价是一个 sigmoid。

## 实现要点

- `x * torch.sigmoid(x)` 即可；不要写成 `x*0.5*(1+erf(...))` 之类近似（SiLU 精确式就是 $x\sigma(x)$）。
- 大负 $x$：sigmoid 下溢成 0，乘积为 0，安全；大正 $x$：sigmoid→1，退化为恒等 $x$，不会溢出。

## 梯度

$$ \text{swish}'(x) = \sigma(x) + x\,\sigma(x)(1-\sigma(x)) = \sigma(x)\,[1 + x(1-\sigma(x))] $$

## 关联

GELU $=x\Phi(x)$ 用高斯 CDF 门，Mish $=x\tanh(\text{softplus}(x))$ 用另一路门——同族「$x$ × 某门」。三者区别只在门的形状。
