# GELU 详解

## 定义

Gaussian Error Linear Unit：用标准正态 CDF $\Phi$ 作为门：

$$ \text{gelu}(x) = x\cdot\Phi(x),\qquad \Phi(x)=\tfrac12\big(1+\operatorname{erf}(x/\sqrt2)\big) $$

直观：每个输入按「它落在正半部分的概率」加权保留。

## 两种近似（易踩口径）

1. **erf 精确式**（torch `approximate='none'`）：`0.5*x*(1+erf(x/sqrt(2)))`。
2. **tanh 近似**（`approximate='tanh'`）：
   $$ 0.5x\Big(1+\tanh\big[\tfrac{\sqrt2}{\sqrt\pi}(x+0.044715x^3)\big]\Big) $$

两者数值不同（$x=1$ 处约差 $10^{-3}$ 量级）。**判题按哪种口径要对齐**——本合集用 erf 精确式，别用 tanh 近似糊过去。

## 实现要点

- 用 `torch.erf`；$x/\sqrt2$ 的系数别漏。
- 大负 $x$：$\Phi\to0$，gelu $\to 0$；大正 $x$：$\Phi\to1$，gelu $\to x$。erf 在两端饱和，无溢出。

## 梯度

$$ \text{gelu}'(x)=\Phi(x)+x\cdot\phi(x),\quad \phi(x)=\tfrac{1}{\sqrt{2\pi}}e^{-x^2/2} $$

Transformer FFN 默认激活即 GELU；它与 Swish 的差别只在门函数是高斯 CDF 而非 sigmoid。
