# 激活函数反向合集：逐元素导数与零点次梯度约定

## 题目描述

激活反向看起来是八条解析式的默写，真正的考点全在**折点与非初等函数**上：`relu` 在 \(x=0\) 取 0（PyTorch 的次梯度约定，取 0.5 必错）、`elu` 在 0 处连续可导值为 1、`gelu` 要用精确形式 \(\Phi(x)+x\phi(x)\) 而不是 tanh 近似、`silu` 是 \(\sigma+x\sigma(1-\sigma)\)。本题给你 `x, dY, kind`，返回逐元素的 `dx`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `kind` 取 `'relu'`、`'leaky_relu'`、`'elu'`、`'sigmoid'`、`'tanh'`、`'gelu'`、`'silu'`、`'softplus'`，其它值抛 ValueError。
- 解析式：relu \(=\mathbb 1[x>0]\)；leaky_relu \(=1\) 或 `negative_slope`；elu \(=1\) 或 \(e^x\)；sigmoid \(=s(1-s)\)；tanh \(=1-t^2\)；softplus \(=\sigma(x)\)；gelu \(=\Phi(x)+x\phi(x)\)；silu \(=\sigma(x)+x\sigma(x)(1-\sigma(x))\)。
- \(x=0\) 处的导数值依次为 relu 0 / leaky_relu `negative_slope` / elu 1 / silu 0.5 / softplus 0.5。
- `negative_slope` 只被 `leaky_relu` 使用，其余 kind 忽略它但不得报错；`x` 与 `dY` 形状不符抛 ValueError。

## 示例

`kind='relu'`、`dY` 全 1，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
x  = [-2.0, -0.5, 0.0, 0.5, 2.0]
dY = [ 1.0,  1.0, 1.0, 1.0, 1.0]
dx = [ 0.0,  0.0, 0.0, 1.0, 1.0]      # x == 0 gives 0, not 0.5
```

## 约束

- `x` 任意形状（含 0 维与标量张量）；内部 float64 计算，返回 dtype 与 `x` 一致
- 与对应 `F.*` 的 `torch.autograd.grad` 逐位对拍，float64 atol=1e-9；`gelu` 用精确 erf 形式，与 tanh 近似在 \(|x|\approx 2\) 处差约 \(10^{-3}\)
