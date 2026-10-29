# LayerNorm 反向：dX 的双减项与 γ/β 的折叠求和

## 题目描述

LayerNorm 是 Transformer 每一个 block 的进出各一次的正则化算子，它的反向是「归一化类算子」里最容易写错的一题：因为前向有减均值与除标准差**两条**耦合路径，dX 就必须减**两项**——\(\text{mean}(g)\) 与 \(\hat x\cdot\text{mean}(g\circ\hat x)\)。少写任何一项都能通过一部分测试，却过不了 autograd 对拍。本题给你 `x, gamma, beta, dY`，返回 `(dX, dGamma, dBeta)`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 归一化沿 `x` 的**最后一个轴**；方差取有偏形式（`unbiased=False`），\(\hat x=(x-\mu)/\sqrt{var+eps}\)。
- 令 \(g=dY\circ\gamma\)，则 \(dX=\big(g-\text{mean}(g)-\hat x\cdot\text{mean}(g\circ\hat x)\big)/\sigma\)，两个 mean 都沿末轴。
- \(d\gamma=\sum_{\text{前导轴}} dY\circ\hat x\)、\(d\beta=\sum_{\text{前导轴}} dY\)，形状恒为 `(H,)`，即使 `x` 是一维。
- `beta` 不参与任何中间量：改动 `beta` 后三个返回值必须逐位不变。形状不符、`eps` 非法一律抛 ValueError。

## 示例

一行 4 个特征、`eps=0.0`，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
x      = [[-1, 1, -1, 1]]     gamma = [2, 1, 2, 1]   beta = [0, 0, 0, 0]
dY     = [[ 1, 2,  3, 4]]
mu = 0, var = 1, xhat = x,  g = dY*gamma = [2, 2, 6, 4]
mean(g) = 3.5,  mean(g*xhat) = -0.5
dX     = [[-2, -1, 2, 1]]     dGamma = [-1, 2, -3, 4]   dBeta = [1, 2, 3, 4]
```

第 0 列：\(2-3.5-(-1)(-0.5)=-2\)；第 2 列：\(6-3.5-(-1)(-0.5)=2\)。

## 约束

- 支持任意前导轴（`(H,)`、`(B,H)`、`(B,T,H)`）与 float32/float64；内部 float64 累计后折回原 dtype
- 与 `torch.nn.functional.layer_norm` 的 `torch.autograd.grad` 对拍，float64 atol=1e-9、float32 atol=1e-5
