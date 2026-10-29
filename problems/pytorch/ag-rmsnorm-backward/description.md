# RMSNorm 反向：没有减均值的那一项

## 题目描述

LLaMA 系模型把 LayerNorm 换成 RMSNorm：只除均方根、不减均值，于是一次归一化退化成「沿末轴的一次平方和归约」，省掉一条归约路径。但反向并不是把 LayerNorm 公式照抄后删掉 \(\mu\) 就完事——\(g\) 的均值那一项确实消失了，\(\hat x\cdot\text{mean}(g\circ\hat x)\) 那一项却仍然保留，且 \(\hat x\) 此时就是 \(x\) 本身而不是去均值后的残差。本题给你 `x, gamma, dY`，返回 `(dX, dGamma)`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 前向：\(rms=\sqrt{\text{mean}(x^2,\text{末轴})+eps}\)，\(\hat x=x/rms\)，\(y=\gamma\circ\hat x\)。
- 令 \(g=dY\circ\gamma\)，则 \(dX=\big(g-\hat x\cdot\text{mean}(g\circ\hat x)\big)/rms\)：**没有** \(\text{mean}(g)\) 这一项。
- \(d\gamma=\sum_{\text{前导轴}} dY\circ\hat x\)，形状恒为 `(H,)`。
- `eps` 允许取 0.0、负值抛 ValueError；`gamma` 长度或 `dY` 形状不符一律抛 ValueError。

## 示例

与 LayerNorm 题同形的一行 4 特征、`eps=0.0`，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
x     = [[-1, 1, -1, 1]]   gamma = [2, 1, 2, 1]   dY = [[1, 2, 3, 4]]
rms = 1, xhat = x, g = dY*gamma = [2, 2, 6, 4], mean(g*xhat) = -0.5
dX     = [[1.5, 2.5, 5.5, 4.5]]     dGamma = [-1, 2, -3, 4]
```

第 0 列：\(2-(-1)(-0.5)=1.5\)。同一组入参下 LayerNorm 还给 \(-2\)，差的就是被减掉的 \(\text{mean}(g)=3.5\)。

## 约束

- 支持任意前导轴与 float32/float64；内部 float64 累计后折回原 dtype
- 与 `x.pow(2).mean(-1)` 手搭前向的 `torch.autograd.grad` 对拍，float64 atol=1e-9、float32 atol=1e-5
