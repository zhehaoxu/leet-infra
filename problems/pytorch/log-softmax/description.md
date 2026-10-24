# 数值稳定 Log-Softmax（Log-Softmax）

## 题目描述

log-softmax 先做 softmax 再取对数：$\log\text{softmax}(x)_i = x_i - \log\sum_j e^{x_j}$。
分类损失（交叉熵 = `log_softmax` + `NLL`）几乎处处用到它。朴素写法 `log(softmax(x))`
会在 logits 很大时 `exp` 上溢成 `inf`、很小时下溢导致 `log(0)=-inf`。请实现**减最大值**
的稳定版本。

## 规则

- 实现 `solve(x, dim=-1)`，沿 `dim` 计算 log-softmax，形状与输入相同。
- 稳定公式：
  $$ \log\text{softmax}(x)_i = (x_i - m) - \log\!\sum_j e^{x_j - m},\quad m=\max_j x_j $$
- 减去最大值不改变结果（分子分母同乘 $e^{-m}$ 抵消），但把指数参数压到 $\le 0$ 防上溢。
- 结果须与 `F.log_softmax(x, dim=dim)` 一致。

## 示例

```
输入：x = [[1, 2, 3],
          [3, 1, 0]]（沿最后一维）
输出：[[-2.4076, -1.4076, -0.4076],
      [-0.1698, -2.1698, -3.1698]]
（每行 exp 之后重新构成该行的 softmax 概率，和为 1）
```

## 约束

- float32 输入；即便 logits 达上千也不得出现 `inf/nan`
- `exp(输出)` 沿 `dim` 求和应为 1
