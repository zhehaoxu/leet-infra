# BatchNorm 前向（训练/推理）

## 题目描述

Batch Normalization 沿通道维（dim=1）跨 batch（及空间维）做归一化，训练与推理行为不同：

- **训练**：用当前 batch 的统计量归一化，并把统计量滑动平均进 running stats：
  $$ \mu_B,\ \sigma_B^2 = \text{mean/var over dims } (0,2,3,\dots),\quad y = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}\gamma + \beta $$
  $$ \text{running\_mean} \leftarrow (1-m)\cdot\text{running\_mean} + m\cdot\mu_B,\qquad \text{running\_var} \leftarrow (1-m)\cdot\text{running\_var} + m\cdot\hat\sigma^2 $$
- **推理**：直接用 running stats 归一化，且不更新它们。

## 规则

- 实现 `solve(x, weight, bias, running_mean, running_var, training=False, momentum=0.1, eps=1e-5)`，返回三元组 `(y, running_mean_new, running_var_new)`。
- 输入为 `(N, C)` 或 `(N, C, *)`，统计维是除 `dim=1` 外的所有维；$\gamma,\beta$ 按通道广播。
- **两个经典口径陷阱**（与 `F.batch_norm` 完全一致）：
  1. **归一化分母**用的是**有偏**方差（除 $N$）；
  2. **running_var 更新**用的却是**无偏**方差（除 $N-1$，即 $\hat\sigma^2 = \frac{n}{n-1}\sigma_B^2$，$n$ 为每通道的元素个数）。
- 不得原地修改传入的 running 张量，返回更新后的新值。

## 示例

```
输入：x = [[1,2],[3,4]] (N=2,C=2), γ=[1,1], β=[0,0],
      running_mean=[0,0], running_var=[1,1], training=True, momentum=0.1
batch 统计：μ=[2,3]，有偏方差=[1,1]，无偏方差=[2,2]
输出：y=[[≈-1.0, ≈-1.0],[≈1.0, ≈1.0]]        # 实为 ±0.99999（eps=1e-5）
      running_mean_new=[0.2, 0.3]
      running_var_new=[1.1, 1.1]              # 0.9*1 + 0.1*2（无偏方差！）
```

## 约束

- float32/float64 输入；每通道统计元素数 $n \ge 2$
- 结果与 `torch.nn.functional.batch_norm` 一致
