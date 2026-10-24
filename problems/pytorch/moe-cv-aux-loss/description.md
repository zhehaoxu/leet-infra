# MoE 路由：变异系数（CV）辅助损失

## 题目描述

Shazeer「Outrageously Large Neural Networks」提出的另一种负载均衡惩罚：直接惩罚**重要度**与**负载**的**变异系数**（Coefficient of Variation, CV）平方。CV 无量纲，天然对专家个数与 token 数不变，方便与主 loss 相加。

$$ \mathrm{CV}(x) = \frac{\mathrm{std}(x)}{\mathrm{mean}(x)}\,,\qquad
\mathcal{L}_{\text{cv}} = \mathrm{CV}(\text{importance})^2 + \mathrm{CV}(\text{load})^2 $$

## 规则

实现 `solve(logits, k)`，`logits` 形状 $(T, E)$：

1. `probs = softmax(logits, -1)`。
2. `importance_e = Σ_t probs[t, e]`（软，可微）。
3. `selected = topk(probs, k, -1).indices`；`load_e = count(e ∈ selected)`（硬）。
4. $\mathrm{CV}(x) = x.\mathrm{std}(\text{unbiased}=\mathrm{False}) / (x.\mathrm{mean}() + 10^{-6})$。
5. 返回标量 $\mathrm{CV}(\text{importance})^2 + \mathrm{CV}(\text{load})^2$。

> 分母 eps 固定 $10^{-6}$；std 用**有偏估计**（除以 $E$ 而非 $E-1$）。

## 示例

```
logits = [[2,0,0,0], [0,2,0,0], [0,0,2,0], [0,0,0,2],
          [1,1,0,0], [0,1,1,0]]，k=1
cv_aux = 0.122975
```

极端对照：所有 4 个 token 都路由到同一专家 → cv_aux ≈ **5.05**。

## 约束

- CPU float32；tie 按较小专家索引优先。
