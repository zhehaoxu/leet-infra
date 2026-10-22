# QK 归一化注意力（QK-Norm Attention Score）

## 题目描述

QK-Norm 在点积之前先对 Query、Key 各做 **RMSNorm**，把每个向量的尺度固定下来，再用 $1/\sqrt{d}$ 缩放。这样注意力 logits 有界，不会因为向量范数膨胀而爆炸（ViT-22B、Chameleon 等采用）。

$$ \hat{q} = \frac{q}{\sqrt{\mean(q^2)+\epsilon}}\cdot w_q,\qquad
\hat{k} = \frac{k}{\sqrt{\mean(k^2)+\epsilon}}\cdot w_k,\qquad
\text{score} = \frac{\hat{q}\,\hat{k}^{\top}}{\sqrt{d}} $$

## 规则

- 实现 `solve(q, k, weight_q, weight_k, eps=1e-6)`，`q/k` 形状 `(..., L, d)`，返回 `(..., L_q, L_k)` 的分数矩阵。
- RMSNorm 沿**最后一维**做，且**不减均值**（区别于 LayerNorm）：除以 $\sqrt{\mean(x^2)+\epsilon}$，$\epsilon$ 在开方**内部**。`weight_q`、`weight_k` 分别广播到最后一维，可不对称。
- 归一化后再做点积并除以 $\sqrt{d}$。
- **尺度不变性**：把 $q,k$ 整体乘以任意正常数，分数不变（这是本关核心卖点，会被测试检出）。
- weight 全 1 时每个向量 RMS 归一后行范数为 $\sqrt{d}$，由柯西-施瓦茨 $|\text{score}| \le \sqrt{d}$。

## 示例

```
输入：q=[[[1,1]]], k=[[[1,1]]], weight_q=weight_k=[1,1], eps=1e-6, d=2
   q̂ = (1,1)/sqrt(mean([1,1]^2)+eps) = (1,1)/1.0 ≈ (1,1)   # mean(q²)=1
   score = (1*1+1*1)/sqrt(2) = 2/√2
输出：[[[1.4142122]]]   # ≈ √2
```

## 约束

- float32/float64 输入
- 与手推 `rmsnorm(q,wq) @ rmsnorm(k,wk)^T / sqrt(d)` 一致
