# MoE 路由：z-loss 稳定性正则

## 题目描述

PaLM / ST-MoE 提出：路由器 logits 一旦爆炸（尤其 bf16/fp16），softmax 就会饱和、梯度消失。**z-loss** 惩罚 $\log(\sum_e e^{x_e})$ 的平方，把 logits 的绝对大小压回原点。

$$ \mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \big[\log(\sum_e e^{x_{t,e}})\big]^2 = \frac{1}{T}\sum_t \big[\mathrm{logsumexp}_e(x_{t,e})\big]^2 $$

## 规则

实现 `solve(logits)`，`logits` 形状 $(T, E)$：返回标量 $\mathrm{mean}(\mathrm{logsumexp}(\text{logits}, -1)^2)$。

## 示例

```
logits = [[1, 2, 0, -1],
          [0.5, 0.5, 0.5, 0.5],
          [3, 0, 0, 0]]

每行 logsumexp: 2.440190, 1.886294, 3.139206
平方: 5.954525, 3.558106, 9.854615
z_loss = 6.455749
```

## 约束

- CPU float32；不乘任何系数（返回原始 mean）。
