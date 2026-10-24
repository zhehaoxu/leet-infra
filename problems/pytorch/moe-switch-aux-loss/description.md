# MoE 路由：Switch 负载均衡辅助损失

## 题目描述

MoE 路由器容易「塌方」到少数几个热门专家。Switch Transformer 提出一个**可微**的辅助损失，同时压两个方向：
- 让「被实际分派到的比例 $f_e$」均匀；
- 让「路由器输出的平均概率 $P_e$」均匀。

$$ \mathcal{L}_{\text{aux}} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e $$

## 规则

实现 `solve(logits, k)`，`logits` 形状 $(T, E)$：

1. `probs = softmax(logits, dim=-1)`。
2. 每行取 `probs` 最大的 $k$ 个 → `selected (T, k)` long。
3. $f_e = \dfrac{\text{count}(e \in \text{selected})}{T}$（**注意分母是 $T$ 而不是 $T \cdot k$**，因此 $\sum_e f_e = k$）。
4. $P_e = \dfrac{1}{T}\sum_t \text{probs}[t,e]$（$\sum_e P_e = 1$）。
5. $\text{aux} = E \cdot \sum_e f_e P_e$。

返回 float 标量。

> 与 HF `transformers.models.mixtral.load_balancing_loss_func` 一致（attention_mask=None 分支）。

## 示例

```
logits = [[2,0,0,0],
          [0,2,0,0],
          [0,0,2,0],
          [0,0,0,2],
          [1,1,0,0],
          [0,1,1,0]]，k=1

aux = 1.025673
```

极端对照：
- 全 4 个 token 都选同一专家：aux ≈ **3.48**；
- 每专家各分到 1 个 token、每行 softmax 尖峰：aux = **1.0**（下界）。

## 约束

- 全部 CPU float32。
- tie 情况按较小专家索引优先（与 `torch.topk` 一致）。
