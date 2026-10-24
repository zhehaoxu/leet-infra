# 解题说明 · Switch 负载均衡辅助损失

## 为什么是 $f_e \cdot P_e$

- $f_e$ 是**不可微**的（argmax 出来的整数计数）；
- $P_e$ 是**可微**的（softmax 后的均值）。

两者的乘积对 $P_e$ 有梯度：如果某专家 $f_e$ 很大（大家都在选它），梯度就推着 $P_e$ 变小；反之 $f_e$ 小的专家会被推大。这是一个「用硬指标做权重、软指标做被优化对象」的经典技巧。

若只用 $P_e$ 单独惩罚（如熵最大化），梯度会均匀分摊给所有 token；若只用 $f_e$，不可微。乘积形式刚好把「实际负载」信号注入到「路由分布」的梯度上。

## 归一化细节

不同实现有细微差异：
- **HF Mixtral**：$f_e = \text{count}_e / T$（分母 $T$），故 $\sum_e f_e = k$；
- **Megatron-LM** `switch_load_balancing_loss_func`：$\tilde{f}_e = \text{count}_e / (T \cdot k)$（分母 $T \cdot k$），故 $\sum_e \tilde{f}_e = 1$。

两者相差 $k$ 倍。题面按 HF 版本；$k=1$（Switch 原论文场景）时二者一致。

## 下界分析

由 $\sum_e f_e = k$、$\sum_e P_e = 1$，柯西不等式给出：
$$ \sum_e f_e P_e \ge \frac{1}{E} \cdot k \cdot 1 = \frac{k}{E} $$
即 $E \cdot \sum_e f_e P_e \ge k$。$k=1$ 时下界为 1，题面示例吻合（完美均衡 aux=1.0）。

## 数值细节

`torch.bincount` 只接受 1D `long`；`selected.reshape(-1)` 展平。`minlength=E` 保证即使某专家被选 0 次也占一位。
