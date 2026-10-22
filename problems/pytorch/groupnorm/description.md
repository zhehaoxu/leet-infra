# GroupNorm

## 题目描述

GroupNorm = BatchNorm 与 InstanceNorm 的折中：把 $C$ 个通道分成 $G$ 组，**组内**（跨该组所有通道与全部空间位置）做统计归一化，与 batch 无关，小 batch 场景稳定：

$$
y = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}\,\gamma_c + \beta_c,\qquad
\mu_g,\sigma_g^2 = \text{统计第 } g \text{ 组的 } (C/G)\times H\times W \text{ 个元素}
$$

$G=1$ 退化为逐样本 LayerNorm，$G=C$ 退化为 InstanceNorm。

## 规则

- 实现 `solve(x, weight, bias, num_groups, eps=1e-5)`，x 形状 `(N, C)` 或 `(N, C, *)`，`num_groups` 整除 $C$。
- 手写思路：`reshape` 成 $(N, G, C/G, M)$，沿 $(2,3)$ 两维 `keepdim` 统计，归一化后 `reshape` 回原形，再按通道乘 $\gamma$ 加 $\beta$。
- 方差用**有偏**口径（与 `F.group_norm` 一致）。
- $\gamma,\beta$ 形状 $(C,)$，是**逐通道**的（组内不同通道可以有不同缩放）。

## 示例

```
输入：x 形状 (1,4,1,4)，通道值依次为 [1,2,3,4]、[5,6,7,8]、[9,10,11,12]、[13,14,15,16]
      γ=[1,1,1,1], β=[0,0,0,0], G=2
第 0 组（通道 0,1 共 8 个值 1..8）：μ=4.5，σ²(有偏)=5.25
输出通道 0：[[-1.52752376, -1.09108841, -0.65465307, -0.21821769]]
输出通道 1：[[ 0.21821769,  0.65465307,  1.09108841,  1.52752376]]
通道 2,3（第 1 组，值 9..16）与通道 0,1 的输出完全相同
```

## 约束

- float32/float64；`num_groups` ∈ [1, C] 且整除 C
- 结果与 `torch.nn.functional.group_norm` 一致
