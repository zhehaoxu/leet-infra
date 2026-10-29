# DAPO：clip-higher、动态采样与 token 级损失

## 题目描述

**DAPO**（Yu et al. 2025, *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476）是对 GRPO 的四处工程化改造，其中三处落在损失函数上：

1. **Clip-Higher**：把 PPO 的对称裁剪拆成 $\epsilon_\text{low}$ 与 $\epsilon_\text{high}$ 两个独立半径，通常取 $\epsilon_\text{high}>\epsilon_\text{low}$。对称裁剪会同时限制「把低概率 token 的概率抬上去」的幅度，导致策略熵过早塌缩；放宽上界能让少数关键 token 获得更大的提升空间。
2. **动态采样（组过滤）**：组内正确率 $acc_b$ 为 0 或 1 的组，组内优势全为 0，对梯度毫无贡献却照样消耗一次前向。DAPO 直接把这些组从损失里剔除，只保留 $0<acc_b<1$ 的组。
3. **Token 级损失**：不再「每条回答先平均、再对回答平均」，而是把**所有保留组的有效 token 汇成一个池**，做一次全局平均。这样长回答里的每个 token 与短回答等权，不再被长度重新加权。

本题给定逐 token 对数概率、掩码与 0/1 正确性奖励，要求按上述三步算出损失。

## 规则

给定 $\text{logp}/\text{logp\_old}$ 形状 $(B,G,T)$、掩码同形 bool、奖励 $(B,G)$ 且元素严格为 0/1、$\epsilon_\text{low}\in[0,1)$、$\epsilon_\text{high}\ge0$：

1. **组内优势**（GRPO 口径）：$A[b,i]=\big(r[b,i]-\bar r_b\big)/\mathrm{std}_b$，std 用总体标准差；该组 std 为 0 时整组 $A=0$。
2. **组过滤**：$acc_b=\bar r_b$，保留 $0<acc_b<1$ 的组；输出 `keep`（$(B,)$ bool）。
3. **非对称裁剪**：$\rho=\exp(\text{logp}-\text{logp\_old})$，$\bar\rho=\mathrm{clamp}(\rho,1-\epsilon_\text{low},1+\epsilon_\text{high})$，surrogate $=\min(\rho A,\ \bar\rho A)$。
4. **全局 token 平均**：$L=-\dfrac{\sum_{b,i,t}\text{valid}\cdot\text{surrogate}}{N_\text{valid}}$，其中 $\text{valid}=\text{mask}\wedge\text{keep}$；若 $N_\text{valid}=0$ 则 $L=0$（不返回 NaN）。`num_tokens` 就是 $N_\text{valid}$。

返回 `loss`（0 维 float64）、`adv`（$(B,G)$ float64）、`keep`（$(B,)$ bool）、`num_tokens`（0 维 int64）。

## 示例

```
rewards = [[1.0, 0.0],      # acc = 0.5 -> 保留
           [1.0, 1.0]]      # acc = 1.0 -> 丢弃
logp = [[[ 0.20, -0.10,  0.05], [ 0.00,  0.10, -0.20]],
        [[ 0.05,  0.05,  0.05], [-0.05,  0.00,  0.10]]]
mask = [[[1,1,1], [1,0,1]],
        [[1,1,1], [1,1,1]]]
ε_low = 0.2, ε_high = 0.28

A = [[ 1.0, -1.0], [0.0, 0.0]]
keep = [True, False]      num_tokens = 5
第 0 组：正优势回答的 ρ = [1.221403, 0.904837, 1.051271]，均在 [0.8, 1.28] 内，sum = 3.177511
        负优势回答（1 个 padding）ρ = [1.0, ·, 0.818731]，surrogate = [-1.0, ·, -0.818731]，sum = -1.818731
L = -(3.177511 - 1.818731) / 5 = -0.271756
```

第 1 组虽然也有 6 个有效 token，但它全对、优势为 0，整组被过滤，`num_tokens` 只计 5。

## 约束

- 返回四个键固定；`adv` float64、`keep` bool、`num_tokens` int64、`loss` 0 维 float64。
- 全对 / 全错组必须被丢弃；若整批都被丢弃，损失为 0 且 `num_tokens=0`，不得出现 NaN。
- 上界放宽必须让正优势样本的 surrogate 更大（损失更负）；负优势侧按 $\min$ 保留更负的未裁剪项。
- 非法输入抛 `ValueError`：非张量 / 非浮点 / 含 NaN·Inf / 形状不符 / $G<2$ / $T<1$ / 奖励非 0/1 / mask 非 bool 或形状不符 / 有空回答 / $\epsilon_\text{low}\notin[0,1)$ / $\epsilon_\text{high}<0$。
- CPU、确定性、不调用随机数。
