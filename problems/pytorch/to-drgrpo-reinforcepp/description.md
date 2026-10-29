# Dr. GRPO 与 REINFORCE++：同一批 rollout 的两种归一化

## 题目描述

GRPO 一族方法在「同 prompt 采一组回答、组内算相对优势」这一点上一致，分歧全在**归一化**。本题要求在同一批 rollout 上同时算出两套优势与两套损失，把差异摆在同一张表里：

- **Dr. GRPO**（Liu et al. 2025, *Understanding R1-Zero-Like Training: A Critical Perspective*, arXiv:2503.20783）指出 GRPO 有两个偏置：按每条回答长度归一化带来**长度偏置**（长回答的每个 token 权重更小），按组内 std 归一化带来**难度偏置**（简单/困难组的梯度被放大）。它的修法是两项都去掉：优势只做组内中心化，损失对所有 token 用一个常数分母 $B\cdot G\cdot T$（$T$ 是张量宽度 $L_\text{max}$，不是每条回答的有效长度）。
- **REINFORCE++**（Hu 2025, *REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models*, arXiv:2501.03262）保留白化，但把统计量放到**整个 batch** 上（全局均值 + 全局总体标准差），并且每条回答先按**自身有效 token 数**取均值、再对回答取均值。

两者都用对称 PPO 裁剪 $\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)$ 与 $\min(\rho A,\ \mathrm{clip}\cdot A)$ 的 surrogate，差别只在归一化的位置与统计范围。

## 规则

给定奖励 $(B,G)$、当前/行为策略的逐 token 对数概率 $(B,G,T)$、有效 token 掩码同形 bool，$\epsilon\in(0,1)$：

1. **优势**
   - $A_\text{dr}[b,i]=r[b,i]-\frac{1}{G}\sum_j r[b,j]$（组内中心化，不除 std）；
   - $\mu,\sigma$ 为全体 $B\cdot G$ 个奖励的均值与**总体**标准差（`unbiased=False`），
     $A_\text{rp}[b,i]=(r[b,i]-\mu)/\sigma$；若 $\sigma\le10^{-12}$ 则整批优势取 0。
2. **逐 token 重要性比** $\rho=\exp(\text{logp}-\text{logp\_old})$，裁剪 $\bar\rho=\mathrm{clamp}(\rho,1-\epsilon,1+\epsilon)$，surrogate $=\min(\rho A,\ \bar\rho A)$。
3. **Dr. GRPO 损失**：$L_\text{dr}=-\dfrac{\sum_{b,i,t}\text{mask}\cdot\text{surrogate}}{B\cdot G\cdot T}$。分母含 padding 位置，padding 会稀释损失。
4. **REINFORCE++ 损失**：$L_\text{rp}=-\dfrac{1}{BG}\sum_{b,i}\dfrac{\sum_t\text{mask}\cdot\text{surrogate}}{|o_{b,i}|}$。分母是每条回答的有效长度，padding 不影响结果。

返回 `adv_drgrpo`、`adv_reinforcepp`（各 $(B,G)$ float64）与 `loss_drgrpo`、`loss_reinforcepp`（各 0 维 float64）。

## 示例

```
rewards = [[1.0, 0.0]]          G = 2
logp    = [[[ 0.30, -0.30,  0.00],
            [ 0.05,  0.00, -0.05]]]     logp_old = 全 0，mask 全 True
ε = 0.2

A_dr = [[ 0.5, -0.5]]        A_rp = [[ 1.0, -1.0]]
ρ    = [[[1.349859, 0.740818, 1.0], [1.051271, 1.0, 0.951229]]]
surrogate（Dr.，A=±0.5）= [[0.6, 0.370409, 0.5], [-0.525636, -0.5, -0.475615]]
L_dr = -(-0.030842) / (1·2·3) = 0.005140
L_rp = -mean( 1.470409/3, -1.501251/3 ) = 0.010280
```

REINFORCE++ 的损失恰是 Dr. GRPO 的两倍：全局 std 为 0.5，白化把优势放大了 2 倍，而两条回答长度相同、除数一致。

## 约束

- 返回四个量全部 float64；float32 输入内部升到 float64。`adv_*` 形状 $(B,G)$，`loss_*` 是 0 维。
- 追加一列全 mask 的 padding：$L_\text{dr}$ 缩小到原来的 $3/4$，$L_\text{rp}$ 完全不变——这是两种归一化最直接的可观测差异。
- 全相同奖励时 $\sigma=0$，REINFORCE++ 优势与两项损失都为 0（容差 $10^{-12}$，不做除零）。
- 非法输入抛 `ValueError`：非张量 / 非浮点 / 含 NaN·Inf / 形状不符 / $G<2$ / $T<1$ / mask 非 bool 或形状不符 / 有空回答 / $\epsilon\notin(0,1)$。
- CPU、确定性、不调用随机数。
