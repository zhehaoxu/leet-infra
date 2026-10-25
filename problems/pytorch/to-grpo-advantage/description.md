# 对齐训练：GRPO 组内标准化优势

## 题目描述

**GRPO**（Group Relative Policy Optimization, Shao et al., arXiv:2402.03300）省掉 PPO 的价值网络：对同一个 prompt 采样一组 \(G\) 个回复、各得一个 reward，用**组内**统计量当 baseline，把 reward 标准化成优势：

\[ A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)} \]

每个组独立归一化（组均值/组标准差），因此优势天然零均值、单位方差，无需 critic。标准化后的优势再喂给 PPO-clip 目标。

## 规则

输入扁平的 `rewards`（\((N,)\)，\(N\) 是 \(G\) 的整数倍，按连续 \(G\) 个分为一组）与整数 `group_size`（\(G\)）。对每组做 \((r-\text{mean})/\text{std}\)（标准差用总体公式 ddof=0），返回同形状 \((N,)\) 的 float32 `advantages`。禁止现成 RL 库。

## 示例

`rewards=[1.0,2.0,3.0,0.0,0.0,6.0]`、`group_size=3`（两组）：

```
组0 [1,2,3]: mean=2, std=sqrt((1+0+1)/3)=0.816497 → [-1.224745, 0, 1.224745]
组1 [0,0,6]: mean=2, std=sqrt((4+4+16)/3)=2.828427 → [-0.707107, -0.707107, 1.414214]
advantages = [-1.224745, 0.0, 1.224745, -0.707107, -0.707107, 1.414214]
```

## 约束

- CPU float32；`N % G == 0`；分组是**连续**的（reshape \((N/G, G)\)）。
- 标准差用总体公式（除以 \(G\)，非 \(G-1\)）。
- 每组均值、方差只在本组内统计，不跨组。
