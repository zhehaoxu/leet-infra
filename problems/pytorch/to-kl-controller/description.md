# 对齐训练：PPO 自适应 KL 系数控制器

## 题目描述

RLHF/PPO 里 KL 惩罚系数 \(\beta\) 不能用固定值：太大会把策略锁死在参考模型附近、学不动，太小又会让策略跑飞、语言质量崩塌。Ziegler et al.（arXiv:1909.08593）与 HuggingFace TRL 的 `AdaptiveKLController` 用一个**比例控制器**在线调节 \(\beta\)：

每个控制周期观测到的平均 KL \(d\) 与目标 \(d_\text{tgt}\) 比较，算出**比例误差**并截断到 \([-0.2, 0.2]\)：

\[ e = \mathrm{clip}\!\left(\frac{d}{d_\text{tgt}} - 1,\ -0.2,\ 0.2\right) \]

再按「本周期走了多少步 \(n\)、控制器horizon \(H\)」平滑地缩放 \(\beta\)：

\[ \beta' = \beta \cdot \left(1 + e\cdot\frac{n}{H}\right) \]

直觉：实际 KL **高于**目标（\(e>0\)）就**抬高** \(\beta\) 加强惩罚；**低于**目标（\(e<0\)）就**压低** \(\beta\) 放松约束。截断 \(\pm0.2\) 防止单步把系数调得太猛。

## 规则

给定标量 `beta`（当前 KL 系数）、`current_kl`（本周期实测平均 KL \(d\)）、`target_kl`（目标 \(d_\text{tgt}\)）、`horizon`（\(H>0\)）、`n_steps`（本周期步数 \(n\)）。按上式计算并返回新的 KL 系数 \(\beta'\)（0 维 float32 标量）。禁止现成 RLHF 库。

## 示例

`beta=0.1`、`current_kl=0.3`、`target_kl=0.2`、`horizon=1000`、`n_steps=1`：

```
e = clip(0.3/0.2 - 1, -0.2, 0.2) = clip(0.5, -0.2, 0.2) = 0.2
beta' = 0.1 * (1 + 0.2 * 1/1000) = 0.1 * 1.0002 = 0.100020
```

实测 KL 高于目标 → 系数被抬高。

## 约束

- CPU float32；`target_kl>0`、`horizon>0`。
- \(d=d_\text{tgt}\) 时 \(e=0\)、\(\beta\) 不变。
- \(|d/d_\text{tgt}-1|\ge 0.2\) 时误差被截断（饱和），更大的偏差不再改变结果。
