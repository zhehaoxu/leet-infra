# KL 散度（前向与反向）

## 题目描述

KL 散度（Kullback–Leibler divergence）衡量用分布 \(Q\) 去近似真实分布 \(P\) 所付出的额外信息量，它**不对称**：\(\mathrm{KL}(P\|Q)\ne\mathrm{KL}(Q\|P)\)。前向 KL（mean-seeking）在变分推断/MLE 里常见，反向 KL（mode-seeking）在知识蒸馏里更偏爱——这正是为什么蒸馏损失要写成 \(\mathrm{KL}(\text{teacher}\|\text{student})\)。

\[ \mathrm{KL}(P\|Q) = \sum_i P_i \log\frac{P_i}{Q_i},\qquad 0\log 0 := 0 \]

## 规则

实现 `solve(P, Q)`，返回一个二元组 `(forward_kl, reverse_kl)`，其中 `forward_kl = KL(P‖Q)`、`reverse_kl = KL(Q‖P)`。`P`、`Q` 为 float32、形状 \((N, C)\) 的**逐行归一化**概率分布（每行和为 1）。对每一行按定义 \(\sum_c P_{c}\log(P_c/Q_c)\) 求和（约定 \(0\log 0=0\)、且当 \(P_c>0\) 时 \(Q_c>0\)），再对 \(N\) 行取**均值**，得到一个 0 维标量。两个返回值都是标量张量。

## 示例

\(N=2, C=2\)，`P=[[0.7,0.3],[0.2,0.8]]`，`Q=[[0.4,0.6],[0.5,0.5]]`：

```
forward_kl = KL(P‖Q) = mean([0.7·ln(0.7/0.4)+0.3·ln(0.3/0.6),
                             0.2·ln(0.2/0.5)+0.8·ln(0.8/0.5)]) = 0.18826580
reverse_kl = KL(Q‖P) = 0.20759282
```

## 约束

- CPU float32；`P`、`Q` 每行均为归一化概率分布。
- 约定 \(0\log 0 = 0\)（即 \(P_c=0\) 的项贡献 0）。
- 返回值必须是 `(forward_kl, reverse_kl)` 两个 0 维张量；二者一般不相等（非对称）。
- 当 `P` 与 `Q` 完全相同，两个返回值都应约等于 0。
