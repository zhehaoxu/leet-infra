# 稀疏化：三次渐稀疏调度 (GSL)

## 题目描述

一次性把网络剪到目标稀疏率会破坏训练稳定性。**渐稀疏率调度（Gradual Sparsity / GSL）** 由 Zhu & Gupta（*To Prune, or Not to Prune*, arXiv:1710.01878）提出：在一段训练窗口内，用**三次函数**把稀疏率从初值 \(p_i\) 平滑地推到终值 \(p_f\)，越早剪得越猛、后期趋稳：

\[ p_t = p_f + (p_i - p_f)\left(1 - \frac{t - t_0}{t_1 - t_0}\right)^3 \]

其中 \(t\) 为当前步、\([t_0,t_1]\) 为剪枝窗口。窗口之前保持 \(p_i\)、窗口之后恒为 \(p_f\)。

## 规则

实现 `solve(initial_sparsity, final_sparsity, current_step, first_step, final_step)`，返回当前步的目标稀疏率 \(p_t\)（0 维 float32 标量）：
- 若 \(\text{current\_step} \le t_0\)：返回 \(p_i\)；
- 若 \(\text{current\_step} \ge t_1\)：返回 \(p_f\)；
- 否则按上式三次插值。

禁止现成调度库。

## 示例

`initial_sparsity=0.0`、`final_sparsity=0.8`、`first_step=0`、`final_step=1000`：

```
current_step=500 → frac=0.5, (1-0.5)^3=0.125, p=0.8+(0-0.8)*0.125 = 0.7
current_step=0   → p_i = 0.0
current_step=1000→ p_f = 0.8
```

## 约束

- CPU float32；\(t_1 > t_0\)。
- 三次项随 \(t\) 从 \(t_0\to t_1\) 单调衰减，稀疏率单调趋近 \(p_f\)。
- 边界闭区间：窗口两端点分别取 \(p_i\)、\(p_f\)。
