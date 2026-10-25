# 对齐训练：广义优势估计 GAE

## 题目描述

策略梯度需要「优势」 \(A_t\)——动作比状态价值好多少。单步 TD 误差方差小但偏差大、蒙特卡洛回报无偏但方差爆炸。**GAE**（Generalized Advantage Estimation, Schulman et al., arXiv:1506.02438）用一个混合参数 \(\lambda\in[0,1]\) 在两者间平滑插值。

定义单步 TD 误差
\[ \delta_t = r_t + \gamma\,V_{t+1} - V_t \]
则 GAE 优势是带 \((\gamma\lambda)\) 折扣的无限窗累加，且有反向递推：
\[ \hat A_t = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l} = \delta_t + \gamma\lambda\,\hat A_{t+1},\qquad \hat A_{T-1}=\delta_{T-1} \]
价值回归目标（return）为
\[ R_t = \hat A_t + V_t \]

轨迹末端之后视为终止：\(V_T=0\)（越界不 bootstrap）。

## 规则

输入 `rewards`（\((T,)\)）、`values`（\((T,)\)，即 \(V_0..V_{T-1}\)）、标量 `gamma`、`lam`。返回二元组 `(advantages, returns)`，均为形状 \((T,)\) 的 float32 张量。禁止现成 RL 库。

## 示例

`rewards=[1.0,0.0,1.0]`、`values=[0.5,0.7,0.3]`、`gamma=0.99`、`lam=0.95`：

```
delta = [1.0+0.99*0.7-0.5, 0.0+0.99*0.3-0.7, 1.0+0.99*0-0.3] = [1.193, -0.403, 0.7]
A_2 = 0.7
A_1 = -0.403 + 0.99*0.95*0.7 = 0.255350
A_0 = 1.193 + 0.99*0.95*0.255350 = 1.433157
advantages = [1.433157, 0.255350, 0.699999]
returns    = [1.933157, 0.955350, 1.0]
```

## 约束

- CPU float32；\(T\ge1\)；\(\gamma,\lambda\in[0,1]\)。
- 递推从 \(t=T-1\) 反向到 \(0\)，\(\hat A\) 初值（\(t=T\)）为 0，等价于 \(V_T=0\)。
- \(\lambda=0\) 退化为单步 TD \(\hat A_t=\delta_t\)；\(\lambda=1,\gamma=1\) 趋近蒙特卡洛 \(R_t=\sum_{l\ge t}r_l\)（本实现仍以 \(V_T=0\) 终止）。
