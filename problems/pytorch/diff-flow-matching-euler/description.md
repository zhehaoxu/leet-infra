# Diffusion：Flow Matching / Rectified Flow 的 Euler 积分

## 题目描述

Flow Matching / Rectified Flow 不再预测噪声，而是学一个**速度场** \(v_\theta(x,t)\)，并把「噪声 → 数据」写成一条从 \(t=1\) 到 \(t=0\) 的常微分方程

\[ \frac{dx}{dt}=v_\theta(x,t) \]

RF 的理想路径是直线：\(x_t=(1-t)x_0+t\epsilon\)，与之配套的速度是常数 \(v=\epsilon-x_0\)。采样就是对这个速度场做显式 Euler：

\[ t_k=t_{\text{start}}+k\cdot\frac{t_{\text{end}}-t_{\text{start}}}{n},\qquad x_{k+1}=x_k+(t_{k+1}-t_k)\,v_\theta(x_k,t_k) \]

注意一阶 Euler 对常速度是**精确**的（直线路径任意步数都落在直线上），这一点与 DDPM 的随机差分完全不同，也是 RF 能用极少步采样的原因。

## 规则

给定 `x_init`（\(t=t_{\text{start}}\) 处的状态）、`velocity_fn(x, t)`、步数 `n_steps`、起止时间 `t_start` / `t_end`（要求 `t_end < t_start`），返回 `(trajectory, x_final)`：`trajectory` 是含起点在内的 `n_steps + 1` 个 float32 张量，`x_final` 即末元素。速度场可以返回同形状张量，也可以返回标量（按广播处理）。

## 示例

\[ x_{\text{init}}=1.0,\ v(x,t)=-x,\ n=4,\ t:\ 1\to0\quad(\Delta t=-0.25) \]

```
trajectory = [1.0, 1.25, 1.5625, 1.953125, 2.44140625]
x_final    = 2.44140625
```

## 约束

- `n_steps` 必须是正整数，`t_start` / `t_end` 必须是有限实数且 `t_end < t_start`；`velocity_fn` 必须可调用。
- 速度场返回的张量形状必须与输入一致，否则抛 `ValueError`。
- **退化性质**：常速度 \(v\equiv c\) 时 \(x_{\text{final}}=x_{\text{init}}+(t_{\text{end}}-t_{\text{start}})c\)；RF 直线路径下 \(x_1=\epsilon\) 出发、一步即可精确还原 \(x_0\)，多步也一样精确。
- 轨迹首元素必须仍是起点（实现不能原地改写已保存的状态）。
