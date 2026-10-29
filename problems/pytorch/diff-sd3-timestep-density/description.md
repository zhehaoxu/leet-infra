# 扩散训练：SD3 的时间步采样密度

## 题目描述

SD3 训练不按均匀分布抽时间步。它先抽一个连续位置 \(\sigma\in[0,1]\)（再乘 `num_train_timesteps` 得到整数时间步），`compute_density_for_timestep_sampling` 用三种 scheme 决定这个位置怎么来：

- `logit_normal`：\(z\) 服从标准正态，位置为 \(\sigma(\mu+\varsigma z)\)。\(\mu\) 控制整体偏向（负值偏噪声侧），\(\varsigma\) 控制集中程度。
- `mode`：\(u\) 服从 \(U(0,1)\)，位置为 \(1-u-s\left(\cos^2\frac{\pi u}{2}-1+u\right)\)，\(s\) 是 `mode_scale`。\(s>0\) 时它把均匀密度压成两端高、中间低的形状；SD3 训练脚本默认 \(s=1.29\)。
- `uniform`：位置就是 \(u\) 本身。

本题要求给定 scheme 与**已经抽好的样本**，算出对应的连续时间位置。随机数由调用方显式提供，题目本身不含随机性。

## 规则

逐元素计算：

| scheme | 输入样本 | 输出位置 |
| --- | --- | --- |
| `logit_normal` | \(z\sim N(\mu,\varsigma^2)\) | \(1/(1+e^{-(\mu+\varsigma z)})\) |
| `mode` | \(u\sim U(0,1)\) | \(1-u-s\left(\cos^2\frac{\pi u}{2}-1+u\right)\) |
| `uniform` | \(u\sim U(0,1)\) | \(u\) |

返回与 `samples` 同形状的一维 float32 张量。

## 示例

`logit_normal`，`z = [-1, 0, 1]`，`logit_mean = 0`，`logit_std = 1`：

```
[sigmoid(-1), sigmoid(0), sigmoid(1)] = [0.268941, 0.5, 0.731059]
```

`mode`，`u = [0, 0.25, 0.5, 0.75, 1]`，`mode_scale = 1.29`：

```
[1.0, 0.616416, 0.5, 0.383584, 0.0]
```

## 约束

- `scheme` 必须是 `logit_normal`、`mode`、`uniform` 之一（源实现对未知字符串静默回落到 uniform，本题改为抛 `ValueError`）。
- `samples` 必须是一维 float 张量且不含 nan / inf；`mode` 与 `uniform` 还要求取值落在 \([0,1]\)。
- `logit_normal` 必须同时给出有限的 `logit_mean` 与 `logit_std >= 0`；`mode` 必须给出有限的 `mode_scale >= 0`。
- 违反以上任一条时抛 `ValueError`。
- 退化性质：`mode` 在 \(u=0\) 处精确为 1、\(u=1\) 处精确为 0、\(u=0.5\) 处精确为 0.5，且关于中心对称 \(v(u)+v(1-u)=1\)；`mode_scale=0` 时退化为 \(1-u\)。`logit_normal` 在 \(z=0\) 处等于 \(\sigma(\mu)\)，`logit_std=0` 时整批为同一常数。
