# 扩散训练：Min-SNR 损失加权

## 题目描述

扩散训练里不同时间步的信噪比差好几个数量级：干净端 \(\bar\alpha_t\to1\) 时 SNR 很大，模型几乎在抄输入，梯度信号却主导总损失。Min-SNR 策略把高 SNR 时间步的损失权重大幅压低、低 SNR 时间步保持原样：

\[ w_t=\frac{\min(\mathrm{SNR}_t,\gamma)}{\mathrm{SNR}_t}=\min\!\left(1,\frac{\gamma}{\mathrm{SNR}_t}\right) \]

其中 \(\mathrm{SNR}_t=\bar\alpha_t/(1-\bar\alpha_t)\)，\(\gamma\) 是 SNR 的上限（SD3 等训练脚本常用 5.0）。注意被截断的是 **SNR**，不是权重：权重恒不超过 1，只在 SNR 超过 \(\gamma\) 时按 \(\gamma/\mathrm{SNR}\) 衰减。

本题要求按调度的累积 alpha 与一批时间步，算出 SNR 和对应的 Min-SNR 权重。

## 规则

给定一维 `alphas_cumprod`（长度 \(T\)）与一维整数 `timesteps`（下标）：

1. 取 \(\alpha=\sqrt{\bar\alpha_t}\)、\(\sigma=\sqrt{1-\bar\alpha_t}\)；
2. \(\mathrm{SNR}=(\alpha/\sigma)^2\)（等于 \(\bar\alpha_t/(1-\bar\alpha_t)\)，但按源码的平方路径计算）；
3. \(w=\min(\mathrm{SNR},\gamma)/\mathrm{SNR}\)。

返回 `(snr, weight)`，二者都是与 `timesteps` 同形状的 float32 张量。

## 示例

`alphas_cumprod = [0.75, 0.5, 0.25, 0.125]`、`timesteps = [0, 1, 2, 3]`、`gamma = 2.0`：

```
snr    = [3.0, 1.0, 0.333333, 0.142857]
weight = [0.666667, 1.0, 1.0, 1.0]
```

前三个时间步的 SNR 都不超过 2，权重保持 1；只有 \(\bar\alpha=0.75\)（SNR=3）被压到 \(2/3\)。

## 约束

- `alphas_cumprod` 必须是一维浮点张量、不含 nan / inf，且每个元素**严格**落在 \((0,1)\) 内；端点 0 或 1 会让 SNR 变成 0 或无穷。
- `timesteps` 必须是一维 `int32` / `int64` 张量，取值在 \([0,T)\) 内；允许重复与空张量。
- `gamma` 必须是正的有限实数（布尔值不算）。
- 违反以上任一条时抛 `ValueError`。
- 性质：\(w\in(0,1]\)；\(\mathrm{SNR}\le\gamma\) 时 \(w=1\)，\(\mathrm{SNR}>\gamma\) 时 \(w=\gamma/\mathrm{SNR}\)；\(\gamma\) 增大时权重逐点不减；沿调度从干净到噪声（\(\bar\alpha\) 递减）SNR 单调递减、权重单调不减。
