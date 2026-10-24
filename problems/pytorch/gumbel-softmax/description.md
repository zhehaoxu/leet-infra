# Gumbel-Softmax 采样（重参数化）

## 题目描述

要从类别分布 $\text{softmax}(\text{logits})$ 里**可导地**采样，直接 `argmax`/`multinomial`
是不可导的离散操作。Gumbel-Softmax 用重参数化把随机性挪到一外部噪声上：给 logits 加上
标准 Gumbel 噪声再过温度 softmax，得到一个「软 one-hot」样本，梯度能流过它。VAE 离散潜变量、
神经架构搜索(NAS)、硬注意力都用它。

> 为保证判题**确定性**，本题把均匀随机数 `u` 作为显式输入传入，而不是在函数里采样。

## 规则

- 实现 `solve(logits, u, tau=1.0)`，`logits` 形状 `(..., K)`，`u` 为同形状、取值 $(0,1)$ 的
  均匀随机数。
- Gumbel 变换：$g = -\log(-\log(u))$（标准 Gumbel$(0,1)$ 噪声）。
- 输出：$y = \text{softmax}((\text{logits} + g)/\tau)$，沿最后一维。
- $\tau\to 0$ 时逼近 one-hot（等价 Gumbel-max 采样），$\tau$ 大时更平滑。

## 示例

```
输入：logits = [2, 1, 0.5, 0.1]
     u = [0.5, 0.9, 0.1, 0.3]，tau = 1
  g = -log(-log(u)) = [0.36651, 2.25037, -0.83403, -0.18563]
  logits+g = [2.367, 3.250, -0.334, -0.086] → /1 → softmax
输出：[0.2798, 0.6773, 0.0188, 0.0241]   （和为 1；第 1 类因噪声大而被抬高）
```

## 约束

- float32 输入；`u` 严格在 $(0,1)$ 内（勿传 0 或 1，会 log(0)）
- 输出每行和为 1；`tau` 极小时近似 one-hot
