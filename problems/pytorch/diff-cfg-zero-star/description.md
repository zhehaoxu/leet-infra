# Diffusion：CFG-Zero* 的零初始化与最优尺度

## 题目描述

Flow-matching 采样的最初几步里，模型输出还很不靠谱：两支预测都偏离真实速度场，此时做强引导只会把误差放大并一路带到后续所有步。**CFG-Zero\***（arXiv:2503.18886）用两个改动修这个问题：

1. **零初始化**：前 \(k\) 步直接把引导结果置零（采样器对这些步的更新量为零）；
2. **最优尺度**：之后不再用固定的无条件预测 \(\epsilon_u\) 做外推基准，而是先求一个标量 \(\alpha\)，让 \(\alpha\epsilon_u\) 在最小二乘意义下最接近 \(\epsilon_c\)，再以 \(\alpha\epsilon_u\) 为基准做 CFG 外推。

## 规则

给定同形状的 `eps_cond` / `eps_uncond`、`guidance_scale=s`、当前步 `step`、零初始化步数 `zero_init_steps=k` 与稳定项 `eps`：

- 若 `step < k`，直接返回与输入同形状的全零 float32 张量；
- 否则按样本求标量
  \[ \alpha=\frac{\langle\epsilon_c,\epsilon_u\rangle}{\lVert\epsilon_u\rVert_2^2+\varepsilon} \]
  令 \(\tilde{\epsilon}_u=\alpha\epsilon_u\)，返回
  \[ \text{out}=\tilde{\epsilon}_u+s\left(\epsilon_c-\tilde{\epsilon}_u\right) \]

内积与平方和都在除第 0 维外的全部维度上归约（每个样本一个标量）。

## 示例

\[ \epsilon_c=\begin{bmatrix}0.4&-1.2&0.6\\0.0&0.5&-0.5\end{bmatrix},\ \epsilon_u=\begin{bmatrix}0.1&-0.4&0.2\\0.2&-0.1&0.0\end{bmatrix},\ s=3,\ k=1 \]

```
step=0 -> [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]           （零初始化窗口内）
step=3 -> [[ 0.590476, -1.161905,  0.580952], [ 0.400000,  1.300000, -1.500000]]
```

## 约束

- `step` 与 `zero_init_steps` 必须是非负整数（布尔值不算整数）；`s` 非负有限，`eps` 为正有限实数。
- 输入至少二维，两支形状必须一致。
- **退化性质**：\(s=1\) 时输出恰为 `eps_cond`（外推端点回到条件支）；两支正交时 \(\alpha=0\)，输出退化为 \(s\epsilon_c\)；两支同向时 \(\alpha\approx 1\)，输出回到 `eps_cond`。
- \(\alpha\) 可以是负数（两支反向时），不得裁剪到非负。
- \(\alpha\) 使残差 \(\epsilon_c-\alpha\epsilon_u\) 与 \(\epsilon_u\) 正交——这是最小二乘口径，不是余弦相似度。
- \(\varepsilon\) 加在分母的平方和上，因此它只会让 \(\lvert\alpha\rvert\) 变小，不会参与符号。
