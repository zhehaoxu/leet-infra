# Diffusion：ε / x₀ / v 三种预测目标互转

## 题目描述

扩散模型可以让网络预测噪声 \(\epsilon\)、干净样本 \(x_0\)，或者二者在同一 \(\bar\alpha_t\) 下的「旋转组合」\(v\)（Salimans & Ho 的 v-prediction，SD 2.x / SDXL / Imagen 都在用）。三种预测目标之间是线性可逆的：令 \(\alpha=\sqrt{\bar\alpha_t}\)、\(\sigma=\sqrt{1-\bar\alpha_t}\)，前向过程与 v 的定义是

\[ x_t = \alpha x_0 + \sigma \epsilon, \qquad v = \alpha \epsilon - \sigma x_0 \]

于是已知任意一个量都能解出另外两个（三种情况分别对应三支公式，见 `notes.md`）。注意 \((x_0,\epsilon)\mapsto(x_t,v)\) 是一个正交旋转，所以「换预测目标」不改信息量，只改网络训练时面对的数值尺度。

## 规则

给定 `x_t`、标量 `alpha_bar`、模型输出 `pred` 与目标参数化 `mode`（`"eps"` / `"x0"` / `"v"`，`"epsilon"` 与 `"x_zero"` 视为同义），一次返回 `(x0, eps, v)` 三个同形状 float32 张量（内部用 float64 计算）。`pred` 必须是 `mode` 所指的那个量，已知量原样回填。

## 示例

\[ x_t=[0.5,-1.0,2.0],\ \bar\alpha_t=0.64\ (\alpha=0.8,\ \sigma=0.6),\ \epsilon=[0.3,0.0,-0.9] \]

```
x0  = [0.4, -1.25, 3.175]
eps = [0.3, 0.0, -0.9]
v   = [0.0, 0.75, -2.625]
```

## 约束

- \(\bar\alpha_t\in[0,1]\)；`mode="eps"` 需要 \(\bar\alpha_t>0\)，`mode="x0"` 需要 \(\bar\alpha_t<1\)（否则公式要除以 0，抛 `ValueError`）。
- **退化性质**：\(\bar\alpha_t=1\) 时 \(x_0=x_t\) 且 \(v=\epsilon\)；\(\bar\alpha_t=0\) 时 \(v=-x_0\) 且 \(\epsilon=x_t\)。
- 三种 mode 互为逆运算：把算出的 \(x_0\) 当输入重新解，必须回到同一个 \(\epsilon\)；把 \(v\) 当输入重新解亦然。
