# 扩散采样：RePaint 的掩码拼接步与前向回跳

## 题目描述

RePaint 做图像补全（inpainting）时，每一步要做两件相反的事：先像普通 DDPM/DDIM 那样把当前样本去噪到 \(x_{t-\text{prev}}\)，但**只在这个式子作用于「要补的区域」**；掩码圈出的已知区域根本不进网络，而是直接拿原图重新加噪到同一个时间点，两段逐格拼起来。拼错了方向、或者让网络输出漏进已知区，补出来的图就不再等于原图。

diffusers 的 `RePaintScheduler` 还把这套过程配了一个反向操作 `undo_step`：把样本**往前**推 \(n = T/N\) 个子步、逐步重新加噪，用于「已经降到的时间点又跳回去」。它用的是前向扩散自己的系数（每张 \(\beta\) 单独一次），不是一步到位的大跳。

本题把这两段合在一道题里，并且把上游自己生成的东西全部摊成显式入参：\(\bar\alpha\) 表用一维 `betas` 给出，两处随机噪声（本步的一张、回跳的 \(n\) 张）也给出。没有真实网络，也没有随机数生成。

## 规则

记 \(T=\)`len(betas)`，\(n=T\,//\,N\)（整除），\(\bar\alpha_t=\prod_{j\le t}(1-\beta_j)\)（前缀积，含下标 \(t\)），`timestep` 直接作为 \(\beta\) 与 \(\bar\alpha\) 两张表的下标。下文用 \(x/\epsilon/x_{\text{orig}}/z\) 依次指 `sample / model_output / original_image / noise`。所有中间量在 float64 下计算。

**去噪加拼接**（\(t\to t-n\)）：

\[ \text{prev}=t-n,\qquad \bar\alpha_{\text{prev}}=\begin{cases}\bar\alpha_{t-n} & \text{prev}\ge 0\\[2pt] 1.0 & \text{prev}<0\end{cases} \]

\[ \hat{x}_0=\frac{x-\sqrt{1-\bar\alpha_t}\;\epsilon}{\sqrt{\bar\alpha_t}},\qquad \text{clip\_sample 时截到 } [-1,1] \]

\[ \tilde\beta=\frac{1-\bar\alpha_{\text{prev}}}{1-\bar\alpha_t}\Big(1-\frac{\bar\alpha_t}{\bar\alpha_{\text{prev}}}\Big),\qquad \sigma=\eta\sqrt{\tilde\beta},\qquad \text{add}=\begin{cases}\sigma\,z & t>0 \text{ 且 } \eta>0\\ 0 & \text{否则}\end{cases} \]

\[ \text{unknown}=\sqrt{\bar\alpha_{\text{prev}}}\,\hat{x}_0+\sqrt{1-\bar\alpha_{\text{prev}}-\sigma^2}\;\epsilon+\text{add} \]

\[ \text{known}=\sqrt{\bar\alpha_{\text{prev}}}\,x_{\text{orig}}+\sqrt{1-\bar\alpha_{\text{prev}}}\;z \]

\[ \text{prev\_sample}=\text{mask}\cdot\text{known}+(1-\text{mask})\cdot\text{unknown} \]

**前向回跳**（从 `sample` 出发，逐个 \(\beta\) 走 \(n\) 步）：

\[ x \leftarrow \sqrt{1-\beta_{t+i}}\;x+\sqrt{\beta_{t+i}}\;\text{undo\_noise}_i,\qquad i=0,\dots,n-1 \]

五个容易写错的地方：

1. `mask` 的 **1 表示保留原图**，0 才是要补的区域；已知区的系数用的是 \(\bar\alpha_{\text{prev}}\) 而不是 \(\bar\alpha_t\)。
2. \(\sigma=\eta\sqrt{\tilde\beta}\)（少开根号就把 \(\eta{=}0.5\) 写成 \(\eta\tilde\beta\)），而方向项的平方根里减的是 \(\sigma^2\) 且用的是 \(\bar\alpha_{\text{prev}}\)。
3. \(\eta\) 项只在 \(t>0\) 且 \(\eta>0\) 时进入**未知区**；已知区的 \(\sqrt{1-\bar\alpha_{\text{prev}}}\,z\) 在任何 \(t\) 都要加。两处用的是**同一张** `noise`。
4. \(\text{prev}<0\) 时 \(\bar\alpha_{\text{prev}}\) 取 **1.0**（上游写死的 `final_alpha_cumprod`），不是 \(\bar\alpha_0\)。此时 unknown 退化成 \(\hat x_0\)、known 退化成原图，整式变成 `mask*orig + (1-mask)*x0_hat`。
5. 回跳的累积系数是 \(\sqrt{\prod_{i=0}^{n-1}(1-\beta_{t+i})}=\sqrt{\bar\alpha_{t+n-1}/\bar\alpha_{t-1}}\)（\(t=0\) 时 \(\bar\alpha_{-1}\) 视作 1），写成 \(\sqrt{\bar\alpha_{t+n}/\bar\alpha_t}\) 就整体多乘了一个 \(\sqrt{\alpha_{t+n}}\)，是个错位。

返回三元组 `(prev_sample, x0_hat, undone)`，都是与 `sample` 同形状的 float32 张量；第二项是（可能已截断的）\(\hat x_0\)。

## 示例

```
betas    = [0.05, 0.08, 0.12, 0.18, 0.25, 0.33, 0.40, 0.50]   # T = 8
abar     = [0.95, 0.874, 0.76912, 0.630678, 0.473009, 0.316916, 0.19015, 0.095075]
N = 4  ->  n = 2
```

取 `timestep=4`、`eta=0.5`、`clip_sample=True`，则各中间量为：

```
abar_t(4)      = 0.473009   sqrt = 0.687756    1 - abar_t     = 0.526991   sqrt = 0.725942
abar_prev(2)   = 0.769120   sqrt = 0.876995    1 - abar_prev  = 0.230880   sqrt = 0.480500
beta_tilde     = 0.168672   sqrt = 0.410697    sigma = 0.205349    sigma^2 = 0.042168
direction_coef = sqrt(0.230880 - 0.042168) = sqrt(0.188712) = 0.434410
x0_hat_preclip = [[0.053841, -0.753922], [1.534764, 0.185249], [0.197112, 0.886394]]
undo_coef      = sqrt((1-0.25)*(1-0.33)) = sqrt(0.502500) = 0.708872
```

给定

```
sample         = [[0.4, -0.7], [1.6, 0.2], [-0.3, 0.9]]
model_output   = [[0.5, -0.25], [0.75, 0.1], [-0.6, 0.4]]
original_image = [[-0.8, 0.6], [0.3, -0.5], [0.9, 0.2]]
mask           = [[1, 0], [0, 1], [1, 0]]
noise          = [[0.7, -0.4], [0.2, 0.9], [-0.5, 0.3]]
undo_noise     = [[[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1]],
                  [[-0.2, 0.3], [0.4, -0.5], [0.1, 0.6]]]
```

则：

```
prev_sample          = [[-0.365246, -0.851928], [1.243872, -0.006048], [0.549046, 1.012732]]
x0_hat               = [[0.053841, -0.753922], [1.0, 0.185249], [0.197112, 0.886394]]
undone               = [[0.209584, -0.24202], [1.241198, 0.018253], [0.049418, 0.941732]]
```

`prev_sample[0][0]` 是掩码格，完全由原图与噪声决定：\(0.876995\times(-0.8)+0.480500\times 0.7=-0.365246\)，与 `model_output` 无关。

三个掩码格（`mask=1`）在任何 `eta`/`clip_sample` 组合下逐位不变——已知区不吃 \(\eta\)，也不经过网络；非掩码格则全部随 \(\eta\) 移动：\(\epsilon\) 的系数从 \(0.434410\) 变成 \(\sqrt{0.230880}=0.480500\)，同时 \(\sigma z\) 项消失。本例 \(\eta\) 的上界是 \(\sqrt{(1-\bar\alpha_t)/(1-\bar\alpha_t/\bar\alpha_{\text{prev}})}=1.169961\)，越过即报错。

```
clip_sample=False:  x0_hat[1][0] = 1.534764    prev_sample[1][0] = 1.712857
eta=0.0:            prev_sample = [[-0.365246, -0.781311], [1.23737, -0.006048], [0.549046, 0.969563]]
eta=1.0:            prev_sample = [[-0.365246, -0.887818], [1.146195, -0.006048], [0.549046, 1.000338]]
eta=1.5:            ValueError
```

`timestep` 落到第一个跳跃之前（\(t<n\)，本例 \(t\in\{0,1\}\)）时 \(\bar\alpha_{\text{prev}}=1\)，`eta` 取多少都不影响结果，且

```
t=1: prev_sample = [[-0.8, -0.653837], [1.0, -0.5], [0.9, 0.810815]]
```

这一行恰好逐位等于 `mask * original_image + (1 - mask) * x0_hat`——此时 `x0_hat` 用的仍是 \(t=1\) 的表值，且 `[1][0]` 那格被截断成了 \(1.0\)。

## 约束

- `sample / model_output / original_image / mask / noise` 必须是同形状张量（第 0 维为 batch，维数 \(\ge 1\)）、全为有限值；`mask` 的取值还得落在 \([0,1]\)。
- `betas` 是一维有限张量，元素落在 \([0,1)\)，且前缀积 \(\bar\alpha\) 不得触到 0。
- `undo_noise` 形状必须是 `(n,) + sample.shape`。
- `timestep`、`num_inference_steps` 必须是 Python int（`bool` 非法）；\(N\ge 1\) 且 \(T\bmod N=0\)；\(0\le t\) 且 \(t+n\le T\)（否则 `undo_step` 会索引到表外）。
- `eta` 是有限非负实数（可以大于 1），但要满足 \(1-\bar\alpha_{\text{prev}}-\sigma^2\ge 0\)；`clip_sample` 是 bool。
- 中间量用 float64，三段返回都是 float32。禁止 `import diffusers`。
