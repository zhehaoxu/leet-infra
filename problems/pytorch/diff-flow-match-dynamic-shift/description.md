# Flow Matching 采样：动态时间平移与 terminal 拉伸

## 题目描述

Rectified Flow / Flow Matching 的采样器把时间轴 \(t\in[0,1]\) 当作「噪声浓度」（1 为纯噪声、0 为干净数据），σ 序列与时间刻度是同一组数的两种写法（`timesteps = sigmas * num_train_timesteps`）。有两个步骤会改变最终 σ 的分布：**time shift** 按分辨率或用户参数把整个栅格往噪声侧推，**terminal stretch** 再把栅格线性拉伸到指定的末端 σ（LTX-Video 引入的 `shift_terminal`）。两者都在 `FlowMatchEulerDiscreteScheduler` 里，而且顺序不能交换。

本题实现从「\(N\) 个均匀 σ」到「最终 timesteps + sigmas」的完整流程：静态平移或动态平移（指数 / 线性两型）、可选的 terminal 拉伸、可选的 σ 反相，以及终端 σ 的追加。

## 规则

记 \(T = \) `num_train_timesteps`（默认 1000）。

**1. 初始栅格**：`sigmas = linspace(sigma_max, sigma_min, N)`，从噪声端递减。

**2. 平移**：

- `use_dynamic_shifting=False`：静态平移 \(s' = \dfrac{\text{shift}\cdot s}{1+(\text{shift}-1)\,s}\)；
- `use_dynamic_shifting=True`：必须给 `mu`，逐元素按 `time_shift_type`：
  - `"exponential"`：\(s' = \dfrac{e^{\mu}}{e^{\mu} + (1/s - 1)}\)
  - `"linear"`：\(s' = \dfrac{\mu}{\mu + (1/s - 1)}\)

三者在 \(\text{shift}=s\)、\(\mu=\ln s\)（指数）、\(\mu=s\)（线性）时是同一个变换——静态 \(s\cdot x/(1+(s-1)x)\) 与线性 \(\mu x/(1+(\mu-1)x)\) 完全相等——但题面与实现都按上游的三条独立公式给出。

**3. terminal 拉伸**（`shift_terminal` 非空时）：\(\text{scale} = \dfrac{1-s'[-1]}{1-\text{shift\_terminal}}\)，\(s'' = 1 - \dfrac{1-s'}{\text{scale}}\)。这一步必须在平移之后、取 timesteps 之前：先算 scale 再平移，末项就不再等于 `shift_terminal`。

**4. timesteps**：\(t = s''\cdot T\)。

**5. σ 反相**（`invert_sigmas=True`，Mochi 用）：再取 \(s = 1 - s''\)、\(t = s\cdot T\)，并且终端追加 \(1\) 而不是 \(0\)。

返回 `(timesteps, sigmas)`：`timesteps` 长度 \(N\)，`sigmas` 长度 \(N+1\)（末尾是终端值），均为 float32。

## 示例

\(N=4\)、`sigma_min=0`、`sigma_max=1`、\(T=1000\)：

```
shift=2               -> timesteps = [1000.0, 800.0, 500.0, 0.0]           sigmas = [1.0, 0.8, 0.5, 0.0, 0.0]
exponential, mu=1     -> timesteps = [1000.0, 844.637573, 576.116882, 0.0] sigmas = [1.0, 0.844638, 0.576117, 0.0, 0.0]
linear, mu=2          -> timesteps = [1000.0, 800.0, 500.0, 0.0]           sigmas = [1.0, 0.8, 0.5, 0.0, 0.0]
shift=2, terminal=0.2 -> timesteps = [1000.0, 840.0, 600.0, 200.0]         sigmas = [1.0, 0.84, 0.6, 0.2, 0.0]
shift=2, invert       -> timesteps = [0.0, 200.0, 500.0, 1000.0]           sigmas = [0.0, 0.2, 0.5, 1.0, 1.0]
shift=2, terminal+inv -> timesteps = [0.0, 160.0, 400.0, 800.0]            sigmas = [0.0, 0.16, 0.4, 0.8, 1.0]
```

初始栅格是 \([1,\,2/3,\,1/3,\,0]\)。静态 `shift=2` 与线性 `mu=2` 给出同一行（\(\mu/(\mu+(1/s-1)) = 2s/(1+s)\)）；指数 `mu=1` 等价于 `shift` \(=e\approx2.718\)，所以中段更靠噪声侧。`terminal=0.2` 的 scale 为 \(1/0.8=1.25\)，把 \([1,0.8,0.5,0]\) 映成 \([1,0.84,0.6,0.2]\)，末项正好落在 0.2。反相后栅格单调递增，终端追加 1。

另一个边界例子：`sigma_min=0.1`、`sigma_max=0.9`、`shift=2`、\(N=4\) 时 `sigmas = [0.947368, 0.77551, 0.536585, 0.181818, 0.0]`——两端不再是 1 和 0，说明端点只在不设 `sigma_min` 时才被钉住。

## 约束

- `num_inference_steps` 与 `num_train_timesteps` 都是正整数（`bool` 不算）；设置 `shift_terminal` 时要求 \(N\ge2\)（否则 scale 退化成 0、拉伸得到 NaN）。
- `shift` 必须是有限正数；`sigma_min` / `sigma_max` 必须满足 \(0\le\sigma_{\min}<\sigma_{\max}\le1\)。
- `use_dynamic_shifting=True` 时必须提供 `mu` 且其为有限数；`time_shift_type="linear"` 还要求 \(\mu>0\)（\(\mu=0\) 会把整串 σ 压成 0）。
- `shift_terminal` 必须落在开区间 \((0,1)\)；`time_shift_type` 只接受 `"exponential"` / `"linear"`，其余抛 `ValueError`。
- 内部用 float64 计算，`timesteps` 与 `sigmas` 都返回 float32。
