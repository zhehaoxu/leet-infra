# 扩散采样：EulerDiscrete 的时间栅格与 σ 转换

## 题目描述

连续扩散的采样器（Euler、Euler-Ancestral、Heun 等）在构造时都要先把「训练用的 \(T\) 个离散时间步」压成「推理用的 \(N\) 个时间步」，再据此取出一串噪声强度 \(\sigma\)。diffusers 的 `EulerDiscreteScheduler` 把这件事拆成三层可以独立开关的配置：时间栅格（`timestep_spacing`）、σ 插值方式（`interpolation_type`）和 σ 转换器（Karras / exponential / beta）。这三层的组合顺序决定了最终的时间刻度和 \(\sigma\) 序列，也决定了采样时每一步走多大的步长。

本题要求实现这条完整的流水线：给定训练侧的 `alphas_cumprod` 与推理步数 \(N\)，返回推理时间刻度 `timesteps` 与 σ 序列 `sigmas`。

## 规则

设训练步数 \(T = \) `alphas_cumprod` 的长度，逐元素基线为

\[ \sigma_{\text{full}}[i] = \sqrt{\frac{1-\bar\alpha_i}{\bar\alpha_i}},\qquad i = 0,\dots,T-1 \]

它随 \(i\) 单调递增：\(i=0\) 是干净端（\(\sigma\) 小），\(i=T-1\) 是噪声端（\(\sigma\) 大）。

**1. 时间栅格**（`timestep_spacing`，一律从噪声端向干净端递减）：

- `"linspace"`：`linspace(0, T-1, N)` 再翻转，浮点等距；
- `"leading"`：`step_ratio = T // N`（整除），`(arange(N) * step_ratio).round()` 翻转后加 `steps_offset`；
- `"trailing"`：`step_ratio = T / N`（真除），取 `count = ceil(T / step_ratio)` 个点 `grid[k] = T - k * step_ratio`，再 `round() - 1`。当 \(T/N\) 的浮点除法不是整数时 `count` 会是 \(N+1\)，与 numpy 的 `arange(T, 0, -step_ratio)` 等长，因此 `timesteps` 可能多出一个 `-1`。

**2. σ 插值**（`interpolation_type`）：

- `"linear"`：在整数下标 \(0..T-1\) 上按 `timesteps` 对 \(\sigma_{\text{full}}\) 线性插值；
- `"log_linear"`：\(\exp\big(\text{linspace}(\log\sigma_{\text{full}}[T-1],\ \log\sigma_{\text{full}}[0],\ N+1)\big)\)。这一支**完全忽略时间栅格**，并且产生 \(N+1\) 个点。

**3. σ 转换器**（`use_karras_sigmas` / `use_exponential_sigmas` / `use_beta_sigmas`，至多开一个）：

- Karras：\(\sigma_k = \big(\text{hi}^{1/\rho} + r_k\,(\text{lo}^{1/\rho} - \text{hi}^{1/\rho})\big)^{\rho}\)，其中 \(r_k = \text{linspace}(0,1,N)_k\)、\(\rho = 7\)；
- exponential：\(\sigma_k = \exp\big(\text{linspace}(\log\text{hi}, \log\text{lo}, N)_k\big)\)。

两者的端点默认取上一步 \(\sigma\) 序列的两头：\(\text{hi} = \) 噪声端第一项、\(\text{lo} = \) 干净端最后一项，可分别被 `sigma_max` / `sigma_min` 覆盖。生成之后必须把每个 \(\sigma\) 反查回时间轴：令 \(u_k = \log\sigma_k\)、\(\ell_i = \log\sigma_{\text{full}}[i]\)（递增），取满足 \(\ell_{\text{low}} \le u_k\) 的最大下标 `low`（夹到 \([0, T-2]\)），再在 log 空间线性插值

\[ w = \operatorname{clamp}\!\left(\frac{\ell_{\text{low}}-u_k}{\ell_{\text{low}}-\ell_{\text{low}+1}},\,0,\,1\right),\qquad t_k = (1-w)\cdot\text{low} + w\cdot(\text{low}+1) \]

**4. 终端 σ**（`final_sigmas_type`）：`"zero"` 追加 \(0\)；`"sigma_min"` 追加 \(\sqrt{(1-\bar\alpha_0)/\bar\alpha_0}\)，即训练栅格最干净端的 σ。终端值追加在末尾，所以 `sigmas` 通常比 `timesteps` 多一个；`log_linear` 且不挂转换器时多两个。

返回 `(timesteps, sigmas)`，均为 float32 张量。

## 示例

取 \(T=10\) 的 `alphas_cumprod = [0.99, 0.9, 0.7, 0.5, 0.3, 0.15, 0.06, 0.02, 0.005, 0.001]`、\(N=3\)：

```
linspace    -> timesteps = [9.0, 4.5, 0.0]       sigmas = [31.606960, 1.954001, 0.100504, 0.0]
leading     -> timesteps = [6.0, 3.0, 0.0]       sigmas = [3.958114, 1.0, 0.100504, 0.0]
trailing    -> timesteps = [9.0, 6.0, 2.0]       sigmas = [31.606960, 3.958114, 0.654654, 0.0]
karras      -> timesteps = [9.0, 5.561106, 0.0]  sigmas = [31.606960, 3.166429, 0.100504, 0.0]
exponential -> timesteps = [9.0, 4.347704, 0.0]  sigmas = [31.606960, 1.782307, 0.100504, 0.0]
log_linear  -> timesteps = [9.0, 4.5, 0.0]       sigmas = [31.606960, 4.647819, 0.683464, 0.100504, 0.0]
```

这里的 σ 基线是 \(\sigma_{\text{full}} = [0.100504, 0.333333, 0.654654, 1.0, 1.527525, 2.380476, 3.958114, 7.0, 14.106736, 31.606961]\)。

- `linspace` 的 4.5 落在下标 4 与 5 之间，插出 \((1.527525+2.380476)/2 = 1.954001\)；
- `leading` 的整除比是 \(10//3=3\)，从 0 起算的栅格是 \([0,3,6]\)，翻转得 \([6,3,0]\)，所以首项是 6 而不是 9；
- `trailing` 的真除比是 \(10/3\)，栅格 \([10,\,6.67,\,3.33]\) 四舍五入得 \([10,7,3]\)，减 1 得 \([9,6,2]\)；
- `karras` / `exponential` 的 σ 只由两个端点与点数决定，时间刻度是反查出来的非整数；
- `log_linear` 完全无视 \(N\) 个时间点，直接生成 \(N+1=4\) 个 σ，再加终端共 5 个；它的 `timesteps` 仍是普通的 `linspace` 结果。

`final_sigmas_type="sigma_min"` 且 \(N=4\) 时 `sigmas = [31.606960, 3.958114, 1.0, 0.100504, 0.100504]`，最后两项重合——因为终端值就是训练栅格最干净那端的 σ。

## 约束

- `alphas_cumprod` 必须是一维浮点张量，长度 \(T\ge 2\)、全部有限且严格落在开区间 \((0,1)\)；
- `num_inference_steps` 为整数（`bool` 不算）且 \(1\le N\le T\)；`steps_offset` 为不小于 0 的整数，只对 `leading` 生效；
- `timestep_spacing`、`interpolation_type`、`final_sigmas_type` 取到未支持的枚举值时抛 `ValueError`；
- 三个 σ 转换器同时开多个抛 `ValueError`；`use_beta_sigmas=True` 直接抛 `ValueError`（上游要 `scipy.stats.beta.ppf`，本题不引入 scipy）；
- `sigma_min` / `sigma_max` 若给出必须是有限正数，且 `sigma_min < sigma_max`；
- 内部用 float64 计算，`timesteps` 与 `sigmas` 均返回 float32。
