# 详解：EulerDiscrete 的时间栅格与 σ 转换

## 出处

HuggingFace diffusers 的 `src/diffusers/schedulers/scheduling_euler_discrete.py` 里的 `EulerDiscreteScheduler.set_timesteps`。这段代码是 Euler / Euler-Ancestral / Heun / DPM2 等一族采样器共享的栅格构造入口：`timestep_spacing` 决定 \(T\to N\) 的压缩方式，`interpolation_type` 决定 σ 怎么取出，Karras / exponential 转换器改变 σ 的分布形态后再反查时间。同一套逻辑在 `scheduling_flow_match_euler_discrete.py` 里被 FlowMatch 版本复用。

## 核心与易错点

1. **σ 基线是「反着」的**：`sigmas_full = sqrt((1 - alphas_cumprod) / alphas_cumprod)` 随下标递增，所以下标 0 是干净端、\(T-1\) 是噪声端；而 `timesteps` 是从 \(T-1\) 往 0 递减的。两套方向相反的索引混在一起，是本题最主要的心智负担：插值用 `timesteps` 当下标，Karras 的端点却要按「σ 序列的第 0 项是噪声端」来取。
2. **两种除法的差异**：`leading` 用 `T // N`（整除，栅格步长偏小、最后会停在 0），`trailing` 用 `T / N`（真除，栅格会贴近噪声端）。同一个 \(T,N\) 用错分支，首项就从 \(T-1\) 变成别的数，是本题最常见的整题错。
3. **`leading` 的栅格从 0 起算**：`(arange(N) * step_ratio).round()` 是 \([0, r, 2r, \dots]\)，翻转后才递减。所以 \(T=10,N=3\) 的首项是 6 而不是 9。
4. **`steps_offset` 只加在 `leading` 上**：`linspace` 与 `trailing` 完全不受它影响，但参数照样可以传。测试里专门覆盖了「传了 offset 也不变」这一点。
5. **`trailing` 的 `arange` 长度可能比 \(N\) 多 1**：`ceil(T / (T/N))` 在浮点除法不整时会算出 \(N+1\)，于是最后一个点落在 0 附近、`round() - 1` 得到 \(-1\)。上游直接写 `np.arange(num_train_timesteps, 0, -step_ratio)`，行为完全一致。实现时不要假设「`timesteps` 一定长 \(N\)」，否则 \(T=60,N=13\) 这类输入会当场对不上。
6. **`log_linear` 完全忽略时间栅格**：`exp(linspace(log σ[-1], log σ[0], N+1))` 只用两端点与点数，产生 \(N+1\) 个 σ（不是 \(N\) 个）。于是它比 `timesteps` 多两个（多出的一个 + 终端）。这是上游的真实怪癖，改 `timestep_spacing` 也改不动它。
7. **Karras 的 \(\rho=7\) 偏噪声端**：\(\rho>1\) 时中段值高于几何平均、低于线性插值 \((hi+lo)/2\)；\(\rho\to\infty\) 时整串 σ 逼近 hi。把 \(\rho\) 写成 1 或把 hi/lo 弄反，都会让 σ 序列整体走形，而端点仍然正确——只看首尾两项是查不出来的。
8. **反查用 log-σ 空间**：`_sigma_to_t` 找到满足 \(\ell_{\text{low}} \le u\) 的最大下标（用 `cumsum(log_sigmas <= log_sigma).argmax()` 的等价写法），再做**对数**线性插值，不是线性插值。σ 跨好几个数量级，用错空间会让中间几项的时间刻度明显偏移。
9. **终端 σ 的 `sigma_min` 会与倒数第二项重合**：终端值取的是 `sqrt((1 - abar[0]) / abar[0])`，而线性插值的最后一项若正好落在 `timesteps = 0`，两者相等。这不是 bug，是上游的正常现象。
10. **`use_beta_sigmas` 依赖 scipy**：`_convert_to_beta` 要 `scipy.stats.beta.ppf`，而判题环境只有 torch / numpy / transformers。本题因此把该开关定义为「为真即 `ValueError`」，并在题面写明，而不是伪造一个 beta 分位数。

## 对拍口径

判题用 numpy 独立实现整条流水线（`_grid_oracle` / `_sigma_to_t_oracle` / `_oracle`），再与参考解逐元素比对：

- **示例值**：\(T=10,N=3\) 的六种配置（三种 spacing + Karras + exponential + log_linear）的 `timesteps` 与 `sigmas` 全部硬编码，保留六位小数；
- **spacing 变体**：\(T=10\)、\(T=9\) 下 \(N\in\{1,3,4,7\}\) 三个分支全对拍，并显式断言 `leading` 用整除、`trailing` 用真除、`np.round(4.5) == 4`（banker's rounding）；
- **offset 隔离**：`steps_offset=10` 只在 `leading` 上表现为整体平移，另两个分支逐位不变；
- **插值命中**：60 点训练表下 \(N\in\{2,5,13\}\)，整数时间点必须原样命中训练栅格的 σ（`trailing` 多出的 \(-1\) 点排除在断言外）；
- **终端 σ**：`zero` 严格为 0，`sigma_min` 等于训练栅格最干净端的 σ 且与倒数第二项重合；
- **Karras**：与 \(\rho=7\) 的显式公式逐元素对拍，另加「端点 = σ_max / σ_min」「单调不增」「中段高于几何平均、低于线性插值」三条形态断言，以及 `sigma_min`/`sigma_max` 覆盖端点；
- **exponential**：log-σ 间隔恒定、两端点正确；
- **log_linear**：点数 \(N+1\)（加终端后 \(N+2\)）、换 spacing 不改 σ、挂转换器后点数回到 \(N+1\)；
- **非法输入**：20 组（非张量、整型/二维/过短/越界/NaN 的 alphas、\(N\) 越界或非整数、枚举值非法、转换器冲突、beta、负 offset、`sigma_min >= sigma_max`）都必须抛 `ValueError`。

共九组断言。

## 复杂度

\(O(T)\) 构造基线、\(O(N)\) 插值与转换器、\(O(N\log T)\) 反查，时间与空间都是线性量级（反查实现为一次性向量化，实际是 \(O(NT)\) 的布尔比较，本题规模下可忽略）。
