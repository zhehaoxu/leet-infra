# 解题说明 · DDIM 逆向一步

## 出处

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_ddim_inverse.py`：`DDIMInverseScheduler` 是 `DDIMScheduler` 的反向调度器，本课逐行对齐它的 `step`（第 311–384 行）。`set_timesteps` 与 `DDIMScheduler` 一致，`step` 的公式注释指向 DDIM 论文 arXiv:2010.02502 的式 (12)。
- 当前 diffusers 仓库里并没有 `pipeline_ddim_inversion.py`，逆向能力就以这个调度器为唯一入口，本课也只取调度器这一层，不涉及 pipeline。
- 本题的计算只用到 `alphas_cumprod` 中若干下标，因此契约直接传入一维累积 alpha 序列，避免把 beta 调度、`trained_betas` 等构造细节一起搬进题目。

## 核心与易错点

**`prev_timestep` 这个名字是反的。** 源码第一步写 `prev_timestep = timestep`，随后就把 `timestep` 覆盖成 `min(timestep - T//N, T-1)`。覆盖之后：`timestep` 是**预测级**（更干净），`prev_timestep` 是**输出级**（更噪）。照字面把「上一级」理解成更干净的一级，整个逆向就会变成采样方向，`prev_sample` 直接错到底。

**两个 alpha 来自不同下标。** `alpha_prod_t` 用预测级 \(t\)，`alpha_prod_t_prev` 用输出级 `timestep`。它们差一个步长比例 \(T//N\)，而步长比例是**整数除法**：`T=1000, N=50` 时是 20，不是 20.0。

**初始 alpha 只在 \(t<0\) 时启用。** 逆向循环的第一步 `timestep` 常常是 0，于是 \(t=-r\)，`alpha_prod_t` 取 `initial_alpha_cumprod`。默认 `set_alpha_to_one=True` 取 1.0，此时 `beta_prod_t = 0`——epsilon 分支里 `/sqrt(alpha_prod_t)` 分母恰为 1，没问题；但 `sample` 分支要除以 `sqrt(beta_prod_t)`，会除零。源码会静默产出 inf/nan，本题契约显式抛 `ValueError`。

**`pred_epsilon` 必须在截断之前算。** 源码的顺序是：算 `pred_original_sample` 与 `pred_epsilon` → 截断 `pred_original_sample` → 用 `pred_epsilon` 算方向项。若把截断提到前面、用夹过的 x0 再算 v 分支的 eps，`prev_sample` 就和源码不一致。测试的 `clip` 用例专门盯这一点。

**返回的是截断后的 x0。** `pred_original_sample` 被 `clamp` 覆盖后原样返回，所以 `prev_sample` 与返回的 x0 一定自洽（同一组截断值）；拿未截断值当返回值的实现会在 `prev == sqrt(ᾱ_prev)*x0 + direction` 上对不上。

**三种预测类型共享同一个合成公式。** epsilon / sample / v_prediction 的差别只在于怎么拆出 \((x_0,\epsilon)\)；拆完之后方向项与输出公式完全一样。因此用同一组 \((x_0,\epsilon)\) 构造三种输入，三者应给出相同的 `prev_sample`——这是比逐个对数更强的语义检查。

**精度。** 内部 float64、出口 float32；`alphas_cumprod` 允许 float32 输入，转 float64 之后再求幂，避免 float32 的 `0.5` 次幂在跨步累积时放大误差。

## 与其他题的分工

- `diff-ddim`：正向确定性去噪一步，输出更干净的 \(x_{t-1}\)；本题方向相反，输出更噪的一级，且多了初始 alpha 兜底与「预测级 / 输出级不同」这一层。
- `diff-v-prediction-convert`：只做 ε / x₀ / v 的互转；本题把三种预测类型接到同一条 DDIM 更新式上，并验证三者一致。
- `diff-ddpm-step`：随机性后验一步，需要方差与噪声；DDIM 逆向是确定性的，没有方差项。
- `diff-sdedit-strength`：决定逆向从哪一级起步（截断起止步）；本题是给定时间步之后的单步算术。
