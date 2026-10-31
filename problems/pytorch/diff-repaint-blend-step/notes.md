# Notes: RePaint 的掩码拼接步与前向回跳

## 出处

- diffusers v0.42.1 `src/diffusers/schedulers/scheduling_repaint.py`（全文 419 行），类 `RePaintScheduler` 在 99-419，`order = 1` 在 126。
- RePaint 本体是 arXiv:2201.09865（Lugmayr 等，RePaint: Inpainting using Denoising Diffusion Probabilistic Models）。源码在四处点名它的 Algorithm 1：317-322 说明未知区把论文的式 (7) 换成了 DDIM 的式子、340-344 对应 Line 5、347-348 对应 Line 8、390-391 对应 Line 10。
- 后验本身：311 的 x̂₀ 是 DDPM（arXiv:2006.11239）式 (15)；`_get_variance`（234-262）按 253-259 的注释取 DDPM 式 (6)(7)，并声明在 `eta=0` 时等价于 DDIM（arXiv:2010.02502）式 (16)；方向项 335 是 DDIM 式 (12)。
- 构造期：四种 β 表在 141-152（`linear` / `scaled_linear` / `squaredcos_cap_v2` 走 47-96 的 `betas_for_alpha_bar` / `sigmoid`），`alphas_cumprod` 在 156-157，`final_alpha_cumprod = torch.tensor(1.0)` 在 160（**写死**，不是 `alphas_cumprod[0]`）。
- 116-118 的配置文档写明 `eta=0.0` 对应 DDIM、`1.0` 对应 DDPM；317-322 的注释进一步说 `eta=1.0` 时 DDIM 调度与 DDPM 结果一致。

## 核心与易错点

- **已知区根本不经过网络。** 348 是逐格 `mask*known + (1-mask)*unknown`，不是「整体去噪完再覆盖」：`known`（345）直接由 `original_image` 重加噪得到，`model_output` 与 `clip_sample` 在 `mask=1` 的格子上没有任何影响，实测这些格子在 `eta∈{0,0.5,1.0}`、`clip_sample` 两种取值下逐位不变。
- **两处都是 ᾱ_prev，只有一处是 ᾱ_t。** 307 的 `beta_prod_t` 只服务于 x̂₀（311）和方差的分母（260）；`known`（345）与方向项（335）用的是 306 的 `alpha_prod_t_prev`。把 `known` 写成 ᾱ_t 尺度是最常见的错。
- **`std_dev_t` 带根号，且它同时进两条支路的系数。** 327 是 `eta * variance**0.5`；335 减去的是 `std_dev_t**2`（不是 `variance`），331 乘的是 `std_dev_t` 本身。题面示例里 `variance=0.168672`、`std=0.205349`，写成 `eta*variance` 会把两者都压低约一倍。
- **η 项的门只关未知区。** 329-331 的 `if t > 0 and eta > 0` 让 `variance` 退化成标量 0，但 345 的 `sqrt(1-ᾱ_prev)*noise` 无条件加。另外 326 只 draw 一次噪声，321-322 的注释明说这张噪声被 7. 和 8. 两步复用——题面因此只给一张 `noise`。
- **`prev < 0` 走的是硬编码 1.0。** 306/249 的三元式意味着前 `n` 步（`t=0..n-1`）满足 ᾱ_prev=1 ⇒ `variance=0`、方向项系数 `sqrt(1-1-0)=0`、`known=original_image`，整式塌成 `mask*orig + (1-mask)*x̂₀`，`eta` 完全不参与。误取 `ᾱ[0]` 会立刻在这些格子上露出偏差。
- **`1 - ᾱ_prev - std_dev_t**2` 可以为负，上游会静默出 NaN。** 由 `variance <= 1 - ᾱ_prev`（因为 `(1-ᾱ_t/ᾱ_prev) <= (1-ᾱ_t)` 当 ᾱ_prev<1）可知 `eta=1` 恒安全，上界是 `sqrt((1-ᾱ_t)/(1-ᾱ_t/ᾱ_prev))`；题面示例里等于 1.169961，`eta=1.5` 即越界。本题把它改成显式 `ValueError`。
- **回跳是 n 个逐 β 子步。** 379-391：`n = T // N`，每步 `sqrt(1-beta)*x + sqrt(beta)*noise`，用的是 `betas[t..t+n-1]` 而不是任何 ᾱ 比值。累积下来 x 的系数是 `sqrt(prod alpha) = sqrt(ᾱ_{t+n-1}/ᾱ_{t-1})`（`t=0` 时 ᾱ₋₁ 视作 1），第 i 张噪声的系数是 `sqrt(beta_{t+i} * prod_{k>i} alpha_{t+k})`；写成 `sqrt(ᾱ_{t+n}/ᾱ_t)` 整体多乘了一个 `sqrt(alpha_{t+n})`，是下标错位而不是量级差。

## 与上游的差异

- 上游 326 用 `randn_tensor` 自己生成噪声、`undo_step` 在 383-388 每轮再 draw 一张（mps 分支只是可复现性替代，与算法无关）。本题把 `noise` 与 `undo_noise`（`n` 张）全部显式给出。
- 上游的 `alphas_cumprod` 由构造器按四种 schedule 生成；本题直接吃一维 `betas`，等价于走 `trained_betas` 那条路（139-140），因此 `beta_schedule` 名字不再入参。
- 上游 `num_train_timesteps` 是 config、`num_inference_steps` 由 `set_timesteps` 赋值，两处整除（302、379）都不校验余数；本题要求 `len(betas) % N == 0`。
- 上游的 `timestep` 来自 231 行乘过 `(T // N)` 的 `timesteps` 数组，本来就是 β 表下标；本题沿用这一约定，但显式要求 `t + n <= T`——上游 382 越界时抛的是 `IndexError`。
- 上游 `mask`/`original_image` 允许广播、`step` 不做形状校验；本题要求五个张量同形、`mask` 落在 [0,1]、维数 >= 1。
- 上游 `step` 的返回只有 `(prev_sample, pred_original_sample)`（350-356），`undo_step` 是**独立**方法，`step` 内部不调用它。本题把两者放在一次调用里，`undone` 是 `undo_step(sample, t)` 的输出，与拼接步无数据依赖。
- v0.42.1 里 `src/diffusers/pipelines/repaint/` 目录已不存在（GitHub raw 404），`step`/`undo_step` 只能被直接调用；`add_noise` 仍在 395-416 抛 `NotImplementedError`（提示改用 `DDPMScheduler.add_noise`）。

## 与其他题的分工

- `diff-repaint-jump-schedule`（同一个调度器的 188-232）只管 `set_timesteps` 的 `jump_length/jump_n_sample` 跳表怎么排，不含任何逐步数学；本题正好接在跳表落地的这两支上。
- `diff-ddim-inverse-step` 也是「把样本推向更噪」，但那是 DDIM 反演、要过网络；本题的 `undo_step` 是纯前向加噪，`model_output` 完全不参与。
- `diff-ddim` / `diff-ddpm-step` 负责标准的一步更新式；本题未知区 = DDIM 式 (12) 的推广（ᾱ_prev 用跳到的那一格）+ η 修正，已知区则完全另起炉灶。
- `diff-consistency-decoder-2step` 已经考过 `betas_for_alpha_bar` 的表构建；本题只吃现成 β 表，不重建调度。
- `diff-amused-topk-remask` 与 `diff-entropy-bound-unmask` 同属「掩码 + 反复提交」家族，但那是离散/潜变量解码侧；本题是连续图像侧的空间混合。
- `diff-sigma-grid-construct` 管连续 σ 栅格；RePaint 始终在训练用的整数 β 表上走，二者不重叠。
