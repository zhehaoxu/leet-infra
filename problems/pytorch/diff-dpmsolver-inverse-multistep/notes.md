# 踩坑记录（order 832）

## 回源口径

- 上游 `huggingface/diffusers` `scheduling_dpmsolver_multistep_inverse.py`（`DPMSolverMultistepInverseScheduler`，2025 TSAIL/HF）。用 `curl` 拉到 `/tmp/dpm_inverse_src.py` 逐行钉死，再用 `/tmp/verify832.py` 把官方 `_sigma_to_alpha_sigma_t` / `convert_model_output` / 三个 update / `step` 原样 `exec` 出来与自写 `solve` 对拍：**5 组配置 × 8 步 = 40 步，最大绝对误差 0.0**。
- 更新函数是从**前向** `DPMSolverMultistepScheduler` 逐字复制的（源码注释 `Copied from ...`），公式本身与前向相同；反演的独特性全在 σ 表方向：`sigmas` 严格递增 ⇒ `λ = log(α) − log(σ_t)` 递减 ⇒ `h < 0`、`exp(−h) > 1`、`exp(h) < 1`。因此题面把「σ 表严格递增」写成硬校验，避免它退化成前向题。
- `_sigma_to_alpha_sigma_t` 的 flow 分支是 `α = 1 − σ`（不是 VP 的 `1/√(1+σ²)`）；flow 下 `σ_t = σ`（未开方换算），且需 `0 < σ < 1`，否则 `log(1−σ)` 无定义。
- `convert_model_output` 的 `flow_prediction` 分支用的是**未换算的原始 σ**：官方写 `sigma_t = self.sigmas[self.step_index]; x0_pred = sample - sigma_t * model_output`，局部变量 `sigma_t` 在此被覆盖为原始 σ，不是前面换算出的 `sigma_t`。首版照抄前三个分支会错，必须单独钉死（并特意用 VP σ=0.25 的用例覆盖，flow 时两者恰好相等不会暴露）。
- 官方变量名遮蔽严重：`sigma_t, sigma_s = sigmas[i+1], sigmas[i]` 随后 `alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)`，`sigma_t/sigma_s` 用的是换算后的值；`alpha_t/alpha_s` 用 α 值。实现时拆成 `_lam()` 返回 `(lambda, alpha, sigma_t)` 三元组才没绕晕。
- 降阶标志与前向 `diff-dpm-solver-order-schedule` 不同：反演调度器**没有** `final_sigmas_type == "zero"` 这一项，只有 `euler_at_final or (lower_order_final and N < 15)`；`lower_order_second` 为 `(i == N−2) and lower_order_final and N < 15`。三阶分支调用时官方漏传 `noise`（`sde-dpmsolver++` 会断言失败），本题不含 SDE 分支故不受影响。

## 查重与判题域裁切

- `diff-ddim-inverse-step`（498）：DDIM 一阶反演 + 累积 α 表 + 可选 clip，与本题的 DPM-Solver 多步 λ/h 系数体系不同。
- `diff-dpm-solver-order-schedule`（520）：纯整数选阶调度；本题把选阶压进 `step` 单调用状态机，重点在系数而非整段轨迹，且降阶条件口径不同（见上）。
- `diff-cosine-dpmsolver-sde-step`（759）：Stable Audio 的 CosineDPMSolver，EDM/VE 约定（α≡1）、只 1/2 阶、每步带 SDE 噪声；本题 VP/flow 约定、1/2/3 阶、无噪声。
- `diff-dpmpp-2m`（350）/ `diff-dpm-x0-conversion-threshold`（521）：分别只覆盖 2M 二阶与 x0/ε 换算 + 动态阈值；本题的 3 阶 `D1_0/D1_1/D1/D2` 与 `c_d2` 未在任何既有题出现。
- 官方 `set_timesteps` 的 σ 表构造（linspace/leading/trailing、karras/exponential/beta/flow、scipy 插值）全部剔除，σ 表作显式入参。

## 测试设计踩坑

- 手算锚点选 flow 约定下 `σ = 1/(1+2^k)`，使 `λ = log(2^k)`、`h = h_0 = h_1 = −log 2`、`r0 = r1 = 1`，于是 `exp(−h) = 2`、`B++ = α_t`、`c_sample = σ_t/σ_s0` 全部可逐字手算。
- **但全用等比 σ 表会让 r0/r1 类变异成为真等价**（`h_0/h` 与 `h/h_0` 都等于 1）。已在 `first/second/third` 的 oracle 对拍里各补 2 组非等比表（flow `[0.05,0.1,0.2,0.5]` / `[0.02,0.05,0.1,0.2,0.5]`，VP `[0.1,0.4,1.0,3.0]` / `[0.1,0.4,1.0,3.0,9.0]`），保证 `r0 ≠ 1`、`r0 ≠ r1`。
- **`m-conv-dpm-sample` 首版存活**：所有 `dpmsolver/sample` 用例恰好取了 `alpha_t == sigma_t`（flow σ=0.5 或 VP σ=1），`/sigma_t` → `/alpha_t` 退化成真等价。补 flow σ=0.25 与 VP σ=2.0 两组分母不等价的用例后 CAUGHT。
- `step` 的 `sigmas` 必须长度恰为 `N+1`（官方 `set_timesteps` 末尾追加 `sigma_max`）；`state=None` 时历史全 `None`，warmup 1→2→3 阶靠 `lower_order_nums` 计数，官方计数器在降阶终步也自增。
- 独立 oracle 用纯 Python `math` 从规则重写，张量逐元素对拍；`close()` 统一处理 Tensor→list，避免 `T(tensor)` 的 UserWarning。

## mut 设计（29 变异全 CAUGHT）

覆盖：`σ→(α,σ_t)` 三条（flow 线性 / VP 开方 / σ_t 换算）、`λ` 符号、`h` 方向、一阶 3 条、`convert` 5 条、二阶 7 条（r0 / D1 / pp midpoint / pp heun / dpm midpoint / dpm heun / 分母）、三阶 5 条（r1 / D1 / D2 / 两条 c_d2）、`step` 5 条（降阶两标志 / 选阶 / 历史移位 / 计数器）。锚点用多行上下文保证唯一（`c_d1` 的 pp/dpm 两分支、三阶同名系数）；`m-first-sample` 的锚点须带换行，否则 `sigma_t/sigma_s` 会前缀命中 `sigma_s0` 三处。剔除 1 个真等价候选后定稿 29 个。
