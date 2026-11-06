# DPMSolverMultistep 反演调度器的多步更新账

## 背景

`DPMSolverMultistepInverseScheduler` 是 diffusers 里 `DPMSolverMultistepScheduler` 的反向调度器：把干净样本沿**递增**的 σ 表一路推回更噪的一级，供图像编辑「先反演成噪声、再改写噪声、再采样」使用。它的一/二/三阶 update 函数逐字复制自前向调度器，但因为 σ 表方向相反，`h = λ_t − λ_s < 0`、`exp(−h) > 1`、`exp(h) < 1`，各阶系数都落在与前向不同的数值区间。

本题只重建官方 `step()` 链路里可判定的账。`sde-*` 分支的 Brownian 噪声项、dynamic thresholding、`learned_range` 通道切片、`set_timesteps` 的 σ 表构造均不进判题域（前两类已有 `diff-cosine-dpmsolver-sde-step` 与 `diff-dpm-x0-conversion-threshold`）。

## 规则

σ 表 `sigmas` 一维、严格递增、全正有限，`step_index = i`；flow 约定另需 `0 < σ < 1`。张量一律 fp64。

### 1. σ→(α, σ_t) 与 λ

```
VP   : alpha = 1 / sqrt(1 + sigma^2),  sigma_t = sigma * alpha
flow : alpha = 1 - sigma,             sigma_t = sigma
lam  = log(alpha) - log(sigma_t)
```

于是 `h = lam[i+1] − lam[i]`（反演方向恒为负），二阶另需 `h_0 = lam[i] − lam[i−1]`，三阶再需 `h_1 = lam[i−1] − lam[i−2]`，并取 `r0 = h_0/h`、`r1 = h_1/h`。注意变量名 `sigma_t` 在官方代码里被换算结果覆盖，故 `sigma_t/sigma_s` 用的是换算后的值。

### 2. convert_model_output

`algorithm_type = "dpmsolver++"` 时求解器量是 `x0_pred`，`"dpmsolver"` 时是 `epsilon`；`prediction_type` 决定换算方式。`sigma` 取 `sigmas[i]`，换算量 `(alpha_t, sigma_t)` 由第 1 节给出：

```
++  epsilon      : (sample - sigma_t * mo) / alpha_t
++  sample       : mo
++  v_prediction : alpha_t * sample - sigma_t * mo
++  flow_prediction : sample - sigma * mo
dpm epsilon      : mo
dpm sample       : (sample - alpha_t * mo) / sigma_t
dpm v_prediction : alpha_t * mo + sigma_t * sample
```

`flow_prediction` 用的是**未换算的原始 σ**（官方局部变量同名遮蔽），不是 `sigma_t`。非法 `prediction_type` / `algorithm_type` 抛 `ValueError`。

### 3. 一阶更新

`m0` 为本步已换算的模型输出；`c_sample`、`c_model` 为 `sample` 与 `m0` 的系数，`prev = c_sample·sample + c_model·m0`：

```
++  : c_sample = sigma_t / sigma_s,  c_model = -(alpha_t * (exp(-h) - 1))
dpm : c_sample = alpha_t / alpha_s, c_model = -(sigma_t * (exp(h) - 1))
```

### 4. 二阶更新

历史 `model_outputs = [m1, m0]`（旧→新），`D0 = m0`、`D1 = (1/r0)·(m0 − m1)`。设 `B++ = alpha_t·(exp(−h) − 1)`、`Bdpm = sigma_t·(exp(h) − 1)`：

```
++  midpoint : c_sample = sigma_t/sigma_s0, c_d0 = -B++,      c_d1 = -0.5 * B++
++  heun     : c_sample = sigma_t/sigma_s0, c_d0 = -B++,      c_d1 = B++/h + alpha_t
dpm midpoint: c_sample = alpha_t/alpha_s0, c_d0 = -Bdpm,     c_d1 = -0.5 * Bdpm
dpm heun    : c_sample = alpha_t/alpha_s0, c_d0 = -Bdpm,     c_d1 = -(Bdpm/h - sigma_t)
```

`prev = c_sample·sample + c_d0·D0 + c_d1·D1`。

### 5. 三阶更新

历史 `[m2, m1, m0]`，`D1_0 = (1/r0)·(m0 − m1)`、`D1_1 = (1/r1)·(m1 − m2)`、`D1 = D1_0 + r0/(r0+r1)·(D1_0 − D1_1)`、`D2 = (1/(r0+r1))·(D1_0 − D1_1)`、`D0 = m0`。三阶是固定公式，没有 `solver_type`：

```
++  : c_d0 = -B++, c_d1 = B++/h + alpha_t,
      c_d2 = -(alpha_t * ((exp(-h) - 1 + h) / h^2 - 0.5))
dpm : c_d0 = -Bdpm, c_d1 = -(Bdpm/h - sigma_t),
      c_d2 = -(sigma_t * ((exp(h) - 1 - h) / h^2 - 0.5))
```

`c_sample` 同二阶；`prev = c_sample·sample + c_d0·D0 + c_d1·D1 + c_d2·D2`。

### 6. step 状态机

单调用按固定顺序执行：换算本步 `model_output` → 历史整体左移一格、末位写入本步结果 → 算两个降阶标志 → 选阶 → 计数器自增。注意反演调度器的降阶条件里**没有**前向的 `final_sigmas_type == "zero"` 项：

```
lower_order_final  = (i == N-1) and (euler_at_final or (lower_order_final and N < 15))
lower_order_second = (i == N-2) and lower_order_final and N < 15
order 1 : solver_order == 1 or lower_order_nums < 1 or lower_order_final
order 2 : solver_order == 2 or lower_order_nums < 2 or lower_order_second
order 3 : otherwise
```

`N = n_steps = len(sigmas) − 1`；`state = None` 表示全新开始（历史全 `None`、计数 0）。退出时 `lower_order_nums = min(lower_order_nums + 1, solver_order)`，新 `state = {model_outputs, lower_order_nums}` 返回。

## 判题输入与返回

单入口 `solve(mode, **kw)`。六个 mode 的入参与返回：

- `sigma_to_alpha`：`sigma`、`use_flow_sigmas` → `{alpha, sigma_t}`；
- `convert`：`model_output`、`sample`、`sigma`、`algorithm_type`、`prediction_type`、`use_flow_sigmas` → `{converted, kind}`；
- `first`：`sigmas`、`step_index`、`sample`、`model_output`、`algorithm_type`、`use_flow_sigmas` → `prev_sample` 及全部中间量；
- `second`：另需 `model_outputs`（长度 2）、`solver_type` → 同上；
- `third`：另需 `model_outputs`（长度 3）→ 同上；
- `step`：`sigmas`、`step_index`、`n_steps`、`sample`、`model_output`、`state` 及全部 config → `prev_sample`、`used_order`、`kind`、`converted`、两个降阶标志、计数器前后值、`state` 与该阶中间量。

## 示例

flow 约定、`sigmas = [0.2, 0.4]`、`step_index = 0`：`λ_s = log 4`、`λ_t = log 1.5`、`h = log(3/8)`、`exp(−h) = 8/3`、`exp(h) = 3/8`，`alpha_t = 0.6`、`sigma_t = 0.4`、`sigma_s = 0.2`。

```
++  : c_sample = 0.4/0.2 = 2.0, c_model = -(0.6 * (8/3 - 1)) = -1.0
      prev = 2.0 * 1.0 - 1.0 * 0.5 = 1.5
dpm : c_sample = 0.6/0.8 = 0.75, c_model = -(0.4 * (3/8 - 1)) = 0.25
      prev = 0.75 * 1.0 + 0.25 * 0.5 = 0.875
```

## 约束

- `sigmas` 严格递增、全正有限；`flow` 时 `0 < σ < 1`；`step_index` 与阶数需保证所需前置 σ 存在，否则 `ValueError`。
- `algorithm_type ∈ {dpmsolver++, dpmsolver}`、`solver_type ∈ {midpoint, heun}`、`solver_order ∈ {1,2,3}`、`prediction_type` 见第 2 节；越界抛 `ValueError`。
- 不含任何随机性；张量逐元素计算，形状由输入决定，返回 fp64。
