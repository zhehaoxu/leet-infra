## 题意

`CosineDPMSolverMultistepScheduler` 是 diffusers 里 Stable Audio Open（arXiv 2407.14358）使用的调度器：把 DPMSolver 多步法搬到 **EDM/VE 约定**下——σ 表由 exponential / karras schedule 生成后追加 `sigma_last`（0 或 `sigma_min`），网络输出先过 **EDM 预条件**（`c_skip`/`c_out` 缩放）再进求解器，且两条 update 都带 **SDE 噪声项**（官方每步由 `BrownianTreeNoiseSampler` 采样）。`step()` 每次调用按固定顺序跑「state 解析 → convert → history 移位 → 降阶判定 → 一阶或二阶 update → state 推进」，靠 `model_outputs` 历史与 `lower_order_nums` 计数把 N 次调用串成采样轨迹。本题要求实现这台状态机的**单调用**，状态显式经 `state` 进出。

判题输入：

- `model_output` / `sample`：同形状浮点张量；
- `sigmas`：一维浮点张量，官方 `set_timesteps` 产出的表 `[σ0, …, σ_{N−1}, σ_last]`（长度 N+1，N ≥ 2），有限、单调不增、**除末位外全为正**——末位是 0（`final_sigmas_type="zero"`，默认）或 `sigma_min`；
- `step_index`：当前索引，`0 ≤ step_index ≤ N−1`；
- `state`：`None` = 全新开始（官方 `begin_index` 场景，任意 `step_index` 合法）；否则是上一次调用返回的 `state` dict；
- `noise`：**每次调用必需**（官方由 BrownianTree 采样器生成、依赖 generator/seed，属管线侧；本题改显式入参），形状须与 `model_output` 一致；
- `prediction_type` ∈ {`epsilon`, `v_prediction`}（官方 config 默认 `v_prediction`；注意本调度器**不支持 `sample`**）；
- `solver_order` ∈ {1, 2}、`solver_type` ∈ {`midpoint`, `heun`}、`lower_order_final`、`euler_at_final`、`final_sigmas_type` ∈ {`zero`, `sigma_min`}、`sigma_data` > 0：EDM 数据方差，默认 1.0。

## 计算规则

**σ 约定**：α ≡ 1（输入在进 unet 前已按 `c_in = 1/√(σ²+σd²)` 缩放），故 λ = log α − log σ_t ≡ **−log σ**；σ = 0（末位）时 λ = +∞（官方 `torch.log(0) = −∞` 的语义，后续 `exp(−∞) = 0`）。h = λ_t − λ_s0 ≥ 0；恒等式 **e^{−h} = σ_t/σ_s0** 给出两个闭式：1 阶 sample 系数 = (σ_t/σ_s0)²、衰减因子 decay = 1 − (σ_t/σ_s0)²。

**convert**（EDM 预条件，`precondition_outputs` 逐字）：σ = `sigmas[step_index]`、d = `sigma_data`：

- c_skip = d²/(σ²+d²)；
- `epsilon`：c_out = σ·d/√(σ²+d²)；`v_prediction`：c_out = **−**σ·d/√(σ²+d²)；其他抛 `ValueError`（官方文案 `Prediction type ... is not supported.`）；
- x0 = c_skip·sample + c_out·mo。

**history 移位**：`model_outputs = 旧表[1:] + [x0]`，长度 = `solver_order` 不变。

**降阶判定**：N = len(sigmas) − 1。lower_order_flag = (`step_index` == N−1) and (`euler_at_final` **或** (`lower_order_final` and N < 15) **或** `final_sigmas_type == "zero"`)。一阶条件 = `solver_order == 1` or `lower_order_nums < 1` or lower_order_flag；否则二阶（官方 `elif` 的 `solver_order == 2` 在 {1,2} 校验下恒真，`lower_order_second` 是死代码）。

**一阶 update**（`dpm_solver_first_order_update`，α_t = 1 已折叠）：σ_t = `sigmas[step_index+1]`、σ_s0 = `sigmas[step_index]`、h = λ_t − λ_s0：

- prev = (σ_t/σ_s0)·e^{−h}·sample + (1 − e^{−2h})·x0 + σ_t·√(1 − e^{−2h})·noise；
- 末步 zero 收尾时 σ_t = 0、h = +∞：三系数分别为 0、1、0 → **prev 恒等 x0**（官方 torch 语义）。

**二阶 update**（`multistep_dpm_solver_second_order_update`）：另取 σ_s1 = `sigmas[step_index−1]`、h_0 = λ_s0 − λ_s1、**r0 = h_0/h**：

- D0 = x0（移位后表末位）、D1 = (1/r0)·(x0 − model_outputs[−2])；
- `midpoint`：d1c = 0.5·(1 − e^{−2h})；`heun`：d1c = (1 − e^{−2h})/(−2h) + 1；
- prev = (σ_t/σ_s0)·e^{−h}·sample + (1 − e^{−2h})·D0 + d1c·D1 + σ_t·√(1 − e^{−2h})·noise；
- h = 0 或 h_0 = 0（相邻 σ 相等）时官方产生 0/0 未定义结果，本题拒绝（`ValueError`）。

**返回 dict**：`prev_sample` / `pred_original_sample`（float32）、`used_first_order`（bool）、`this_order`（1 或 2）、`r0` / `d1_coefficient`（浮点，一阶时为 None）、`state`。新 `state` = {`model_outputs`, `lower_order_nums`（小于 `solver_order` 时 +1）}。校验失败一律抛 `ValueError`。

## 示例

```
sigmas = [3.0, 1.0, 0.0] (N=2, zero ending), solver_order=2, midpoint,
prediction_type=v_prediction, sigma_data=1.0, lower_order_final=True

step 0 (state=None): mo = 0.5, sample = 1.0, noise = 0.2
  c_skip = 1/(3^2+1) = 1/10, c_out = -3/sqrt(10)
  x0 = (1/10)*1.0 - (3/sqrt(10))*0.5 = -0.374342
  h = lam(1) - lam(3) = log(3), e^{-h} = 1/3
  c_sample = (1/3)*(1/3) = 1/9, decay = 1 - 1/9 = 8/9
  noise term = 1*sqrt(8/9)*0.2 = 0.188562
  prev = (1/9)*1.0 + (8/9)*(-0.374342) + 0.188562 = -0.033075
  state = {model_outputs: [None, -0.374342], lower_order_nums: 1}

step 1 (state from step 0, last step): mo = 0.3, sample = -0.033075, noise = 0.1
  c_skip = 1/2, c_out = -1/sqrt(2)
  x0 = (1/2)*(-0.033075) - (1/sqrt(2))*0.3 = -0.228670
  lower_order_flag: step_index == N-1 and final_sigmas_type == "zero" -> True
  sigma_t = 0 -> lam_t = +inf, h = inf: e^{-h} = 0, decay = 1
  c_sample = 0, noise term = 0
  prev = 0*sample + 1*x0 + 0*noise = x0 = -0.228670
  state = {model_outputs: [-0.374342, -0.228670], lower_order_nums: 2}
```
