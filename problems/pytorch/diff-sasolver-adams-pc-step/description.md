## 题意

SA-Solver（Xue et al. 2023, arXiv 2309.05019）把扩散采样做成一对随机多步法：**SA-Predictor**（随机 Adams-Bashforth，显式）负责向前迈一步，**SA-Corrector**（随机 Adams-Moulton，隐式）负责在下一步用新拿到的网络输出**回头修正**上一步的落点。`SASolverScheduler.step()` 每次调用按固定顺序跑完「convert → corrector → history 移位 → warmup 定阶 → predictor」五拍，靠一组内部状态（`model_outputs` 历史、`last_sample`、`last_noise`、`lower_order_nums`、`this_corrector_order`）把 N 次调用串成一条采样轨迹。本题要求实现这台状态机的**单调用**，状态显式经 `state` 进出。

判题输入：

- `model_output` / `sample`：同形状浮点张量；
- `sigmas`：一维浮点张量，官方 `set_timesteps` 产出的表 `[σ0, …, σ_{N−1}, σ_last]`（长度 N+1，N ≥ 2），**全为正**（官方各 schedule 均无 0 收尾）、有限、单调不增；
- `step_index`：当前索引，`0 ≤ step_index ≤ N−1`；
- `state`：`None` = 冷启动（含 image-to-image 的中段冷启动）；否则是上一次调用返回的 `state` dict；
- `noise`：**每次调用必需**（官方每步由 generator 生成，本题改显式入参），形状须与 `model_output` 一致——即使 τ = 0 也要给（官方照样抽取并暂存给下一步 corrector）；
- `tau` / `tau_corrector`：随机性系数（官方 `tau_func(t)` 只输出 0 或 1）。corrector 在**上一步**的 timestep 上取值、predictor 在**当前步**的 timestep 上取值，故拆成两个入参；无 corrector 的调用忽略 `tau_corrector`；
- `predictor_order` / `corrector_order` ∈ [1, 4]：配置阶数；`lower_order_final`：末段降阶开关（官方默认 True）；
- `prediction_type` ∈ {`epsilon`, `v_prediction`}。

本题钉死官方默认路径：`algorithm_type = data_prediction`（predict_x0 = True）、无 thresholding、无 flow 分支。

## 计算规则

**σ 约定**（每次调用由 `sigmas` 重算，纯函数）：α = 1/√(σ²+1)、σ_t = σ·α、λ = log α − log σ_t。数学上 λ ≡ −log σ、e^λ ≡ 1/σ（VE 约定的恒等式，官方仍走双 log 路径，本题逐字保留）。

**convert**（`prediction_type`）：在 σ_now = `sigmas[step_index]` 处把网络输出转成 x0 预测：

- `epsilon`：x0 = (sample − σ_t·mo)/α = √(σ²+1)·sample − σ·mo；
- `v_prediction`：x0 = α·sample − σ_t·mo。

**corrector**（仅当 `step_index > 0` 且 `state` 给出）：阶数 tc 取**上一次调用保存的** `state["this_corrector_order"]`，τ 用 `tau_corrector`：

- σ_t 在 `sigmas[step_index]`、σ_s0 在 `sigmas[step_index−1]`；h = λ_t − λ_s0；
- λ 节点表 `[λ(sigmas[step_index−k]), k = 0..tc−1]`；
- 系数 = Lagrange(tc−1, 节点) × 正指数积分 ∫exp(x(1+τ²))·x^(tc−1−j)dx（区间 [λ_s0, λ_t]，官方 `get_coefficients_exponential_positive` 换元逐字）；
- **tc = 2 修正**（UniPC 同款 O(h³) 项）：Δ = exp((1+τ²)λ_t)·(h/2 − (h(1+τ²) − 1 + exp(−(1+τ²)h))/((1+τ²)²·h))，coefs[0] += Δ、coefs[1] −= Δ；
- 模型输出表 = `state["model_outputs"]`（**移位前**）+ [x0]，取末 tc 条；
- gradient = Σ_k (1+τ²)·σ_t·exp(−τ²λ_t)·coefs[k]·mo_list[−(k+1)]；
- noise_part = σ_t·√(1 − exp(−2τ²h))·`state["last_noise"]`——**上一步 predictor 注入的那段噪声**，保证两侧随机性一致；
- corrected = exp(−τ²h)·(σ_t/σ_s0)·`state["last_sample"]` + gradient + noise_part（出发点是上一次 predictor **之前**的样本）。

**history 移位**：`model_outputs = 旧表[1:] + [x0]`，长度 M = max(predictor_order, corrector_order − 1) 不变（−1 正是给 corrector 的 x0 腾位）。

**warmup 定阶**：`lower_order_final` 时 tp = min(predictor_order, N − step_index)、tc_new = min(corrector_order, N − step_index + 1)，否则直接取配置阶数；再统一 tp = min(tp, lower_order_nums + 1)、tc_new = min(tc_new, lower_order_nums + 2)。**末步（step_index = N−1）predictor 必降到 1 阶**；warmup 保证 history 的 None 头永不被读。

**predictor**：阶数 tp、τ 用 `tau`：

- σ_t 在 `sigmas[step_index+1]`、σ_s0 在 `sigmas[step_index]`；节点表同构；
- **tp = 2 修正**：Δ = exp((1+τ²)λ_t)·(h²/2 − (h(1+τ²) − 1 + exp(−(1+τ²)h))/(1+τ²)²)/(λ_s0 − λ(sigmas[step_index−1]))——分母是**相邻 λ 差**，与 corrector 版不同公式；step 0 因 warmup tp = 1 永不触雷；
- 模型输出表 = **移位后**的 history（x0 在末位），取末 tp 条；
- gradient / noise_part 公式同 corrector，noise 用**本轮入参** `noise`；
- prev = exp(−τ²h)·(σ_t/σ_s0)·(corrected 或 sample) + gradient + noise_part。

**返回 dict**：`prev_sample` / `pred_original_sample`（float32）、`corrected_sample`（float32，无 corrector 时 None）、`used_corrector`（bool）、`this_predictor_order` / `this_corrector_order`（本轮 warmup 后的实际阶数）、`predictor_coefficients` / `corrector_coefficients`（系数列表，浮点；无 corrector 时后者为 None）、`state`。新 `state` = {`model_outputs`, `last_sample`（corrected 或 sample）, `last_noise`（本轮 noise）, `lower_order_nums`（小于 M 时 +1）, `this_corrector_order` = tc_new}。校验失败一律抛 `ValueError`。

## 示例

```
sigmas = [4/3, 3/4, 1/4], orders 2/2, lower_order_final = True, tau = 0

step 0 (state = None): mo = 0.3, sample = 1.0, noise = 0.9
  alpha(4/3) = 3/5, sigma_t(4/3) = 4/5
  x0 = sqrt(25/9)*1.0 - (4/3)*0.3 = 5/3 - 2/5 = 19/15
  tp = min(2, 2-0) capped by (0+1) -> 1 (warmup), tc_new = min(2, 3, 0+2) = 2
  coef[0] = exp(lam(3/4)) - exp(lam(4/3)) = 1/(3/4) - 1/(4/3) = 4/3 - 3/4 = 7/12
  noise_part = (3/5)*sqrt(1-exp(0))*noise = 0   (tau = 0, but noise still stored)
  prev = (3/5)/(4/5)*1.0 + (3/5)*(7/12)*(19/15) = 3/4 + 133/300 = 179/150 = 1.1933
  state = {model_outputs: [None, 19/15], last_sample: 1.0, last_noise: 0.9,
           lower_order_nums: 1, this_corrector_order: 2}

step 1 (state from step 0): mo = -0.2, sample = 179/150, noise = 0.9, tau_corrector = 0
  x0 = (5/4)*(179/150) - (3/4)*(-0.2) = 197/120 = 1.6417
  corrector (tc = 2 saved): coefs = [0.3836, 0.1998]  (order-2 mod, delta = 0.0641)
  corrected = (3/5)/(4/5)*1.0 + (3/5)*(0.3836*(197/120) + 0.1998*(19/15)) = 1.2796
  predictor: lower_order_final -> tp = min(2, 2-1) = 1 (final step drops to order 1)
  coef[0] = 1/(1/4) - 1/(3/4) = 4 - 4/3 = 8/3
  prev = (1/sqrt(17))/(3/5)*1.2796 + (1/sqrt(17))*(8/3)*(197/120) = 1.5790
```
