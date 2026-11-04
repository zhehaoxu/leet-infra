## 题意

LMS-Discrete（`LMSDiscreteScheduler`）是最经典的线性多步法（Linear Multi-Step）采样器：把 ODE 的右端项 `dσ = (sample − x0)/σ` 的**历史导数**攒起来，用一条过最近若干个数据点的 Lagrange 插值多项式在 σ 空间上积分，一步外推到下一个 σ。它的全部机制浓缩在两件事上：① 系数 = **σ 空间 Lagrange 基的积分**（官方用 `scipy.integrate.quad` 现算，每次调用逐系数重算）；② **导数历史**的追加与封顶淘汰（保留最近 order 条）。本题要求实现单调用，历史经 `state` 显式进出。

判题输入：

- `model_output` / `sample`：同形状浮点张量；
- `sigmas`：一维浮点张量，官方 `set_timesteps` 的表 `[σ0, …, σ_{N−1}, 0]`（长度 N+1，N ≥ 2），前 N 个为正、末位恰为 0、有限、单调不增（与 DPMSolver/KDPM2 同款 0 收尾）；
- `step_index`：当前索引，`0 ≤ step_index ≤ N−1`；
- `state`：`None` = 冷启动；否则是上一次调用返回的 `state` dict（`step_index = 0` 时必须传 None）；
- `prediction_type` ∈ {`epsilon`, `v_prediction`, `sample`}（三分支官方全有）；
- `order` ∈ [1, 4]：线性多步法阶数（官方默认 4）。

## 计算规则

每次调用四拍（官方 `step` L545-623 逐字同构）：

**1. convert**：在 σ = `sigmas[step_index]` 处把网络输出转成 x0 预测：

- `epsilon`：x0 = sample − σ·mo；
- `v_prediction`：x0 = mo·(−σ/√(σ²+1)) + sample/(σ²+1)；
- `sample`：x0 = mo。

**2. 导数与历史**：derivative = (sample − x0)/σ（分母是当前 σ，前 N 个 σ 为正故必有限）；追加进历史，超长 pop 头部——保留最近 `order` 条。

**3. LMS 系数**：`order_used = min(step_index + 1, order)`（warmup：前几步历史不足自动降阶）。对 curr_order = c ∈ [0, order_used) 取官方 `get_lms_coefficient(order_used, step_index, c)`：

- 被积函数 ℓ_c(τ) = Π_{k∈[0,order_used), k≠c} (τ − σ_{t−k}) / (σ_{t−c} − σ_{t−k})——Lagrange 基（k=c 时为 1、其余节点为 0）；
- 在区间 [`sigmas[step_index]`, `sigmas[step_index+1]`] 上积分。**σ 表递减 → 区间反向 → 积分为负**（官方 `quad` 同款语义）；末步区间 [σ_{N−1}, 0] 积到 0 收尾；
- 本题用**幂系数展开解析积分**（ℓ_c 是 ≤3 次多项式，展开后逐项 ∫τⁱ = (b^{i+1} − a^{i+1})/(i+1)）——与官方 `quad(epsrel=1e-4)` 数学等价（quad 对光滑多项式实际精度 ~1e-12）。

**4. 落地**：prev = sample + Σ_c coeffs[c]·derivs[−(c+1)]——**coeff[0] 配最新导数**（官方 `zip(lms_coeffs, reversed(self.derivatives))`）。

**返回 dict**：`prev_sample` / `pred_original_sample` / `derivative`（float32）、`lms_coefficients`（系数列表，浮点）、`order_used`（本轮实际阶数）、`state` = {`derivatives`: 导数历史}（float64 张量）。调用前历史长度契约 = min(step_index, order)、返回后 = min(step_index + 1, order)——LMS 历史从第一步起全是真实导数，**没有 None 头**。校验失败一律抛 `ValueError`。

## 示例

```
sigmas = [14, 7, 3, 0.8, 0.1, 0], order = 4

step 0 (state = None): mo = 0.4, sample = 2.0
  x0 = 2.0 - 14*0.4 = -3.6, derivative = (2.0 - (-3.6))/14 = 0.4
  order_used = min(0+1, 4) = 1
  coef[0] = integral over [14, 7] of 1 dtau = 7 - 14 = -7
  prev = 2.0 + (-7)*0.4 = -0.8          (first step degenerates to Euler)
  state = {derivatives: [0.4]}

step 1 (state from step 0): mo = -0.1, sample = -0.8
  x0 = -0.8 - 7*(-0.1) = -0.1, derivative = (-0.8 - (-0.1))/7 = 0.1
  order_used = min(1+1, 4) = 2, history = [0.4, 0.1]
  nodes = (sigma[1], sigma[0]) = (7, 14)
  l_0(tau) = (tau - 14)/(7 - 14), l_1(tau) = (tau - 7)/(14 - 7)
  coef[0] = integral over [7, 3] of l_0 = -36/7, coef[1] = integral of l_1 = 8/7
  (partition of unity: coef[0] + coef[1] = sigma[2] - sigma[1] = -4)
  prev = -0.8 + (-36/7)*0.1 + (8/7)*0.4 = -0.8 - 0.5143 + 0.4571 = -0.8571
  state = {derivatives: [0.4, 0.1]}
```
