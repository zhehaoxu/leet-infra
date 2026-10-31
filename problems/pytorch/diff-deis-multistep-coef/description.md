## 题意

DPMSolver（520 题）用固定 β 系数表做多步更新；DEIS（Diffusion Explicit Integral Sampler，2204.13902）走另一条路：把 ODE 积分**精确**做在 log-σ 空间——对插值多项式逐段解析积分，得到随 σ 区间变化的 `ind_fn` 系数。本题复现 diffusers `DEISMultistepScheduler` 的单步更新。

判题输入：

- `sample` / `model_output`：同形状浮点张量；
- `sigmas`：长度 N+1 的一维浮点张量（N 个推理步，**末位是上游追加的 `sigma_last` 而非 0**），全为正、单调不增；
- `step_index`：当前步索引 `[0, N-1]`；
- `converted_history`：本步**之前**各步已转换输出 `d` 的列表（最新在尾）——上游 `model_outputs` 移位窗口的可见部分；
- `solver_order` ∈ {1,2,3}、`lower_order_final`、`prediction_type`、`use_flow_sigmas`。

## 计算规则

**第 1 步：α/σ 变换**（上游 `_sigma_to_alpha_sigma_t`）：

- 常规：`α = 1/√(σ²+1)`，`σ_t = σ·α`；
- flow（`use_flow_sigmas=True`，σ 须 ∈ (0,1)）：`α = 1−σ`，`σ_t = σ`。

**第 2 步：模型输出转换**（`convert_model_output`，用当前步 `sigmas[step_index]`）：

| prediction_type | x0 |
| --- | --- |
| `epsilon` | `sample − σ_c·mo` |
| `sample` | `mo` |
| `v_prediction` | `α_c·sample − σ_c·mo` |
| `flow_prediction` | `sample − sigmas[step_index]·mo`（**用原始 sigma**，不经 α/σ 变换） |

再投到 deis 口径：`d = (sample − α_c·x0) / σ_c`。

**第 3 步：降阶选择**（`low = len(converted_history)`）：

```
lof = (step_index == N-1) and lower_order_final and N < 15
los = (step_index == N-2) and lower_order_final and N < 15
一阶：solver_order==1 或 low < 1 或 lof
二阶：否则若 solver_order==2 或 low < 2 或 los
三阶：否则
```

（上游 DEIS 的降阶条件比 DPMSolver 少 `euler_at_final` 与 `final_sigmas_type` 两项。）

**第 4 步：更新公式**：

- 一阶（DDIM 等价，`sigmas[i] → sigmas[i+1]`）：`h = λ_t − λ_s`（`λ = ln α − ln σ_t`），
  `x = (α_t/α_s)·sample − σ_t·(e^h − 1)·d`；
- 二阶（`m1 = history[-1]`）：`ρ = σ/α`，
  `ind2(t,b,c) = t·(−ln c + ln t − 1)/(ln b − ln c)`，
  `coef1 = ind2(ρt,ρ0,ρ1) − ind2(ρ0,ρ0,ρ1)`，`coef2 = ind2(ρt,ρ1,ρ0) − ind2(ρ0,ρ1,ρ0)`，
  `x = α_t·(sample/α_0 + coef1·d + coef2·m1)`；
- 三阶（另用 `m2 = history[-2]`）：`ind3` 为 log-ρ 二次多项式的精确积分（见 solution），
  三个系数各由「区间端点对」差分给出，`x = α_t·(sample/α_0 + c1·d + c2·m1 + c3·m2)`。

返回 `{"converted", "used_order", "x_next", "coefs"}`：`coefs` 一阶为 `[]`、二阶/三阶为系数列表（python float）。

## 手算示例

```
sample = [2, -1]，mo = [1, 1]，sigmas = [8, 2, 0.5]，step 0（一阶）
σ_s = 8/√65，α_s = 1/√65 → λ_s = −ln 8；σ_t = 2/√5，α_t = 1/√5 → λ_t = −ln 2
h = ln 4 → e^h − 1 = 3
x0 = [1.007722, −1.992278]；d = [1.889599, −0.758747]
x = √13·[2,−1] − (2/√5)·3·d = [2.140776, −1.569618]
```

**降阶示例**（`sigmas=[10,4,1,0.5]`，N=3，`solver_order=3`）：

- step 1（history 1 条）：预热不足 → 二阶，即使 `lower_order_final=False`；
- step 2 末步：N=3 < 15 → 一阶；`lower_order_final=False` 且 history 满 2 条 → 三阶；
- N=15 的调度器末步**不降**（不满足 <15）；N=14 的末步强制一阶。

**convert 反例**：`flow_prediction` 与 `epsilon` 在常规 σ 下转换结果不同——前者用原始 σ（不受 `use_flow_sigmas` 影响），后者用变换后的 `σ_c`；flow 口径下两者 x0 的 σ 相同但 deis 投影的 `α_c` 不同（`1−σ` vs `1/√(σ²+1)`）。

## 边界与约束

- `sigmas` 全为**正**且单调不增（上游末位 `sigma_last > 0`；含 0 一律 `ValueError`）；
- `converted_history` 长度必须恰为 `min(step_index, solver_order)`（历史窗口与步数一致的契约）；
- `use_flow_sigmas=True` 时要求所有 σ ∈ (0,1)（由 log 定义域自然保证）；
- 二阶需 `step_index ≥ 1`、三阶需 `step_index ≥ 2`（由 history 契约兜底）；
- 内部 float64 计算，`x_next` 返回 float32；违例抛 `ValueError`。

## 与现有题的边界

- 520 `diff-dpm-solver-order-schedule`：只考 DPMSolver 的**阶数表**；本题考 DEIS 的**系数数值**与更新公式（DEIS 的降阶条件也更简）；
- 350 `diff-dpmpp-2m`：DPM++(2M) 的固定中点系数表，非 log-ρ 精确积分；
- 521 `diff-dpm-x0-conversion-threshold`：考 DPMSolver++ 的 convert + 动态阈值；本题 convert 的 deis 投影 `(sample−α·x0)/σ` 与四 prediction_type 分支不同；
- 522 `diff-euler-ancestral-step`：单步欧拉 + 祖先噪声（随机），与多步确定性 DEIS 无重叠。
