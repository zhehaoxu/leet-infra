# 扩散：SCM trigflow 一致性步（跳回数据端再重加噪）

## 题意

sCM（Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models）把扩散的 timestep 换成了**角度**：trigflow 参数化定义

```text
x(θ) = cos(θ)·x0 + sin(θ)·σ_data·ε,    θ = arctan(σ/σ_data) ∈ [0, π/2]
```

实现 `solve(model_output, sample, timesteps, step_index, noise=None, sigma_data=0.5, prediction_type="trigflow")`：给定角度表（单调不增，长度 N+1）与当前步索引，执行一致性步

```text
s = timesteps[step_index]        # 当前角度
t = timesteps[step_index + 1]    # 下一角度（更干净）
pred_x0     = cos(s)·sample − sin(s)·model_output
prev_sample = cos(t)·pred_x0 + sin(t)·(σ_data·noise)
```

返回 `dict(prev_sample, pred_original_sample, s_angle, t_angle, noise_term)`，其中 `noise_term = sin(t)·σ_data·noise`。

## 关键语义

- **一致性 = 跳回数据端再重加噪**：每步先用模型输出还原 `x0`，再直接从 `x0` 加噪到下一角度。这不是 Markov 链上的小步欧拉，而是 sCM 的非 Markov 跳跃。
- **单步采样自然免噪**：Sana-Sprint 单步表 `[θ_max, 0]` 的末步 `t = 0`，`sin(0) = 0` 使噪声项系数为零，`prev == pred_x0`。噪声不需要显式门控——三角函数自己完成。
- **noise=None 的合法性**：仅当 `t` **精确等于** 0.0 时可省略 noise；`t != 0` 时必须显式给。noise 非 None 但 `t == 0.0` 合法（官方语义是 randn 照常采样、但被乘零），此时 `prev` 与 noise 取值无关。
- **σ_data 缩放**：只作用于噪声项（官方 `randn_tensor(...) * sigma_data`），不影响 `pred_x0`。
- **相邻相等角度合法**：官方 `index_for_timestep` 专门处理表中重复值，判题域的单调不增校验允许相邻相等、只拒绝严格递增对。

## 判题域约定

- `step_index` 显式入参（官方 `_step_index` 状态机与 `begin_index` 样板外置），须满足 `0 ≤ step_index ≤ len(timesteps) − 2`。
- `noise` 为显式入参（官方 `randn_tensor` 在管线侧）；本函数内乘 `σ_data`。
- `prediction_type` 只支持 `"trigflow"`（官方唯一实现，其余值报 `Unsupported parameterization`）。
- 官方 `len(self.timesteps) > 1` 门控的 else 分支只在 `num_inference_steps=0` 的退化表下可达，属死代码，不入实现。

## 例子

两步 Sana-Sprint 默认表 `[1.57080, 1.3, 0]`（`set_timesteps(2, intermediate_timesteps=1.3)`）：

- step 0：`s = 1.57080, t = 1.3`，`cos(s) ≈ 0.0000007967`——起点几乎纯噪声；
- step 1：`s = 1.3, t = 0`，`sin(t) = 0`，噪声项消失，`prev == pred_x0`。

数据规模：`timesteps` 长度 2–32，张量形状 `(1, 2)`–`(2, 2)`，float64。
