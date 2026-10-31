## 上游

- 仓库：`huggingface/diffusers`，`src/diffusers/schedulers/scheduling_deis_multistep.py`（v0.42.x，2026-10-31 经 contents API 下载；本地副本 `.workbuddy/upstream/d93/`）
- 论文：DEIS——Zhang et al., "Fast Sampling of Diffusion Models with Exponential Integrator"（2204.13902），原始仓库 `qsh-zh/deis`

## 逐行对照

| 上游 | 本题 |
| --- | --- |
| `convert_model_output` L581-639 | convert 步：四种 `prediction_type` 分支 + deis 投影 `(sample − α_t·x0)/σ_t`；注意 `flow_prediction` 用原始 `self.sigmas[step_index]`（L625），不经 α/σ 变换 |
| `_sigma_to_alpha_sigma_t` L437-457 | `alpha_sigma()`：`use_flow_sigmas → (1−σ, σ)`，否则 `(1/√(σ²+1), σ/√(σ²+1))` |
| `deis_first_order_update` L641-700 | 一阶：`x = (α_t/α_s)·sample − σ_t·(e^h − 1)·d`，`h = λ_t − λ_s` |
| `multistep_deis_second_order_update` L702-773 | `ind_fn(t,b,c) = t(−ln c + ln t − 1)/(ln b − ln c)`（L763-765，是 `(ln t − ln c)/(ln b − ln c)` 对 t 的精确积分）；coef1/coef2 由端点差分给出 L767-770 |
| `multistep_deis_third_order_update` L775-861 | `ind_fn(t,b,c,d)` L840-851（log-ρ 二次多项式积分）；coef1/2/3 以 `(ρ0,ρ1,ρ2)` 轮换 (b) 与 (c,d) 配对 L853-855 |
| `step` L917-979 | 降阶：`lower_order_final = (step_index == len(timesteps)-1) and config.lower_order_final and len < 15`（L951-956，**没有** DPMSolver 的 `euler_at_final`/`final_sigmas_type` 两项）；`model_outputs` 移位 L959-961；`lower_order_nums` 自增 L970-971 |

## 刻意偏离（已记录）

- **不依赖 diffusers**：判题环境无 diffusers；`sigmas` 为显式入参（调用方需要时可复现 `set_timesteps` 的 `interp + 追加 sigma_last`）。调度侧的 timestep/sigma 构造（linspace/leading/trailing、Karras/exponential/beta sigmas、flow sigmas）不在范围内——那是与 517/518 共享的调度族机制。
- **历史窗口契约**：上游用定长 `model_outputs` 列表（`None` 哨兵）加独立的 `lower_order_nums` 计数；本题强制 `len(converted_history) == min(step_index, solver_order)`——这正是降阶判断（`low < 1`、`low < 2`）用到的全部信息。
- **thresholding 不出**：`config.thresholding` 默认 False，且动态阈值已在 521（DPMSolver++ 侧）覆盖。
- **`solver_type` 只有 `"logrho"`**：上游把其他取值静默改写回 `logrho`（L231-235），实践中只存在这一条路径。
- 内部 float64 计算（上游 float32）；`x_next` 返回 float32。

## 测试设计

- 独立 numpy oracle（`_oracle_solve`）从论文积分公式重写，8 组随机三阶配置逐元素对拍（1e-9）。
- 手算一阶示例带硬编码锚点（λ_s = −ln 8、h = ln 4、e^h−1 = 3 → 整洁乘子）。
- 降阶矩阵：预热（step 1 → 二阶，即使无 lof）、N=3 末步降阶、`lower_order_final=False` 逃逸、N=15（不降）vs N=14（降一阶）在末步对照。
- 4 种 `prediction_type` 分支，含 flow（要求 σ ∈ (0,1)——越界时 `α = 1−σ < 0`，`ln α` 无定义）。
- 变异测试 8/8 抓获：`exp(h)−1 → h`、一阶基底 sample→d、系数端点互换、ind2 丢 `−1` 项、ind3 常数 `2→1`、m1/m2 历史槽位、`<15 → <20`、convert 减→加。
