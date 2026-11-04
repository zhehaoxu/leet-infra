## 题意

PLMS（pseudo linear multistep）是 PNDM（Liu et al. 2022, arXiv 2202.09778）调度器的 **skip_prk_steps=True 全程路径**——Stable Diffusion 的默认用法：不做 prk 的 4 子步 Runge-Kutta 热身，从第一次调用起就用「ets 历史」跑 Adams-Bashforth 阶梯。它的机制浓缩在三件事：① **ets 记账**（截断到 3 条再追加当前输出；唯独 counter 1 不追加）；② **AB 阶梯**（按 ets 长度 1→2→3→4 依次用恒等 / 梯形 / 2 阶 / 3 阶 / 4 阶系数组合输出）；③ **counter 1 的重启对**（从 counter 0 暂存的样本在 (t+δ, t) 上重走整步）。落地用 PNDM 论文的 formula(9)。本题要求实现单调用，状态显式经 `state` 进出。

判题输入：

- `model_output` / `sample`：同形状浮点张量；
- `alphas_cumprod`：一维浮点张量，官方训练侧 ᾱ 表（长度 ≥ 2，全部 ∈ (0, 1]，单调不增；**ᾱ(0) = 1−β₀ ≈ 0.999 而非 1**——正是 formula(9) 分母不为零的保证）；
- `timestep`：当前训练时间步索引，`0 ≤ timestep ≤ len−1`；
- `step_ratio = num_train_timesteps // num_inference_steps ≥ 1`（步距 δ）；
- `counter`：全局调用计数（官方 `self.counter`，显式入参）；
- `state`：`None` ⇔ `counter = 0`；否则 {`ets`, `cur_sample`}；
- `prediction_type` ∈ {`epsilon`, `v_prediction`}。

## 计算规则

**state 契约**（官方 append/pop 记账的自然形状）：调用前 ets 长度恰为 [c1→1, c2→1, c3→2, c4→3, c≥5→4]（截断 `[-3:]` 发生在调用内，故 c≥5 的调用方持有 4 条）；`cur_sample` 仅 counter = 1 时必须非 None（counter ≥ 2 必须为 None）。

**1. ets 记账**：counter ≠ 1 时 `ets = ets[-3:]` 截断再 append 当前输出（旧 3 条 + 新 1 条 = 4 条封顶）；counter = 1 **不 append**（ets 尾仍是上一轮输出）。

**2. AB 阶梯**（按 append 后的 ets 长度选分支）：

- len 1 且 counter 0：组合 = mo 恒等，同时暂存 `cur_sample = sample`；
- len 1 且 counter 1：组合 = (mo + ets[-1]) / 2（两点梯形），且 **sample 换成 `state["cur_sample"]`**（从 counter 0 暂存的样本重启）；
- len 2：(3·ets[-1] − ets[-2]) / 2；
- len 3：(23·ets[-1] − 16·ets[-2] + 5·ets[-3]) / 12；
- len ≥ 4：(55·ets[-1] − 59·ets[-2] + 37·ets[-3] − 9·ets[-4]) / 24。

**3. timestep 记账**：counter 1 的 (timestep, prev_timestep) = (t+δ, t)——与 counter 0 的组合输出配同一对相邻格；其余 = (t, t−δ)。counter 1 的 t+δ 不得越过 ᾱ 表尾。

**4. 落地 formula(9)**：α_t = `alphas_cumprod[timestep]`、α_prev = `alphas_cumprod[prev_timestep]`（**prev_timestep < 0 时取表头 `alphas_cumprod[0]`**，官方 `final_alpha_cumprod`）；v_prediction 先把网络输出转回 ε：ε = √α_t·mo + √(1−α_t)·sample（sample 在 counter 1 已换成 cur_sample——转换发生在记账之后）；

- sample_coeff = √(α_prev/α_t)；
- denom = α_t·√(1−α_prev) + √(α_t·(1−α_t)·α_prev)；
- prev = sample_coeff·sample − (α_prev − α_t)·ε / denom。

**返回 dict**：`prev_sample` / `model_output_combined`（float32）、`prev_timestep`（int）、`state` = {`ets`, `cur_sample`}（float64）。末步 timestep = 0 → prev_timestep = −δ < 0 → α_prev 取表头；若 timestep 也为 0（t == prev）则 formula(9) 退化为恒等（sample_coeff = 1、α 差 = 0）。校验失败一律抛 `ValueError`。

## 示例

```
alphas = linear-beta cumprod (T=1000), timesteps = [999, 500, 500, 250, 0], delta = 200

counter 0 (state = None): t = 500, mo = 0.3, sample = 2.0
  ets = [0.3] (append), combined = 0.3 (identity), cur_sample = 2.0 (stashed)
  (timestep, prev_timestep) = (500, 300)
  prev = formula(9) with alpha_t = a[500], alpha_prev = a[300]

counter 1 (state from counter 0): t = 500, mo = -0.2
  no append (ets stays [0.3])
  combined = (0.3 + (-0.2)) / 2 = 0.05   (two-point trapezoid)
  sample = cur_sample = 2.0 (restart from the stashed sample!)
  (timestep, prev_timestep) = (700, 500)  <- same adjacent pair, rewalked

counter 2: ets = [m0, m2] (append) -> combined = (3*m2 - m0) / 2
counter 3: ets = [m0, m2, m3] -> combined = (23*m3 - 16*m2 + 5*m0) / 12
counter 4: ets = [m0, m2, m3, m4] -> combined = (55*m4 - 59*m3 + 37*m2 - 9*m0) / 24
counter >= 5: ets[-3:] truncates first, then appends -> stays at 4 entries
final step (t = 0): prev_timestep = -200 < 0 -> alpha_prev = alphas_cumprod[0]
```
