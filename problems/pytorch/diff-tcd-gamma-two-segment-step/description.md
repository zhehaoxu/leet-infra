# 扩散：TCD gamma 中间噪声级与两段式一致性步

## 题意

TCD（Trajectory Consistency Distillation，arXiv 2410.05964）在 LCM 一致性模型上引入 gamma 插值与两段式采样。实现 `solve(model_output, sample, alphas_cumprod, timesteps, step_index, noise=None, eta=0.3, prediction_type="epsilon")`：

```text
prev_t = timesteps[step_index + 1]        # 越过表尾取 0
s      = floor((1 − eta) · prev_t)        # gamma 中间噪声级（论文的 gamma，代码的 eta）
# 参数化（epsilon 为例，sample / v_prediction 同理推导 x0 与 ε）：
x0 = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t
x_s = √ᾱ_s·x0 + √(1−ᾱ_s)·ε               # 第一段：跳到中间噪声级 s
# 第二段（eta > 0 且非末步）：
prev = √(ᾱ_prev/ᾱ_s)·x_s + √(1−ᾱ_prev/ᾱ_s)·noise
# 末步（step_index == len(timesteps)−1）或 eta == 0：prev = x_s
```

返回 `dict(prev_sample, pred_noised_sample, pred_original_sample, pred_epsilon, timestep_s, prev_timestep, noise_applied)`。

## 关键语义

- **两段式**：TCD 不直接从 `x_t` 走到 `x_prev`，而是先跳到 gamma 中间级 `x_s`（用同一个 ε 重新加噪），再从 `x_s` 加噪到 `prev`——这是轨迹一致性的边缘分布拼接。
- **eta 的两个作用**：① 缩短中间级 `s = floor((1−eta)·prev_t)`（eta=0 时 s 恰为 prev_t，两段合一）；② 控制第二段噪声注入（eta=0 完全确定性）。末步永远不加噪。
- **三种参数化**：epsilon / sample / v_prediction 各自推导 `x0` 与 `ε`，但 `x_s` 与 `prev` 两段公式完全共用。
- **prev_t 定位**：`timesteps[step_index+1]`，越过表尾取 0（官方 `torch.tensor(0)`）；官方 `final_alpha_cumprod` 分支（prev<0）不可达，属死代码。
- **校验**：timesteps 必须整型、**严格递减**（官方自定义路径校验）且落在 ᾱ 索引范围内；`eta ∈ [0,1]`（官方 assert，判题域 ValueError）；`eta > 0` 且非末步时必须显式给 noise，其余情况忽略。

## 判题域约定

- timestep/prev_t/timestep_s 是 `alphas_cumprod`（训练表，1D 浮点）的索引，判题域显式校验索引范围（官方越界会 IndexError）。
- 官方 `randn_tensor`（generator）在管线侧，noise 显式入参。
- 官方 eta 校验是 `assert`（AssertionError），判题域统一 ValueError 并保留 "gamma must be less than or equal to 1.0" 关键词。

## 例子

训练表 ᾱ 取 β=linspace(1e-4, 0.02, 50) 的 cumprod；推理表 `[49, 36, 23, 10, 0]`（N=5）、eta=0.3：step 2 的 `prev_t = 10`，`s = floor(0.7·10) = 7`；eta=0.25 时 `s = floor(7.5) = 7`（round 会给 8——floor 语义的判别案例）。单步表 `[49]` 的 step 0 即末步，`prev_t = 0`、`prev == x_s`。

数据规模：ᾱ 长度 50，表长 1–5，张量形状 `(1, 2)`–`(2, 2)`，float64。
