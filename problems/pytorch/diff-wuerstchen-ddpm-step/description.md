## 题意

`DDPMWuerstchenScheduler` 是 Wuerstchen / Kandinsky / Stable Cascade 等两级扩散流水线的调度器（prior 与 decoder 各持一个实例，靠 `scaler` 区分）。它与普通 DDPM 的根本差异：**没有离散 β/ᾱ 表**——ᾱ 是连续 t ∈ [0,1] 的**解析 cosine 函数**（带 s offset、除以 `_init_alpha_cumprod` 归一化、clamp 到 [0.0001, 0.9999]），再经 `scaler` 幂变换扭曲时间轴；`prev_t` 不是显式入参而是**查表**（在 `set_timesteps` 产出的表上做 argmin 最近邻取下一位）；后验一步在 `prev_t` 精确为 0 时**门控掉噪声项**（DDIM 式确定性收尾）。本题要求实现它的单步 `step`，噪声显式入参。

判题输入：

- `model_output` / `sample` / `noise`：同形状浮点张量，batch 维 B 与 `timestep` 一致；
- `timestep`：一维浮点张量（batch 维），值域 [0, 1]，**batch 内全部相等**（官方管线每步共享一个 t；官方只用 `timestep[0]` 查 prev_t）；
- `timesteps_table`：一维浮点张量，官方 `set_timesteps` 产出的 `linspace(1.0, 0.0, N+1)`，长度 ≥ 2、单调不增；
- `noise`：**每次调用必需**（官方 `randn_tensor` 生成）；
- `scaler` > 0（默认 1.0；>1 / <1 / =1 三分支）、`s` > 0（默认 0.008）。

## 计算规则

**ᾱ 函数**（官方 `_alpha_cumprod` 逐字）：

- `_init_alpha_cumprod` = cos(s/(1+s)·π/2)²；
- scaler 变换：scaler > 1 → t = 1−(1−t)^scaler；scaler < 1 → t = t^scaler；=1 恒等；
- ᾱ(t) = cos((t+s)/(1+s)·π/2)² / _init_alpha_cumprod，clamp 到 [0.0001, 0.9999]——t=1 → 0 触发下界、t=0 → 1 触发上界。

**查表**（官方 `previous_timestep` 逐字）：index = argmin(|timesteps_table − timestep[0]|)（平局取首个，torch 语义），prev_t = timesteps_table[index+1]；index+1 越界（timestep 已是表尾）拒绝。

**后验一步**（官方 `step` 逐字）：ᾱ_t = ᾱ(timestep)、ᾱ_prev = ᾱ(prev_t)（各 view 成 (B, 1, …)）、α = ᾱ_t/ᾱ_prev：

- mu = √(1/α)·(sample − (1−α)·model_output/√(1−ᾱ_t))；
- std = √((1−α)·(1−ᾱ_prev)/(1−ᾱ_t))·noise；
- 门控：prev = mu + std·(prev_t != 0)——prev_t 精确为 0（末步）不加噪声，浮点比较；
- ᾱ_t = ᾱ_prev（如相邻表值映射到同一 ᾱ）时 α = 1 → mu = sample、std = 0 → prev = sample（恒等步）。

**返回 dict**：`prev_sample` / `mu` / `alpha_cumprod` / `alpha_cumprod_prev` / `alpha_ratio`（float32）、`noise_applied`（bool）、`prev_t`（浮点标量）。校验失败一律抛 `ValueError`。

## 示例

```
timesteps_table = [1.0, 0.75, 0.5, 0.25, 0.0], scaler = 1.0, s = 0.008
t = 0.5 (B=1), mo = -0.2, sample = 0.8, noise = 0.1

init = cos(0.008/1.008 * pi/2)^2 = 0.999845
ac  = cos(0.508/1.008 * pi/2)^2 / init = 0.493844
acp = cos(0.258/1.008 * pi/2)^2 / init = 0.847012
alpha = ac / acp = 0.583042   (< 1: t 越大 ᾱ 越小)
prev_t = table[argmin(|tab - 0.5|) + 1] = 0.25 -> noise_applied = True

mu  = sqrt(1/0.583042) * (0.8 - (1-0.583042)*(-0.2)/sqrt(1-0.493844))
    = 1.201215
std = sqrt(0.416958*0.152988/0.506156) * 0.1 = 0.035500
prev = mu + std = 1.236716
```
