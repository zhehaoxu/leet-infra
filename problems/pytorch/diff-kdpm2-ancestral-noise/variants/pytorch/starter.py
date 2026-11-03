import math

import torch


def solve(
    model_output,
    sample,
    sigmas,
    step_index,
    state=None,
    noise=None,
    prediction_type="epsilon",
):
    """KDPM2-Ancestral：交错表 + σ_up/σ_down 噪声拆分的两调用状态机单步。

    返回 dict(prev_sample, pred_original_sample, state, derivative, dt,
    sigma_up, sigma_down)。

    入参 `sigmas` 是未展开基础表 [σ0, …, σ_{N−1}, 0]（N ≥ 2，前 N 个为正、
    末位恰为 0、单调不增）。辅助表（官方 set_timesteps）：
      up[k]   = √(σ_next[k]²·(σ[k]² − σ_next[k]²) / σ[k]²)
      down[k] = √(σ_next[k]² − up[k]²)，再强制 down[N] = 0
      interpol[k] = exp(lerp(log σ[k], log down[k], 0.5))，再强制 [-2:] = 0
    四张表（基础 σ + up/down/interpol）都按「头 1 条 + repeat_interleave(2) + 尾 1 条」
    展开成长度 2N+2。

    首调（state=None，step_index 偶数，0 ≤ i ≤ 2N−2）：σ = sigE[i]、
    m = sigIE[i]（interpol 中点，ancestral 无 [0,…] 死头，首调与次调读同一格）；
    x0 = pred(sample, mo, σ)；derivative = (sample − x0)/σ；dt = m − σ；
    prev = sample + derivative·dt；state = {"sample": sample}（只存原样本）；
    noise 被忽略；返回 sigma_up / sigma_down 为 None。

    次调（state 为只含 sample 的 dict，step_index 奇数，1 ≤ i ≤ 2N−3）：
    σ = sigE[i−1]、m = sigIE[i−1]、u = sigUp[i−1]、d = sigDown[i−1]；
    x0 = pred(sample, mo, m)；derivative = (sample − x0)/m（分母是中点）；
    dt = d − σ（不是整步长——噪声吃掉 up 部分）；
    prev = state["sample"] + derivative·dt + noise·u（从暂存样本出发并注入噪声，
    noise 必需且形状须与 model_output 一致）；state = None；首调导数被丢弃。
    末逻辑步（j = N−1）只有首调：σ_next = 0 → up = down = interpol = 0，一步落到 x0。

    prediction_type：epsilon → sample − sigma_input·mo；
    v_prediction → mo·(−sigma_input/√(sigma_input²+1)) + sample/(sigma_input²+1)
    （sigma_input 首调用 σ、次调用 m）。sigma_hat = σ·(gamma+1)，gamma = 0 恒等。

    校验失败一律抛 ValueError。
    """
    raise NotImplementedError
