import math

import torch


def solve(
    model_output,
    sample,
    sigmas,
    step_index,
    state=None,
    prediction_type="epsilon",
):
    """KDPM2（确定性 DPM-Solver-2）交错表上的两调用状态机单步。

    返回 dict(prev_sample, pred_original_sample, state, derivative, dt)。

    入参 `sigmas` 是未展开的基础表：长度 N+1（N ≥ 2），[σ0, …, σ_{N−1}, 0]，
    有限、非负、单调不增，前 N 个 > 0、末位恰为 0.0。

    你需要：
    1. 构造中点表 I[k] = √(σ[k−1]·σ[k])（k = 1…N）、I[0] = 0，以及两张
       长度 2N+2 的展开表 sigE / sigIE（头 1 条 + 主体 repeat_interleave(2) + 尾 1 条）；
    2. 按 step_index 的奇偶与 state 是否为 None 分派首调/次调：
       - 首调（state=None，偶数 i）：σ = sigE[i]，中点 m = sigIE[i+1]，
         x0 = pred(sample, mo, σ)，derivative = (sample − x0)/σ，dt = m − σ，
         prev = sample + derivative·dt，state = {"sample": sample}；
       - 次调（state 为 dict，奇数 i，1 ≤ i ≤ 2N−3）：σ = sigE[i−1]，
         m = sigIE[i]，σ_next = sigE[i]，x0 = pred(sample, mo, m)，
         derivative = (sample − x0)/m，dt = σ_next − σ（整步长），
         prev = state["sample"] + derivative·dt（从暂存样本出发），state = None；
    3. pred 两种类型：epsilon → x − σ_in·mo；
       v_prediction → mo·(−σ_in/√(σ_in²+1)) + x/(σ_in²+1)。

    返回 prev_sample / pred_original_sample（float32）、state、derivative、dt。
    """
    raise NotImplementedError
