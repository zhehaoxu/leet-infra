import math

import torch


def solve(
    model_output,
    sample,
    alphas_cumprod,
    timestep,
    step_ratio,
    counter,
    state=None,
    prediction_type="epsilon",
):
    """PLMS（PNDM 的 skip_prk_steps=True 全程路径）：ets 历史的 Adams-Bashforth
    阶梯 + cur_sample 重启的单调用状态机。

    返回 dict(prev_sample, model_output_combined, prev_timestep, state)。

    `alphas_cumprod` 一维浮点张量（长度 ≥ 2，全部 ∈ (0, 1]，单调不增）；
    timestep / step_ratio / counter 均为整数。state = {"ets", "cur_sample"}；
    counter = 0 ⇔ state 为 None；counter ≥ 1 时 ets 长度契约
    [c1→1, c2→1, c3→2, c4→3, c≥5→3]，cur_sample 仅 counter = 1 时必须非 None。

    四拍：
    1. ets 记账：counter ≠ 1 时 ets[-3:] 截断 + append 当前输出；counter = 1
       不 append。
    2. AB 阶梯（按 append 后 ets 长度）：len 1 & counter 0 → 恒等（暂存
       cur_sample = sample）；len 1 & counter 1 → (mo + ets[-1])/2（梯形），且
       sample 换成 state["cur_sample"] 重启；len 2 → (3e[-1] − e[-2])/2；
       len 3 → (23e[-1] − 16e[-2] + 5e[-3])/12；len ≥ 4 →
       (55e[-1] − 59e[-2] + 37e[-3] − 9e[-4])/24。
    3. timestep 记账：counter 1 → (timestep, prev) = (t+δ, t)；否则 (t, t−δ)，
       δ = step_ratio；counter 1 的 t+δ 不得越过表尾。
    4. 落地 formula(9)：α_t = alphas[timestep]、α_prev = alphas[prev]（prev < 0
       取表头 alphas[0]）；v_prediction 先转 ε = √α_t·mo + √(1−α_t)·sample；
       prev = √(α_prev/α_t)·sample − (α_prev − α_t)·ε /
       (α_t·√(1−α_prev) + √(α_t·(1−α_t)·α_prev))。

    校验失败一律抛 ValueError。
    """
    raise NotImplementedError
