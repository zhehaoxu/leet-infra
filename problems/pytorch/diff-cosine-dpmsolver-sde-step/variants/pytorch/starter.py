import math

import torch


def solve(
    model_output,
    sample,
    sigmas,
    step_index,
    state=None,
    noise=None,
    prediction_type="v_prediction",
    solver_order=2,
    solver_type="midpoint",
    lower_order_final=True,
    euler_at_final=False,
    final_sigmas_type="zero",
    sigma_data=1.0,
):
    """CosineDPMSolverMultistep：EDM 预条件 + SDE 噪声的 multistep DPMSolver
    单调用状态机（Stable Audio Open / CosXL 使用的调度器）。

    返回 dict(prev_sample, pred_original_sample, used_first_order, this_order,
    r0, d1_coefficient, state)。

    `sigmas` 是官方 set_timesteps 产出的表：长度 N+1（N ≥ 2）、有限、单调不增；
    除末位外全为正（末位 = 0 或 sigma_min）。α_t ≡ 1 → λ = −log σ
    （σ = 0 → λ = +∞，torch 语义）。每次调用：

    1. state 解析：state = {"model_outputs"(长度 solver_order),
       "lower_order_nums"}；None 表示全新开始（任意 step_index 合法）；
       给 state 时 1 ≤ lower_order_nums ≤ min(solver_order, step_index)，
       model_outputs 前 solver_order−lower_order_nums 条为 None、末
       lower_order_nums 条为真实输出（与 sample 同形状）。
    2. convert（EDM 预条件）：σ = sigmas[step_index]、d = sigma_data；
       c_skip = d²/(σ²+d²)；epsilon → c_out = σ·d/√(σ²+d²)；
       v_prediction → c_out = −σ·d/√(σ²+d²)；其他抛 ValueError
       （"Prediction type ... is not supported."）。x0 = c_skip·sample + c_out·mo。
    3. history 移位：model_outputs = 旧表[1:] + [x0]。
    4. 降阶判定：lower_order_flag = (step_index == N−1) and (euler_at_final or
       (lower_order_final and N < 15) or final_sigmas_type == "zero")；
       一阶条件 = solver_order == 1 or lower_order_nums < 1 or
       lower_order_flag，否则二阶。
    5. 一阶 update：σ_t = sigmas[step_index+1]、σ_s0 = sigmas[step_index]、
       h = λ_t − λ_s0；prev = (σ_t/σ_s0)·e^{−h}·sample + (1 − e^{−2h})·x0 +
       σ_t·√(1 − e^{−2h})·noise（zero 收尾末步 σ_t = 0、h = +∞ → prev 恒等 x0）。
    6. 二阶 update：σ_s1 = sigmas[step_index−1]、h_0 = λ_s0 − λ_s1、
       r0 = h_0/h；D0 = x0、D1 = (1/r0)·(x0 − model_outputs[−2])；
       midpoint → d1c = 0.5·(1 − e^{−2h})；heun → d1c = (1 − e^{−2h})/(−2h) + 1；
       prev = (σ_t/σ_s0)·e^{−h}·sample + (1 − e^{−2h})·D0 + d1c·D1 +
       σ_t·√(1 − e^{−2h})·noise。h = 0 或 h_0 = 0 抛 ValueError（官方 0/0 未定义）。
    7. state 推进：lower_order_nums 在小于 solver_order 时 +1。

    noise 每次调用必需（官方 BrownianTreeNoiseSampler 属管线侧）。校验失败
    一律抛 ValueError。
    """
    raise NotImplementedError
