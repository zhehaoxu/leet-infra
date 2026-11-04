import math

import torch


def solve(
    model_output,
    sample,
    sigmas,
    step_index,
    state=None,
    prediction_type="epsilon",
    order=4,
):
    """LMS-Discrete：σ 空间 Lagrange 积分系数 + 导数历史的线性多步单步。

    返回 dict(prev_sample, pred_original_sample, derivative, lms_coefficients,
    order_used, state)。

    `sigmas = [σ0, …, σ_{N−1}, 0]`（长度 N+1，N ≥ 2），前 N 个为正、末位恰为 0、
    有限、单调不增。每次调用四拍：

    1. convert：σ = sigmas[step_index] 处转 x0 预测——epsilon → sample − σ·mo；
       v_prediction → mo·(−σ/√(σ²+1)) + sample/(σ²+1)；sample → x0 = mo。
    2. 导数：derivative = (sample − x0)/σ；追加进 state["derivatives"]，
       超长 pop 头部（保留最近 order 条）。
    3. 系数：order_used = min(step_index + 1, order)；对 c ∈ [0, order_used) 取
       get_lms_coefficient：ℓ_c(τ) = Π_{k≠c}(τ − σ_{t−k})/(σ_{t−c} − σ_{t−k})
       （Lagrange 基，k=c 时为 1、其余节点为 0），在 [σ_t, σ_{t+1}] 上积分
       （区间反向积分为负——σ 表递减）。幂系数展开后解析积分。
    4. 落地：prev = sample + Σ_c coeffs[c]·derivs[−(c+1)]——coeff[0] 配最新导数。

    state = {"derivatives": [导数历史]}；调用前长度契约 = min(step_index, order)。
    首步 order_used = 1：系数 = σ1 − σ0（负），prev = sample − (σ0 − σ1)·derivative
    ——退化为欧拉步。末步积分区间 [σ_{N−1}, 0] 积到 0 收尾。

    校验失败一律抛 ValueError。
    """
    raise NotImplementedError
