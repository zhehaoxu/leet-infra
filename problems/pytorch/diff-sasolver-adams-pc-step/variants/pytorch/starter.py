import math

import torch


def solve(
    model_output,
    sample,
    sigmas,
    step_index,
    state=None,
    noise=None,
    tau=0.0,
    tau_corrector=None,
    predictor_order=2,
    corrector_order=2,
    lower_order_final=True,
    prediction_type="epsilon",
):
    """SA-Solver：随机 Adams-Bashforth 预估 + Adams-Moulton 校正的单调用状态机。

    返回 dict(prev_sample, pred_original_sample, corrected_sample, used_corrector,
    this_predictor_order, this_corrector_order, predictor_coefficients,
    corrector_coefficients, state)。

    本题钉死官方默认路径：data_prediction（predict_x0 = True）、无 thresholding。
    `sigmas` 长度 N+1（N ≥ 2），全为正、有限、单调不增；α = 1/√(σ²+1)、
    σ_t = σ·α、λ = log α − log σ_t。每次调用五步：

    1. convert：x0 = conv(sample, mo, sigmas[step_index])；
       epsilon → (sample − σ_t·mo)/α；v_prediction → α·sample − σ_t·mo。
    2. corrector（step_index > 0 且 state 给出）：阶数 tc = 上一次调用保存的
       state["this_corrector_order"]，τ = tau_corrector（上一步 timestep 的取值）；
       σ_t 在 sigmas[step_index]、σ_s0 在 sigmas[step_index−1]；h = λ_t − λ_s0；
       节点表 [λ(sigmas[step_index−k]), k = 0..tc−1]；系数 = Lagrange(tc−1) ×
       正指数积分；tc = 2 时加 UniPC 同款 O(h³) 修正（coefs[0] += Δ、
       coefs[1] −= Δ，Δ = exp((1+τ²)λ_t)·(h/2 − (h(1+τ²) − 1 +
       exp(−(1+τ²)h))/((1+τ²)²·h))）；模型输出表 = 移位前 history + [x0] 取末
       tc 条；gradient = Σ (1+τ²)·σ_t·exp(−τ²λ_t)·coefs[k]·mo_list[−(k+1)]；
       noise_part = σ_t·√(1 − exp(−2τ²h))·state["last_noise"]；
       corrected = exp(−τ²h)·(σ_t/σ_s0)·state["last_sample"] + gradient + noise_part。
    3. history 移位：model_outputs = 旧表[1:] + [x0]，长度 M =
       max(predictor_order, corrector_order − 1) 不变。
    4. warmup：lower_order_final 时 tp = min(predictor_order, N − step_index)、
       tc_new = min(corrector_order, N − step_index + 1)；再 tp = min(tp,
       lower_order_nums + 1)、tc_new = min(tc_new, lower_order_nums + 2)；
       末步 predictor 必降到 1 阶。
    5. predictor：阶数 tp、τ = tau（当前步）；σ_t 在 sigmas[step_index+1]；
       节点表同构；tp = 2 修正的分母是 λ_s0 − λ(sigmas[step_index−1])；模型输出
       表 = 移位后 history 取末 tp 条；noise 用本轮入参 noise；
       prev = exp(−τ²h)·(σ_t/σ_s0)·(corrected 或 sample) + gradient + noise_part。

    state = {"model_outputs", "last_sample"(corrected 或 sample), "last_noise",
    "lower_order_nums"(小于 M 时 +1), "this_corrector_order" = tc_new}；
    warmup 保证 history 的 None 头永不被读。noise 每次调用必需。

    校验失败一律抛 ValueError。
    """
    raise NotImplementedError
