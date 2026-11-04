import math

import torch

_PREDICTION_TYPES = ("epsilon", "v_prediction")


def _float_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def _sigma_to_alpha_sigma_t(sigma):
    """官方 _sigma_to_alpha_sigma_t（非 flow 分支）逐字：α = 1/√(σ²+1)，σ_t = σ·α。"""
    alpha = 1.0 / math.sqrt(sigma**2 + 1.0)
    return alpha, sigma * alpha


def _lambda(sigma):
    """官方 λ = log(α) − log(σ_t)；数学上恒等于 −log σ，但保留官方双 log 路径。"""
    alpha, sigma_t = _sigma_to_alpha_sigma_t(sigma)
    return math.log(alpha) - math.log(sigma_t)


def _exp_integral_positive(order, interval_start, interval_end, tau):
    """官方 get_coefficients_exponential_positive 逐字（predict_x0 分支）：
    ∫ exp(x(1+τ²))·x^order dx，先换元 x → x·(1+τ²)。order ∈ [0, 3]。"""
    if order not in (0, 1, 2, 3):
        raise ValueError("order is only supported for 0, 1, 2 and 3")
    c = 1.0 + tau**2
    end_cov = c * interval_end
    start_cov = c * interval_start
    decay = math.exp(-(end_cov - start_cov))
    if order == 0:
        return math.exp(end_cov) * (1.0 - decay) / c
    if order == 1:
        return math.exp(end_cov) * ((end_cov - 1.0) - (start_cov - 1.0) * decay) / c**2
    if order == 2:
        return (
            math.exp(end_cov)
            * ((end_cov**2 - 2.0 * end_cov + 2.0) - (start_cov**2 - 2.0 * start_cov + 2.0) * decay)
            / c**3
        )
    return (
        math.exp(end_cov)
        * (
            (end_cov**3 - 3.0 * end_cov**2 + 6.0 * end_cov - 6.0)
            - (start_cov**3 - 3.0 * start_cov**2 + 6.0 * start_cov - 6.0) * decay
        )
        / c**4
    )


def _lagrange_polynomial_coefficient(order, lambda_list):
    """官方 lagrange_polynomial_coefficient 逐字：过点 (λ_i, i) 的 Lagrange 多项式
    展开系数（∑ c_k x^k），order 为多项式次数（= 阶数 − 1）。"""
    if order == 0:
        return [[1.0]]
    if order == 1:
        l0, l1 = lambda_list
        d01 = l0 - l1
        return [
            [1.0 / d01, -l1 / d01],
            [1.0 / (l1 - l0), -l0 / (l1 - l0)],
        ]
    if order == 2:
        l0, l1, l2 = lambda_list
        d1 = (l0 - l1) * (l0 - l2)
        d2 = (l1 - l0) * (l1 - l2)
        d3 = (l2 - l0) * (l2 - l1)
        return [
            [1.0 / d1, (-l1 - l2) / d1, l1 * l2 / d1],
            [1.0 / d2, (-l0 - l2) / d2, l0 * l2 / d2],
            [1.0 / d3, (-l0 - l1) / d3, l0 * l1 / d3],
        ]
    l0, l1, l2, l3 = lambda_list
    d1 = (l0 - l1) * (l0 - l2) * (l0 - l3)
    d2 = (l1 - l0) * (l1 - l2) * (l1 - l3)
    d3 = (l2 - l0) * (l2 - l1) * (l2 - l3)
    d4 = (l3 - l0) * (l3 - l1) * (l3 - l2)
    return [
        [1.0 / d1, (-l1 - l2 - l3) / d1, (l1 * l2 + l1 * l3 + l2 * l3) / d1, (-l1 * l2 * l3) / d1],
        [1.0 / d2, (-l0 - l2 - l3) / d2, (l0 * l2 + l0 * l3 + l2 * l3) / d2, (-l0 * l2 * l3) / d2],
        [1.0 / d3, (-l0 - l1 - l3) / d3, (l0 * l1 + l0 * l3 + l1 * l3) / d3, (-l0 * l1 * l2) / d3],
        [1.0 / d4, (-l0 - l1 - l2) / d4, (l0 * l1 + l0 * l2 + l1 * l2) / d4, (-l0 * l1 * l2) / d4],
    ]


def _coefficients(order, interval_start, interval_end, lambda_list, tau):
    """官方 get_coefficients_fn（predict_x0 分支）：Lagrange 系数 × 正指数积分。"""
    lagrange_coefficient = _lagrange_polynomial_coefficient(order - 1, lambda_list)
    coefficients = []
    for i in range(order):
        coefficient = 0.0
        for j in range(order):
            coefficient += lagrange_coefficient[i][j] * _exp_integral_positive(
                order - 1 - j, interval_start, interval_end, tau
            )
        coefficients.append(coefficient)
    return coefficients


def _check_tau(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须有限")
    if value < 0.0:
        raise ValueError(f"{name} 必须非负（官方 tau_func 只输出 0 或 1）")
    return value


def _check_order(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if value < 1 or value > 4:
        raise ValueError(f"{name} 必须落在 [1, 4]")
    return value


def _conv_model_output(output64, sample64, sigma, prediction_type):
    """官方 convert_model_output（data_prediction 分支，无 thresholding）逐字。"""
    alpha, sigma_t = _sigma_to_alpha_sigma_t(sigma)
    if prediction_type == "epsilon":
        return (sample64 - sigma_t * output64) / alpha
    return alpha * sample64 - sigma_t * output64


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

    本题钉死官方默认路径：algorithm_type = data_prediction（predict_x0 = True）、
    无 thresholding、无 variance_type 切片。prediction_type 暴露
    epsilon / v_prediction；阶数 predictor_order / corrector_order ∈ [1, 4] 显式入参；
    lower_order_final 显式入参（官方默认 True）。τ 由调用方按官方
    tau_func(timestep) 预先算好：corrector 用**上一步** timestep 的取值
    （tau_corrector），predictor 用**当前步** timestep 的取值（tau）。

    ## σ 约定（每次调用由入参 sigmas 重算，纯函数）

    `sigmas` 是官方 set_timesteps 产出的表：长度 N+1（N ≥ 2），**全为正**、
    有限、单调不增（官方各 schedule 均无 0 收尾，karras 末位相邻相等允许）。
    α = 1/√(σ²+1)、σ_t = σ·α、λ = log α − log σ_t（数学恒等于 −log σ）。

    ## 单调用状态机（官方 step 逐字同构）

    每次调用依次做五件事：

    1. **convert**：x0 = conv(sample, mo, sigmas[step_index])。
       epsilon → (sample − σ_t·mo)/α；v_prediction → α·sample − σ_t·mo。
    2. **corrector**（仅当 step_index > 0 且 state 给出）：阶数 tc 取
       **上一次调用保存的** state["this_corrector_order"]；τ 用 tau_corrector。
       σ_t 在 sigmas[step_index]、σ_s0 在 sigmas[step_index−1]；h = λ_t − λ_s0；
       λ 节点表 [λ(sigmas[step_index−k]), k = 0..tc−1]；
       系数 = Lagrange(tc−1, 节点) × 正指数积分(tc−1−j, [λ_s0, λ_t], τ)；
       **tc = 2 修正**（UniPC 同款，O(h³) 项）：Δ = exp((1+τ²)λ_t)·
       (h/2 − (h(1+τ²) − 1 + exp(−(1+τ²)h))/((1+τ²)²·h))，coefs[0] += Δ、
       coefs[1] −= Δ；模型输出表 = state["model_outputs"] + [x0]，取末 tc 条
       （注意：corrector 用的是**移位前**的 history 加上当前 x0）；
       gradient = Σ (1+τ²)·σ_t·exp(−τ²λ_t)·coefs[k]·mo_list[−(k+1)]；
       noise_part = σ_t·√(1 − exp(−2τ²h))·state["last_noise"]（**上一步 predictor
       注入的噪声**，保证 predictor/corrector 随机性一致）；
       corrected = exp(−τ²h)·(σ_t/σ_s0)·state["last_sample"] + gradient + noise_part
       （出发点是 last_sample——上一次 predictor **之前**的样本）。
    3. **history 移位**：model_outputs = 旧表[1:] + [x0]（长度 M =
       max(predictor_order, corrector_order − 1) 不变）。
    4. **warmup 阶数**：lower_order_final 时 tp = min(predictor_order, N −
       step_index)、tc_new = min(corrector_order, N − step_index + 1)，否则
       取配置阶数；再 tp = min(tp, lower_order_nums + 1)、
       tc_new = min(tc_new, lower_order_nums + 2)。**末步（step_index = N−1）
       predictor 必降到 1 阶**。
    5. **predictor**：阶数 tp、τ 用 tau。σ_t 在 sigmas[step_index+1]、σ_s0 在
       sigmas[step_index]；节点表 [λ(sigmas[step_index−k]), k = 0..tp−1]；
       **tp = 2 修正**：Δ = exp((1+τ²)λ_t)·(h²/2 − (h(1+τ²) − 1 +
       exp(−(1+τ²)h))/(1+τ²)²)/(λ_s0 − λ(sigmas[step_index−1]))（分母是相邻
       λ 差，只会在 step_index ≥ 1 时走到——step 0 warmup 后 tp = 1）；
       模型输出表 = **移位后**的 history（x0 在末位），取末 tp 条；
       gradient / noise_part 同 corrector，noise 用**本轮入参 noise**；
       prev = exp(−τ²h)·(σ_t/σ_s0)·(corrected 或 sample) + gradient + noise_part。

    state = {"model_outputs", "last_sample"(corrected 或 sample), "last_noise",
    "lower_order_nums"(lower_order_nums < M 时 +1), "this_corrector_order" =
    tc_new}。warmup 保证 history 的 None 头永不被读。

    校验失败一律抛 ValueError。state 各张量以 float64 存取（判题环境按
    float32 往返时容差 1e-6）。
    """
    output64 = _float_tensor("model_output", model_output)
    sample64 = _float_tensor("sample", sample)
    if sample64.shape != output64.shape:
        raise ValueError("sample 与 model_output 形状必须一致")

    if not torch.is_tensor(sigmas) or not sigmas.is_floating_point() or sigmas.dim() != 1:
        raise ValueError("sigmas 必须是一维浮点张量")
    values = sigmas.to(torch.float64)
    if values.numel() < 3:
        raise ValueError("sigmas 长度至少为 3（N ≥ 2 个推理步）")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("sigmas 必须全部有限")
    if bool((values <= 0).any()):
        raise ValueError("sigmas 必须全为正（官方各 schedule 均无 0 收尾）")
    if bool((values[1:] > values[:-1]).any()):
        raise ValueError("sigmas 必须单调不增")

    num_steps = values.numel() - 1  # N = num_inference_steps
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index > num_steps - 1:
        raise ValueError(f"step_index 必须落在 [0, {num_steps - 1}]")

    predictor_order = _check_order("predictor_order", predictor_order)
    corrector_order = _check_order("corrector_order", corrector_order)
    history_len = max(predictor_order, corrector_order - 1)

    if not isinstance(lower_order_final, bool):
        raise ValueError("lower_order_final 必须是布尔值")
    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")

    tau = _check_tau("tau", tau)

    if noise is None:
        raise ValueError("必须显式给 noise（官方每步由 generator 生成并暂存给下步 corrector）")
    noise64 = _float_tensor("noise", noise)
    if noise64.shape != output64.shape:
        raise ValueError("noise 形状必须与 model_output 一致")

    sigma_now = float(values[step_index])
    x0 = _conv_model_output(output64, sample64, sigma_now, prediction_type)

    # ---- 状态解析 ----
    if state is None:
        history = [None] * history_len
        last_sample = None
        last_noise = None
        lower_order_nums = 0
        saved_corrector_order = None
        use_corrector = False
    else:
        if not isinstance(state, dict) or set(state.keys()) != {
            "model_outputs",
            "last_sample",
            "last_noise",
            "lower_order_nums",
            "this_corrector_order",
        }:
            raise ValueError(
                "state 必须是只含 model_outputs / last_sample / last_noise / "
                "lower_order_nums / this_corrector_order 键的 dict"
            )
        if step_index == 0:
            raise ValueError("step_index 为 0 时不能传 state（无 corrector 可跑）")
        history_raw = state["model_outputs"]
        if not isinstance(history_raw, list) or len(history_raw) != history_len:
            raise ValueError(f"state.model_outputs 必须是长度 {history_len} 的 list")
        history = []
        for idx, item in enumerate(history_raw):
            if item is None:
                history.append(None)
            else:
                stored = _float_tensor(f"state.model_outputs[{idx}]", item)
                if stored.shape != sample64.shape:
                    raise ValueError("state.model_outputs 条目必须与 sample 同形状")
                history.append(stored)
        lower_order_nums = state["lower_order_nums"]
        if isinstance(lower_order_nums, bool) or not isinstance(lower_order_nums, int):
            raise ValueError("state.lower_order_nums 必须是整数")
        if lower_order_nums < 0 or lower_order_nums > history_len or lower_order_nums > step_index:
            raise ValueError(
                f"state.lower_order_nums 必须落在 [0, min({history_len}, {step_index})]"
            )
        for idx in range(history_len - lower_order_nums):
            if history[idx] is not None:
                raise ValueError("state.model_outputs 前 lower_order_nums 移位空位必须为 None")
        for idx in range(history_len - lower_order_nums, history_len):
            if history[idx] is None:
                raise ValueError("state.model_outputs 末 lower_order_nums 条必须是真实输出")
        last_sample = _float_tensor("state.last_sample", state["last_sample"])
        if last_sample.shape != sample64.shape:
            raise ValueError("state.last_sample 必须与 sample 同形状")
        last_noise = _float_tensor("state.last_noise", state["last_noise"])
        if last_noise.shape != output64.shape:
            raise ValueError("state.last_noise 必须与 model_output 同形状")
        saved_corrector_order = state["this_corrector_order"]
        if isinstance(saved_corrector_order, bool) or not isinstance(saved_corrector_order, int):
            raise ValueError("state.this_corrector_order 必须是整数")
        if saved_corrector_order < 1 or saved_corrector_order > corrector_order:
            raise ValueError(
                f"state.this_corrector_order 必须落在 [1, {corrector_order}]"
            )
        use_corrector = True

    # ---- corrector（随机 Adams-Moulton；用上一步的阶数/样本/噪声）----
    corrected_sample = None
    corrector_coefficients = None
    if use_corrector:
        tau_c = _check_tau("tau_corrector", tau_corrector)
        tc = saved_corrector_order
        lam_t = _lambda(float(values[step_index]))
        lam_s0 = _lambda(float(values[step_index - 1]))
        h = lam_t - lam_s0
        lambda_list = [_lambda(float(values[step_index - k])) for k in range(tc)]
        coefs = _coefficients(tc, lam_s0, lam_t, lambda_list, tau_c)
        if tc == 2:
            delta = (
                math.exp((1.0 + tau_c**2) * lam_t)
                * (
                    h / 2.0
                    - (h * (1.0 + tau_c**2) - 1.0 + math.exp((1.0 + tau_c**2) * (-h)))
                    / ((1.0 + tau_c**2) ** 2 * h)
                )
            )
            coefs[0] += delta
            coefs[1] -= delta
        corrector_coefficients = coefs
        alpha_t, sigma_t = _sigma_to_alpha_sigma_t(float(values[step_index]))
        _, sigma_s0 = _sigma_to_alpha_sigma_t(float(values[step_index - 1]))
        model_prev = history + [x0]
        gradient = torch.zeros_like(sample64)
        for k in range(tc):
            gradient = gradient + (
                (1.0 + tau_c**2)
                * sigma_t
                * math.exp(-(tau_c**2) * lam_t)
                * coefs[k]
                * model_prev[-(k + 1)]
            )
        noise_part = sigma_t * math.sqrt(1.0 - math.exp(-2.0 * tau_c**2 * h)) * last_noise
        corrected_sample = (
            math.exp(-(tau_c**2) * h) * (sigma_t / sigma_s0) * last_sample
            + gradient
            + noise_part
        )

    # ---- history 移位 ----
    history = history[1:] + [x0]

    # ---- warmup 阶数 ----
    if lower_order_final:
        this_predictor_order = min(predictor_order, num_steps - step_index)
        this_corrector_order = min(corrector_order, num_steps - step_index + 1)
    else:
        this_predictor_order = predictor_order
        this_corrector_order = corrector_order
    this_predictor_order = min(this_predictor_order, lower_order_nums + 1)
    this_corrector_order = min(this_corrector_order, lower_order_nums + 2)

    # ---- predictor（随机 Adams-Bashforth；用本轮阶数/样本/噪声）----
    tp = this_predictor_order
    lam_t = _lambda(float(values[step_index + 1]))
    lam_s0 = _lambda(float(values[step_index]))
    h = lam_t - lam_s0
    lambda_list = [_lambda(float(values[step_index - k])) for k in range(tp)]
    coefs = _coefficients(tp, lam_s0, lam_t, lambda_list, tau)
    if tp == 2:
        temp_lambda_s = _lambda(float(values[step_index - 1]))
        delta = (
            math.exp((1.0 + tau**2) * lam_t)
            * (
                h**2 / 2.0
                - (h * (1.0 + tau**2) - 1.0 + math.exp((1.0 + tau**2) * (-h)))
                / ((1.0 + tau**2) ** 2)
            )
            / (lam_s0 - temp_lambda_s)
        )
        coefs[0] += delta
        coefs[1] -= delta
    predictor_coefficients = coefs
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(float(values[step_index + 1]))
    _, sigma_s0 = _sigma_to_alpha_sigma_t(float(values[step_index]))
    gradient = torch.zeros_like(sample64)
    for k in range(tp):
        gradient = gradient + (
            (1.0 + tau**2)
            * sigma_t
            * math.exp(-(tau**2) * lam_t)
            * coefs[k]
            * history[-(k + 1)]
        )
    noise_part = sigma_t * math.sqrt(1.0 - math.exp(-2.0 * tau**2 * h)) * noise64
    base = corrected_sample if use_corrector else sample64
    prev = math.exp(-(tau**2) * h) * (sigma_t / sigma_s0) * base + gradient + noise_part

    # ---- 状态推进 ----
    new_lower = lower_order_nums + 1 if lower_order_nums < history_len else lower_order_nums
    new_state = {
        "model_outputs": history,
        "last_sample": base,
        "last_noise": noise64,
        "lower_order_nums": new_lower,
        "this_corrector_order": this_corrector_order,
    }

    return {
        "prev_sample": prev.to(torch.float32),
        "pred_original_sample": x0.to(torch.float32),
        "corrected_sample": corrected_sample.to(torch.float32) if use_corrector else None,
        "used_corrector": use_corrector,
        "this_predictor_order": this_predictor_order,
        "this_corrector_order": this_corrector_order,
        "predictor_coefficients": predictor_coefficients,
        "corrector_coefficients": corrector_coefficients,
        "state": new_state,
    }
