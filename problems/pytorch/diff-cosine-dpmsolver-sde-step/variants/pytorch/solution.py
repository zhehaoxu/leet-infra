import math

import torch

_SOLVER_TYPES = ("midpoint", "heun")
_FINAL_SIGMAS_TYPES = ("zero", "sigma_min")


def _float_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def _sigma_to_alpha_sigma_t(sigma):
    """官方 _sigma_to_alpha_sigma_t 逐字：输入在进入 unet 前已按 EDM 预条件
    缩放（c_in = 1/√(σ²+σd²)），所以 α_t 恒为 1、σ_t = σ。"""
    return 1.0, sigma


def _lambda(sigma):
    """官方 λ = log(α) − log(σ_t)；本题 α ≡ 1 → λ 恒等于 −log σ。
    σ = 0（final_sigmas_type="zero" 的末位）时官方 torch.log(0) = −∞ →
    λ = +∞，保留同一语义（后续 exp(−h) = 0、exp(−2h) = 0）。"""
    alpha, sigma_t = _sigma_to_alpha_sigma_t(sigma)
    if sigma_t == 0.0:
        return math.inf
    return math.log(alpha) - math.log(sigma_t)


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

    与 DPMSolverMultistep 的差异：σ 表由 exponential / karras schedule 生成后
    追加 sigma_last（0 或 sigma_min）；模型输出先过 EDM 预条件再进 DPMSolver；
    α_t ≡ 1（VE），两条 update 都带 SDE 噪声项（官方每步由
    BrownianTreeNoiseSampler 采样，依赖 generator/seed，属管线侧——本题由
    调用方算好 noise 显式传入）。

    ## EDM 预条件（convert，官方 precondition_outputs 逐字）

    σ = sigmas[step_index]，d = sigma_data：
    c_skip = d²/(σ²+d²)；epsilon → c_out = σ·d/√(σ²+d²)；
    v_prediction → c_out = **−**σ·d/√(σ²+d²)；其他 prediction_type 抛
    ValueError（"Prediction type ... is not supported."，官方同文案）。
    x0 = c_skip·sample + c_out·model_output。

    ## λ 与系数

    α_t ≡ 1 → λ = −log σ（σ = 0 → λ = +∞，torch 语义）。h = λ_t − λ_s0 ≥ 0
    （σ 表单调不增）；e^{−h} = σ_t/σ_s0，故 1 阶 sample 系数 = (σ_t/σ_s0)·e^{−h}。

    ## 单调用状态机（官方 step 逐字同构）

    1. **state 解析**：state = {"model_outputs"(长度 solver_order),
       "lower_order_nums"}。state=None 表示全新开始（官方 begin_index 场景，
       任意 step_index 合法）；给 state 时必须 1 ≤ lower_order_nums ≤
       min(solver_order, step_index)，且 model_outputs **前
       solver_order−lower_order_nums 条为 None、末 lower_order_nums 条为真实
       输出**（与 sample 同形状）。
    2. **convert**：x0 = EDM 预条件（上式）。
    3. **history 移位**：model_outputs = 旧表[1:] + [x0]（长度不变）。
    4. **降阶判定**：N = len(sigmas) − 1。
       lower_order_flag = (step_index == N−1) and (euler_at_final or
       (lower_order_final and N < 15) or final_sigmas_type == "zero")。
       一阶条件 = solver_order == 1 or lower_order_nums < 1 or
       lower_order_flag。官方 elif 的 `solver_order == 2 or ...` 在
       solver_order ∈ {1, 2} 校验下恒真（lower_order_second 为死代码），合并
       为否则走二阶。
    5. **一阶 update**（官方 dpm_solver_first_order_update，α_t = 1 已折叠）：
       σ_t = sigmas[step_index+1]、σ_s0 = sigmas[step_index]、h = λ_t − λ_s0；
       prev = (σ_t/σ_s0)·e^{−h}·sample + (1 − e^{−2h})·x0 +
       σ_t·√(1 − e^{−2h})·noise。末步 zero 收尾时 σ_t = 0、h = +∞：三系数分别
       为 0、1、0 → prev 恒等 x0（官方 torch 语义，exp(−∞) = 0）。
    6. **二阶 update**（官方 multistep_dpm_solver_second_order_update）：
       另取 σ_s1 = sigmas[step_index−1]，h_0 = λ_s0 − λ_s1，r0 = h_0/h；
       D0 = x0（移位后表末位）、D1 = (1/r0)·(x0 − model_outputs[−2])；
       midpoint → d1c = 0.5·(1 − e^{−2h})；
       heun → d1c = (1 − e^{−2h})/(−2h) + 1；
       prev = (σ_t/σ_s0)·e^{−h}·sample + (1 − e^{−2h})·D0 + d1c·D1 +
       σ_t·√(1 − e^{−2h})·noise。h = 0 或 h_0 = 0（相邻 σ 相等）时官方产生
       0/0 未定义结果，本题拒绝（ValueError）。
    7. **state 推进**：lower_order_nums 在小于 solver_order 时 +1。

    state 张量以 float64 存取（判题环境按 float32 往返时容差 1e-6）。
    校验失败一律抛 ValueError。
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
    if bool((values < 0).any()):
        raise ValueError("sigmas 不能为负")
    if bool((values[:-1] <= 0).any()):
        raise ValueError("sigmas 除末位外必须全为正（末位可为 0 或 sigma_min）")
    if bool((values[1:] > values[:-1]).any()):
        raise ValueError("sigmas 必须单调不增")

    num_steps = values.numel() - 1  # N = num_inference_steps
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index > num_steps - 1:
        raise ValueError(f"step_index 必须落在 [0, {num_steps - 1}]")

    if isinstance(solver_order, bool) or not isinstance(solver_order, int):
        raise ValueError("solver_order 必须是整数")
    if solver_order not in (1, 2):
        raise ValueError(f"solver_order 必须是 1 或 2， got {solver_order}")
    if solver_type not in _SOLVER_TYPES:
        raise ValueError(f"solver_type {solver_type!r} is not implemented （仅 midpoint / heun）")
    if final_sigmas_type not in _FINAL_SIGMAS_TYPES:
        raise ValueError(f"final_sigmas_type 必须是 'zero' 或 'sigma_min'， got {final_sigmas_type!r}")
    if not isinstance(lower_order_final, bool):
        raise ValueError("lower_order_final 必须是布尔值")
    if not isinstance(euler_at_final, bool):
        raise ValueError("euler_at_final 必须是布尔值")
    if isinstance(sigma_data, bool) or not isinstance(sigma_data, (int, float)):
        raise ValueError("sigma_data 必须是数值")
    sigma_data = float(sigma_data)
    if not math.isfinite(sigma_data) or sigma_data <= 0.0:
        raise ValueError("sigma_data 必须为正有限")

    if noise is None:
        raise ValueError("必须显式给 noise（官方每步由 BrownianTreeNoiseSampler 采样）")
    noise64 = _float_tensor("noise", noise)
    if noise64.shape != output64.shape:
        raise ValueError("noise 形状必须与 model_output 一致")

    # ---- state 解析 ----
    if state is None:
        history = [None] * solver_order
        lower_order_nums = 0
    else:
        if not isinstance(state, dict) or set(state.keys()) != {"model_outputs", "lower_order_nums"}:
            raise ValueError("state 必须是只含 model_outputs / lower_order_nums 键的 dict")
        history_raw = state["model_outputs"]
        if not isinstance(history_raw, list) or len(history_raw) != solver_order:
            raise ValueError(f"state.model_outputs 必须是长度 {solver_order} 的 list")
        lower_order_nums = state["lower_order_nums"]
        if isinstance(lower_order_nums, bool) or not isinstance(lower_order_nums, int):
            raise ValueError("state.lower_order_nums 必须是整数")
        if lower_order_nums < 1 or lower_order_nums > solver_order or lower_order_nums > step_index:
            raise ValueError(
                f"state.lower_order_nums 必须落在 [1, min({solver_order}, {step_index})]"
                "（给 state 意味着此前至少调用过一次）"
            )
        history = []
        for idx in range(solver_order):
            item = history_raw[idx]
            if idx < solver_order - lower_order_nums:
                if item is not None:
                    raise ValueError("state.model_outputs 前 solver_order−lower_order_nums 移位空位必须为 None")
                history.append(None)
            else:
                if item is None:
                    raise ValueError("state.model_outputs 末 lower_order_nums 条必须是真实输出")
                stored = _float_tensor(f"state.model_outputs[{idx}]", item)
                if stored.shape != sample64.shape:
                    raise ValueError("state.model_outputs 条目必须与 sample 同形状")
                history.append(stored)

    # ---- convert（EDM 预条件）----
    sigma_now = float(values[step_index])
    sd2 = sigma_data**2
    c_skip = sd2 / (sigma_now**2 + sd2)
    denom = math.sqrt(sigma_now**2 + sd2)
    if prediction_type == "epsilon":
        c_out = sigma_now * sigma_data / denom
    elif prediction_type == "v_prediction":
        c_out = -sigma_now * sigma_data / denom
    else:
        raise ValueError(f"Prediction type {prediction_type!r} is not supported.")
    x0 = c_skip * sample64 + c_out * output64

    # ---- history 移位 ----
    history = history[1:] + [x0]

    # ---- 降阶判定 ----
    lower_order_flag = (step_index == num_steps - 1) and (
        euler_at_final or (lower_order_final and num_steps < 15) or final_sigmas_type == "zero"
    )
    use_first = solver_order == 1 or lower_order_nums < 1 or lower_order_flag

    sigma_t = float(values[step_index + 1])
    sigma_s0 = float(values[step_index])
    lam_t = _lambda(sigma_t)
    lam_s0 = _lambda(sigma_s0)
    h = lam_t - lam_s0

    r0 = None
    d1_coefficient = None
    if use_first:
        decay = 1.0 - math.exp(-2.0 * h)
        c_sample = (sigma_t / sigma_s0) * math.exp(-h)
        prev = c_sample * sample64 + decay * x0 + sigma_t * math.sqrt(decay) * noise64
        this_order = 1
    else:
        sigma_s1 = float(values[step_index - 1])
        h_0 = lam_s0 - _lambda(sigma_s1)
        if not h > 0.0 or not h_0 > 0.0:
            raise ValueError("二阶更新要求相邻 λ 严格递减（h > 0 且 h_0 > 0；官方在相等时产生 0/0 未定义结果）")
        r0 = h_0 / h
        d1 = (1.0 / r0) * (x0 - history[-2])
        decay = 1.0 - math.exp(-2.0 * h)
        if solver_type == "midpoint":
            d1_coefficient = 0.5 * decay
        else:
            d1_coefficient = decay / (-2.0 * h) + 1.0
        c_sample = (sigma_t / sigma_s0) * math.exp(-h)
        prev = (
            c_sample * sample64
            + decay * x0
            + d1_coefficient * d1
            + sigma_t * math.sqrt(decay) * noise64
        )
        this_order = 2

    # ---- state 推进 ----
    new_lower = lower_order_nums + 1 if lower_order_nums < solver_order else lower_order_nums
    new_state = {
        "model_outputs": history,
        "lower_order_nums": new_lower,
    }

    return {
        "prev_sample": prev.to(torch.float32),
        "pred_original_sample": x0.to(torch.float32),
        "used_first_order": use_first,
        "this_order": this_order,
        "r0": r0,
        "d1_coefficient": d1_coefficient,
        "state": new_state,
    }
