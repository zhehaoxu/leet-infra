import math

import torch

_PREDICTION_TYPES = ("epsilon", "sample", "v_prediction", "flow_prediction")


def _float_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def solve(
    sample,
    model_output,
    sigmas,
    step_index,
    converted_history,
    solver_order=2,
    lower_order_final=True,
    prediction_type="epsilon",
    use_flow_sigmas=False,
):
    """DEIS 多步单步更新：返回 (converted, used_order, x_next, coefs)。

    sample / model_output：同形状浮点张量。
    sigmas：一维浮点张量，长度 N+1（N 个推理步，末位是上游 set_timesteps 追加的
      sigma_last），全为正、有限、单调不增。
    step_index：整数，落在 [0, N-1]。
    converted_history：本步**之前**各步已转换输出 d 的列表（最新在尾），长度必须
      恰为 min(step_index, solver_order)——对应上游 lower_order_nums 计数与
      model_outputs 移位窗口。
    solver_order：1/2/3；lower_order_final：bool；use_flow_sigmas：bool。

    α/σ 变换（上游 _sigma_to_alpha_sigma_t）：
      use_flow_sigmas=False -> alpha = 1/sqrt(sigma²+1), sigma_t = sigma·alpha
      use_flow_sigmas=True  -> alpha = 1 - sigma,       sigma_t = sigma

    convert（上游 convert_model_output，用当前步 sigma = sigmas[step_index]）：
      epsilon      -> x0 = sample - σ_c·mo
      sample       -> x0 = mo
      v_prediction -> x0 = alpha_c·sample - σ_c·mo
      flow_prediction -> x0 = sample - sigmas[step_index]·mo（用原始 sigma）
      d = (sample - alpha_c·x0) / σ_c

    降阶（上游 step 的三分支；lower_order_nums 以 len(converted_history) 计）：
      lof = (step_index == N-1) and lower_order_final and N < 15
      los = (step_index == N-2) and lower_order_final and N < 15
      low < 1 或 solver_order==1 或 lof        -> 1 阶（DDIM 等价）
      否则 low < 2 或 solver_order==2 或 los   -> 2 阶
      否则                                      -> 3 阶

    一阶（sigmas[i] -> sigmas[i+1]）：
      h = (ln α_t - ln σ_t) - (ln α_s - ln σ_s)
      x = (α_t/α_s)·sample - σ_t·(e^h - 1)·d

    二阶（m1 = converted_history[-1]）：
      ρ = σ/α；ind2(t,b,c) = t·(−ln c + ln t − 1)/(ln b − ln c)
      coef1 = ind2(ρt,ρ0,ρ1) − ind2(ρ0,ρ0,ρ1)；coef2 = ind2(ρt,ρ1,ρ0) − ind2(ρ0,ρ1,ρ0)
      x = α_t·(sample/α_0 + coef1·d + coef2·m1)

    三阶（另需 m2 = converted_history[-2]）：
      ind3(t,b,c,d) = t·(ln c·(ln d − ln t + 1) − ln d·ln t + ln d + ln²t − 2·ln t + 2)
                      / ((ln b − ln c)·(ln b − ln d))
      coef1 = ind3(ρt,ρ0,ρ1,ρ2) − ind3(ρ0,ρ0,ρ1,ρ2)
      coef2 = ind3(ρt,ρ1,ρ2,ρ0) − ind3(ρ0,ρ1,ρ2,ρ0)
      coef3 = ind3(ρt,ρ2,ρ0,ρ1) − ind3(ρ0,ρ2,ρ0,ρ1)
      x = α_t·(sample/α_0 + coef1·d + coef2·m1 + coef3·m2)

    返回 dict：converted（float64）、used_order（int）、x_next（float32）、
    coefs（一阶 []、二阶 [coef1,coef2]、三阶 [c1,c2,c3]，python float）。
    """
    sample64 = _float_tensor("sample", sample)
    output64 = _float_tensor("model_output", model_output)
    if sample64.shape != output64.shape:
        raise ValueError("sample 与 model_output 形状必须一致")

    if not torch.is_tensor(sigmas) or not sigmas.is_floating_point() or sigmas.dim() != 1:
        raise ValueError("sigmas 必须是一维浮点张量")
    if sigmas.numel() < 2:
        raise ValueError("sigmas 长度至少为 2（1 个推理步 + 末位 sigma_last）")
    values = sigmas.to(torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("sigmas 必须全部有限")
    if bool((values <= 0).any()):
        raise ValueError("sigmas 必须全为正（上游末位是 sigma_last 而非 0）")
    if bool((values[1:] > values[:-1]).any()):
        raise ValueError("sigmas 必须单调不增")

    num_steps = values.numel() - 1
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index > num_steps - 1:
        raise ValueError(f"step_index 必须落在 [0, {num_steps - 1}]")

    order = solver_order
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2, 3):
        raise ValueError("solver_order 必须是 1/2/3")
    if not isinstance(lower_order_final, bool):
        raise ValueError("lower_order_final 必须是 bool")
    if not isinstance(use_flow_sigmas, bool):
        raise ValueError("use_flow_sigmas 必须是 bool")
    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")

    if not isinstance(converted_history, (list, tuple)):
        raise ValueError("converted_history 必须是张量列表")
    history = [_float_tensor(f"converted_history[{k}]", h) for k, h in enumerate(converted_history)]
    for h in history:
        if h.shape != sample64.shape:
            raise ValueError("converted_history 各项必须与 sample 同形状")
    low = len(history)
    if low != min(step_index, order):
        raise ValueError(
            f"converted_history 长度必须恰为 min(step_index, solver_order) = {min(step_index, order)}"
        )

    def alpha_sigma(s):
        if use_flow_sigmas:
            return 1.0 - s, s
        a = 1.0 / math.sqrt(s * s + 1.0)
        return a, s * a

    # ---- convert_model_output（当前步） ----
    sigma_cur = float(values[step_index])
    alpha_c, sigma_c = alpha_sigma(sigma_cur)
    if prediction_type == "epsilon":
        x0 = sample64 - sigma_c * output64
    elif prediction_type == "sample":
        x0 = output64
    elif prediction_type == "v_prediction":
        x0 = alpha_c * sample64 - sigma_c * output64
    else:  # flow_prediction
        x0 = sample64 - sigma_cur * output64
    converted = (sample64 - alpha_c * x0) / sigma_c

    # ---- 降阶选择 ----
    lof = (step_index == num_steps - 1) and lower_order_final and num_steps < 15
    los = (step_index == num_steps - 2) and lower_order_final and num_steps < 15
    if order == 1 or low < 1 or lof:
        used = 1
    elif order == 2 or low < 2 or los:
        used = 2
    else:
        used = 3

    log = math.log
    if used == 1:
        sigma_s = float(values[step_index])
        sigma_t = float(values[step_index + 1])
        alpha_s, sigma_s2 = alpha_sigma(sigma_s)
        alpha_t, sigma_t2 = alpha_sigma(sigma_t)
        lambda_s = log(alpha_s) - log(sigma_s2)
        lambda_t = log(alpha_t) - log(sigma_t2)
        h = lambda_t - lambda_s
        x_next = (alpha_t / alpha_s) * sample64 - sigma_t2 * (math.exp(h) - 1.0) * converted
        coefs = []
    elif used == 2:
        st = float(values[step_index + 1])
        s0 = float(values[step_index])
        s1 = float(values[step_index - 1])
        alpha_t, sigma_t = alpha_sigma(st)
        alpha_0, sigma_a0 = alpha_sigma(s0)
        alpha_1, sigma_a1 = alpha_sigma(s1)
        rho_t, rho_0, rho_1 = sigma_t / alpha_t, sigma_a0 / alpha_0, sigma_a1 / alpha_1
        m0 = converted
        m1 = history[-1]

        def ind2(t, b, c):
            return t * (-log(c) + log(t) - 1.0) / (log(b) - log(c))

        coef1 = ind2(rho_t, rho_0, rho_1) - ind2(rho_0, rho_0, rho_1)
        coef2 = ind2(rho_t, rho_1, rho_0) - ind2(rho_0, rho_1, rho_0)
        x_next = alpha_t * (sample64 / alpha_0 + coef1 * m0 + coef2 * m1)
        coefs = [coef1, coef2]
    else:
        st = float(values[step_index + 1])
        s0 = float(values[step_index])
        s1 = float(values[step_index - 1])
        s2 = float(values[step_index - 2])
        alpha_t, sigma_t = alpha_sigma(st)
        alpha_0, sigma_a0 = alpha_sigma(s0)
        alpha_1, sigma_a1 = alpha_sigma(s1)
        alpha_2, sigma_a2 = alpha_sigma(s2)
        rho_t, rho_0, rho_1, rho_2 = (
            sigma_t / alpha_t,
            sigma_a0 / alpha_0,
            sigma_a1 / alpha_1,
            sigma_a2 / alpha_2,
        )
        m0 = converted
        m1 = history[-1]
        m2 = history[-2]

        def ind3(t, b, c, d):
            numerator = t * (
                log(c) * (log(d) - log(t) + 1.0)
                - log(d) * log(t)
                + log(d)
                + log(t) ** 2
                - 2.0 * log(t)
                + 2.0
            )
            denominator = (log(b) - log(c)) * (log(b) - log(d))
            return numerator / denominator

        coef1 = ind3(rho_t, rho_0, rho_1, rho_2) - ind3(rho_0, rho_0, rho_1, rho_2)
        coef2 = ind3(rho_t, rho_1, rho_2, rho_0) - ind3(rho_0, rho_1, rho_2, rho_0)
        coef3 = ind3(rho_t, rho_2, rho_0, rho_1) - ind3(rho_0, rho_2, rho_0, rho_1)
        x_next = alpha_t * (sample64 / alpha_0 + coef1 * m0 + coef2 * m1 + coef3 * m2)
        coefs = [coef1, coef2, coef3]

    return {
        "converted": converted,
        "used_order": used,
        "x_next": x_next.to(torch.float32),
        "coefs": coefs,
    }
