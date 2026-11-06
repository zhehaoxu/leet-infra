"""DPMSolverMultistep 反演调度器的多步更新账（order 832）。

回源：huggingface/diffusers
``src/diffusers/schedulers/scheduling_dpmsolver_multistep_inverse.py``
(``DPMSolverMultistepInverseScheduler``)。

它是 ``DPMSolverMultistepScheduler`` 的反向（inversion）调度器：把干净样本
沿**递增**的 σ 表一路推回更噪的一级，供图像编辑「先反演成噪声、再改写、
再采样」使用。与前向采样相比，更新函数是逐字复制的，但因为 σ 表方向相反，
``h = λ_t − λ_s < 0``，``exp(−h) > 1``、``exp(h) < 1``，各阶系数全部落在
与前向不同的数值区间。本题把官方单步链路拆成可判定的账：

1. ``sigma_to_alpha``：``_sigma_to_alpha_sigma_t``——VP 约定
   ``α = 1/√(1+σ²)``、``σ_t = σ·α``；flow 约定 ``α = 1−σ``、``σ_t = σ``。
2. ``convert``：``convert_model_output``——按 algorithm_type
   (``dpmsolver++`` 收 x0_pred / ``dpmsolver`` 收 ε) 与 prediction_type
   (epsilon / sample / v_prediction / flow_prediction) 把模型输出换成求解器量。
3. ``first`` / ``second`` / ``third``：一/二/三阶多步更新，逐字复刻
   ``dpm_solver_first_order_update``、
   ``multistep_dpm_solver_second_order_update``、
   ``multistep_dpm_solver_third_order_update`` 的 DPMSolver/DPMSolver++ 两个
   非 SDE 分支（midpoint / heun；三阶为固定公式、无 solver_type）。
4. ``step``：``step()`` 的单调用状态机——降阶标志、历史移位、按
   ``lower_order_nums`` 选阶、计数器自增，全部经显式 ``state`` 进出。

不进判题域（避免与既有题重复）：``sde-*`` 分支的 Brownian 噪声项
（见 ``diff-cosine-dpmsolver-sde-step``）、dynamic thresholding 与
``learned_range`` 通道切片（见 ``diff-dpm-x0-conversion-threshold``）、
``set_timesteps`` 的 σ 表构造算法。官方三阶分支漏传 ``noise`` 的缺陷同样
不进判题域（本题不含 SDE 分支）。

判题约定：单入口 ``solve(mode, **kw)``；σ 表严格递增（反演方向）、全正有限，
flow 约定另需 ``0 < σ < 1``；张量一律 fp64。
"""

import math

import torch
from torch import Tensor

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _sigma_to_alpha_sigma_t(sigma: float, use_flow_sigmas: bool):
    """官方 _sigma_to_alpha_sigma_t：VP 与 flow 两条约定。"""
    if use_flow_sigmas:
        alpha = 1.0 - sigma
        sigma_t = sigma
    else:
        alpha = 1.0 / math.sqrt(sigma * sigma + 1.0)
        sigma_t = sigma * alpha
    return alpha, sigma_t


def _lam(sigma: float, use_flow_sigmas: bool):
    """返回 (lambda, alpha, sigma_t)，其中 lambda = log(alpha) - log(sigma_t)。"""
    alpha, sigma_t = _sigma_to_alpha_sigma_t(sigma, use_flow_sigmas)
    return math.log(alpha) - math.log(sigma_t), alpha, sigma_t


def _prep(sigmas, step_index: int, need_prev: int, use_flow_sigmas: bool):
    """校验并取回 fp64 σ 表；need_prev 为本阶需要的前置步数（1 阶 0 / 2 阶 1 / 3 阶 2）。"""
    if not isinstance(sigmas, Tensor):
        sigmas = torch.as_tensor(sigmas, dtype=torch.float64)
    sigmas = sigmas.detach().to(torch.float64).reshape(-1)
    if sigmas.numel() < 2 or not bool(torch.isfinite(sigmas).all()):
        raise ValueError("sigmas must be finite and length >= 2")
    if bool((sigmas <= 0).any()):
        raise ValueError("sigmas must be positive")
    if not bool((sigmas[1:] > sigmas[:-1]).all()):
        raise ValueError("sigmas must be strictly increasing (inverse direction)")
    if use_flow_sigmas and bool((sigmas >= 1).any()):
        raise ValueError("flow sigmas require 0 < sigma < 1")
    if step_index < 0 or step_index + 1 >= sigmas.numel():
        raise ValueError("step_index out of range")
    if step_index - need_prev < 0:
        raise ValueError("not enough preceding sigmas for this order")
    return sigmas


def _as_f64(x):
    if not isinstance(x, Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    return x.detach().to(torch.float64)


def _first_update(sigmas, step_index, sample, m0, algorithm_type, use_flow_sigmas):
    sigmas = _prep(sigmas, step_index, 0, use_flow_sigmas)
    s0 = float(sigmas[step_index].item())
    st = float(sigmas[step_index + 1].item())
    lambda_s, alpha_s, sigma_s = _lam(s0, use_flow_sigmas)
    lambda_t, alpha_t, sigma_t = _lam(st, use_flow_sigmas)
    h = lambda_t - lambda_s
    if h == 0.0:
        raise ValueError("degenerate h")
    if algorithm_type == "dpmsolver++":
        c_sample = sigma_t / sigma_s
        c_model = -(alpha_t * (math.exp(-h) - 1.0))
    elif algorithm_type == "dpmsolver":
        c_sample = alpha_t / alpha_s
        c_model = -(sigma_t * (math.exp(h) - 1.0))
    else:
        raise ValueError("algorithm_type must be dpmsolver++ or dpmsolver")
    prev = c_sample * sample + c_model * m0
    diag = {"h": h, "lambda_t": lambda_t, "lambda_s": lambda_s,
            "alpha_t": alpha_t, "alpha_s": alpha_s,
            "sigma_t": sigma_t, "sigma_s": sigma_s,
            "c_sample": c_sample, "c_model": c_model}
    return prev, diag


def _second_update(sigmas, step_index, sample, model_outputs, algorithm_type,
                   solver_type, use_flow_sigmas):
    sigmas = _prep(sigmas, step_index, 1, use_flow_sigmas)
    s1 = float(sigmas[step_index - 1].item())
    s0 = float(sigmas[step_index].item())
    st = float(sigmas[step_index + 1].item())
    lambda_s1, alpha_s1, sigma_s1 = _lam(s1, use_flow_sigmas)
    lambda_s0, alpha_s0, sigma_s0 = _lam(s0, use_flow_sigmas)
    lambda_t, alpha_t, sigma_t = _lam(st, use_flow_sigmas)
    h = lambda_t - lambda_s0
    h_0 = lambda_s0 - lambda_s1
    if h == 0.0 or h_0 == 0.0:
        raise ValueError("degenerate h / h_0")
    r0 = h_0 / h
    m1, m0 = model_outputs[-2], model_outputs[-1]
    D0 = m0
    D1 = (1.0 / r0) * (m0 - m1)
    if algorithm_type == "dpmsolver++":
        base = alpha_t * (math.exp(-h) - 1.0)
        c_sample = sigma_t / sigma_s0
        c_d0 = -base
        if solver_type == "midpoint":
            c_d1 = -0.5 * base
        elif solver_type == "heun":
            c_d1 = alpha_t * ((math.exp(-h) - 1.0) / h + 1.0)
        else:
            raise ValueError("solver_type must be midpoint or heun")
    elif algorithm_type == "dpmsolver":
        base = sigma_t * (math.exp(h) - 1.0)
        c_sample = alpha_t / alpha_s0
        c_d0 = -base
        if solver_type == "midpoint":
            c_d1 = -0.5 * base
        elif solver_type == "heun":
            c_d1 = -(sigma_t * ((math.exp(h) - 1.0) / h - 1.0))
        else:
            raise ValueError("solver_type must be midpoint or heun")
    else:
        raise ValueError("algorithm_type must be dpmsolver++ or dpmsolver")
    prev = c_sample * sample + c_d0 * D0 + c_d1 * D1
    diag = {"h": h, "h_0": h_0, "r0": r0, "D0": D0, "D1": D1,
            "alpha_t": alpha_t, "sigma_t": sigma_t, "sigma_s0": sigma_s0,
            "c_sample": c_sample, "c_d0": c_d0, "c_d1": c_d1}
    return prev, diag


def _third_update(sigmas, step_index, sample, model_outputs, algorithm_type,
                  use_flow_sigmas):
    sigmas = _prep(sigmas, step_index, 2, use_flow_sigmas)
    s2 = float(sigmas[step_index - 2].item())
    s1 = float(sigmas[step_index - 1].item())
    s0 = float(sigmas[step_index].item())
    st = float(sigmas[step_index + 1].item())
    lambda_s2, alpha_s2, sigma_s2 = _lam(s2, use_flow_sigmas)
    lambda_s1, alpha_s1, sigma_s1 = _lam(s1, use_flow_sigmas)
    lambda_s0, alpha_s0, sigma_s0 = _lam(s0, use_flow_sigmas)
    lambda_t, alpha_t, sigma_t = _lam(st, use_flow_sigmas)
    h = lambda_t - lambda_s0
    h_0 = lambda_s0 - lambda_s1
    h_1 = lambda_s1 - lambda_s2
    if h == 0.0 or h_0 == 0.0 or h_1 == 0.0 or (h_0 + h_1) == 0.0:
        raise ValueError("degenerate h / h_0 / h_1")
    r0 = h_0 / h
    r1 = h_1 / h
    m2, m1, m0 = model_outputs[-3], model_outputs[-2], model_outputs[-1]
    D1_0 = (1.0 / r0) * (m0 - m1)
    D1_1 = (1.0 / r1) * (m1 - m2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
    D0 = m0
    if algorithm_type == "dpmsolver++":
        c_sample = sigma_t / sigma_s0
        c_d0 = -(alpha_t * (math.exp(-h) - 1.0))
        c_d1 = alpha_t * ((math.exp(-h) - 1.0) / h + 1.0)
        c_d2 = -(alpha_t * ((math.exp(-h) - 1.0 + h) / (h * h) - 0.5))
    elif algorithm_type == "dpmsolver":
        c_sample = alpha_t / alpha_s0
        c_d0 = -(sigma_t * (math.exp(h) - 1.0))
        c_d1 = -(sigma_t * ((math.exp(h) - 1.0) / h - 1.0))
        c_d2 = -(sigma_t * ((math.exp(h) - 1.0 - h) / (h * h) - 0.5))
    else:
        raise ValueError("algorithm_type must be dpmsolver++ or dpmsolver")
    prev = c_sample * sample + c_d0 * D0 + c_d1 * D1 + c_d2 * D2
    diag = {"h": h, "h_0": h_0, "h_1": h_1, "r0": r0, "r1": r1,
            "D0": D0, "D1_0": D1_0, "D1_1": D1_1, "D1": D1, "D2": D2,
            "alpha_t": alpha_t, "sigma_t": sigma_t, "sigma_s0": sigma_s0,
            "c_sample": c_sample, "c_d0": c_d0, "c_d1": c_d1, "c_d2": c_d2}
    return prev, diag


def _convert(model_output, sample, sigma, algorithm_type, prediction_type,
             use_flow_sigmas):
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(float(sigma), use_flow_sigmas)
    if algorithm_type == "dpmsolver++":
        kind = "x0_pred"
        if prediction_type == "epsilon":
            converted = (sample - sigma_t * model_output) / alpha_t
        elif prediction_type == "sample":
            converted = model_output
        elif prediction_type == "v_prediction":
            converted = alpha_t * sample - sigma_t * model_output
        elif prediction_type == "flow_prediction":
            # 官方此处用未换算的 sigma（局部变量同名遮蔽），不是 sigma_t
            converted = sample - float(sigma) * model_output
        else:
            raise ValueError("unsupported prediction_type for dpmsolver++")
    elif algorithm_type == "dpmsolver":
        kind = "epsilon"
        if prediction_type == "epsilon":
            converted = model_output
        elif prediction_type == "sample":
            converted = (sample - alpha_t * model_output) / sigma_t
        elif prediction_type == "v_prediction":
            converted = alpha_t * model_output + sigma_t * sample
        else:
            raise ValueError("unsupported prediction_type for dpmsolver")
    else:
        raise ValueError("algorithm_type must be dpmsolver++ or dpmsolver")
    return converted, kind


def solve(mode: str, **kw):
    flow = bool(kw.get("use_flow_sigmas", False))

    if mode == "sigma_to_alpha":
        sigma = float(kw["sigma"])
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if flow and sigma >= 1.0:
            raise ValueError("flow sigmas require 0 < sigma < 1")
        alpha, sigma_t = _sigma_to_alpha_sigma_t(sigma, flow)
        return {"alpha": alpha, "sigma_t": sigma_t}

    if mode == "convert":
        model_output = _as_f64(kw["model_output"])
        sample = _as_f64(kw["sample"])
        sigma = float(kw["sigma"])
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if flow and sigma >= 1.0:
            raise ValueError("flow sigmas require 0 < sigma < 1")
        converted, kind = _convert(model_output, sample, sigma,
                                   kw["algorithm_type"],
                                   kw["prediction_type"], flow)
        return {"converted": converted, "kind": kind}

    if mode == "first":
        sample = _as_f64(kw["sample"])
        m0 = _as_f64(kw["model_output"])
        prev, diag = _first_update(kw["sigmas"], int(kw["step_index"]), sample,
                                   m0, kw["algorithm_type"], flow)
        return {"prev_sample": prev, **diag}

    if mode == "second":
        sample = _as_f64(kw["sample"])
        mos = [_as_f64(x) for x in kw["model_outputs"]]
        if len(mos) != 2:
            raise ValueError("second order needs exactly 2 model outputs")
        prev, diag = _second_update(kw["sigmas"], int(kw["step_index"]), sample,
                                    mos, kw["algorithm_type"],
                                    kw["solver_type"], flow)
        return {"prev_sample": prev, **diag}

    if mode == "third":
        sample = _as_f64(kw["sample"])
        mos = [_as_f64(x) for x in kw["model_outputs"]]
        if len(mos) != 3:
            raise ValueError("third order needs exactly 3 model outputs")
        prev, diag = _third_update(kw["sigmas"], int(kw["step_index"]), sample,
                                   mos, kw["algorithm_type"], flow)
        return {"prev_sample": prev, **diag}

    if mode == "step":
        sigmas = _prep(kw["sigmas"], int(kw["step_index"]), 0, flow)
        n_steps = int(kw["n_steps"])
        if sigmas.numel() != n_steps + 1:
            raise ValueError("len(sigmas) must equal n_steps + 1")
        step_index = int(kw["step_index"])
        solver_order = int(kw["solver_order"])
        if solver_order not in (1, 2, 3):
            raise ValueError("solver_order must be 1, 2 or 3")
        lower_order_final_cfg = bool(kw["lower_order_final"])
        euler_at_final = bool(kw["euler_at_final"])
        algorithm_type = kw["algorithm_type"]
        prediction_type = kw["prediction_type"]
        solver_type = kw.get("solver_type", "midpoint")
        state = kw["state"]
        if state is None:
            model_outputs = [None] * solver_order
            lower_order_nums = 0
        else:
            model_outputs = list(state["model_outputs"])
            lower_order_nums = int(state["lower_order_nums"])
            if len(model_outputs) != solver_order:
                raise ValueError("state.model_outputs length mismatch")

        sample = _as_f64(kw["sample"])
        raw = _as_f64(kw["model_output"])
        converted, kind = _convert(raw, sample,
                                   float(sigmas[step_index].item()),
                                   algorithm_type, prediction_type, flow)

        # 官方 history 移位：整体左移一格，末位写入本步转换结果
        model_outputs = model_outputs[1:] + [converted]

        # 官方降阶标志：反演调度器没有 forward 的 final_sigmas_type 条件
        lower_order_final = (step_index == n_steps - 1) and (
            euler_at_final or (lower_order_final_cfg and n_steps < 15)
        )
        lower_order_second = (
            (step_index == n_steps - 2) and lower_order_final_cfg and n_steps < 15
        )

        if solver_order == 1 or lower_order_nums < 1 or lower_order_final:
            used_order = 1
            prev, diag = _first_update(sigmas, step_index, sample,
                                       converted, algorithm_type, flow)
        elif solver_order == 2 or lower_order_nums < 2 or lower_order_second:
            used_order = 2
            hist = model_outputs[-2:]
            if any(x is None for x in hist):
                raise ValueError("second order needs 2 filled history entries")
            prev, diag = _second_update(sigmas, step_index, sample, hist,
                                        algorithm_type, solver_type, flow)
        else:
            used_order = 3
            hist = model_outputs[-3:]
            if any(x is None for x in hist):
                raise ValueError("third order needs 3 filled history entries")
            prev, diag = _third_update(sigmas, step_index, sample, hist,
                                       algorithm_type, flow)

        new_lower_order_nums = min(lower_order_nums + 1, solver_order)
        new_state = {"model_outputs": model_outputs,
                     "lower_order_nums": new_lower_order_nums}
        return {"prev_sample": prev, "used_order": used_order,
                "kind": kind, "converted": converted,
                "lower_order_final": lower_order_final,
                "lower_order_second": lower_order_second,
                "lower_order_nums_before": lower_order_nums,
                "lower_order_nums_after": new_lower_order_nums,
                "state": new_state, **diag}

    raise ValueError(f"unknown mode: {mode}")
