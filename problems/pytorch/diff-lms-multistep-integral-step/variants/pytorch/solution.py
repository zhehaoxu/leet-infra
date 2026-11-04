import math

import torch

_PREDICTION_TYPES = ("epsilon", "v_prediction", "sample")


def _float_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch 张量")
    if not value.is_floating_point():
        raise ValueError(f"{name} 必须是浮点张量")
    if value.numel() == 0:
        raise ValueError(f"{name} 不能为空张量")
    return value.to(torch.float64)


def _lagrange_integral(nodes, c, lo, hi):
    """官方 get_lms_coefficient 的解析替代：ℓ_c(τ) 是过节点 nodes[k]（k≠c 时
    为 0、k=c 时为 1）的 Lagrange 基，幂系数展开后逐项积分 ∫[lo, hi]。

    与官方 scipy.integrate.quad(epsrel=1e-4) 数学等价——被积函数是 ≤3 次多项式，
    quad 实际精度 ~1e-12，解析积分精确。nodes[k] = σ[t−k]。"""
    n = len(nodes)
    coef = [1.0]
    denom = 1.0
    for k in range(n):
        if k == c:
            continue
        denom *= nodes[c] - nodes[k]
        new = [0.0] * (len(coef) + 1)
        for i, a in enumerate(coef):
            new[i] -= a * nodes[k]
            new[i + 1] += a
        coef = new
    total = 0.0
    for i, a in enumerate(coef):
        total += (a / denom) * (hi ** (i + 1) - lo ** (i + 1)) / (i + 1)
    return total


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

    官方 set_timesteps 的表带 0 收尾：`sigmas = [σ0, …, σ_{N−1}, 0]`（长度 N+1，
    N ≥ 2），前 N 个为正、末位恰为 0、有限、单调不增。

    每次调用四拍（官方 step L545-623 逐字同构）：

    1. **convert**：在 σ = sigmas[step_index] 处把网络输出转成 x0 预测——
       epsilon → sample − σ·mo；v_prediction → mo·(−σ/√(σ²+1)) + sample/(σ²+1)；
       sample → x0 = mo。
    2. **导数**：derivative = (sample − x0)/σ（分母是当前 σ，必为正）；
       追加进 `state["derivatives"]` 历史，超长 pop 头部（保留最近 order 条）。
    3. **系数**：order_used = min(step_index + 1, order)（warmup：前几步降阶）；
       对 curr_order = c ∈ [0, order_used) 取官方 get_lms_coefficient(order_used,
       t, c)——被积函数 ℓ_c(τ) = Π_{k≠c}(τ − σ_{t−k})/(σ_{t−c} − σ_{t−k}) 是
       Lagrange 基（k=c 时为 1、其余节点为 0），在 [σ_t, σ_{t+1}] 上积分
       （区间反向时积分为负——σ 表递减，官方 quad 同款语义）。本题用幂系数
       展开解析积分。
    4. **落地**：prev = sample + Σ_c coeffs[c]·derivs[−(c+1)]——coeff[0] 配
       **最新**导数（官方 `zip(lms_coeffs, reversed(self.derivatives))`）。

    state = {"derivatives": [导数历史]}（float64 张量，长度 = min(step_index,
    order) 的调用前契约、返回后 = min(step_index + 1, order)）。无 None 头
    （与 756 不同，LMS 历史从第一步起全是真实导数）。

    首步（step_index = 0）order_used = 1：系数 = ∫[σ0, σ1] 1 dτ = σ1 − σ0（负值），
    prev = sample − (σ0 − σ1)·derivative——**退化为欧拉步**。末步积分区间
    [σ_{N−1}, 0] 积到 0 收尾。

    校验失败一律抛 ValueError。state 各张量以 float64 存取。
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
        raise ValueError("sigmas 必须非负")
    if bool((values[1:] > values[:-1]).any()):
        raise ValueError("sigmas 必须单调不增")
    if float(values[-1]) != 0.0:
        raise ValueError("sigmas 末位必须恰为 0（官方 set_timesteps 会追加）")
    if bool((values[:-1] <= 0).any()):
        raise ValueError("sigmas 前 N 个必须为正")

    num_steps = values.numel() - 1  # N = num_inference_steps
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index > num_steps - 1:
        raise ValueError(f"step_index 必须落在 [0, {num_steps - 1}]")

    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError("order 必须是整数")
    if order < 1 or order > 4:
        raise ValueError("order 必须落在 [1, 4]")

    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")

    # ---- 状态解析 ----
    if state is None:
        history = []
    else:
        if not isinstance(state, dict) or set(state.keys()) != {"derivatives"}:
            raise ValueError("state 必须是只含 derivatives 键的 dict")
        if step_index == 0:
            raise ValueError("step_index 为 0 时不能传 state（冷启动请传 None）")
        history_raw = state["derivatives"]
        if not isinstance(history_raw, list):
            raise ValueError("state.derivatives 必须是 list")
        if len(history_raw) != min(step_index, order):
            raise ValueError(
                f"state.derivatives 长度必须恰为 min(step_index, order) = {min(step_index, order)}"
            )
        history = []
        for idx, item in enumerate(history_raw):
            stored = _float_tensor(f"state.derivatives[{idx}]", item)
            if stored.shape != sample64.shape:
                raise ValueError("state.derivatives 条目必须与 sample 同形状")
            history.append(stored)

    # ---- 1. convert ----
    sigma = float(values[step_index])
    if prediction_type == "epsilon":
        pred_original = sample64 - sigma * output64
    elif prediction_type == "v_prediction":
        pred_original = output64 * (-sigma / math.sqrt(sigma**2 + 1.0)) + sample64 / (sigma**2 + 1.0)
    else:  # sample
        pred_original = output64

    # ---- 2. 导数与历史 ----
    derivative = (sample64 - pred_original) / sigma
    history = history + [derivative]
    if len(history) > order:
        history = history[1:]

    # ---- 3. LMS 系数（σ 空间 Lagrange 基积分）----
    order_used = min(step_index + 1, order)
    nodes = [float(values[step_index - k]) for k in range(order_used)]
    lo = float(values[step_index])
    hi = float(values[step_index + 1])
    lms_coefficients = [_lagrange_integral(nodes, c, lo, hi) for c in range(order_used)]

    # ---- 4. 落地（coeff[0] 配最新导数）----
    prev = sample64
    for c in range(order_used):
        prev = prev + lms_coefficients[c] * history[-(c + 1)]

    return {
        "prev_sample": prev.to(torch.float32),
        "pred_original_sample": pred_original.to(torch.float32),
        "derivative": derivative.to(torch.float32),
        "lms_coefficients": lms_coefficients,
        "order_used": order_used,
        "state": {"derivatives": history},
    }
