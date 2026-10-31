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


def solve(
    model_output,
    sample,
    sigmas,
    step_index,
    state=None,
    prediction_type="epsilon",
    clip_sample=False,
    clip_sample_range=1.0,
):
    """Heun 预测-校正半步：返回 dict(prev_sample, pred_original_sample, state)。

    上游把 N 个逻辑步展开成 2N 个 σ 索引（timesteps 长 2N−1，sigmas 长 2N）：
    sigmas = [s0, s1, s1, s2, s2, …, s_{N-1}, s_{N-1}, 0]。
    逻辑步 j（0 ≤ j ≤ N−2）占索引 2j（预测）与 2j+1（校正）；**最后一个逻辑步
    只有预测**（索引 2N−2，σ 一步降到 0）——sigmas 比 timesteps 多一位，故校正
    半步永远不会遇到 σ=0。

    预测半步（state=None，step_index 须为偶数）：
      σ = sigmas[i]，σ_next = sigmas[i+1]；σ 必须 > 0。
      σ_input = σ；x0 = f(sample, mo, σ_input)（见下）
      derivative = (sample − x0) / σ          ← 除以的是当前 σ（σ_hat == σ）
      dt = σ_next − σ
      prev = sample + derivative·dt
      返回 state = {"prev_derivative", "dt", "sample"}（供校正半步使用）。

    校正半步（state 为 dict，step_index 须为奇数且 1 ≤ i ≤ 2N−3）：
      要求 sigmas[i] == sigmas[i+1]（成对结构），σ_next = sigmas[i] > 0。
      σ_input = σ_next；x0 = f(sample, mo, σ_input)——sample 是**欧拉半步结果**
      （pipeline 传入），mo 是在新点上的模型输出。
      derivative = (sample − x0) / σ_next      ← 除以的是 σ_next
      avg = (state.prev_derivative + derivative) / 2
      prev = state.sample + avg·state.dt      ← 从**暂存样本**出发、用**暂存步长**
      返回 state = None（状态清空）。

    x0 的三种 prediction_type（σ_input 为预测半步的 σ / 校正半步的 σ_next）：
      epsilon      → sample − σ_input·mo
      v_prediction → mo·(−σ_input/√(σ_input²+1)) + sample/(σ_input²+1)
      sample       → mo

    clip_sample=True 时对 x0 clamp 到 ±clip_sample_range（发生在导数计算之前）。

    sigmas：一维浮点张量，有限、非负、单调不增，长度 ≥ 3。
    返回 dict：prev_sample / pred_original_sample（float32）、state（dict 或 None）、
    derivative（本半步算得的导数，float32）。
    """
    output64 = _float_tensor("model_output", model_output)
    sample64 = _float_tensor("sample", sample)
    if sample64.shape != output64.shape:
        raise ValueError("sample 与 model_output 形状必须一致")

    if not torch.is_tensor(sigmas) or not sigmas.is_floating_point() or sigmas.dim() != 1:
        raise ValueError("sigmas 必须是一维浮点张量")
    if sigmas.numel() < 3:
        raise ValueError("sigmas 长度至少为 3（成对结构 + 末位 0）")
    values = sigmas.to(torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("sigmas 必须全部有限")
    if bool((values < 0).any()):
        raise ValueError("sigmas 必须非负")
    if bool((values[1:] > values[:-1]).any()):
        raise ValueError("sigmas 必须单调不增")

    total = values.numel()  # 2N
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise ValueError("step_index 必须是整数")
    if step_index < 0 or step_index > total - 2:
        raise ValueError(f"step_index 必须落在 [0, {total - 2}]")

    if prediction_type not in _PREDICTION_TYPES:
        raise ValueError(f"未知的 prediction_type: {prediction_type!r}")
    if not isinstance(clip_sample, bool):
        raise ValueError("clip_sample 必须是 bool")
    if not isinstance(clip_sample_range, (int, float)) or isinstance(clip_sample_range, bool):
        raise ValueError("clip_sample_range 必须是数")

    if state is None:
        # ---- 预测半步 ----
        if step_index % 2 != 0:
            raise ValueError("预测半步（state=None）的 step_index 必须为偶数")
        sigma = float(values[step_index])
        sigma_next = float(values[step_index + 1])
        if sigma <= 0.0:
            raise ValueError("预测半步的 sigma 必须为正")
        sigma_input = sigma
        if prediction_type == "epsilon":
            x0 = sample64 - sigma_input * output64
        elif prediction_type == "v_prediction":
            x0 = output64 * (-sigma_input / math.sqrt(sigma_input**2 + 1.0)) + sample64 / (
                sigma_input**2 + 1.0
            )
        else:
            x0 = output64
        if clip_sample:
            x0 = torch.clamp(x0, -float(clip_sample_range), float(clip_sample_range))
        derivative = (sample64 - x0) / sigma
        dt = sigma_next - sigma
        prev = sample64 + derivative * dt
        new_state = {
            "prev_derivative": derivative,
            "dt": dt,
            "sample": sample64,
        }
        return {
            "prev_sample": prev.to(torch.float32),
            "pred_original_sample": x0.to(torch.float32),
            "state": new_state,
            "derivative": derivative.to(torch.float32),
        }

    # ---- 校正半步 ----
    if not isinstance(state, dict) or set(state.keys()) != {"prev_derivative", "dt", "sample"}:
        raise ValueError("state 必须是含 prev_derivative/dt/sample 三键的 dict")
    if step_index % 2 != 1:
        raise ValueError("校正半步的 step_index 必须为奇数")
    if step_index > total - 3:
        raise ValueError("最后一个逻辑步只有预测半步，不存在校正（索引 2N−1 不应被调用）")
    if float(values[step_index]) != float(values[step_index + 1]):
        raise ValueError("sigmas 成对结构被破坏：sigmas[i] 必须等于 sigmas[i+1]")
    sigma_next = float(values[step_index])
    if sigma_next <= 0.0:
        raise ValueError("校正半步的 sigma_next 必须为正")

    prev_derivative = _float_tensor("state.prev_derivative", state["prev_derivative"])
    stored_sample = _float_tensor("state.sample", state["sample"])
    if prev_derivative.shape != sample64.shape or stored_sample.shape != sample64.shape:
        raise ValueError("state 的张量必须与 sample 同形状")
    dt = state["dt"]
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise ValueError("state.dt 必须是数")
    dt = float(dt)

    sigma_input = sigma_next
    if prediction_type == "epsilon":
        x0 = sample64 - sigma_input * output64
    elif prediction_type == "v_prediction":
        x0 = output64 * (-sigma_input / math.sqrt(sigma_input**2 + 1.0)) + sample64 / (
            sigma_input**2 + 1.0
        )
    else:
        x0 = output64
    if clip_sample:
        x0 = torch.clamp(x0, -float(clip_sample_range), float(clip_sample_range))
    derivative = (sample64 - x0) / sigma_next
    avg = (prev_derivative + derivative) / 2.0
    prev = stored_sample + avg * dt
    return {
        "prev_sample": prev.to(torch.float32),
        "pred_original_sample": x0.to(torch.float32),
        "state": None,
        "derivative": derivative.to(torch.float32),
    }
