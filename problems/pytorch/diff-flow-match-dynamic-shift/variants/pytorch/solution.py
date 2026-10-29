import math

import torch

_SHIFT_TYPES = ("exponential", "linear")


def _int(name, value, low):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return value


def _float(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数")
    return value


def _time_shift(sigmas, mu, shift_type):
    ratio = 1.0 / sigmas - 1.0
    if shift_type == "exponential":
        scale = math.exp(mu)
        return scale / (scale + ratio)
    return mu / (mu + ratio)


def solve(
    num_inference_steps,
    shift=1.0,
    use_dynamic_shifting=False,
    mu=None,
    time_shift_type="exponential",
    shift_terminal=None,
    invert_sigmas=False,
    sigma_min=0.0,
    sigma_max=1.0,
    num_train_timesteps=1000,
):
    """构造 FlowMatchEuler 的 σ 序列与时间刻度，覆盖动态平移与 terminal 拉伸。

    num_inference_steps:N，要求 >= 1；num_train_timesteps:T（默认 1000），要求 >= 1。
    sigma_min / sigma_max:σ 栅格两端，要求 0 <= sigma_min < sigma_max <= 1。

    1) 初始 σ 栅格：sigmas = linspace(sigma_max, sigma_min, N)（从噪声端递减）。
    2) 平移：
       use_dynamic_shifting=False -> 静态平移 sigmas = shift·s / (1 + (shift-1)·s)；
       use_dynamic_shifting=True  -> 必须给 mu，按 time_shift_type 逐元素：
            "exponential" -> e^mu / (e^mu + (1/s - 1))
            "linear"      -> mu / (mu + (1/s - 1))
    3) 若给 shift_terminal（要求 0 < shift_terminal < 1 且 N >= 2）：
       先用平移后的末项算 scale = (1 - sigmas[-1]) / (1 - shift_terminal)，
       再 sigmas = 1 - (1 - sigmas) / scale。注意这一步在平移之后、取 timesteps 之前。
    4) timesteps = sigmas · T。
    5) invert_sigmas=True 时再取 sigmas = 1 - sigmas、timesteps = sigmas · T，
       终端追加 1.0；否则终端追加 0.0。

    返回 (timesteps, sigmas)：timesteps 为 float32、长度 N；sigmas 为 float32、长度 N+1。
    """
    steps = _int("num_inference_steps", num_inference_steps, 1)
    train_steps = _int("num_train_timesteps", num_train_timesteps, 1)
    shift = _float("shift", shift)
    if shift <= 0.0:
        raise ValueError("shift 必须是正数")
    low = _float("sigma_min", sigma_min)
    high = _float("sigma_max", sigma_max)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("需要 0 <= sigma_min < sigma_max <= 1")
    if time_shift_type not in _SHIFT_TYPES:
        raise ValueError(f"未知的 time_shift_type: {time_shift_type!r}")
    terminal = None if shift_terminal is None else _float("shift_terminal", shift_terminal)
    if terminal is not None:
        if not 0.0 < terminal < 1.0:
            raise ValueError("shift_terminal 必须落在开区间 (0, 1)")
        if steps < 2:
            raise ValueError("设置 shift_terminal 时 num_inference_steps 不能小于 2")

    if use_dynamic_shifting:
        if mu is None:
            raise ValueError("use_dynamic_shifting=True 时必须提供 mu")
        mu = _float("mu", mu)
        if time_shift_type == "linear" and mu <= 0.0:
            raise ValueError("linear 平移要求 mu > 0")

    sigmas = torch.linspace(high, low, steps, dtype=torch.float64)

    if use_dynamic_shifting:
        sigmas = _time_shift(sigmas, mu, time_shift_type)
    else:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

    if terminal is not None:
        scale = (1.0 - float(sigmas[-1])) / (1.0 - terminal)
        sigmas = 1.0 - (1.0 - sigmas) / scale

    timesteps = sigmas * train_steps
    if invert_sigmas:
        sigmas = 1.0 - sigmas
        timesteps = sigmas * train_steps
        tail = 1.0
    else:
        tail = 0.0

    sigmas = torch.cat([sigmas, torch.tensor([tail], dtype=torch.float64)])
    return timesteps.to(torch.float32), sigmas.to(torch.float32)
