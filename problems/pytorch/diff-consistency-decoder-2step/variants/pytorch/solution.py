import math

import torch


def _alpha_bar_fn(t, kind):
    if kind == "cosine":
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    if kind == "laplace":
        lmb = -0.5 * math.copysign(1.0, 0.5 - t) * math.log(1.0 - 2.0 * math.fabs(0.5 - t) + 1e-6)
        snr = math.exp(lmb)
        return math.sqrt(snr / (1.0 + snr))
    if kind == "exp":
        return math.exp(t * -12.0)
    raise ValueError(f"Unsupported alpha_transform_type: {kind}")


def _real(name, value, low=None, high=None, integer=False, exclusive_low=False):
    if integer:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是整数")
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    if low is not None and (value <= low if exclusive_low else value < low):
        raise ValueError(f"{name} 的取值非法")
    if high is not None and value > high:
        raise ValueError(f"{name} 的取值非法")
    return value


def _tensor(name, value, shape):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() < 1:
        raise ValueError(f"{name} 至少要有一维 batch")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} 的形状必须与 x_init 一致")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 必须全为有限值")
    return value.to(torch.float64)


def solve(
    x_init,
    f_first,
    f_second,
    noise,
    num_train_timesteps,
    alpha_transform="cosine",
    sigma_data=0.5,
    timesteps=(1008, 512),
    max_beta=0.999,
):
    """Consistency Decoder 的两步采样：betas_for_alpha_bar -> c_skip/c_out/c_in -> 重加噪。"""
    if alpha_transform not in ("cosine", "laplace", "exp"):
        raise ValueError("alpha_transform 只能是 cosine / laplace / exp")
    train_t = _real("num_train_timesteps", num_train_timesteps, integer=True)
    if train_t < 2:
        raise ValueError("num_train_timesteps 必须 >= 2")
    sd = _real("sigma_data", sigma_data, low=0.0, exclusive_low=True)
    mb = _real("max_beta", max_beta, low=0.0, exclusive_low=True, high=1.0)

    if isinstance(timesteps, (str, bytes)) or not isinstance(timesteps, (tuple, list)):
        raise ValueError("timesteps 必须是长度 2 的整数序列")
    if len(timesteps) != 2:
        raise ValueError("Consistency Decoder 只支持恰好 2 步")
    marks = []
    for item in timesteps:
        marks.append(_real("timesteps 的元素", item, integer=True))
    if marks[0] >= train_t or marks[1] >= train_t:
        raise ValueError("timesteps 的元素必须落在 [0, num_train_timesteps) 内")
    if marks[0] <= marks[1]:
        raise ValueError("timesteps 必须严格递减")

    if not isinstance(x_init, torch.Tensor):
        raise ValueError("x_init 必须是 torch.Tensor")
    if x_init.dim() < 1:
        raise ValueError("x_init 至少要有一维 batch")
    shape = tuple(x_init.shape)
    x0_in = _tensor("x_init", x_init, shape)
    f1 = _tensor("f_first", f_first, shape)
    f2 = _tensor("f_second", f_second, shape)
    z = _tensor("noise", noise, shape)

    betas = [
        min(1.0 - _alpha_bar_fn((i + 1) / train_t, alpha_transform) / _alpha_bar_fn(i / train_t, alpha_transform), mb)
        for i in range(train_t)
    ]
    alpha_bar = []
    acc = 1.0
    for beta in betas:
        acc *= 1.0 - beta
        if acc <= 0.0:
            raise ValueError("调度表退化：某个 alpha_bar 触到 0，sigma 会发散")
        alpha_bar.append(acc)

    c_skip, c_out, c_in = [], [], []
    for t in range(train_t):
        ab = alpha_bar[t]
        sigma = math.sqrt(1.0 / ab - 1.0)
        recip_root = 1.0 / math.sqrt(ab)
        c_skip.append(recip_root * sd**2 / (sigma**2 + sd**2))
        c_out.append(sigma * sd / math.sqrt(sigma**2 + sd**2))
        c_in.append(recip_root / math.sqrt(sigma**2 + sd**2))

    t_first, t_second = marks

    def denoise(x, f, idx):
        return c_out[idx] * f + c_skip[idx] * x

    scaled_first = x0_in * c_in[t_first]
    x0_first = denoise(x0_in, f1, t_first)
    sqrt_ab = math.sqrt(alpha_bar[t_second])
    sqrt_one_minus = math.sqrt(1.0 - alpha_bar[t_second])
    x_second = sqrt_ab * x0_first + sqrt_one_minus * z
    scaled_second = x_second * c_in[t_second]
    x_final = denoise(x_second, f2, t_second)

    return (
        scaled_first.to(torch.float32),
        x_second.to(torch.float32),
        scaled_second.to(torch.float32),
        x_final.to(torch.float32),
    )
