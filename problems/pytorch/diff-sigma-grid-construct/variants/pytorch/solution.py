import math

import torch

_SPACINGS = ("linspace", "leading", "trailing")
_INTERPOLATIONS = ("linear", "log_linear")
_FINAL = ("zero", "sigma_min")
RHO = 7.0


def _int(name, value, low):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    if value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return value


def _optional_positive(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字或 None")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限正数")
    return value


def _timestep_grid(train_steps, steps, spacing, offset):
    if spacing == "linspace":
        return torch.linspace(0, train_steps - 1, steps, dtype=torch.float64).flip(0)
    if spacing == "leading":
        step_ratio = train_steps // steps
        grid = torch.arange(0, steps, dtype=torch.float64) * step_ratio
        return grid.round().flip(0) + offset
    step_ratio = train_steps / steps
    count = int(math.ceil(train_steps / step_ratio))
    index = torch.arange(count, dtype=torch.float64)
    grid = train_steps - index * step_ratio
    return grid.round() - 1


def _interp_linear(grid, values):
    last = values.numel() - 1
    low = torch.clamp(torch.floor(grid), 0, last).to(torch.long).clamp(max=last - 1)
    weight = (grid - low.to(torch.float64)).clamp(0.0, 1.0)
    return values[low] * (1.0 - weight) + values[low + 1] * weight


def _karras_sigmas(in_sigmas, steps, sigma_min, sigma_max):
    low = float(in_sigmas[-1]) if sigma_min is None else sigma_min
    high = float(in_sigmas[0]) if sigma_max is None else sigma_max
    ramp = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    low_inv = low ** (1.0 / RHO)
    high_inv = high ** (1.0 / RHO)
    return (high_inv + ramp * (low_inv - high_inv)) ** RHO


def _exponential_sigmas(in_sigmas, steps, sigma_min, sigma_max):
    low = float(in_sigmas[-1]) if sigma_min is None else sigma_min
    high = float(in_sigmas[0]) if sigma_max is None else sigma_max
    return torch.exp(torch.linspace(math.log(high), math.log(low), steps, dtype=torch.float64))


def _sigma_to_t(sigmas, log_sigmas):
    last = log_sigmas.numel() - 1
    log_sigma = torch.log(torch.clamp(sigmas, min=1e-10))
    count = (log_sigmas[None, :] <= log_sigma[:, None]).sum(dim=1)
    low = (count - 1).clamp(0, last - 1)
    high = low + 1
    weight = (log_sigmas[low] - log_sigma) / (log_sigmas[low] - log_sigmas[high])
    weight = weight.clamp(0.0, 1.0)
    return (1.0 - weight) * low.to(torch.float64) + weight * high.to(torch.float64)


def solve(
    alphas_cumprod,
    num_inference_steps,
    timestep_spacing="linspace",
    interpolation_type="linear",
    use_karras_sigmas=False,
    use_exponential_sigmas=False,
    use_beta_sigmas=False,
    final_sigmas_type="zero",
    sigma_min=None,
    sigma_max=None,
    steps_offset=0,
):
    """构造 EulerDiscrete 的推理时间栅格：返回 (timesteps, sigmas)。

    alphas_cumprod:一维浮点张量，长度 T = num_train_timesteps，元素落在 (0, 1) 且全部有限；
    num_inference_steps:N，要求 1 <= N <= T。

    1) 时间栅格（timestep_spacing，通常长度 N，一律从噪声侧往干净侧递减）：
       "linspace" -> linspace(0, T-1, N) 再翻转（浮点等距）；
       "leading"  -> step_ratio = T // N（整除），(arange(N) * step_ratio).round() 翻转后 + steps_offset；
       "trailing" -> step_ratio = T / N（真除），取 count = ceil(T / step_ratio) 个点
                     grid[k] = T - k * step_ratio，再 round() - 1；
                     浮点除法不整时 count 会是 N+1（与 numpy 的 arange(T, 0, -step_ratio) 等长）。
   2) σ 基线 sigmas_full = sqrt((1 - abar) / abar)，长度 T。
   3) interpolation_type：
       "linear"     -> 在整数下标 0..T-1 上按 timesteps 线性插值 sigmas_full；
       "log_linear" -> exp(linspace(log(sigmas_full[-1]), log(sigmas_full[0]), N+1))，
                       注意这一支完全忽略时间栅格、且产生 N+1 个点。
   4) σ 转换器（三者至多一个为真，否则 ValueError）：
       use_karras_sigmas       -> (hi^(1/ρ) + ramp·(lo^(1/ρ) - hi^(1/ρ)))^ρ，ramp=linspace(0,1,N)，ρ=7；
                                   hi/lo 默认取 in_sigmas[0] / in_sigmas[-1]，可被 sigma_max/sigma_min 覆盖；
                                   随后 timesteps = 把每个 σ 反查回时间轴（log-σ 线性插值）。
       use_exponential_sigmas  -> exp(linspace(log(hi), log(lo), N))，同样反查 timesteps。
       use_beta_sigmas         -> 上游依赖 scipy.stats.beta.ppf，本题不支持，为真即 ValueError。
   5) 终端 σ（final_sigmas_type）："zero" -> 0；"sigma_min" -> sqrt((1 - abar[0]) / abar[0])。
      追加到 sigmas 末尾（所以 sigmas 比 timesteps 多一个，log_linear 无转换器时多两个）。

    返回 (timesteps, sigmas)，均为 float32 张量。
    """
    if not torch.is_tensor(alphas_cumprod):
        raise ValueError("alphas_cumprod 必须是 torch 张量")
    if not alphas_cumprod.is_floating_point():
        raise ValueError("alphas_cumprod 必须是浮点张量")
    if alphas_cumprod.dim() != 1:
        raise ValueError("alphas_cumprod 必须是一维张量")
    train_steps = alphas_cumprod.numel()
    if train_steps < 2:
        raise ValueError("alphas_cumprod 长度至少为 2")
    alpha = alphas_cumprod.to(torch.float64)
    if not bool(torch.isfinite(alpha).all()):
        raise ValueError("alphas_cumprod 必须全部有限")
    if bool((alpha <= 0).any()) or bool((alpha >= 1).any()):
        raise ValueError("alphas_cumprod 的元素必须落在开区间 (0, 1)")

    steps = _int("num_inference_steps", num_inference_steps, 1)
    if steps > train_steps:
        raise ValueError("num_inference_steps 不能大于训练步数")
    offset = _int("steps_offset", steps_offset, 0)
    if timestep_spacing not in _SPACINGS:
        raise ValueError(f"未知的 timestep_spacing: {timestep_spacing!r}")
    if interpolation_type not in _INTERPOLATIONS:
        raise ValueError(f"未知的 interpolation_type: {interpolation_type!r}")
    if final_sigmas_type not in _FINAL:
        raise ValueError(f"未知的 final_sigmas_type: {final_sigmas_type!r}")
    if sum(bool(flag) for flag in (use_karras_sigmas, use_exponential_sigmas, use_beta_sigmas)) > 1:
        raise ValueError("use_karras_sigmas / use_exponential_sigmas / use_beta_sigmas 至多只能开一个")
    if use_beta_sigmas:
        raise ValueError("use_beta_sigmas 依赖 scipy.stats.beta.ppf，本题不支持")
    low = _optional_positive("sigma_min", sigma_min)
    high = _optional_positive("sigma_max", sigma_max)
    if low is not None and high is not None and low >= high:
        raise ValueError("sigma_min 必须小于 sigma_max")

    timesteps = _timestep_grid(train_steps, steps, timestep_spacing, offset)

    sigmas_full = torch.sqrt((1.0 - alpha) / alpha)
    log_sigmas = torch.log(sigmas_full)

    if interpolation_type == "linear":
        sigmas = _interp_linear(timesteps, sigmas_full)
    else:
        sigmas = torch.exp(
            torch.linspace(
                math.log(float(sigmas_full[-1])),
                math.log(float(sigmas_full[0])),
                steps + 1,
                dtype=torch.float64,
            )
        )

    if use_karras_sigmas:
        sigmas = _karras_sigmas(sigmas, steps, low, high)
        timesteps = _sigma_to_t(sigmas, log_sigmas)
    elif use_exponential_sigmas:
        sigmas = _exponential_sigmas(sigmas, steps, low, high)
        timesteps = _sigma_to_t(sigmas, log_sigmas)

    if final_sigmas_type == "zero":
        sigma_last = 0.0
    else:
        sigma_last = math.sqrt(float((1.0 - alpha[0]) / alpha[0]))

    sigmas = torch.cat([sigmas, torch.tensor([sigma_last], dtype=torch.float64)])
    return timesteps.to(torch.float32), sigmas.to(torch.float32)
