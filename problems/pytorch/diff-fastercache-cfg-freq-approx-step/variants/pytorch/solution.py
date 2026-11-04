"""FasterCache CFG-Cache 频域逼近步.

回源: diffusers ``hooks/faster_cache.py`` (654 行, FasterCache 论文
arXiv 2410.19355) 的 ``FasterCacheDenoiserHook`` (L261-361) 与
``_split_low_high_freq`` (L638-654)、``apply_faster_cache`` 的权重缺省
(L524-562)。CFG-Cache: 相邻步的 uncond 与 cond 分支输出之差在频域上缓变,
用上一步的**频域 delta** 逼近本步 uncond 分支, 省掉一半 batch 前向。

每步门控 (L271-293):
    within  = lo < current_timestep < hi               # 开区间 (严格)
    skip_uncond = within and iteration % skip_range != 0 and not distilled
    (官方 `iteration > 0 and ...` 的 >0 是 0%N==0 死代码)
蒸馏模型: 恒整批前向, delta 永不建立 (L295-297)。

非 skip 步 (整批前向, 输出 2B = uncond||cond, L339-351):
    uncond, cond = chunk(2, dim=0)
    各自 permute BCFHW->(B,F,C,H,W)->flatten(0,1) 后 _split_low_high_freq:
        fft2 -> fftshift (全维) -> 圆形低频掩码 (radius = min(H,W)//5,
        中心 (W//2, H//2), (x-cx)²+(y-cy)² <= r²)
        low = shifted*mask, high = shifted*~mask
    delta_low = low(uncond) - low(cond);  delta_high 同理, 存入 state。

skip 步 (只跑 cond, 输出 B, L306-338):
    state.delta <- state.delta * weight               # 权重作用在**存储的** delta 上
    weight = alpha if lo_w < t < hi_w else 1.0         # 各自更新区间, 开区间
    cond 频域拆分后:
        uncond_freq = (delta_low + low_cond) + (delta_high + high_cond)
        uncond = real(ifft2(ifftshift(uncond_freq)))   # 还原 BCFHW 排布
    output = cat([uncond, cond], dim=0)                # 2B

与官方实现的判题域约定 (official -> judgement):
* 官方 hook 包住 denoiser 前向, skip 时先把输入切半再前向; 判题域 model_output
  是「本次实际执行的输出」——skip 步为条件半批输出 (B), 否则整批输出 (2B),
  batch_size 在首个非 skip 步记录并保持。
* 权重是 callback(module) 的返回值; 判题域按官方缺省规则内联:
  alpha_low/high_frequency 在各自 low/high_frequency_weight_update_timestep_range
  开区间内取 alpha, 否则 1.0。
* tensor_format 固定 "BCFHW" (BFCHW/BCHW 分支与 tuple 输出分支不入)。
* 官方不校验; 判题域补张量/区间/alpha/state 四键/skip 步 delta 必须已建立/
  输入 batch 与门控一致等显式校验。
* state 纯函数式: 返回新 dict, delta clone。
"""

import torch


def _split_low_high_freq(x):
    """官方 _split_low_high_freq: fft2 -> fftshift -> 圆形低频掩码拆分."""
    fft = torch.fft.fft2(x)
    fft_shifted = torch.fft.fftshift(fft)
    height, width = x.shape[-2:]
    radius = min(height, width) // 5
    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    center_x, center_y = width // 2, height // 2
    mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius ** 2
    low_freq_mask = mask.unsqueeze(0).unsqueeze(0)
    high_freq_mask = ~low_freq_mask
    return fft_shifted * low_freq_mask, fft_shifted * high_freq_mask


def _to_flat(hidden_states):
    """BCFHW -> (B*F, C, H, W) (官方 permute(0,2,1,3,4) + flatten(0,1))."""
    return hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)


def _from_flat(flat, batch_size):
    """(B*F, C, H, W) -> BCFHW (官方 unflatten + permute 还原)."""
    return flat.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)


def _freq_weight(timestep, update_range, alpha):
    lo, hi = update_range
    return alpha if lo < timestep < hi else 1.0


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("state 必须是 dict 或 None")
    required = {"iteration", "batch_size", "low_frequency_delta", "high_frequency_delta"}
    extra = set(state) - required
    missing = required - set(state)
    if extra:
        raise ValueError(f"state 含未知键: {sorted(extra)}")
    if missing:
        raise ValueError(f"state 缺少键: {sorted(missing)}")
    iteration = state["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("state['iteration'] 必须是非负整数")
    batch_size = state["batch_size"]
    if batch_size is not None and (isinstance(batch_size, bool)
                                   or not isinstance(batch_size, int) or batch_size < 2
                                   or batch_size % 2 != 0):
        raise ValueError("state['batch_size'] 必须是正偶数或 None")
    for key in ("low_frequency_delta", "high_frequency_delta"):
        delta = state[key]
        if delta is not None and not isinstance(delta, torch.Tensor):
            raise ValueError(f"state['{key}'] 必须是复数张量或 None")


def _validate_range(range_value, name):
    # str/bytes 情形已被 list/tuple 检查子sumed, 不单列
    if not isinstance(range_value, (list, tuple)) or len(range_value) != 2 \
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in range_value):
        raise ValueError(f"{name} 必须是 (lo, hi) 二元数值组")


def solve(model_output, state, current_timestep,
          unconditional_batch_skip_range=5,
          unconditional_batch_timestep_skip_range=(-1, 641),
          low_frequency_weight_update_timestep_range=(99, 901),
          high_frequency_weight_update_timestep_range=(-1, 301),
          alpha_low_frequency=1.1, alpha_high_frequency=1.1,
          is_guidance_distilled=False):
    """FasterCache CFG-Cache 频域逼近步。

    参数
    ----
    model_output : torch.Tensor
        本次实际执行的 denoiser 输出 (BCFHW): 门控判 skip 时为条件半批
        (B = batch_size//2), 否则为整批 (2B = uncond||cond)。
    state : dict or None
        {iteration, batch_size, low_frequency_delta, high_frequency_delta};
        delta 是 fftshift 域的复数张量; 首步传 None。
    current_timestep : int
        当前调度器时间步。
    unconditional_batch_skip_range : int
        每 N 次迭代真算一次 uncond 分支 (iteration % N == 0), 其余跳过。
    unconditional_batch_timestep_skip_range : (lo, hi)
        允许跳过 uncond 的时间步开区间。
    low/high_frequency_weight_update_timestep_range : (lo, hi)
        各自 alpha 权重的生效开区间 (区间外权重 1.0)。
    alpha_low/high_frequency : float
        低/高频 delta 的缩放权重 (官方 Equation 11 的 ⍺1/⍺2)。
    is_guidance_distilled : bool
        蒸馏模型无 uncond 分支: 恒整批, delta 永不建立。

    返回
    ----
    dict(output, uncond_skipped, state)
        * output         : 非 skip 步为 model_output 原样; skip 步为
                           cat([uncond 近似, cond], dim=0) (2B)
        * uncond_skipped : 本步是否跳过了 uncond 分支前向
        * state          : 新 state dict (不修改入参)
    """
    if not isinstance(model_output, torch.Tensor):
        raise ValueError("model_output 必须是 torch 张量")
    if model_output.dim() != 5:
        raise ValueError("model_output 必须是 5 维 BCFHW 张量")
    if isinstance(current_timestep, bool) or not isinstance(current_timestep, int):
        raise ValueError("current_timestep 必须是整数")
    if isinstance(unconditional_batch_skip_range, bool) \
            or not isinstance(unconditional_batch_skip_range, int) \
            or unconditional_batch_skip_range < 1:
        raise ValueError("unconditional_batch_skip_range 必须 >= 1")
    _validate_range(unconditional_batch_timestep_skip_range,
                    "unconditional_batch_timestep_skip_range")
    _validate_range(low_frequency_weight_update_timestep_range,
                    "low_frequency_weight_update_timestep_range")
    _validate_range(high_frequency_weight_update_timestep_range,
                    "high_frequency_weight_update_timestep_range")
    for alpha_name, alpha in (("alpha_low_frequency", alpha_low_frequency),
                              ("alpha_high_frequency", alpha_high_frequency)):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) \
                or not torch.isfinite(torch.tensor(float(alpha))):
            raise ValueError(f"{alpha_name} 必须是有限数值")
    if not isinstance(is_guidance_distilled, bool):
        raise ValueError("is_guidance_distilled 必须是 bool")
    if state is None:
        state = {"iteration": 0, "batch_size": None,
                 "low_frequency_delta": None, "high_frequency_delta": None}
    _validate_state(state)
    iteration = int(state["iteration"])
    batch_size = state["batch_size"]
    cur_batch = model_output.size(0)

    lo, hi = unconditional_batch_timestep_skip_range
    is_within_timestep_range = lo < current_timestep < hi
    should_skip_uncond = (is_within_timestep_range
                          and iteration % unconditional_batch_skip_range != 0
                          and not is_guidance_distilled)
    if should_skip_uncond:
        if batch_size is None:
            raise ValueError("跳过 uncond 需要 state['batch_size'] 已记录 (首步必须整批前向)")
        if cur_batch * 2 != batch_size:
            raise ValueError("skip 步的输入必须是条件半批 (batch == batch_size // 2)")
        if state["low_frequency_delta"] is None or state["high_frequency_delta"] is None:
            raise ValueError("skip 步的频域 delta 必须已建立 (前置整批步先更新)")

        low_weight = _freq_weight(current_timestep,
                                  low_frequency_weight_update_timestep_range,
                                  alpha_low_frequency)
        high_weight = _freq_weight(current_timestep,
                                   high_frequency_weight_update_timestep_range,
                                   alpha_high_frequency)
        delta_low = state["low_frequency_delta"] * low_weight
        delta_high = state["high_frequency_delta"] * high_weight

        flat = _to_flat(model_output).float()
        low_cond, high_cond = _split_low_high_freq(flat)
        uncond_freq = (delta_low + low_cond) + (delta_high + high_cond)
        uncond_states = torch.fft.ifftshift(uncond_freq)
        uncond_states = torch.fft.ifft2(uncond_states).real
        uncond_states = _from_flat(uncond_states, cur_batch).to(model_output.dtype)
        output = torch.cat([uncond_states, model_output], dim=0)
        uncond_skipped = True
        new_state = {
            "iteration": iteration + 1,
            "batch_size": batch_size,
            "low_frequency_delta": delta_low.detach().clone(),
            "high_frequency_delta": delta_high.detach().clone(),
        }
    else:
        if batch_size is None:
            batch_size = cur_batch
        if cur_batch != batch_size:
            raise ValueError("整批步的输入 batch 必须与 state['batch_size'] 一致")
        if cur_batch % 2 != 0:
            raise ValueError("整批步的输入必须是 uncond||cond 拼接 (batch 为偶数)")
        if not is_guidance_distilled:
            uncond_states, cond_states = model_output.chunk(2, dim=0)
            flat_uncond = _to_flat(uncond_states).float()
            flat_cond = _to_flat(cond_states).float()
            low_uncond, high_uncond = _split_low_high_freq(flat_uncond)
            low_cond, high_cond = _split_low_high_freq(flat_cond)
            delta_low = low_uncond - low_cond
            delta_high = high_uncond - high_cond
        else:
            delta_low = state["low_frequency_delta"]
            delta_high = state["high_frequency_delta"]
        output = model_output
        uncond_skipped = False
        new_state = {
            "iteration": iteration + 1,
            "batch_size": batch_size,
            "low_frequency_delta": (delta_low.detach().clone()
                                    if isinstance(delta_low, torch.Tensor) else delta_low),
            "high_frequency_delta": (delta_high.detach().clone()
                                     if isinstance(delta_high, torch.Tensor) else delta_high),
        }
    return {"output": output, "uncond_skipped": uncond_skipped, "state": new_state}
