"""SeaCache 频域指示器门控步.

回源: diffusers ``hooks/sea_cache.py`` (1097 行, SeaCache 论文
arXiv 2602.18993, Spectral-Evolution-Aware 指示器)。每步三段:

1. **SEA 滤波指示器** (``_apply_sea_filter`` L362-389 + ``_build_indicator``
   L392-405): 原始视觉隐张量 (C,T,H,W) 经 ``movedim(0,-1)`` 变 (T,H,W,C),
   对空间三维做 FFT; 每轴乘 Wiener 型增益
       clean_power = 1/(|f|^power_exp + 1e-16)
       axis_gain   = (1-σ)·clean_power / ((1-σ)²·clean_power + σ² + 1e-16)
   三轴增益连乘后按均值密度归一化 (mean>0 且有限), ifftn 取实部回原 dtype。
   σ 先 clamp 到 [1e-6, 1-1e-6]。
2. **门控** (``resolve_gate`` L182-245): 累积相对 L1 预算——
       distance  = mean|cur−prev| / (mean|prev| + 1e-16)
       candidate = accumulated_distance + distance
       compute   <=>  强制条件 ∨ candidate >= threshold      # 注意是 >=
   强制条件 (or): retention 窗 (step < retention_steps)、cache_end 窗
   (step >= N − cache_end_steps)、首次观测 (previous_indicator is None)、
   连续缓存达上限 (max_consecutive_cached>0 且 consecutive>=max)、
   指示器形状/dtype 变化 (distance=inf)。步非相邻 (≠上次+1) 先整条轨迹
   重置。compute 步 accumulated 归零, skip 步 accumulated = candidate。
3. **残差复用** (leader 路径 L827-878 + ``_record_full_execution`` L288-311):
   compute 步记录 (step, residual = gen_output − gen_input) 并把 history
   截断到 residual_order+1; skip 步取最近残差, residual_order=1 且有两条
   不同步记录时线性外推
       cached = latest + (latest − previous) · (step−latest_step)/(latest_step−prev_step)
   输出 = gen_input + cached; 无 history 却判 skip → fail-open compute 且
   accumulated 归零; history 残差形状/dtype 与 gen_input 不符 → 清空 history
   fail-open compute。compute 步 consecutive_cached=0, skip 步 +1。

与官方实现的判题域约定 (official -> judgement):
* 官方 hook 挂 leader/block 共享 state_manager, gate 有 (step,sigma) 备忘
  (一次 forward 内 leader+root 双入口); 判题域每步显式调用一次, 备忘省略。
* 官方 raw_vision 是 latent 列表、distance 逐项平均; 判题域单 latent,
  逐项平均退化为单项。
* 官方 und 序列 (Cosmos 双流) 与 FSDP/并行同步不在判题域; 取 generic
  视频 leader 路径, history 条目只含 (step, residual)。
* 官方 fail-open (无 context/autograd 等) 不入实现; 无 history 判 skip 与
  残差形状不符两条 fail-open 保留 (归零 accumulated 与官方 leader 路径一致)。
* 指示器在返回 dict 中显式给出 (官方内部态), 供对拍观测。
* state 纯函数式: 返回新 dict, 不修改入参。
"""

import math

import torch


def sea_filter(hidden_states, sigma, power_exp):
    """官方 _apply_sea_filter: 三轴 Wiener 型频域增益 + 密度归一化.

    入参 (T, H, W, C) 张量; 空间三维 FFT, 频域乘三轴增益连乘 (均值归一),
    ifftn 取实部回原 dtype。
    """
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 4:
        raise ValueError("hidden_states 必须是 4 维 (T, H, W, C) 张量")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) \
            or not math.isfinite(sigma):
        raise ValueError("sigma 必须是有限数值")
    if isinstance(power_exp, bool) or not isinstance(power_exp, (int, float)) \
            or not math.isfinite(power_exp) or power_exp <= 0:
        raise ValueError("power_exp 必须是正的有限数值")
    dtype = hidden_states.dtype
    x = hidden_states.contiguous().float()
    dims = (0, 1, 2)
    spectrum = torch.fft.fftn(x, dim=dims)
    sigma = max(1e-6, min(1.0 - 1e-6, float(sigma)))
    signal_scale = 1.0 - sigma
    noise_scale = sigma
    gain = None
    for axis in dims:
        freqs = torch.fft.fftfreq(x.shape[axis], dtype=torch.float32)
        clean_power = 1.0 / (freqs.abs().pow(power_exp) + 1e-16)
        axis_gain = signal_scale * clean_power \
            / (signal_scale ** 2 * clean_power + noise_scale ** 2 + 1e-16)
        shape = [1] * x.dim()
        shape[axis] = axis_gain.shape[0]
        gain = axis_gain.reshape(shape) if gain is None else gain * axis_gain.reshape(shape)
    mean_gain = gain.mean()
    if torch.isfinite(mean_gain) and mean_gain > 0:
        gain = gain / mean_gain
    return torch.fft.ifftn(spectrum * gain, dim=dims).real.to(dtype)


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("state 必须是 dict 或 None")
    required = {"history", "previous_indicator", "accumulated_distance",
                "consecutive_cached", "last_step"}
    extra = set(state) - required
    missing = required - set(state)
    if extra:
        raise ValueError(f"state 含未知键: {sorted(extra)}")
    if missing:
        raise ValueError(f"state 缺少键: {sorted(missing)}")
    history = state["history"]
    if not isinstance(history, list):
        raise ValueError("state['history'] 必须是 list")
    for entry in history:
        if not isinstance(entry, dict) or set(entry) != {"step", "residual"}:
            raise ValueError("history 条目必须是 {'step', 'residual'} dict")
        if isinstance(entry["step"], bool) or not isinstance(entry["step"], int) \
                or entry["step"] < 0:
            raise ValueError("history 条目的 step 必须是非负整数")
        if not isinstance(entry["residual"], torch.Tensor):
            raise ValueError("history 条目的 residual 必须是 torch 张量")
    indicator = state["previous_indicator"]
    if indicator is not None and not isinstance(indicator, torch.Tensor):
        raise ValueError("state['previous_indicator'] 必须是 torch 张量或 None")
    acc = state["accumulated_distance"]
    if isinstance(acc, bool) or not isinstance(acc, (int, float)) or acc < 0:
        raise ValueError("state['accumulated_distance'] 必须是非负数值")
    cached_n = state["consecutive_cached"]
    if isinstance(cached_n, bool) or not isinstance(cached_n, int) or cached_n < 0:
        raise ValueError("state['consecutive_cached'] 必须是非负整数")
    last_step = state["last_step"]
    if last_step is not None and (isinstance(last_step, bool)
                                  or not isinstance(last_step, int) or last_step < 0):
        raise ValueError("state['last_step'] 必须是非负整数或 None")


def solve(vision_latent, gen_input, gen_output, state, step_index, sigma,
          num_inference_steps, threshold=0.25, residual_order=1,
          retention_steps=1, cache_end_steps=1, max_consecutive_cached=2,
          power_exp=3.0):
    """SeaCache 门控步 (generic 视频路径)。

    参数
    ----
    vision_latent : torch.Tensor
        原始视觉隐张量 (C, T, H, W); 滤波后作距离指示器。
    gen_input / gen_output : torch.Tensor
        本步块栈输入与「假设完整执行」的输出 (同形状)。
    state : dict or None
        {history, previous_indicator, accumulated_distance,
         consecutive_cached, last_step}; 首步传 None。
    step_index : int
        当前调度器步 (0 <= step < num_inference_steps)。
    sigma : float
        本步噪声水平 ∈ [0, 1] (滤波内再 clamp)。
    num_inference_steps : int
        总步数 (>= 1)。
    threshold : float
        累积相对 L1 预算 (>= 0); candidate >= threshold 即 compute。
    residual_order : int
        0 = 直接复用最近残差; 1 = 两条不同步记录时线性外推。
    retention_steps / cache_end_steps : int
        开头/结尾强制完整执行的步数 (>= 0)。
    max_consecutive_cached : int
        连续缓存上限 (>= 0, 0 = 不限)。
    power_exp : float
        SEA 干净信号功率先验指数 (> 0, 官方默认 3.0)。

    返回
    ----
    dict(output, computed, distance, accumulated_distance, state)
        * output               : compute 步为 gen_output; skip 步为
                                 gen_input + (外推) 残差
        * computed             : bool
        * distance             : 本步相对 L1 (强制 compute 时 None)
        * accumulated_distance : 门控后的累积值 (compute 步归零)
        * state                : 新 state dict (不修改入参)
    """
    for name, t in (("vision_latent", vision_latent), ("gen_input", gen_input),
                    ("gen_output", gen_output)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是 torch 张量")
    if vision_latent.dim() != 4:
        raise ValueError("vision_latent 必须是 4 维 (C, T, H, W) 张量")
    if gen_input.shape != gen_output.shape:
        raise ValueError("gen_input 与 gen_output 形状必须一致")
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index 必须是非负整数")
    if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, int) \
            or num_inference_steps < 1:
        raise ValueError("num_inference_steps 必须 >= 1")
    if step_index >= num_inference_steps:
        raise ValueError("step_index 必须小于 num_inference_steps")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) \
            or not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma 必须落在 [0, 1]")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold 必须是非负有限数值")
    if isinstance(residual_order, bool) or residual_order not in (0, 1):
        raise ValueError("residual_order 必须是 0 或 1")
    if isinstance(retention_steps, bool) or not isinstance(retention_steps, int) \
            or retention_steps < 0:
        raise ValueError("retention_steps 必须是非负整数")
    if isinstance(cache_end_steps, bool) or not isinstance(cache_end_steps, int) \
            or cache_end_steps < 0:
        raise ValueError("cache_end_steps 必须是非负整数")
    if isinstance(max_consecutive_cached, bool) or not isinstance(max_consecutive_cached, int) \
            or max_consecutive_cached < 0:
        raise ValueError("max_consecutive_cached 必须是非负整数")
    if state is None:
        state = {"history": [], "previous_indicator": None,
                 "accumulated_distance": 0.0, "consecutive_cached": 0,
                 "last_step": None}
    _validate_state(state)
    history = list(state["history"])
    previous_indicator = state["previous_indicator"]
    accumulated = float(state["accumulated_distance"])
    consecutive_cached = int(state["consecutive_cached"])
    last_step = state["last_step"]

    indicator = sea_filter(vision_latent.movedim(0, -1), sigma=sigma,
                           power_exp=power_exp)

    # 步非相邻: 整条轨迹重置 (官方 fail-open "non-adjacent scheduler steps")
    if last_step is not None and step_index != last_step + 1:
        history = []
        previous_indicator = None
        accumulated = 0.0
        consecutive_cached = 0

    is_retained = step_index < retention_steps
    is_in_cache_end = step_index >= num_inference_steps - cache_end_steps
    is_first_observation = previous_indicator is None
    is_max_consecutive = bool(max_consecutive_cached
                              and consecutive_cached >= max_consecutive_cached)
    forced_compute = is_retained or is_in_cache_end or is_first_observation \
        or is_max_consecutive

    distance = None
    if forced_compute:
        should_compute = True
        candidate = 0.0
    else:
        if indicator.shape != previous_indicator.shape \
                or indicator.dtype != previous_indicator.dtype:
            distance = float("inf")
        else:
            numerator = (indicator.float() - previous_indicator.float()).abs().mean()
            denominator = previous_indicator.float().abs().mean() + 1e-16
            distance = float(numerator / denominator)
        if not math.isfinite(distance):
            should_compute = True
            candidate = 0.0
        else:
            candidate = accumulated + distance
            should_compute = candidate >= threshold
    accumulated = 0.0 if should_compute else candidate

    # 残差复用 (generic leader 路径, 含两条 fail-open)
    cached_residual = None
    if not should_compute and not history:
        should_compute = True
        accumulated = 0.0
    elif not should_compute:
        residual_history = history[-(residual_order + 1):]
        cached_residual = residual_history[-1]["residual"]
        if any(entry["residual"].shape != gen_input.shape
               or entry["residual"].dtype != gen_input.dtype
               for entry in residual_history):
            history = []
            accumulated = 0.0
            should_compute = True
        elif residual_order == 1 and len(residual_history) >= 2:
            prev_entry = residual_history[-2]
            latest_entry = residual_history[-1]
            if latest_entry["step"] != prev_entry["step"]:
                step_scale = (step_index - latest_entry["step"]) \
                    / (latest_entry["step"] - prev_entry["step"])
                cached_residual = latest_entry["residual"] \
                    + (latest_entry["residual"] - prev_entry["residual"]) * step_scale

    if should_compute:
        computed = True
        output = gen_output
        history = history + [{"step": step_index,
                              "residual": (gen_output - gen_input).detach().clone()}]
        history = history[-(residual_order + 1):]
        consecutive_cached = 0
    else:
        computed = False
        output = gen_input + cached_residual
        consecutive_cached += 1

    new_state = {
        "history": history,
        "previous_indicator": indicator.detach().clone(),
        "accumulated_distance": accumulated,
        "consecutive_cached": consecutive_cached,
        "last_step": step_index,
    }
    return {"output": output, "computed": computed, "distance": distance,
            "accumulated_distance": accumulated, "state": new_state}
