"""MagCache 累积误差门控步.

回源: diffusers ``hooks/mag_cache.py`` (469 行, MagCache 论文/仓库
github.com/Zehong-Ma/MagCache)。与 FBCache (764) 的单步残差比较不同, MagCache
用**累积误差**决定跳过:

1. mag_ratios: 模型相关的每步幅度比 (checkpoint 校准得到); 长度 !=
   num_inference_steps 时用 nearest_interp 最近邻插值:
       target == 1            -> 取 src[-1:]
       scale = (src_len - 1) / (target_len - 1)
       idx = round(grid * scale)                      # grid = 0..target_len-1
2. retention_step = int(retention_ratio * num_inference_steps + 0.5)
   (四舍五入); step < retention_step 永不跳过。
3. step >= retention_step 时 (先累积后判定):
       accumulated_ratio *= current_scale             # current_scale = ratios[step]
       accumulated_steps += 1
       accumulated_err   += abs(1.0 - accumulated_ratio)
       skip 条件 = previous_residual is not None
                   and accumulated_err <= threshold
                   and accumulated_steps <= max_skip_steps
   不满足 skip 条件 -> 重置 (ratio=1.0, steps=0, err=0.0) 并 compute;
   满足 -> 跳过整段块, output = hidden_states + previous_residual。
4. compute 步: output = tail_output (真实执行结果),
   previous_residual = tail_output - hidden_states (供后续 skip 使用)。

与官方实现的判题域约定 (official -> judgement):
* 官方 hook 挂 head/tail 共享 state, step_index 由 tail hook 推进; 判题域 step
  显式入参。
* 官方 calibrate 模式 (测 ratios 用) 不入实现; mag_ratios 必须显式给。
* 官方 skip 时形状不匹配的文本+图像拼接分支 (Flux/SD3 专用) 不入实现, 判题域
  要求形状一致。
* current_scale = ratios[step] if step < len(ratios) else 1.0 (官方同款越界回退,
  scale=1.0 时 ratio 不变但 err 仍按 abs(1-ratio) 累积)。
* state 纯函数式: 返回新 dict, 不修改入参。
"""

import torch


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("state 必须是 dict 或 None")
    required = {"previous_residual", "accumulated_ratio",
                "accumulated_steps", "accumulated_err"}
    extra = set(state) - required
    missing = required - set(state)
    if extra:
        raise ValueError(f"state 含未知键: {sorted(extra)}")
    if missing:
        raise ValueError(f"state 缺少键: {sorted(missing)}")
    res = state["previous_residual"]
    if res is not None and not isinstance(res, torch.Tensor):
        raise ValueError("state['previous_residual'] 必须是 torch 张量或 None")
    for k in ("accumulated_ratio", "accumulated_err"):
        v = state[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"state['{k}'] 必须是数值")
    steps = state["accumulated_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("state['accumulated_steps'] 必须是非负整数")


def nearest_interp(src, target_length):
    """官方 nearest_interp: 最近邻索引插值 (返回 1D float 张量)."""
    if not isinstance(src, torch.Tensor) or src.dim() != 1 or src.numel() < 1:
        raise ValueError("src 必须是非空一维 torch 张量")
    if not src.dtype.is_floating_point:
        raise ValueError("src 必须是浮点张量")
    if not isinstance(target_length, int) or isinstance(target_length, bool) \
            or target_length < 1:
        raise ValueError("target_length 必须是 >= 1 的整数")
    if target_length == 1:
        return src[-1:].clone()
    scale = (src.numel() - 1) / (target_length - 1)
    grid = torch.arange(target_length, dtype=torch.float32)
    mapped = torch.round(grid * scale).long()
    return src[mapped].clone()


def solve(hidden_states, tail_output, state, step, num_inference_steps,
          mag_ratios, threshold=0.06, retention_ratio=0.2, max_skip_steps=3):
    """MagCache 累积误差门控步。

    参数
    ----
    hidden_states : torch.Tensor
        模型输入 (首块输入)。
    tail_output : torch.Tensor
        假设完整执行时的尾块输出 (compute 路径的最终输出)。
    state : dict or None
        {"previous_residual", "accumulated_ratio", "accumulated_steps",
         "accumulated_err"}; 首步传 None (ratio=1.0, steps=0, err=0.0)。
    step : int
        当前去噪步索引 (>= 0)。
    num_inference_steps : int
        总推理步数 (>= 1), 用于 retention 与 mag_ratios 插值。
    mag_ratios : torch.Tensor or list[float]
        每步幅度比; 长度 != num_inference_steps 时最近邻插值。
    threshold : float
        累积误差阈值 (>= 0)。
    retention_ratio : float
        前置保留比例 ∈ [0, 1], retention_step = int(ratio*N + 0.5)。
    max_skip_steps : int
        连续 skip 步数上限 (accumulated_steps <= max_skip_steps)。

    返回
    ----
    dict(output, computed, accumulated_err, state)
        * output          : compute 步为 tail_output; skip 步为
                            hidden_states + previous_residual
        * computed        : bool
        * accumulated_err : 判定时的累积误差 (float)
        * state           : 新 state dict (不修改入参)
    """
    for name, t in (("hidden_states", hidden_states), ("tail_output", tail_output)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是 torch 张量")
    if hidden_states.shape != tail_output.shape:
        raise ValueError("hidden_states 与 tail_output 形状必须一致")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step 必须是非负整数")
    if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, int) \
            or num_inference_steps < 1:
        raise ValueError("num_inference_steps 必须 >= 1")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError("threshold 必须是非负数值")
    if isinstance(retention_ratio, bool) or not isinstance(retention_ratio, (int, float)) \
            or not (0.0 <= retention_ratio <= 1.0):
        raise ValueError("retention_ratio 必须落在 [0, 1]")
    if isinstance(max_skip_steps, bool) or not isinstance(max_skip_steps, int) or max_skip_steps < 1:
        raise ValueError("max_skip_steps 必须 >= 1")
    if mag_ratios is None:
        raise ValueError("mag_ratios 必须显式提供 (官方校准模式不在判题域)")
    if not isinstance(mag_ratios, torch.Tensor):
        mag_ratios = torch.tensor(mag_ratios, dtype=torch.float64)
    if mag_ratios.dim() != 1 or mag_ratios.numel() < 1:
        raise ValueError("mag_ratios 必须是非空一维张量")
    if not mag_ratios.dtype.is_floating_point:
        raise ValueError("mag_ratios 必须是浮点张量")
    if mag_ratios.numel() != num_inference_steps:
        mag_ratios = nearest_interp(mag_ratios, num_inference_steps)
    if state is None:
        state = {"previous_residual": None, "accumulated_ratio": 1.0,
                 "accumulated_steps": 0, "accumulated_err": 0.0}
    _validate_state(state)

    retention_step = int(retention_ratio * num_inference_steps + 0.5)

    acc_ratio = float(state["accumulated_ratio"])
    acc_steps = int(state["accumulated_steps"])
    acc_err = float(state["accumulated_err"])
    computed = True

    if step >= retention_step:
        current_scale = float(mag_ratios[step].item()) if step < mag_ratios.numel() else 1.0
        acc_ratio *= current_scale
        acc_steps += 1
        acc_err += abs(1.0 - acc_ratio)
        if state["previous_residual"] is not None and acc_err <= threshold \
                and acc_steps <= max_skip_steps:
            computed = False
        else:
            acc_ratio = 1.0
            acc_steps = 0
            acc_err = 0.0

    if computed:
        new_state = {
            "previous_residual": (tail_output - hidden_states).clone(),
            "accumulated_ratio": acc_ratio,
            "accumulated_steps": acc_steps,
            "accumulated_err": acc_err,
        }
        output = tail_output
    else:
        new_state = {
            "previous_residual": state["previous_residual"].clone(),
            "accumulated_ratio": acc_ratio,
            "accumulated_steps": acc_steps,
            "accumulated_err": acc_err,
        }
        output = hidden_states + state["previous_residual"]

    return {"output": output, "computed": computed, "accumulated_err": acc_err,
            "state": new_state}
