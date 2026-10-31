import math

import torch


def _int(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 Python 整数")
    return value


def _real(name, value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    if low is not None and value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} 不能大于 {high}")
    return value


def _token_tensor(name, value, shape, vocab):
    if not torch.is_tensor(value):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} 必须是 int32 / int64")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} 的形状必须是 {tuple(shape)}")
    if int(value.min()) < 0 or int(value.max()) >= vocab:
        raise ValueError(f"{name} 的取值必须落在 [0, vocab_size) 内")
    return value


def _temperature(name, value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} 为元组/列表时必须恰好含两个元素")
        pair = [_real(f"{name}[{i}]", v, low=-8.0, high=8.0) for i, v in enumerate(value)]
        return pair[0], pair[1]
    single = _real(name, value, low=-8.0, high=8.0)
    return single, 0.01


def solve(
    logits,
    sample,
    drawn_tokens,
    gumbel_uniforms,
    timestep,
    num_inference_steps,
    mask_token_id,
    masking_schedule="cosine",
    temperature=(2.0, 0.0),
    starting_mask_ratio=1.0,
):
    """AMUS（masked token generation）单步：按「加噪置信度」重掩码 top-k 个位置。

    逐步复刻 diffusers `schedulers/scheduling_amused.py` 的 `AmusedScheduler.step`。
    """
    if not torch.is_tensor(logits):
        raise ValueError("logits 必须是 torch.Tensor")
    if logits.dim() != 3:
        raise ValueError("logits 必须是三维张量 (batch, seq_len, vocab_size)")
    if logits.dtype not in (torch.float32, torch.float64):
        raise ValueError("logits 必须是 float32 / float64")
    batch, seq_len, vocab = int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2])
    if batch < 1 or seq_len < 2 or vocab < 1:
        raise ValueError("batch / vocab_size 必须 >= 1，seq_len 必须 >= 2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits 不能含 nan / inf")

    shape = (batch, seq_len)
    sample_ids = _token_tensor("sample", sample, shape, vocab)
    drawn = _token_tensor("drawn_tokens", drawn_tokens, shape, vocab)

    if not torch.is_tensor(gumbel_uniforms):
        raise ValueError("gumbel_uniforms 必须是 torch.Tensor")
    if gumbel_uniforms.dtype not in (torch.float32, torch.float64):
        raise ValueError("gumbel_uniforms 必须是 float32 / float64")
    if tuple(gumbel_uniforms.shape) != shape:
        raise ValueError(f"gumbel_uniforms 的形状必须是 {shape}")
    if not bool(torch.isfinite(gumbel_uniforms).all()):
        raise ValueError("gumbel_uniforms 不能含 nan / inf")
    if bool((gumbel_uniforms < 0.0).any()) or bool((gumbel_uniforms > 1.0).any()):
        raise ValueError("gumbel_uniforms 必须落在 [0, 1] 内")

    step = _int("timestep", timestep)
    steps = _int("num_inference_steps", num_inference_steps)
    if steps < 1:
        raise ValueError("num_inference_steps 必须是正整数")
    if not 0 <= step < steps:
        raise ValueError("timestep 必须落在 [0, num_inference_steps) 内")

    token_id = _int("mask_token_id", mask_token_id)
    if not 0 <= token_id < vocab:
        raise ValueError("mask_token_id 必须落在 [0, vocab_size) 内")

    if not isinstance(masking_schedule, str):
        raise ValueError("masking_schedule 必须是字符串")
    if masking_schedule not in ("cosine", "linear"):
        raise ValueError("masking_schedule 只能是 'cosine' 或 'linear'")

    temp_lo, temp_hi = _temperature("temperature", temperature)
    ratio_cap = _real("starting_mask_ratio", starting_mask_ratio, low=0.0, high=1.0)

    logits64 = logits.to(torch.float64)
    probs = torch.softmax(logits64, dim=-1)
    unknown = sample_ids.to(torch.int64).eq(token_id)
    committed = torch.where(unknown, drawn.to(torch.int64), sample_ids.to(torch.int64))

    if step == 0:
        masking = torch.zeros(shape, dtype=torch.bool)
        mask_len = torch.zeros((batch, 1), dtype=torch.int64)
        return committed, committed, masking, mask_len

    step_idx = steps - 1 - step
    ratio = torch.tensor([(step_idx + 1) / steps], dtype=torch.float64)
    if masking_schedule == "cosine":
        mask_ratio = torch.cos(ratio * (math.pi / 2.0))
    else:
        mask_ratio = 1.0 - ratio
    mask_ratio = mask_ratio * ratio_cap

    mask_len = (seq_len * mask_ratio).floor()
    mask_len = torch.min(unknown.to(torch.int64).sum(dim=-1, keepdim=True) - 1, mask_len)
    mask_len = torch.max(torch.tensor([1], dtype=torch.float64), mask_len).to(torch.int64)

    selected_probs = torch.gather(probs, -1, committed.unsqueeze(-1)).squeeze(-1)
    sentinel = torch.full_like(selected_probs, torch.finfo(torch.float64).max)
    selected_probs = torch.where(unknown, selected_probs, sentinel)

    span = torch.linspace(temp_lo, temp_hi, steps, dtype=torch.float64)
    current_temp = span[step_idx]

    uniforms = gumbel_uniforms.to(torch.float64).clamp(1e-20)
    gumbel = -torch.log((-torch.log(uniforms)).clamp(1e-20))
    confidence = torch.log(selected_probs.clamp(1e-20)) + current_temp * gumbel

    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len)
    masking = confidence < cut_off

    prev_sample = torch.where(masking, torch.tensor(token_id, dtype=torch.int64), committed)
    return prev_sample, committed, masking, mask_len
