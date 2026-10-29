import math

import torch


def _int(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 Python 整数")
    return value


def _real(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是实数")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限值")
    return value


def solve(
    logits,
    sampled_tokens,
    random_tokens,
    timestep,
    num_inference_steps,
    entropy_bound=0.1,
    t_max=0.8,
    t_min=0.4,
):
    if not torch.is_tensor(logits) or logits.dim() != 3:
        raise ValueError("logits 必须是三维张量 (batch, block_length, vocab_size)")
    if logits.dtype not in (torch.float32, torch.float64):
        raise ValueError("logits 必须是 float32 / float64")
    batch, block, vocab = logits.shape
    if batch < 1 or block < 1 or vocab < 1:
        raise ValueError("batch / block_length / vocab_size 都必须 >= 1")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits 不能含 nan / inf")

    for name, value in (("sampled_tokens", sampled_tokens), ("random_tokens", random_tokens)):
        if not torch.is_tensor(value) or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} 必须是 int32 / int64 张量")
        if tuple(value.shape) != (batch, block):
            raise ValueError(f"{name} 的形状必须是 (batch, block_length)")
        if value.numel() and (int(value.min()) < 0 or int(value.max()) >= vocab):
            raise ValueError(f"{name} 的取值必须落在 [0, vocab_size) 内")

    num_inference_steps = _int("num_inference_steps", num_inference_steps)
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps 必须 >= 1")
    timestep = _int("timestep", timestep)
    if timestep < 0 or timestep >= num_inference_steps:
        raise ValueError("timestep 必须落在 [0, num_inference_steps) 内")

    entropy_bound = _real("entropy_bound", entropy_bound)
    if entropy_bound < 0.0:
        raise ValueError("entropy_bound 必须 >= 0")
    t_max = _real("t_max", t_max)
    t_min = _real("t_min", t_min)
    if t_max <= 0.0 or t_min <= 0.0:
        raise ValueError("t_max 与 t_min 必须为正")

    fraction = (num_inference_steps - timestep) / num_inference_steps
    temperature = t_min + (t_max - t_min) * fraction
    scaled = logits.to(torch.float64) / temperature

    log_norm = torch.logsumexp(scaled, dim=-1, keepdim=True)
    log_probs = scaled - log_norm
    token_entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)

    sorted_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, stable=True)
    cumulative = torch.cumsum(sorted_entropy, dim=-1)
    sorted_accepted = (cumulative - sorted_entropy) <= entropy_bound
    accepted = torch.zeros_like(sorted_accepted)
    accepted.scatter_(-1, sorted_indices, sorted_accepted)

    prev_sample = torch.where(accepted, sampled_tokens.to(torch.int64), random_tokens.to(torch.int64))
    return prev_sample, accepted, token_entropy.to(torch.float32)
