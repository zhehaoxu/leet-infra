import math

import torch


def _is_int_number(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float))


def _is_integer_tensor(tensor) -> bool:
    return (tensor.dtype != torch.bool and not torch.is_floating_point(tensor)
            and not torch.is_complex(tensor))


def _rank_descending(values: torch.Tensor) -> torch.Tensor:
    """排名张量：0 给最大的那一格；完全打平时下标小的排前。"""
    return values.argsort(dim=-1, descending=True).argsort(dim=-1)


def _transfer_schedule(block_length: int, num_inference_steps: int) -> list[int]:
    base, remainder = divmod(block_length, num_inference_steps)
    return [base + 1] * remainder + [base] * (num_inference_steps - remainder)


def solve(sampled_tokens, sampled_probs, sample, timestep, num_inference_steps,
          mask_token_id, threshold, editing_threshold, prompt_mask, committed, random_tokens):
    step, steps, batch, length = _validate(
        sampled_tokens, sampled_probs, sample, timestep, num_inference_steps, mask_token_id,
        threshold, editing_threshold, prompt_mask, committed, random_tokens)
    confidence = sampled_probs.to(torch.float32)
    changed = sampled_tokens != sample

    if mask_token_id is None:
        # --- 均匀破坏模式：配额是「累计应提交数」，不是每步定值 ---
        base = torch.zeros_like(sample, dtype=torch.bool) if step == 0 else committed
        target = ((step + 1) * length + steps - 1) // steps
        needed = (target - base.sum(dim=-1)).clamp(min=0)

        ranks = _rank_descending(confidence.masked_fill(base, float("-inf")))
        transfer_index = ~base & ((ranks < needed.unsqueeze(-1)) | (confidence > float(threshold)))

        editing_transfer_index = torch.zeros_like(transfer_index)
        if editing_threshold is not None:
            editing_transfer_index = base & changed & (confidence > float(editing_threshold))

        prev = torch.where(transfer_index | editing_transfer_index, sampled_tokens, sample)
        after = base | transfer_index
        prev = torch.where(after, prev, random_tokens)
        return prev.to(torch.int64), transfer_index, editing_transfer_index

    # --- 掩码填充模式 ---
    active = sample == int(mask_token_id)

    transfer_index = torch.zeros_like(sample, dtype=torch.bool)
    if active.any():
        quota = _transfer_schedule(length, steps)[min(step, steps - 1)]
        masked_conf = torch.where(active, confidence,
                                  torch.full_like(confidence, float("-inf")))
        high_conf = masked_conf > float(threshold)
        for b in range(batch):
            if int(high_conf[b].sum().item()) >= quota:
                transfer_index[b] = high_conf[b]
            else:
                k = min(quota, int(active[b].sum().item()))
                if k > 0:
                    transfer_index[b] = _rank_descending(masked_conf[b]) < k

    editing_transfer_index = torch.zeros_like(transfer_index)
    if editing_threshold is not None and float(editing_threshold) > 0.0:
        prompt = (torch.zeros(length, dtype=torch.bool)
                  if prompt_mask is None else prompt_mask.to(torch.bool))
        editable = ~active & ~prompt.unsqueeze(0)
        edit_conf = torch.where(editable, confidence,
                                torch.full_like(confidence, float("-inf")))
        editing_transfer_index = editable & changed & (edit_conf > float(editing_threshold))

    prev = torch.where(transfer_index | editing_transfer_index, sampled_tokens, sample)
    return prev.to(torch.int64), transfer_index, editing_transfer_index


def _int_scalar(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"`{name}` must be a Python int, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}, got {value}.")
    return int(value)


def _validate(sampled_tokens, sampled_probs, sample, timestep, num_inference_steps, mask_token_id,
              threshold, editing_threshold, prompt_mask, committed, random_tokens):
    step = _int_scalar("timestep", timestep, 0)
    steps = _int_scalar("num_inference_steps", num_inference_steps, 1)
    if mask_token_id is not None:
        _int_scalar("mask_token_id", mask_token_id, 0)

    for name, tensor in (("sampled_tokens", sampled_tokens), ("sampled_probs", sampled_probs),
                         ("sample", sample)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"`{name}` must be a torch.Tensor.")
        if tensor.dim() != 2:
            raise ValueError(f"`{name}` must be 2-D, got shape {tuple(tensor.shape)}.")
    batch, length = sample.shape
    for name, tensor in (("sampled_tokens", sampled_tokens), ("sampled_probs", sampled_probs)):
        if tuple(tensor.shape) != (batch, length):
            raise ValueError(f"`{name}` must have shape {(batch, length)}, got {tuple(tensor.shape)}.")
    if not _is_integer_tensor(sample):
        raise ValueError("`sample` must be an integer tensor.")
    if not _is_integer_tensor(sampled_tokens):
        raise ValueError("`sampled_tokens` must be an integer tensor.")
    if not torch.is_floating_point(sampled_probs):
        raise ValueError("`sampled_probs` must be a floating point tensor.")
    if batch < 1 or length < 1:
        raise ValueError("empty batch or empty block is not allowed.")
    if int(sampled_tokens.min().item()) < 0 or int(sample.min().item()) < 0:
        raise ValueError("token ids must be >= 0.")
    if not bool(torch.isfinite(sampled_probs).all()):
        raise ValueError("`sampled_probs` must be finite.")
    probs_min, probs_max = float(sampled_probs.min()), float(sampled_probs.max())
    if probs_min < 0.0 or probs_max > 1.0:
        raise ValueError(f"`sampled_probs` must lie in [0, 1], got [{probs_min}, {probs_max}].")

    if not _is_int_number(threshold) or not math.isfinite(float(threshold)):
        raise ValueError("`threshold` must be a finite real number.")
    if editing_threshold is not None and (not _is_int_number(editing_threshold)
                                          or not math.isfinite(float(editing_threshold))):
        raise ValueError("`editing_threshold` must be None or a finite real number.")

    if prompt_mask is not None:
        if not isinstance(prompt_mask, torch.Tensor):
            raise ValueError("`prompt_mask` must be a torch.Tensor or None.")
        if prompt_mask.dtype != torch.bool:
            raise ValueError("`prompt_mask` must be a bool tensor.")
        if tuple(prompt_mask.shape) != (length,):
            raise ValueError(f"`prompt_mask` must have shape {(length,)}, got {tuple(prompt_mask.shape)}.")
    if committed is not None:
        if not isinstance(committed, torch.Tensor):
            raise ValueError("`committed` must be a torch.Tensor or None.")
        if committed.dtype != torch.bool:
            raise ValueError("`committed` must be a bool tensor.")
        if tuple(committed.shape) != (batch, length):
            raise ValueError(f"`committed` must have shape {(batch, length)}, got {tuple(committed.shape)}.")
    if random_tokens is not None:
        if not isinstance(random_tokens, torch.Tensor):
            raise ValueError("`random_tokens` must be a torch.Tensor or None.")
        if tuple(random_tokens.shape) != (batch, length):
            raise ValueError(
                f"`random_tokens` must have shape {(batch, length)}, got {tuple(random_tokens.shape)}."
            )
        if not _is_integer_tensor(random_tokens):
            raise ValueError("`random_tokens` must be an integer tensor.")
        if int(random_tokens.min().item()) < 0:
            raise ValueError("`random_tokens` must hold token ids >= 0.")

    if mask_token_id is None:
        if committed is None:
            raise ValueError("`committed` is required in uniform corruption mode (`mask_token_id is None`).")
        if random_tokens is None:
            raise ValueError("`random_tokens` is required in uniform corruption mode.")
    elif committed is not None and not isinstance(committed, torch.Tensor):
        raise ValueError("`committed` must be a torch.Tensor or None.")

    return step, steps, batch, length
