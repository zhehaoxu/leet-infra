import math

import torch
import torch.nn.functional as F

_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    return float(value)


def _weighting(sigmas, scheme):
    if not torch.is_tensor(sigmas):
        raise ValueError("sigmas 必须是 torch 张量")
    if not sigmas.is_floating_point():
        raise ValueError("sigmas 必须是浮点张量")
    if sigmas.numel() == 0:
        raise ValueError("sigmas 不能为空")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("sigmas 必须全部是有限值")
    if bool((sigmas <= 0).any()):
        raise ValueError("sigmas 必须全部为正")

    if scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        return 2 / (math.pi * bot)
    return torch.ones_like(sigmas)


def _confidence_loss(logits, labels, lambda_conf, temperature, per_token_weights, ignore_index):
    if not torch.is_tensor(logits):
        raise ValueError("logits 必须是 torch 张量")
    if logits.dim() < 2:
        raise ValueError("logits 至少要有 2 维")
    if not logits.is_floating_point():
        raise ValueError("logits 必须是浮点张量")
    if logits.shape[-1] < 1:
        raise ValueError("logits 的最后一维（词表）不能为空")
    if not torch.is_tensor(labels):
        raise ValueError("labels 必须是 torch 张量")
    if labels.dtype not in _INT_DTYPES:
        raise ValueError("labels 必须是整数张量")
    if tuple(labels.shape) != tuple(logits.shape[:-1]):
        raise ValueError("labels 形状必须等于 logits.shape[:-1]")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise ValueError("ignore_index 必须是整数")
    temperature = _number("temperature", temperature)
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    lambda_conf = _number("lambda_conf", lambda_conf)
    if not math.isfinite(lambda_conf):
        raise ValueError("lambda_conf 必须是有限值")

    labels = labels.to(torch.int64)
    valid = labels.ne(ignore_index)
    if valid.any():
        kept = labels[valid]
        if bool((kept < 0).any()) or bool((kept >= logits.shape[-1]).any()):
            raise ValueError("有效标签必须落在 [0, V)")

    if per_token_weights is None:
        weights = torch.ones_like(labels, dtype=logits.dtype)
    else:
        if not torch.is_tensor(per_token_weights) or not per_token_weights.is_floating_point():
            raise ValueError("per_token_weights 必须是浮点张量")
        if tuple(per_token_weights.shape) != tuple(labels.shape):
            raise ValueError("per_token_weights 形状必须等于 labels 形状")
        weights = per_token_weights.to(dtype=logits.dtype)

    valid_f = valid.to(dtype=weights.dtype)
    per_token_nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape_as(labels)

    denom_sft = (weights * valid_f).sum().clamp_min(1)
    loss_sft = (per_token_nll * weights * valid_f).sum() / denom_sft

    if lambda_conf == 0.0:
        zero = torch.zeros((), device=logits.device, dtype=loss_sft.dtype)
        return loss_sft, loss_sft, zero

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        correct = valid & pred.eq(labels)

    scaled_logits = logits.float()
    if temperature != 1.0:
        scaled_logits = scaled_logits / float(temperature)
    probs = torch.softmax(scaled_logits, dim=-1)
    eps = torch.finfo(probs.dtype).tiny
    log_probs = torch.log(probs.clamp_min(eps))
    entropy = -(probs * log_probs).sum(dim=-1).to(dtype=logits.dtype)

    correct_f = correct.to(dtype=weights.dtype)
    denom_conf = (weights * correct_f).sum().clamp_min(1)
    loss_conf = (entropy * weights * correct_f).sum() / denom_conf
    loss = loss_sft + lambda_conf * loss_conf
    return loss, loss_sft, loss_conf


def solve(
    sigmas,
    weighting_scheme,
    logits,
    labels,
    lambda_conf=0.0,
    temperature=1.0,
    per_token_weights=None,
    ignore_index=-100,
):
    """训练一步里的两件损失工具：σ 损失加权名册 + 置信度感知的 token 交叉熵。

    第一部分（σ 加权，对应 diffusers 的 compute_loss_weighting_for_sd3）：
      sigmas 是浮点张量（元素为有限正数），weighting_scheme 取字符串：
        "sigma_sqrt" -> (sigmas ** -2.0).float()，结果一律是 float32（上游 .float() 的位置）；
        "cosmap"     -> 2 / (pi * (1 - 2*sigmas + 2*sigmas**2))，保持 sigmas 的 dtype；
        其它任意字符串（含 "none"、拼错的名字）-> torch.ones_like(sigmas)，保持 dtype 且不报错。

    第二部分（置信度感知损失，对应 compute_confidence_aware_loss）：
      logits 形如 (..., V)，labels 形如 (...) 的整数张量；labels == ignore_index 的位置不参与任何损失。
      weights  = per_token_weights（形状同 labels、浮点）或全 1，一律转成 logits.dtype；
      valid    = (labels != ignore_index)
      nll      = cross_entropy(logits.view(-1, V), labels.view(-1), reduction="none").view_as(labels)
      loss_sft = sum(nll * weights * valid) / clamp_min(sum(weights * valid), 1)
      lambda_conf == 0.0 时直接返回 (loss_sft, loss_sft, 0)，不算熵；
      否则：
        pred      = argmax(logits, -1)（no_grad）
        correct   = valid & (pred == labels)
        p         = softmax(logits.float() / temperature, -1)   # 温度只缩放熵，不动 CE
        H         = -(p * log(clamp_min(p, finfo(p.dtype).tiny))).sum(-1)
        loss_conf = sum(H * weights * correct) / clamp_min(sum(weights * correct), 1)
        loss      = loss_sft + lambda_conf * loss_conf
      两个分母都过 clamp_min(1)：权重整体偏小时结果不是加权平均，而是被 1 截断后的求和。

    返回 (weighting, loss, loss_sft, loss_conf)：weighting 形状同 sigmas；
    后三个是 0 维张量，dtype 与 logits 相同。

    非法输入抛 ValueError：sigmas 非张量/非浮点/为空/含非正或非有限值；logits 非张量、
    维度 < 2、非浮点或最后一维为空；labels 非整数张量、形状不符或有效标签越界；
    temperature <= 0；lambda_conf 非有限；per_token_weights 非浮点张量或形状不符。
    """
    weighting = _weighting(sigmas, weighting_scheme)
    loss, loss_sft, loss_conf = _confidence_loss(
        logits, labels, lambda_conf, temperature, per_token_weights, ignore_index
    )
    return weighting, loss, loss_sft, loss_conf
