import torch


def solve(grads: torch.Tensor, loss_scale: float, growth_interval: int,
          consec_ok: int, backoff_factor: float = 2.0, growth_factor: float = 2.0) -> dict:
    scaled_fp16 = (grads * loss_scale).to(torch.float16)
    overflow = bool(torch.isinf(scaled_fp16).any() or torch.isnan(scaled_fp16).any())

    if overflow:
        return {
            "overflow": True,
            "skip_update": True,
            "new_loss_scale": float(loss_scale / backoff_factor),
            "new_consec_ok": 0,
            "unscaled_grads": grads.clone(),
        }

    c = consec_ok + 1
    if c >= growth_interval:
        new_scale = loss_scale * growth_factor
        new_consec = 0
    else:
        new_scale = loss_scale
        new_consec = c

    unscaled = (scaled_fp16.float() / loss_scale).to(grads.dtype)
    return {
        "overflow": False,
        "skip_update": False,
        "new_loss_scale": float(new_scale),
        "new_consec_ok": int(new_consec),
        "unscaled_grads": unscaled,
    }
