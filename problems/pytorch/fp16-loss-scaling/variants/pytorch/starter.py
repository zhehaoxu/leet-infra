import torch


def solve(grads: torch.Tensor, loss_scale: float, growth_interval: int,
          consec_ok: int, backoff_factor: float = 2.0, growth_factor: float = 2.0) -> dict:
    """一个优化步的动态 loss scaling 判定。

    scaled=(grads*loss_scale).to(float16)；overflow=是否 inf/NaN。
    溢出：skip_update=True, new_scale=loss_scale/backoff, new_consec_ok=0, unscaled=grads 原样。
    否则：c=consec_ok+1；c>=growth_interval 则 new_scale=loss_scale*growth,new_consec_ok=0 否则不变+1；
         unscaled=(scaled.float()/loss_scale) 转回 grads.dtype。
    返回 {"overflow","skip_update","new_loss_scale","new_consec_ok","unscaled_grads"}。
    """
    # TODO: 实现动态 loss scaling 单步
    raise NotImplementedError
