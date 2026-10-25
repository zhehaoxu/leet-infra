import torch


def solve(token_ids, role_ids, assistant_id, eos_id, supervise_eos=True):
    """返回 (B,L) float32 掩码：role==assistant_id 的连续段置 1；supervise_eos=False
    时仅丢弃该段最后一位（且该位 token==eos_id）。其余为 0。"""
    # TODO
    raise NotImplementedError
