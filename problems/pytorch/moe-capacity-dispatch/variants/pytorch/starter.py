import torch


def solve(logits, k, capacity_factor):
    """位置式容量分派：topk → renorm → 每专家保留前 C 个 (C=ceil(T*k/E*cf))。
    返回 (final_map bool (T,E), final_probs float (T,E), dropped int)。
    """
    # TODO
    raise NotImplementedError
