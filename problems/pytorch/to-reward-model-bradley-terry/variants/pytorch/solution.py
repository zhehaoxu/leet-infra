import torch
import torch.nn.functional as F


def _pool_last_valid(hidden, mask):
    # 每行最后有效 token 下标 = (mask * [0..L-1]) 在该行的最大值
    L = int(hidden.shape[1])
    pos = torch.arange(L, dtype=hidden.dtype)
    idx = (mask.to(hidden.dtype) * pos).max(dim=1).values.long()  # (N,)
    gather = idx.view(-1, 1, 1).expand(-1, 1, hidden.shape[2])
    return hidden.gather(1, gather).squeeze(1)  # (N, D)


def solve(chosen_hidden, rejected_hidden, chosen_mask, rejected_mask, weight, bias):
    """Bradley-Terry 奖励模型损失。"""
    hc = _pool_last_valid(chosen_hidden, chosen_mask)
    hr = _pool_last_valid(rejected_hidden, rejected_mask)
    r_chosen = hc @ weight + bias
    r_rejected = hr @ weight + bias
    return -F.logsigmoid(r_chosen - r_rejected).mean()
