"""参考答案：从 logits 计算语言模型困惑度（右移对齐 + ignore_index 掩码）。"""

import torch


def solve(logits, labels, ignore_index=-100):
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    # 数值稳定的 log_softmax，再按真实标签 gather 出 log p
    logp = torch.log_softmax(shift_logits, dim=-1)
    safe_ids = shift_labels.clamp(min=0).unsqueeze(-1)
    nll = -logp.gather(-1, safe_ids).squeeze(-1)          # (B, L-1)
    valid = shift_labels != ignore_index
    if int(valid.sum().item()) == 0:
        return torch.tensor(-1.0)                          # 空评测集哨兵
    return torch.exp(nll[valid].mean()).float()
