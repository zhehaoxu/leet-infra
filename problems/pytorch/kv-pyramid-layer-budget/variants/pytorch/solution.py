import torch


def solve(q_len, num_layers, compression_ratio, window_size, beta):
    """对齐 kvpress PyramidKVPress.get_layer_budget（无 module 只有 layer_idx）。"""
    assert beta >= 1
    max_cp = window_size + q_len * (1 - compression_ratio)
    min_num = (max_cp - window_size) / beta
    max_num = (max_cp - window_size) * 2 - min_num
    if max_num >= q_len - window_size:
        max_num = q_len - window_size
        min_num = (max_cp - window_size) * 2 - max_num
    if not (q_len >= max_num >= min_num >= window_size):
        v = max(1, round(q_len * (1 - compression_ratio)))
        return torch.tensor([v] * num_layers, dtype=torch.long)
    steps = (max_num - min_num) / (num_layers - 1) if num_layers > 1 else 0.0
    budgets = [round(max_num - l * steps) for l in range(num_layers)]
    return torch.tensor(budgets, dtype=torch.long)
