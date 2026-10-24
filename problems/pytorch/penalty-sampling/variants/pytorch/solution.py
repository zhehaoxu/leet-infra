"""参考解：重复惩罚（repeat + freq + present，对齐 llama.cpp `penalties`）。

对滑动窗口（history 末尾 penalty_last_n 个 token，-1 表示整段）中出现过的每个 token：
1. 乘法项 penalty_repeat：logit <= 0 时乘以 repeat，否则除以 repeat（两种都让 logit 变小）。
2. 加法项：再减去 count*penalty_freq + (count>0)*penalty_present。

顺序敏感：必须先乘/除、后减。只返回惩罚后的 logits（softmax 上游，不做采样）。
"""

import torch


def solve(logits, history, penalty_repeat, penalty_freq, penalty_present, penalty_last_n):
    out = logits.clone()
    win = list(history) if penalty_last_n < 0 else list(history)[-int(penalty_last_n):]
    counts = {}
    for t in win:
        counts[int(t)] = counts.get(int(t), 0) + 1
    for tid, c in counts.items():
        if c <= 0:
            continue
        l = out[tid]
        out[tid] = l * penalty_repeat if float(l) <= 0 else l / penalty_repeat
        out[tid] = out[tid] - c * penalty_freq - 1.0 * penalty_present
    return out
