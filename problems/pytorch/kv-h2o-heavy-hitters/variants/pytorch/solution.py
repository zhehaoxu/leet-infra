import torch


def solve(attn, budget, n_sink):
    """H2O 累计注意力重击手池（float64 内部累加、float32 输出）。"""
    budget = int(budget)
    n_sink = int(n_sink)
    if budget < 0:
        raise ValueError("budget 必须 >= 0")
    s_steps, length = attn.shape
    if n_sink < 0 or n_sink > length:
        raise ValueError("n_sink 必须在 [0, L] 内")

    a = attn.to(torch.float64)
    acc = a.sum(dim=0)
    prompt = length - s_steps
    keep = torch.zeros(s_steps, length, dtype=torch.bool)
    run = torch.zeros(length, dtype=torch.float64)
    for s in range(s_steps):
        run = run + a[s]
        valid = prompt + s
        order = torch.argsort(-run[:valid], stable=True).tolist()  # 并列 → 小下标在前
        keep[s, :min(n_sink, valid)] = True
        picked = 0
        for idx in order:
            if picked >= budget:
                break
            if idx < n_sink:
                continue
            keep[s, idx] = True
            picked += 1
    return acc.to(torch.float32), keep
