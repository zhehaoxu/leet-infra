import torch


def solve(arrival, num_tokens, deadline, max_batch, max_wait, max_steps):
    if max_batch < 1 or max_wait < 1:
        raise ValueError("max_batch / max_wait 必须为正")
    if max_steps < 0:
        raise ValueError("max_steps 不能为负")
    for name, t in (("arrival", arrival), ("num_tokens", num_tokens), ("deadline", deadline)):
        if not isinstance(t, torch.Tensor) or t.dim() != 1:
            raise ValueError(f"{name} 必须是一维张量")
    n = int(arrival.numel())
    if int(num_tokens.numel()) != n or int(deadline.numel()) != n:
        raise ValueError("三个输入长度必须一致")
    arrives = arrival.to(torch.int64).tolist()
    remain = num_tokens.to(torch.int64).tolist()
    dls = deadline.to(torch.int64).tolist()
    if n:
        if min(arrives) < 0 or min(dls) < 0:
            raise ValueError("arrival / deadline 不能为负")
        if min(remain) < 1:
            raise ValueError("num_tokens 至少为 1")

    start = [-1] * n
    finish = [-1] * n
    wait = [0] * n
    promotions = 0

    for step in range(max_steps):
        ready = [r for r in range(n) if arrives[r] <= step and finish[r] < 0]
        if not ready:
            continue
        promoted = [r for r in ready if wait[r] >= max_wait]
        normal = [r for r in ready if wait[r] < max_wait]
        promoted.sort(key=lambda r: (dls[r], r))
        normal.sort(key=lambda r: (dls[r], r))
        order = promoted + normal
        chosen = order[:max_batch]
        promotions += min(len(chosen), len(promoted))

        picked = set(chosen)
        for r in chosen:
            if start[r] < 0:
                start[r] = step
            remain[r] -= 1
            wait[r] = 0
            if remain[r] == 0:
                finish[r] = step
        for r in ready:
            if r not in picked:
                wait[r] += 1

    missed = sum(1 for r in range(n) if finish[r] < 0 or finish[r] > dls[r])
    return (
        torch.tensor(start, dtype=torch.int64),
        torch.tensor(finish, dtype=torch.int64),
        torch.tensor(missed, dtype=torch.int64),
        torch.tensor(promotions, dtype=torch.int64),
    )
