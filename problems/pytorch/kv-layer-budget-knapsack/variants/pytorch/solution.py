import torch


def solve(gain, budget):
    """层间名额背包 DP：按**恰好花费** b 建状态，避免把「等值但更省」的前缀丢掉。"""
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget 必须 >= 0")
    layers, max_keep = gain.shape
    if layers < 1 or max_keep < 1:
        raise ValueError("gain 不能为空")
    if bool((gain < 0).any()):
        raise ValueError("gain 必须非负")

    g = gain.to(torch.float64).tolist()
    cap = min(budget, layers * (max_keep - 1))
    neg = float("-inf")
    dp = [(0.0, ())] + [(neg, None)] * cap
    for l in range(layers):
        nxt = [(neg, None)] * (cap + 1)
        for b in range(cap + 1):
            base_val, base_alloc = dp[b]
            if base_alloc is None:
                continue
            for x in range(min(max_keep - 1, cap - b) + 1):
                nb = b + x
                val = base_val + g[l][x]
                alloc = base_alloc + (x,)
                cur = nxt[nb]
                if cur[1] is None or val > cur[0] + 1e-12 or (
                        abs(val - cur[0]) <= 1e-12 and alloc < cur[1]):
                    nxt[nb] = (val, alloc)
        dp = nxt

    best = None
    for b in range(cap + 1):
        val, alloc = dp[b]
        if alloc is None:
            continue
        if best is None or val > best[0] + 1e-12 or (
                abs(val - best[0]) <= 1e-12 and alloc < best[1]):
            best = (val, alloc)
    return torch.tensor(best[1], dtype=torch.long), torch.tensor(best[0], dtype=torch.float32)
