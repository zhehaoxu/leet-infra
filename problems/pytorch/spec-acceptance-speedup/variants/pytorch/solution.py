import torch


def solve(alphas, cost_ratio):
    """按逐位接受率算期望吐词与加速比，并取最优 γ（内部 float64）。"""
    if alphas.ndim != 1:
        raise ValueError("alphas 必须是一维")
    g_max = int(alphas.numel())
    if g_max < 1:
        raise ValueError("至少需要一个逐位接受率")
    a = alphas.to(torch.float64)
    if bool((a < 0).any()) or bool((a > 1).any()):
        raise ValueError("接受率必须落在 [0, 1]")
    ratio = float(cost_ratio)
    if ratio < 0:
        raise ValueError("cost_ratio 必须 >= 0")

    expected = [1.0]                                  # γ = 0 时只吐 1 个 bonus token
    prod = 1.0
    for i in range(g_max):
        prod *= float(a[i])                           # 前 i+1 位全中的概率
        expected.append(expected[-1] + prod)

    exp = torch.tensor(expected, dtype=torch.float64)
    gammas = torch.arange(g_max + 1, dtype=torch.float64)
    speed = exp / (1.0 + gammas * ratio)
    best = int(torch.argmax(speed))                   # 并列时 argmax 取最小下标
    return (exp.to(torch.float32), speed.to(torch.float32),
            torch.tensor(best, dtype=torch.int64), speed[best].to(torch.float32).reshape(()))
