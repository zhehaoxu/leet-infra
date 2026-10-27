import torch


def solve(alphas, budget):
    if not isinstance(alphas, torch.Tensor) or alphas.dim() != 1:
        raise ValueError("alphas 必须是一维张量")
    n = int(alphas.numel())
    if n == 0:
        raise ValueError("alphas 不能为空")
    a = alphas.to(torch.float64)
    if bool((a < 0).any()) or bool((a > 1).any()):
        raise ValueError("alphas 必须落在 [0, 1]")
    if budget < n:
        raise ValueError("名额不足以让每个请求至少分到 1 个")

    # 每个请求先拿基础 1 个名额（γ_i = 0），剩下的名额按「下一位草稿的边际收益」贪心加
    gammas = [0] * n
    remain = budget - n
    for _ in range(remain):
        best, best_gain = 0, -1.0
        for r in range(n):
            # 从 γ 加到 γ+1 的边际收益是 α^(γ+1)
            gain = float(a[r]) ** (gammas[r] + 1)
            if gain > best_gain:
                best, best_gain = r, gain
        gammas[best] += 1

    expected = [0.0] * n
    for r in range(n):
        total, power = 0.0, 1.0
        for _ in range(gammas[r] + 1):
            total += power
            power *= float(a[r])
        expected[r] = total

    return (
        torch.tensor(gammas, dtype=torch.int64),
        torch.tensor(expected, dtype=torch.float32),
        torch.tensor(sum(expected), dtype=torch.float32),
    )
