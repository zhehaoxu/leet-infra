"""参考答案：动态稀疏一步（按 |w| 剪、按 |grad| 长，互换固定数量以保持激活数不变）。"""

import torch


def solve(mask, weight, grad, sparsity, eps, u):
    n = mask.numel()
    m = mask.flatten().tolist()
    w = weight.flatten().tolist()
    g = grad.flatten().tolist()
    uu = u.flatten().tolist()
    active = [v > 0 for v in m]
    pool = [i for i in range(n) if not active[i]]
    k = n - int(round(sparsity * n))
    react, pruned = [], []

    n_active = sum(active)
    if n_active < k:
        grow = sorted(pool, key=lambda i: (-abs(g[i]), -uu[i], i))[: k - n_active]
        for i in grow:
            active[i] = True
        react += grow
        pool = [i for i in pool if not active[i]]
    elif n_active > k:
        drop = sorted(
            (i for i in range(n) if active[i]),
            key=lambda i: (abs(w[i]), i),
        )[: n_active - k]
        for i in drop:
            active[i] = False
        pruned += drop

    n_active = sum(active)
    n_inactive = n - n_active
    r = min(int(round(eps * n_inactive)), n_active, len(pool))
    if r > 0:
        drop = sorted(
            (i for i in range(n) if active[i]),
            key=lambda i: (abs(w[i]), i),
        )[:r]
        for i in drop:
            active[i] = False
        pruned += drop
        grow = sorted(pool, key=lambda i: (-abs(g[i]), -uu[i], i))[:r]
        for i in grow:
            active[i] = True
        react += grow

    new_mask = torch.tensor([1.0 if a else 0.0 for a in active], dtype=torch.float32)
    return (
        new_mask.reshape(mask.shape),
        torch.tensor(sorted(react), dtype=torch.long),
        torch.tensor(sorted(pruned), dtype=torch.long),
    )
