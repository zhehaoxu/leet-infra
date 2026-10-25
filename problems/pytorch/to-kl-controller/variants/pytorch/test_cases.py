import torch

BETA, CUR, TGT, HORIZON, N = 0.1, 0.3, 0.2, 1000.0, 1


def _oracle(beta, cur, tgt, horizon, n_steps):
    """独立重算：纯 Python max/min 手算截断（结构异于参考解的 torch.clamp）。"""
    pe = max(-0.2, min(0.2, cur / tgt - 1.0))
    return beta * (1.0 + pe * n_steps / horizon)


def check_example(solve):
    y = solve(BETA, CUR, TGT, HORIZON, N)
    assert abs(y.item() - 0.10002000629901886) < 1e-9, f"got {y.item()}"


check_example.example = True


def check_matches_oracle(solve):
    for beta, cur, tgt, horizon, n in [
        (0.1, 0.3, 0.2, 1000.0, 1),
        (0.2, 0.05, 0.1, 500.0, 4),
        (0.05, 0.9, 0.3, 1000.0, 100),
        (0.3, 0.31, 0.3, 2000.0, 20),
        (0.1, 0.0, 0.2, 1000.0, 1),
    ]:
        assert abs(solve(beta, cur, tgt, horizon, n).item() - _oracle(beta, cur, tgt, horizon, n)) < 1e-5


def check_direction(solve):
    # 等于目标 → 不变；高于目标 → 抬高；低于目标 → 压低
    same = solve(0.1, 0.2, 0.2, 1000.0, 1).item()
    up = solve(0.1, 0.3, 0.2, 1000.0, 1).item()
    down = solve(0.1, 0.1, 0.2, 1000.0, 1).item()
    assert abs(same - 0.1) < 1e-6, f"KL 命中目标应几乎不变，got {same}"
    assert up > 0.1 > down, (up, down)


def check_clip_saturation(solve):
    # 误差被截断到 ±0.2：0.4/0.2-1=1.0 与 100/0.2-1=499 都饱和到同一结果
    a = solve(0.1, 0.4, 0.2, 1000.0, 1).item()
    b = solve(0.1, 100.0, 0.2, 1000.0, 1).item()
    assert abs(a - b) < 1e-9, (a, b)
    # 未饱和的中等偏差应严格小于饱和值
    mid = solve(0.1, 0.22, 0.2, 1000.0, 1).item()
    assert mid < a, (mid, a)


def check_returns_scalar(solve):
    y = solve(BETA, CUR, TGT, HORIZON, N)
    assert y.dim() == 0 and y.dtype == torch.float32, (y.dim(), y.dtype)
