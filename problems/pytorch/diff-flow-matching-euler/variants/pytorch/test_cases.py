import torch


def _oracle(x_init, velocity_fn, n_steps, t_start, t_end):
    """独立 oracle：先把时间栅格整段算出来，再逐段用 python float 加乘（结构不同源）。"""
    dt = (t_end - t_start) / n_steps
    grid = [t_start + i * dt for i in range(n_steps + 1)]
    out = [x_init.to(torch.float32).clone()]
    for left, right in zip(grid, grid[1:]):
        cur = out[-1]
        v = velocity_fn(cur, left)
        if not torch.is_tensor(v):
            v = torch.tensor(float(v))
        nxt = [a + (right - left) * b for a, b in zip(cur.flatten().tolist(), v.flatten().tolist())]
        out.append(torch.tensor(nxt).reshape(cur.shape).to(torch.float32))
    return out


def check_example(solve):
    def v(x, t):
        return -x

    trajectory, x_final = solve(torch.tensor(1.0), v, 4)
    want = [1.0, 1.25, 1.5625, 1.953125, 2.44140625]
    got = [float(e) for e in trajectory]
    assert len(got) == 5, f"应是 5 个状态，got {len(got)}"
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, want)), got
    assert abs(float(x_final) - 2.44140625) < 1e-6, float(x_final)


check_example.example = True


def check_grid_and_lengths(solve):
    seen = []

    def v(x, t):
        seen.append(t)
        return torch.zeros_like(x)

    x_init = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    trajectory, x_final = solve(x_init, v, 4, 0.8, 0.2)

    assert len(trajectory) == 5, "轨迹长度必须是 n_steps + 1"
    assert len(seen) == 4, "速度场应被调用 n_steps 次"
    assert all(torch.is_tensor(e) and e.dtype == torch.float32 for e in trajectory)
    assert all(e.shape == (2, 3) for e in trajectory), "形状必须保持不变"
    assert torch.allclose(trajectory[0], x_init.to(torch.float32)), "首元素应是起点"
    assert torch.equal(trajectory[-1], x_final), "返回值应与轨迹末元素一致"

    # 速度恒为零：每一步只走栅格，轨迹保持在起点（也验证没有被原地改写）
    assert all(torch.allclose(e, x_init.to(torch.float32)) for e in trajectory), "零速度不应移动"

    want = [0.8 - k * 0.15 for k in range(4)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(seen, want)), f"时间栅格不对：{seen}"


def check_constant_velocity_exact(solve):
    torch.manual_seed(0)
    c = torch.randn(3, 2, dtype=torch.float64)

    for n in (1, 3, 10):
        _, got = solve(torch.zeros(3, 2), lambda x, t: c, n)
        want = c * (0.0 - 1.0)  # Δt = t_end - t_start = -1
        assert torch.allclose(got, want.to(torch.float32), atol=1e-6), f"n={n}: 常速度积分应精确"

    # 非默认区间 + python float 标量速度（应广播）
    _, got = solve(torch.ones(4), lambda x, t: 2.5, 2, 0.9, -0.1)
    assert torch.allclose(got, torch.full((4,), 1.0 + 2.5 * (-1.0)), atol=1e-6), got.tolist()


def check_rectified_flow_straight(solve):
    """RF 直线路径：x_t = (1-t)·x0 + t·ε，速度恒为 ε-x0，任意步数都应精确落回 x0。"""
    torch.manual_seed(1)
    x0 = torch.randn(2, 3, dtype=torch.float64)
    eps = torch.randn(2, 3, dtype=torch.float64)

    def v(x, t):
        return eps - x0

    for n in (1, 2, 7):
        trajectory, x_final = solve(eps, v, n)
        assert torch.allclose(x_final, x0.to(torch.float32), atol=1e-5), f"n={n}: 单步/多步都应还原 x0"
        for k, state in enumerate(trajectory):
            t_k = 1.0 - k / n
            want = (1.0 - t_k) * x0 + t_k * eps
            assert torch.allclose(state, want.to(torch.float32), atol=1e-5), f"n={n}, k={k}: 偏离直线路径"


def check_oracle_and_invalid(solve):
    torch.manual_seed(2)

    def v(x, t):
        return 0.5 * torch.cos(x) + t * t - 0.3

    x_init = torch.randn(2, 2, 2, dtype=torch.float64)
    trajectory, x_final = solve(x_init, v, 6, 0.75, -0.25)
    want = _oracle(x_init, v, 6, 0.75, -0.25)
    assert len(trajectory) == len(want), "轨迹长度与 oracle 不一致"
    for k, (got_k, want_k) in enumerate(zip(trajectory, want)):
        assert torch.allclose(got_k, want_k, atol=1e-5), f"第 {k} 步与 oracle 不一致"
    assert torch.equal(trajectory[-1], x_final)

    good = torch.zeros(2)
    for bad in (
        lambda: solve([0.0, 1.0], v, 4),
        lambda: solve(good, "velocity", 4),
        lambda: solve(good, v, 0),
        lambda: solve(good, v, 2.0),
        lambda: solve(good, v, True),
        lambda: solve(good, v, 4, 0.0, 0.0),
        lambda: solve(good, v, 4, 0.0, 1.0),
        lambda: solve(good, lambda x, t: torch.zeros(3), 4),
        lambda: solve(good, lambda x, t: None, 4),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
