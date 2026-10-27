import math

import torch


def _sigma(x):
    return 1.0 / (1.0 + math.exp(-x))


def _oracle(times, mu):
    """独立 oracle：在 logit 域平移 µ 后过一次 sigmoid（与 t' = s·t/(1+(s-1)t) 不同源）。"""
    out = []
    for t in times.tolist():
        if t <= 0.0:
            out.append(0.0)
        elif t >= 1.0:
            out.append(1.0)
        else:
            out.append(_sigma(math.log(t / (1.0 - t)) + mu))
    return torch.tensor(out)


def check_example(solve):
    shifted, s = solve(torch.tensor([0.25, 0.5, 0.75]), seq_len=256)
    assert abs(s - 1.6487212707) < 1e-9, f"seq_len=256 时 s 应为 exp(0.5)，got {s}"
    want = [0.354661, 0.622459, 0.831824]
    got = [round(float(e), 6) for e in shifted]
    assert got == want, f"映射结果不对：{got}"
    assert shifted.dtype == torch.float32 and shifted.shape == (3,)


check_example.example = True


def check_endpoints_and_monotone(solve):
    edge = torch.tensor([0.0, 1.0])
    out, _ = solve(edge, mu=0.7)
    assert abs(float(out[0])) < 1e-7, "t=0 必须映射到 0"
    assert abs(float(out[1]) - 1.0) < 1e-7, "t=1 必须映射到 1"

    grid = torch.linspace(0.0, 1.0, 101)
    out_up, _ = solve(grid, shift=2.0)
    assert bool((out_up >= grid - 1e-7).all()), "s>1 应把时间推向噪声侧（t' >= t）"
    assert bool((out_up[1:] > out_up[:-1]).all()), "映射必须严格单调递增"

    out_down, _ = solve(grid, shift=0.5)
    assert bool((out_down <= grid + 1e-7).all()), "s<1 应把时间推向干净侧（t' <= t）"
    assert bool((out_down[1:] > out_down[:-1]).all()), "映射必须严格单调递增"


def check_identity_and_composition(solve):
    grid = torch.linspace(0.0, 1.0, 33)
    for kwargs in ({"shift": 1.0}, {"mu": 0.0}):
        out, s = solve(grid, **kwargs)
        assert abs(s - 1.0) < 1e-12, f"s 应为 1，got {s}"
        assert torch.allclose(out, grid, atol=1e-7), "不做 shift 时必须原样返回"

    # logit 域可加：先 s1 再 s2 等价于一次 s1·s2
    for s1, s2 in ((1.5, 2.0), (0.4, 3.0), (2.5, 0.8)):
        once, s = solve(grid, shift=s1 * s2)
        twice, _ = solve(grid, shift=s1)
        twice, _ = solve(twice, shift=s2)
        assert abs(s - s1 * s2) < 1e-12
        assert torch.allclose(twice, once, atol=1e-6), f"s={s1}->{s2} 与 s={s1 * s2} 不等价"


def check_seq_len_interpolation(solve):
    t = torch.tensor([0.3, 0.6])
    _, s_base = solve(t, seq_len=256)
    _, s_max = solve(t, seq_len=4096)
    assert abs(s_base - math.exp(0.5)) < 1e-9, "base_seq_len 处 µ 应等于 base_shift"
    assert abs(s_max - math.exp(1.15)) < 1e-9, "max_seq_len 处 µ 应等于 max_shift"

    _, s_mid = solve(t, seq_len=(256 + 4096) // 2)
    assert abs(s_mid - math.exp((0.5 + 1.15) / 2)) < 1e-9, "中点应线性插值出 µ 的均值"

    prev = 0.0
    for seq_len in (0, 128, 256, 1024, 4096, 8192):
        _, s = solve(t, seq_len=seq_len)
        assert s > prev, "s 必须随序列长度单调递增"
        prev = s
    # 短序列（低于 base_seq_len）外推为更小的 s，而不是截断到 base_shift
    _, s_short = solve(t, seq_len=64)
    assert s_short < s_base, "低于 base_seq_len 时应外推出更小的 s"

    # 逆映射回原位
    for s in (0.3, 1.0, 1.6487212707, 6.0):
        out, got_s = solve(t, shift=s)
        back = out.to(torch.float64) / (got_s - (got_s - 1.0) * out.to(torch.float64))
        assert torch.allclose(back, t.to(torch.float64), atol=1e-5), f"s={s} 时逆映射未还原"


def check_oracle_and_invalid(solve):
    grid = torch.tensor([0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0])
    for mu in (-1.2, 0.0, 0.5, 1.15):
        got, s = solve(grid, mu=mu)
        assert abs(s - math.exp(mu)) < 1e-9, f"µ={mu} 时 s 应为 exp(µ)"
        assert torch.allclose(got, _oracle(grid, mu).to(torch.float32), atol=1e-6), f"µ={mu} 与 oracle 不一致"

    good = torch.tensor([0.0, 0.5])
    for bad in (
        lambda: solve([0.0, 0.5], shift=2.0),
        lambda: solve(torch.zeros(2, 2), shift=2.0),
        lambda: solve(torch.tensor([-0.1, 0.5]), shift=2.0),
        lambda: solve(torch.tensor([0.5, 1.1]), shift=2.0),
        lambda: solve(torch.tensor([float("nan"), 0.5]), shift=2.0),
        lambda: solve(good),
        lambda: solve(good, shift=2.0, mu=0.5),
        lambda: solve(good, shift=2.0, seq_len=1024),
        lambda: solve(good, shift=0.0),
        lambda: solve(good, shift=-1.0),
        lambda: solve(good, mu=float("inf")),
        lambda: solve(good, seq_len=-1),
        lambda: solve(good, seq_len=1024.0),
        lambda: solve(good, seq_len=1024, base_seq_len=2048, max_seq_len=1024),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
