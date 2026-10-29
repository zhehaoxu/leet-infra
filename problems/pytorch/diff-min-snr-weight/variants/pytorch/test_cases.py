import math

import torch


def _oracle(ac, ts, gamma):
    """独立 oracle：逐时间步用 math.sqrt 重算，权重写成 min(1, gamma/snr)。"""
    acs = [float(a) for a in ac.tolist()]
    snr = []
    for t in ts.tolist():
        a = math.sqrt(acs[t])
        s = math.sqrt(1.0 - acs[t])
        snr.append((a / s) ** 2)
    w = [min(1.0, gamma / v) for v in snr]
    return torch.tensor(snr, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)


def check_example(solve):
    ac = torch.tensor([0.75, 0.5, 0.25, 0.125])
    ts = torch.tensor([0, 1, 2, 3])
    snr, w = solve(ac, ts, 2.0)
    assert snr.dtype == torch.float32 and w.dtype == torch.float32
    assert snr.shape == (4,) and w.shape == (4,)
    assert [round(float(x), 6) for x in snr] == [3.0, 1.0, 0.333333, 0.142857], snr
    assert [round(float(x), 6) for x in w] == [0.666667, 1.0, 1.0, 1.0], w


check_example.example = True


def check_snr_closed_form(solve):
    ac = torch.tensor([0.999, 0.9, 0.5, 0.1, 0.01])
    ts = torch.tensor([0, 1, 2, 3, 4])
    snr, _ = solve(ac, ts, 5.0)
    want = ac.to(torch.float64) / (1.0 - ac.to(torch.float64))
    assert torch.allclose(snr.to(torch.float64), want, rtol=1e-5), snr
    assert bool((snr[1:] < snr[:-1]).all()), "越靠后的时间步 SNR 必须越低"


def check_weight_cap_and_bounds(solve):
    ac = torch.tensor([0.99, 0.9, 0.75, 0.5, 0.25, 0.1])
    ts = torch.arange(6)
    snr, w = solve(ac, ts, 3.0)
    assert bool(((w > 0) & (w <= 1.0 + 1e-6)).all()), w
    capped = snr > 3.0
    assert bool(torch.allclose(w[capped].to(torch.float64), 3.0 / snr[capped].to(torch.float64), rtol=1e-5))
    assert bool((w[~capped] == 1.0).all()), "SNR 不超过 gamma 时权重必须恰为 1"

    # gamma 越大，封顶越少、权重逐点不减
    prev = solve(ac, ts, 1.0)[1]
    for g in (2.0, 5.0, 100.0):
        cur = solve(ac, ts, g)[1]
        assert bool((cur >= prev - 1e-6).all()), f"gamma={g} 时权重应不低于更小的 gamma"
        prev = cur
    # gamma 足够大时全部不封顶
    big = solve(ac, ts, 1e9)[1]
    assert bool(torch.allclose(big, torch.ones_like(big), atol=1e-6))


def check_repeat_shape_and_empty(solve):
    ac = torch.tensor([0.8, 0.4, 0.2])
    ts = torch.tensor([2, 0, 2, 2, 0])
    snr, w = solve(ac, ts, 2.0)
    assert snr.shape == (5,) and w.shape == (5,)
    assert float(snr[0]) == float(snr[2]) == float(snr[3]), "重复时间步必须给出相同结果"
    assert float(snr[1]) == float(snr[4])

    empty = torch.tensor([], dtype=torch.int64)
    snr0, w0 = solve(ac, empty, 2.0)
    assert snr0.numel() == 0 and w0.numel() == 0


def check_bad_input(solve):
    ac = torch.tensor([0.9, 0.5])
    ts = torch.tensor([0, 1])
    bads = [
        lambda: solve([0.9, 0.5], ts, 5.0),
        lambda: solve(ac.reshape(2, 1), ts, 5.0),
        lambda: solve(torch.tensor([1, 0]), ts, 5.0),
        lambda: solve(torch.tensor([0.0, 0.5]), ts, 5.0),
        lambda: solve(torch.tensor([1.0, 0.5]), ts, 5.0),
        lambda: solve(torch.tensor([-0.1, 0.5]), ts, 5.0),
        lambda: solve(torch.tensor([float("nan"), 0.5]), ts, 5.0),
        lambda: solve(ac, [0, 1], 5.0),
        lambda: solve(ac, torch.tensor([[0, 1]]), 5.0),
        lambda: solve(ac, torch.tensor([0.0, 1.0]), 5.0),
        lambda: solve(ac, torch.tensor([0, 2]), 5.0),
        lambda: solve(ac, torch.tensor([-1, 0]), 5.0),
        lambda: solve(ac, ts, 0.0),
        lambda: solve(ac, ts, -1.0),
        lambda: solve(ac, ts, float("inf")),
        lambda: solve(ac, ts, float("nan")),
        lambda: solve(ac, ts, True),
    ]
    for fn in bads:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("非法输入应抛 ValueError")


def check_random_against_oracle(solve):
    g = torch.Generator().manual_seed(20261029)
    for _ in range(20):
        t_len = int(torch.randint(2, 12, (1,), generator=g).item())
        ac = torch.rand(t_len, generator=g) * 0.98 + 0.01
        n = int(torch.randint(1, 10, (1,), generator=g).item())
        ts = torch.randint(0, t_len, (n,), generator=g)
        gamma = float(torch.rand(1, generator=g).item()) * 12.0 + 0.05
        snr, w = solve(ac, ts, gamma)
        osnr, ow = _oracle(ac, ts, gamma)
        assert torch.allclose(snr, osnr, rtol=1e-5, atol=1e-7), (snr, osnr)
        assert torch.allclose(w, ow, rtol=1e-5, atol=1e-7), (w, ow)
