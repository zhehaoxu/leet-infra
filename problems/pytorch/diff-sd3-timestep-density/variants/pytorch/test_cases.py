import math

import torch


def _oracle(scheme, samples, logit_mean=None, logit_std=None, mode_scale=None):
    """独立 oracle：逐元素用 math 重写，并用 exp 手写 sigmoid。"""
    out = []
    for v in samples.tolist():
        if scheme == "logit_normal":
            x = logit_mean + logit_std * v
            out.append(1.0 / (1.0 + math.exp(-x)))
        elif scheme == "mode":
            out.append(1.0 - v - mode_scale * (math.cos(math.pi * v / 2) ** 2 - 1.0 + v))
        else:
            out.append(v)
    return torch.tensor(out, dtype=torch.float32)


def check_example(solve):
    z = torch.tensor([-1.0, 0.0, 1.0])
    got = solve("logit_normal", z, logit_mean=0.0, logit_std=1.0)
    assert got.dtype == torch.float32 and got.shape == (3,)
    assert [round(float(e), 6) for e in got] == [0.268941, 0.5, 0.731059], got

    u = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    got = solve("mode", u, mode_scale=1.29)
    assert got.dtype == torch.float32 and got.shape == (5,)
    assert [round(float(e), 6) for e in got] == [1.0, 0.616416, 0.5, 0.383584, 0.0], got


check_example.example = True


def check_uniform_passthrough(solve):
    u = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    got = solve("uniform", u)
    assert torch.equal(got, u), got
    assert got.dtype == torch.float32
    assert solve("uniform", torch.tensor([0.25])).shape == (1,)


def check_logit_normal_is_sigmoid(solve):
    z = torch.linspace(-3.0, 3.0, 25)
    for mean, std in ((0.0, 1.0), (0.5, 1.5), (-0.8, 0.6)):
        got = solve("logit_normal", z, logit_mean=mean, logit_std=std)
        want = torch.sigmoid(mean + std * z)
        assert torch.allclose(got, want, atol=1e-6), (mean, std)
        assert bool((got[1:] > got[:-1]).all()), "logit-normal 映射必须随 z 严格递增"

    # 零均值时的对称性：sigmoid(z) + sigmoid(-z) = 1
    got = solve("logit_normal", z, logit_mean=0.0, logit_std=1.2)
    assert torch.allclose(got + got.flip(0), torch.ones_like(got), atol=1e-6)

    # logit_std = 0 时退化为常数 sigmoid(mean)
    flat = solve("logit_normal", z, logit_mean=0.4, logit_std=0.0)
    assert torch.allclose(flat, torch.full_like(flat, float(torch.sigmoid(torch.tensor(0.4)))), atol=1e-6)


def check_mode_properties(solve):
    u = torch.linspace(0.0, 1.0, 101)
    got = solve("mode", u, mode_scale=1.29)
    assert abs(float(got[0]) - 1.0) < 1e-6, "u=0 必须映射到 1"
    assert abs(float(got[-1])) < 1e-6, "u=1 必须映射到 0"
    assert bool((got[1:] < got[:-1]).all()), "mode 映射必须严格单调递减"
    assert abs(float(got[50]) - 0.5) < 1e-6, "u=0.5 必须映射到 0.5"
    # 对称性：v(u) + v(1-u) = 1
    assert torch.allclose(got + got.flip(0), torch.ones_like(got), atol=1e-6)

    # mode_scale = 0 时括号整项归零，退化为 1-u
    zero = solve("mode", u, mode_scale=0.0)
    assert torch.allclose(zero, 1.0 - u, atol=1e-7)


def check_bad_input(solve):
    z = torch.tensor([-1.0, 0.0, 1.0])
    ok = torch.tensor([0.1, 0.5])
    bads = [
        lambda: solve("gaussian", ok),
        lambda: solve("logit_normal", z),
        lambda: solve("logit_normal", z, logit_mean=0.0),
        lambda: solve("logit_normal", z, logit_std=1.0),
        lambda: solve("logit_normal", z, logit_mean=0.0, logit_std=-1.0),
        lambda: solve("logit_normal", z, logit_mean=float("inf"), logit_std=1.0),
        lambda: solve("mode", ok),
        lambda: solve("mode", ok, mode_scale=-0.5),
        lambda: solve("mode", ok, mode_scale=float("nan")),
        lambda: solve("mode", torch.tensor([-0.01, 0.5]), mode_scale=1.29),
        lambda: solve("mode", torch.tensor([0.5, 1.01]), mode_scale=1.29),
        lambda: solve("uniform", torch.tensor([0.5, 1.01])),
        lambda: solve("uniform", torch.tensor([float("inf"), 0.2])),
        lambda: solve("uniform", torch.tensor([[0.1, 0.2]])),
        lambda: solve("uniform", [0.1, 0.2]),
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
        n = int(torch.randint(1, 10, (1,), generator=g).item())
        z = torch.randn(n, generator=g)
        mean = float(torch.randn(1, generator=g).item())
        std = float(torch.rand(1, generator=g).item()) + 0.05
        got = solve("logit_normal", z, logit_mean=mean, logit_std=std)
        assert torch.allclose(got, _oracle("logit_normal", z, mean, std), atol=1e-6)

        u = torch.rand(n, generator=g)
        scale = float(torch.rand(1, generator=g).item()) * 1.7
        got = solve("mode", u, mode_scale=scale)
        assert torch.allclose(got, _oracle("mode", u, None, None, scale), atol=1e-6)

        got = solve("uniform", u)
        assert torch.equal(got, u)
