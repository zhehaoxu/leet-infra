import math

import torch


def _oracle(logmel, n_mfcc, lifter, cmvn):
    """独立 oracle：逐 (t, k) 用余弦求和算 DCT，再手写 lifter 与 z-score。"""
    rows, cols = len(logmel), len(logmel[0])
    out = []
    for t in range(rows):
        row = []
        for k in range(n_mfcc):
            acc = 0.0
            for n in range(cols):
                acc += logmel[t][n] * math.cos(math.pi * k * (n + 0.5) / cols)
            scale = math.sqrt(1.0 / cols) if k == 0 else math.sqrt(2.0 / cols)
            v = acc * scale
            if lifter > 0:
                v *= 1.0 + 0.5 * lifter * math.sin(math.pi * k / lifter)
            row.append(v)
        out.append(row)
    if cmvn:
        for k in range(n_mfcc):
            col = [out[t][k] for t in range(rows)]
            mean = sum(col) / rows
            var = sum((v - mean) ** 2 for v in col) / rows
            std = math.sqrt(var) + 1e-8
            for t in range(rows):
                out[t][k] = (out[t][k] - mean) / std
    return out


_LOG_MEL = [
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 2.5, 3.0, 3.5],
    [0.5, 1.5, 2.5, 3.5],
]


def check_example(solve):
    torch.manual_seed(0)
    c = solve(torch.tensor(_LOG_MEL), 4)
    want = torch.tensor(
        [
            [5.0, -2.230442, 0.0, -0.158513],
            [5.5, -1.115221, 0.0, -0.079256],
            [4.0, -2.230442, 0.0, -0.158513],
        ]
    )
    assert tuple(c.shape) == (3, 4), f"形状应为 (3, 4)，got {tuple(c.shape)}"
    assert torch.allclose(c, want, atol=1e-5), f"MFCC 不对：{c.tolist()}"


check_example.example = True


def check_orthonormal_dct(solve):
    """正交 DCT 保能量；常数输入只留 c0 = sqrt(M) * const。"""
    torch.manual_seed(1)
    x = torch.rand(5, 6)
    c = solve(x, 6)
    assert abs(float((c ** 2).sum()) - float((x ** 2).sum())) < 1e-4, "正交变换应保能量"

    const = solve(torch.full((3, 5), 2.0), 5)
    assert torch.allclose(const[:, 0], torch.full((3,), math.sqrt(5) * 2.0), atol=1e-5), "c0 应为 sqrt(M)*2"
    assert torch.all(const[:, 1:].abs() < 1e-6), "常数输入的高阶系数应为 0"

    # 与逐点余弦求和的独立实现一致（含截断到前 n_mfcc 个系数）
    got = solve(x, 3)
    want = torch.tensor(_oracle(x.tolist(), 3, 0, False))
    assert torch.allclose(got, want, atol=1e-5), "随机用例与独立 DCT 实现不一致"


def check_lifter(solve):
    """lifter 只按系数下标加权：w[k] = 1 + (L/2) sin(pi*k/L)。"""
    torch.manual_seed(2)
    x = torch.tensor(_LOG_MEL)
    base = solve(x, 4)
    lifted = solve(x, 4, lifter=4)
    w = torch.tensor([1.0 + 2.0 * math.sin(math.pi * k / 4) for k in range(4)])
    assert abs(float(w[1]) - 1.0 - math.sqrt(2.0)) < 1e-6, "w[1] 应为 1+sqrt(2)"
    assert torch.allclose(lifted, base * w[None, :], atol=1e-5), "lifter 权重不对"
    # k=0 的权重恒为 1：lifter 不动 c0
    assert torch.allclose(lifted[:, 0], base[:, 0], atol=1e-6), "c0 不应被提升"
    # L=2 时 k=3 的权重是 1 + sin(3*pi/2) = 0：该系数被完全抹掉
    dead = solve(x, 4, lifter=2)
    assert torch.all(dead[:, 3].abs() < 1e-6), "lifter=2 应把 c3 权重压成 0"


def check_cmvn(solve):
    """CMVN 后逐系数均值为 0、总体标准差为 1（含 lid 为 0 的系数）。"""
    torch.manual_seed(3)
    x = torch.randn(9, 6)
    c = solve(x, 4, lifter=4, cmvn=True)
    assert torch.all(c.mean(dim=0).abs() < 1e-5), f"均值应为 0，got {c.mean(dim=0).tolist()}"
    std = c.std(dim=0, unbiased=False)
    assert torch.all((std - 1.0).abs() < 1e-4), f"总体标准差应为 1，got {std.tolist()}"

    # 全零输入：方差为 0，除以 1e-8 之后仍是 0（不会 NaN）
    zero = solve(torch.zeros(4, 5), 3, cmvn=True)
    assert torch.all(torch.isfinite(zero)) and float(zero.abs().max()) == 0.0, "全零输入应稳定输出 0"


def check_invalid_inputs(solve):
    good = torch.randn(4, 5)
    for bad in (
        lambda: solve(torch.randn(5), 3),
        lambda: solve(good, 0),
        lambda: solve(good, 6),
        lambda: solve(good, 3, lifter=-1.0),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
