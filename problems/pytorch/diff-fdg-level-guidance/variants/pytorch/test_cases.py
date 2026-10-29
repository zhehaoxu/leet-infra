import math

import torch

_EPS = 1e-12


def _std(vals):
    """无偏标准差（与 torch.std 的默认 correction=1 一致）。"""
    n = len(vals)
    mean = sum(vals) / n
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))


def _projected(shift, reference, weight):
    """把 shift 相对 reference 拆成平行/正交分量，平行分量乘 weight。"""
    norm = math.sqrt(sum(v * v for v in reference))
    divisor = norm if norm > _EPS else _EPS
    unit = [v / divisor for v in reference]
    dot = sum(a * b for a, b in zip(shift, unit))
    parallel = [dot * u for u in unit]
    return [weight * p + (s - p) for s, p in zip(shift, parallel)]


def _oracle(cond, uncond, scales, weights, rescales, active):
    """独立 oracle：逐频率层、逐样本用 python 标量重算。"""
    total = None
    for c_level, u_level, scale, weight, phi, on in zip(cond, uncond, scales, weights, rescales, active):
        rows_c = c_level.reshape(c_level.shape[0], -1).tolist()
        rows_u = u_level.reshape(u_level.shape[0], -1).tolist()
        rows = []
        for rc, ru in zip(rows_c, rows_u):
            if not on:
                rows.append(list(rc))
                continue
            shift = [a - b for a, b in zip(rc, ru)]
            if not math.isclose(weight, 1.0):
                shift = _projected(shift, rc, weight)
            pred = [b + scale * s for b, s in zip(ru, shift)]
            if phi > 0.0:
                std_ref = _std(rc)
                std_pred = _std(pred)
                assert std_pred > 0.0, "oracle 前提：该层每个样本非常数"
                pred = [(1.0 - phi) * p + phi * p * (std_ref / std_pred) for p in pred]
            rows.append(pred)
        level = torch.tensor(rows, dtype=torch.float64).reshape(c_level.shape)
        total = level if total is None else total + level
    return total


def check_example(solve):
    c = [torch.tensor([[1.0, -2.0, 0.5]]), torch.tensor([[0.2, 0.4, -0.6]])]
    u = [torch.tensor([[0.5, -1.0, 0.0]]), torch.tensor([[0.1, 0.2, -0.2]])]

    plain = solve(c, u, [10.0, 5.0])
    assert [[round(float(v), 6) for v in row] for row in plain.tolist()] == [[6.1, -9.8, 2.8]], plain.tolist()
    assert plain.dtype == torch.float32 and plain.shape == c[0].shape

    weighted = solve(c, u, [10.0, 5.0], parallel_weights=[0.0, 1.0])
    assert [[round(float(v), 6) for v in row] for row in weighted.tolist()] == [[0.861905, 0.676190, 0.180952]], (
        weighted.tolist()
    )

    rescaled = solve(c, u, [10.0, 5.0], level_rescale=[0.5, 0.5])
    assert [[round(float(v), 6) for v in row] for row in rescaled.tolist()] == [
        [3.608426, -5.666953, 1.507397]
    ], rescaled.tolist()


check_example.example = True


def check_equal_scales_collapse(solve):
    torch.manual_seed(0)
    cond = [torch.randn(2, 3, 4, dtype=torch.float64) for _ in range(3)]
    uncond = [torch.randn(2, 3, 4, dtype=torch.float64) for _ in range(3)]

    big_c = sum(cond)
    big_u = sum(uncond)
    for scale in (1.0, 4.0, 10.0):
        got = solve(cond, uncond, [scale] * 3)
        want = (big_u + scale * (big_c - big_u)).to(torch.float32)
        assert torch.allclose(got, want, atol=1e-5), f"所有层同尺度时，FDG 应退化为「求和后再做 CFG」（s={scale}）"


def check_parallel_weight(solve):
    torch.manual_seed(1)
    c0 = torch.randn(2, 4, dtype=torch.float64)
    c1 = torch.randn(2, 4, dtype=torch.float64)
    u0 = torch.randn(2, 4, dtype=torch.float64)
    u1 = torch.randn(2, 4, dtype=torch.float64)

    # 只让第 0 层生效、尺度取 1，则 total - c1 - u0 恰为该层的 shift
    on0 = solve([c0, c1], [u0, u1], [1.0, 1.0], parallel_weights=[1.0, 1.0], active_levels=[True, False])
    delta = on0.to(torch.float64) - c1 - u0
    assert torch.allclose(delta, c0 - u0, atol=1e-6), "w=1 时 shift 原样保留"

    # w=0：只剩正交分量，与条件预测正交
    orth = solve([c0, c1], [u0, u1], [1.0, 1.0], parallel_weights=[0.0, 1.0], active_levels=[True, False])
    delta = orth.to(torch.float64) - c1 - u0
    dot = torch.sum(delta * c0, dim=1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5), "w=0 时 shift 应剔除相对 cond 的平行分量"
    # 正交化只会削弱幅值
    assert (delta.norm(dim=1) <= (c0 - u0).norm(dim=1) + 1e-6).all()

    # w=0.5：平行分量减半，正交分量不变
    half = solve([c0, c1], [u0, u1], [1.0, 1.0], parallel_weights=[0.5, 1.0], active_levels=[True, False])
    delta_half = half.to(torch.float64) - c1 - u0
    assert torch.allclose(delta, (c0 - u0) + 2.0 * (delta_half - (c0 - u0)), atol=1e-5), (
        "w=0 的结果应等于「w=0.5 再额外去掉一半平行分量」"
    )


def check_level_rescale(solve):
    torch.manual_seed(2)
    c0 = torch.randn(3, 4, 5, dtype=torch.float64) * 3.0
    u0 = torch.randn(3, 4, 5, dtype=torch.float64)

    full = solve([c0], [u0], [8.0], level_rescale=[1.0]).to(torch.float64)
    assert torch.allclose(full.std(dim=(1, 2)), c0.std(dim=(1, 2)), atol=1e-5), (
        "φ=1 后每个样本的标准差应等于该层条件预测"
    )

    raw = solve([c0], [u0], [8.0], level_rescale=[0.0]).to(torch.float64)
    assert torch.allclose(raw, u0 + 8.0 * (c0 - u0), atol=1e-6), "φ=0 时不做 rescale"

    half = solve([c0], [u0], [8.0], level_rescale=[0.5]).to(torch.float64)
    assert torch.allclose(half, 0.5 * raw + 0.5 * full, atol=1e-5), "φ 是线性混合系数"


def check_disabled_level(solve):
    torch.manual_seed(3)
    cond = [torch.randn(2, 3, dtype=torch.float64) for _ in range(3)]
    uncond = [torch.randn(2, 3, dtype=torch.float64) for _ in range(3)]

    # 中间层关闭：它贡献自己的 cond 层，而不是「上一次循环留下的那层」
    got = solve(cond, uncond, [6.0, 6.0, 6.0], active_levels=[True, False, True]).to(torch.float64)
    want = (
        uncond[0] + 6.0 * (cond[0] - uncond[0])
        + cond[1]
        + uncond[2] + 6.0 * (cond[2] - uncond[2])
    )
    assert torch.allclose(got, want, atol=1e-6), "关闭的频率层必须原样透传它自己的 cond 层"

    # 全关时输出就是各层 cond 之和
    off = solve(cond, uncond, [6.0, 6.0, 6.0], active_levels=[False, False, False])
    assert torch.allclose(off, sum(cond).to(torch.float32), atol=1e-6)


def check_against_oracle(solve):
    torch.manual_seed(4)
    cond = [torch.randn(2, 3, 4, dtype=torch.float64) for _ in range(3)]
    uncond = [torch.randn(2, 3, 4, dtype=torch.float64) for _ in range(3)]

    settings = [
        ([10.0, 5.0, 2.0], None, None, None),
        ([4.0, 4.0, 4.0], [0.0, 1.0, 0.25], None, None),
        ([7.0, 3.0, 1.0], [0.5, 0.5, 0.5], [0.0, 1.0, 0.3], None),
        ([7.0, 3.0, 1.0], None, None, [True, False, True]),
    ]
    for scales, weights, rescales, active in settings:
        got = solve(cond, uncond, scales, weights, rescales, active)
        want = _oracle(
            cond,
            uncond,
            scales,
            [1.0] * 3 if weights is None else weights,
            [0.0] * 3 if rescales is None else rescales,
            [True] * 3 if active is None else active,
        )
        assert torch.allclose(got, want.to(torch.float32), atol=1e-5), (
            f"scales={scales}, weights={weights}, rescales={rescales}, active={active} 与 oracle 不一致"
        )


def check_invalid(solve):
    c = [torch.zeros(2, 3), torch.ones(2, 3)]
    u = [torch.ones(2, 3), torch.zeros(2, 3)]

    for bad in (
        lambda: solve([], [], []),
        lambda: solve(c, u[:1], [1.0, 1.0]),
        lambda: solve(c, u, [1.0]),
        lambda: solve(c, u, [1.0, 1.0], parallel_weights=[1.0]),
        lambda: solve(c, u, [1.0, 1.0], level_rescale=[0.0]),
        lambda: solve(c, u, [1.0, 1.0], active_levels=[True]),
        lambda: solve([c[0], torch.ones(2, 4)], [u[0], torch.ones(2, 4)], [1.0, 1.0]),
        lambda: solve([torch.zeros(3)], [torch.ones(3)], [1.0]),
        lambda: solve([torch.tensor(1.0)], [torch.tensor(1.0)], [1.0]),
        lambda: solve([[[0.0, 1.0]]], [torch.ones(1, 2)], [1.0]),
        lambda: solve(c, u, [-1.0, 1.0]),
        lambda: solve(c, u, [float("nan"), 1.0]),
        lambda: solve(c, u, [1.0, 1.0], parallel_weights=[-0.5, 1.0]),
        lambda: solve(c, u, [1.0, 1.0], level_rescale=[1.5, 0.0]),
        lambda: solve(c, u, [1.0, 1.0], active_levels=[1, 0]),
        # 近似常数的层开启 rescale 时标准差为 0，无法拉回
        lambda: solve([torch.ones(2, 3)], [torch.zeros(2, 3)], [4.0], level_rescale=[0.5]),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
