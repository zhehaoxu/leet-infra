import math

import torch


def _std(x, dims):
    """独立 oracle：先 python 均值再均方偏差，口径为总体标准差。"""
    vals = [float(e) for e in x.flatten().tolist()]
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def _oracle(eps_cond, eps_uncond, w, phi, batch_dims):
    """逐样本走 python 循环（每维展平成一段），与参考解的张量 reduce 不同源。"""
    rows = []
    for i in range(eps_cond.shape[0]):
        c = eps_cond[i]
        u = eps_uncond[i]
        # batch_dims 已保证只允许「整个样本」参与统计
        flat_c = c.reshape(-1)
        flat_u = u.reshape(-1)
        eps_hat = [uc + w * (cc - uc) for cc, uc in zip(flat_c.tolist(), flat_u.tolist())]
        std_c = _std(flat_c, None)
        mean_h = sum(eps_hat) / len(eps_hat)
        std_h = math.sqrt(sum((v - mean_h) ** 2 for v in eps_hat) / len(eps_hat))
        if phi > 0.0:
            ratio = std_c / std_h
            eps_hat = [(1.0 - phi) * v + phi * v * ratio for v in eps_hat]
        rows.append(torch.tensor(eps_hat).reshape(c.shape))
    return torch.stack(rows)


def check_example(solve):
    c = torch.tensor([[0.4, -1.2, 0.6], [0.0, 0.5, -0.5]])
    u = torch.tensor([[0.1, -0.4, 0.2], [0.2, -0.1, 0.0]])

    got = solve(c, u, 3.0, 0.5)
    want = [[0.425607, -1.191700, 0.595850], [-0.123014, 0.522809, -0.461302]]
    assert [[round(float(v), 6) for v in row] for row in got.tolist()] == want, got.tolist()

    raw = solve(c, u, 3.0, 0.5, rescale=0.0)
    assert [[round(float(v), 6) for v in row] for row in raw.tolist()] == [
        [1.0, -2.8, 1.4],
        [-0.4, 1.7, -1.5],
    ], "未做 rescale 时应是裸外推"
    assert got.dtype == torch.float32 and got.shape == c.shape


check_example.example = True


def check_no_guidance_identity(solve):
    torch.manual_seed(0)
    c = torch.randn(2, 4, 3, dtype=torch.float64)
    u = torch.randn(2, 4, 3, dtype=torch.float64)

    # w = 1：外推退化为条件预测，rescale 比值恰为 1
    for phi in (0.0, 1.0):
        got = solve(c, u, 1.0, 0.5, rescale=phi)
        assert torch.allclose(got, c.to(torch.float32), atol=1e-6), "w=1 必须原样返回条件预测"

    # t 落在区间外：无论 w / φ 多大都不做引导
    for t in (0.05, 0.95):
        got = solve(c, u, 7.5, t, t_low=0.2, t_high=0.8, rescale=0.6)
        assert torch.allclose(got, c.to(torch.float32), atol=1e-7), "区间外必须原样返回 eps_cond"


def check_interval_bounds(solve):
    torch.manual_seed(1)
    c = torch.randn(3, 5, dtype=torch.float64)
    u = torch.randn(3, 5, dtype=torch.float64)
    lo, hi = 0.2, 0.8

    guided = solve(c, u, 4.0, 0.5, t_low=lo, t_high=hi, rescale=0.0)
    for t in (lo, hi, 0.5):
        got = solve(c, u, 4.0, t, t_low=lo, t_high=hi, rescale=0.0)
        assert torch.allclose(got, guided, atol=1e-7), f"t={t} 在闭区间内，应做引导"
    for t in (lo - 1e-3, hi + 1e-3):
        got = solve(c, u, 4.0, t, t_low=lo, t_high=hi, rescale=0.0)
        assert torch.allclose(got, c.to(torch.float32), atol=1e-7), f"t={t} 在区间外，不应引导"


def check_rescale_property(solve):
    torch.manual_seed(2)
    c = torch.randn(3, 4, 5, dtype=torch.float64) * 2.0
    u = torch.randn(3, 4, 5, dtype=torch.float64)

    full = solve(c, u, 5.0, 0.4, rescale=1.0).to(torch.float64)
    raw = solve(c, u, 5.0, 0.4, rescale=0.0).to(torch.float64)

    std_c = c.std(dim=(1, 2), unbiased=False)
    std_full = full.std(dim=(1, 2), unbiased=False)
    assert torch.allclose(std_full, std_c, atol=1e-5), "rescale=1 后每样本标准差应等于条件预测"

    # 逐样本只做了一次标量缩放（均值不被保留）
    scale = full / raw
    assert torch.allclose(scale, scale[:, :1, :1].expand_as(scale), atol=1e-5), "缩放必须是每样本一个常数"

    half = solve(c, u, 5.0, 0.4, rescale=0.5).to(torch.float64)
    assert torch.allclose(half, 0.5 * raw + 0.5 * full, atol=1e-5), "φ=0.5 应是两者的线性混合"

    # 显式 dims：只按最后一维统计
    narrow = solve(c, u, 5.0, 0.4, rescale=1.0, dims=(2,)).to(torch.float64)
    assert torch.allclose(narrow.std(dim=2, unbiased=False), c.std(dim=2, unbiased=False), atol=1e-5)


def check_oracle_and_invalid(solve):
    torch.manual_seed(3)
    c = torch.randn(4, 3, 2, dtype=torch.float64)
    u = torch.randn(4, 3, 2, dtype=torch.float64) * 0.5

    for w, phi in ((2.0, 0.0), (2.0, 1.0), (7.5, 0.35)):
        got = solve(c, u, w, 0.6, rescale=phi)
        want = _oracle(c, u, w, phi, None)
        assert torch.allclose(got, want.to(torch.float32), atol=1e-5), f"w={w}, φ={phi} 与 oracle 不一致"

    good_c = torch.zeros(2, 3)
    good_u = torch.ones(2, 3)
    for bad in (
        lambda: solve([[0.0, 1.0]], good_u, 2.0, 0.5),
        lambda: solve(good_c, torch.zeros(3, 3), 2.0, 0.5),
        lambda: solve(torch.zeros(4), torch.ones(4), 2.0, 0.5),
        lambda: solve(good_c, good_u, -1.0, 0.5),
        lambda: solve(good_c, good_u, 2.0, 1.2),
        lambda: solve(good_c, good_u, 2.0, 0.5, t_low=0.8, t_high=0.2),
        lambda: solve(good_c, good_u, 2.0, 0.5, t_low=-0.1),
        lambda: solve(good_c, good_u, 2.0, 0.5, rescale=1.5),
        lambda: solve(good_c, good_u, 2.0, 0.5, dims=(0,)),
        lambda: solve(good_c, good_u, 2.0, 0.5, dims=(1, 1)),
        lambda: solve(good_c, good_u, 2.0, 0.5, dims=(2,)),
        lambda: solve(good_c, good_u, 2.0, 0.5, dims=()),
        # 常数预测外推后标准差为 0，无法 rescale
        lambda: solve(torch.ones(2, 3), torch.zeros(2, 3), 2.0, 0.5),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
