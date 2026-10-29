import math

import torch


def _oracle(cond, uncond, skip, scale, skip_scale, step, steps, start, stop, pstart, pstop):
    """独立 oracle：窗口判定与逐元素组合全部用 python 标量重做。"""
    cfg_on = int(start * steps) <= step < int(stop * steps) and not math.isclose(scale, 1.0)
    pag_on = int(pstart * steps) < step < int(pstop * steps) and not math.isclose(skip_scale, 0.0)

    rows_c = cond.reshape(cond.shape[0], -1).tolist()
    rows_u = uncond.reshape(uncond.shape[0], -1).tolist()
    rows_k = skip.reshape(skip.shape[0], -1).tolist()
    rows = []
    for rc, ru, rk in zip(rows_c, rows_u, rows_k):
        row = []
        for c, u, k in zip(rc, ru, rk):
            if not cfg_on and not pag_on:
                v = c
            elif not cfg_on:
                v = k + skip_scale * (c - k)
            elif not pag_on:
                v = u + scale * (c - u)
            else:
                v = u + scale * (c - u) + skip_scale * (c - k)
            row.append(v)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64).reshape(cond.shape)


def check_example(solve):
    c = torch.tensor([[0.4, -1.2, 0.6], [0.0, 0.5, -0.5]])
    u = torch.tensor([[0.1, -0.4, 0.2], [0.2, -0.1, 0.0]])
    k = torch.tensor([[0.2, -0.8, 0.5], [0.1, 0.3, -0.2]])

    # step=1：PAG 窗口 (0, 2) 与 CFG 窗口 [0, 10) 同时开启
    both = solve(c, u, k, 7.5, 2.8, 1, 10)
    assert [[round(float(v), 6) for v in row] for row in both.tolist()] == [
        [2.91, -7.52, 3.48],
        [-1.58, 4.96, -4.59],
    ], both.tolist()

    # step=0：PAG 窗口是开区间，左端点不生效，只剩 CFG
    only_cfg = solve(c, u, k, 7.5, 2.8, 0, 10)
    assert [[round(float(v), 6) for v in row] for row in only_cfg.tolist()] == [
        [2.35, -6.4, 3.2],
        [-1.3, 4.4, -3.75],
    ], only_cfg.tolist()
    assert only_cfg.dtype == torch.float32 and only_cfg.shape == c.shape


check_example.example = True


def check_window_boundaries(solve):
    torch.manual_seed(0)
    c = torch.randn(2, 4, dtype=torch.float64)
    u = torch.randn(2, 4, dtype=torch.float64)
    k = torch.randn(2, 4, dtype=torch.float64)

    kwargs = dict(start=0.5, stop=1.0, perturbed_start=0.0, perturbed_stop=0.3)
    # CFG 窗口 [5, 10)，PAG 窗口 (0, 3)
    only_skip = solve(c, u, k, 7.5, 2.8, 2, 10, **kwargs).to(torch.float64)
    assert torch.allclose(only_skip, k + 2.8 * (c - k), atol=1e-6), "step=2 应只剩 PAG 分支"

    for step in (0, 3, 4):
        got = solve(c, u, k, 7.5, 2.8, step, 10, **kwargs)
        assert torch.allclose(got, c.to(torch.float32), atol=1e-7), f"step={step} 两个窗口都关闭"

    for step in (5, 9):
        got = solve(c, u, k, 7.5, 2.8, step, 10, **kwargs).to(torch.float64)
        assert torch.allclose(got, u + 7.5 * (c - u), atol=1e-6), f"step={step} 应只剩 CFG 分支"

    # CFG 窗口左闭：start=0.0 时 step=0 就开始引导；PAG 窗口右开：step=int(stop*N) 已经关闭
    left = solve(c, u, k, 7.5, 2.8, 0, 10, start=0.0, stop=1.0).to(torch.float64)
    assert torch.allclose(left, u + 7.5 * (c - u), atol=1e-6), "CFG 窗口左端点应包含在内"

    # 步数很少时两套窗口的整数取整方式不同：N=3 时 PAG 窗口是 (0, 0)，恒不生效
    tiny = solve(c, u, k, 7.5, 2.8, 1, 3).to(torch.float64)
    assert torch.allclose(tiny, u + 7.5 * (c - u), atol=1e-6), "N=3 时 PAG 窗口为空"


def check_scale_window(solve):
    torch.manual_seed(1)
    c = torch.randn(2, 5, dtype=torch.float64)
    u = torch.randn(2, 5, dtype=torch.float64)
    k = torch.randn(2, 5, dtype=torch.float64)

    # guidance_scale 与 1.0 相近：CFG 关闭，但 PAG 仍开（step=1 落在 PAG 窗口内）
    won = solve(c, u, k, 1.0, 2.8, 1, 10).to(torch.float64)
    assert torch.allclose(won, k + 2.8 * (c - k), atol=1e-6), "scale=1 时退回 PAG 分支"

    # perturbed_scale 与 0.0 相近：PAG 关闭，只剩 CFG
    zero = solve(c, u, k, 7.5, 0.0, 1, 10).to(torch.float64)
    assert torch.allclose(zero, u + 7.5 * (c - u), atol=1e-6), "skip_scale=0 时只剩 CFG"

    # 两者都关闭：原样返回条件预测
    plain = solve(c, u, k, 1.0, 0.0, 5, 10)
    assert torch.allclose(plain, c.to(torch.float32), atol=1e-7)


def check_branch_composition(solve):
    torch.manual_seed(2)
    c = torch.randn(3, 4, 3, dtype=torch.float64)
    u = torch.randn(3, 4, 3, dtype=torch.float64)
    k = torch.randn(3, 4, 3, dtype=torch.float64)

    both = solve(c, u, k, 6.0, 3.5, 1, 10).to(torch.float64)
    cfg = solve(c, u, k, 6.0, 0.0, 1, 10).to(torch.float64)
    pag = solve(c, u, k, 1.0, 3.5, 1, 10).to(torch.float64)

    # 双开时两项扰动分别可加：both - cfg 恰好是 PAG 的扰动项
    assert torch.allclose(both - cfg, 3.5 * (c - k), atol=1e-5), "PAG 项与 CFG 项应可分离"
    # 单开 PAG 时基准是扰动预测，所以 both 等于「CFG 结果 + (PAG 结果 - 扰动预测)」
    assert torch.allclose(both, cfg + pag - k.to(torch.float64), atol=1e-5), "双开等于 CFG 结果加上 PAG 的偏移"
    assert not torch.allclose(cfg, pag, atol=1e-3), "两支应给出不同结果（本用例）"


def check_optional_branches(solve):
    torch.manual_seed(3)
    c = torch.randn(2, 3, dtype=torch.float64)
    u = torch.randn(2, 3, dtype=torch.float64)
    k = torch.randn(2, 3, dtype=torch.float64)

    # 分支关闭时对应输入可以缺省
    got = solve(c, None, k, 1.0, 2.8, 1, 10)
    assert torch.allclose(got, (k + 2.8 * (c - k)).to(torch.float32), atol=1e-6)

    got = solve(c, u, None, 7.5, 0.0, 1, 10)
    assert torch.allclose(got, (u + 7.5 * (c - u)).to(torch.float32), atol=1e-6)

    # 两支都关闭时两个分支输入都可以缺省
    got = solve(c, None, None, 1.0, 0.0, 0, 10)
    assert torch.allclose(got, c.to(torch.float32), atol=1e-7)


def check_against_oracle(solve):
    torch.manual_seed(4)
    c = torch.randn(3, 2, 4, dtype=torch.float64)
    u = torch.randn(3, 2, 4, dtype=torch.float64)
    k = torch.randn(3, 2, 4, dtype=torch.float64)

    settings = [
        (7.5, 2.8, 0.0, 1.0, 0.01, 0.2),
        (5.0, 1.0, 0.25, 0.75, 0.0, 0.5),
        (1.0, 2.8, 0.0, 1.0, 0.0, 1.0),
        (9.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    ]
    for scale, skip_scale, start, stop, pstart, pstop in settings:
        for step in range(10):
            got = solve(c, u, k, scale, skip_scale, step, 10, start, stop, pstart, pstop)
            want = _oracle(c, u, k, scale, skip_scale, step, 10, start, stop, pstart, pstop)
            assert torch.allclose(got, want.to(torch.float32), atol=1e-5), (
                f"scale={scale}, skip={skip_scale}, step={step} 与 oracle 不一致"
            )


def check_invalid(solve):
    c = torch.zeros(2, 3)
    u = torch.ones(2, 3)
    k = torch.full((2, 3), 0.5)

    for bad in (
        lambda: solve([[0.0, 1.0]], u, k, 7.5, 2.8, 1, 10),
        lambda: solve(c, torch.ones(3, 3), k, 7.5, 2.8, 1, 10),
        lambda: solve(c, u, torch.ones(2, 2), 7.5, 2.8, 1, 10),
        lambda: solve(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), 7.5, 2.8, 1, 10),
        lambda: solve(c, u, k, -1.0, 2.8, 1, 10),
        lambda: solve(c, u, k, 7.5, -0.1, 1, 10),
        lambda: solve(c, u, k, 7.5, 2.8, -1, 10),
        lambda: solve(c, u, k, 7.5, 2.8, 10, 10),
        lambda: solve(c, u, k, 7.5, 2.8, 1.5, 10),
        lambda: solve(c, u, k, 7.5, 2.8, True, 10),
        lambda: solve(c, u, k, 7.5, 2.8, 1, 0),
        lambda: solve(c, u, k, 7.5, 2.8, 1, 10, start=-0.1),
        lambda: solve(c, u, k, 7.5, 2.8, 1, 10, stop=1.2),
        lambda: solve(c, u, k, 7.5, 2.8, 1, 10, start=0.8, stop=0.2),
        lambda: solve(c, u, k, 7.5, 2.8, 1, 10, perturbed_start=0.5, perturbed_stop=0.1),
        lambda: solve(c, u, k, "7.5", 2.8, 1, 10),
        # 分支生效却缺少对应预测
        lambda: solve(c, None, k, 7.5, 2.8, 1, 10),
        lambda: solve(c, u, None, 7.5, 2.8, 1, 10),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
