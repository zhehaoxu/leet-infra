import torch

# 示例：LX = [[0,1,2],[-100,-99,-98]]（实跑参考答案回填）
EXAMPLE_X = [[0.0, 1.0, 2.0], [-100.0, -99.0, -98.0]]
EXAMPLE_DIM1 = [2.4076058864593506, -97.59239196777344]
EXAMPLE_DIM0 = [0.0, 1.0, 2.0]
EXAMPLE_ALL = 2.4076058864593506


def check_example(solve):
    x = torch.tensor(EXAMPLE_X)
    assert torch.allclose(solve(x, dim=1), torch.tensor(EXAMPLE_DIM1), atol=1e-6)
    assert torch.allclose(solve(x, dim=0), torch.tensor(EXAMPLE_DIM0), atol=1e-6)
    assert abs(float(solve(x, dim=(0, 1))) - EXAMPLE_ALL) < 1e-6
    assert solve(x, dim=1, keepdim=True).shape == (2, 1)


check_example.example = True


def check_no_naive_overflow(solve):
    # 朴素 log(sum(exp(x))) 在两端都会挂：+1e3 上溢成 inf、-1e3 下溢成 -inf
    big = torch.tensor([[1000.0, 1002.0, 999.0]])
    small = torch.tensor([[-1000.0, -1002.0, -999.0]])
    naive_big = big.exp().sum(dim=1).log()
    naive_small = small.exp().sum(dim=1).log()
    assert not torch.isfinite(naive_big).all(), "朴素式在大正值处应当失效"
    assert not torch.isfinite(naive_small).all(), "朴素式在小值处应当失效"
    assert torch.isfinite(solve(big, dim=1)).all()
    assert torch.isfinite(solve(small, dim=1)).all()
    # 大值段的正确答案：max + log Σexp(x-max) = 1002 + log(1+e^-2+e^-3)
    ref = 1002.0 + float(torch.log1p(torch.exp(torch.tensor(-2.0)) + torch.exp(torch.tensor(-3.0))))
    assert abs(float(solve(big, dim=1)) - ref) < 1e-4, (float(solve(big, dim=1)), ref)


def check_matches_torch(solve):
    torch.manual_seed(401)
    for x in [torch.randn(5, 7), torch.randn(4, 6) * 100.0, torch.randn(3, 4, 5)]:
        opts = [None, 0, 1, -1, (0, 1), tuple(range(x.dim()))]
        if x.dim() >= 3:
            opts.append((1, 2))
        for dimv in opts:
            for keep in [False, True]:
                ref = torch.logsumexp(x.reshape(-1), 0) if dimv is None \
                    else torch.logsumexp(x, dim=dimv, keepdim=keep)
                got = solve(x, dim=dimv, keepdim=keep)
                if dimv is None and keep:
                    assert got.shape == (1,) * x.dim(), got.shape
                    continue
                assert got.shape == ref.shape, (dimv, keep)
                assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), (dimv, keep)
                assert got.dtype == x.dtype


def check_degenerate_segments(solve):
    x = torch.tensor([[-float("inf"), -float("inf"), -float("inf")],
                      [0.0, -float("inf"), 2.0],
                      [float("inf"), 1.0, float("inf")]])
    got = solve(x, dim=1)
    ref = torch.logsumexp(x, dim=1)
    assert torch.isneginf(got[0]), got.tolist()          # 整段全 -inf → -inf，不能是 NaN
    assert not torch.isnan(got).any(), got.tolist()
    assert abs(float(got[1]) - (2.0 + float(torch.log1p(torch.exp(torch.tensor(-2.0)))))) < 1e-5
    assert torch.isposinf(got[2]), got.tolist()          # 段内有 +inf → +inf
    assert torch.allclose(got, ref, atol=1e-6, equal_nan=True)
    # NaN 传播
    n = torch.tensor([1.0, float("nan"), 3.0])
    assert torch.isnan(solve(n, dim=0))


def check_log_softmax_consistency(solve):
    # log_softmax(x) = x - logsumexp(x)：与已有的 log-softmax 题口径互验
    torch.manual_seed(402)
    import torch.nn.functional as F
    x = torch.randn(8, 13) * 50.0
    ref = F.log_softmax(x, dim=1)
    got = x - solve(x, dim=1, keepdim=True)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
    # dim=None 是本题扩展口径：整体归约成 0 维；keepdim=True 时所有轴保留为 1
    assert solve(x, dim=None).dim() == 0
    assert solve(x, dim=None, keepdim=True).shape == (1, 1)
    assert abs(float(solve(x, dim=None)) - float(torch.logsumexp(x.reshape(-1), 0))) < 1e-3


def check_invalid_dim(solve):
    x = torch.randn(3, 4)
    for dimv in [2, -3, (0, 0), (0, 1, 2), ()]:
        try:
            solve(x, dim=dimv)
        except ValueError:
            continue
        raise AssertionError(f"dim={dimv} 应抛 ValueError")
