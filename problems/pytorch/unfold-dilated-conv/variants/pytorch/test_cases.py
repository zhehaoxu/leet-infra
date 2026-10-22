import torch


def _ref(x, kH, kW, d):
    B, C, H, W = x.shape
    oh, ow = H - (kH - 1) * d, W - (kW - 1) * d
    out = torch.empty(B, C, oh, ow, kH, kW, dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            out[..., i, j] = x[..., i * d: i * d + oh, j * d: j * d + ow]
    return out


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
    #   0 1 2 3
    #   4 5 6 7
    #   8 9 10 11
    y = solve(x, (2, 2), dilation=2)
    assert y.shape == (1, 1, 1, 2, 2, 2), f"got {tuple(y.shape)}"
    # 块 (0,0) 采样 rows{0,2}×cols{0,2}
    assert torch.equal(y[0, 0, 0, 0], torch.tensor([[0.0, 2.0], [8.0, 10.0]])), f"got {y[0,0,0,0].tolist()}"


def check_dilation_one(solve):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 6)
    y = solve(x, (3, 2), 1)
    assert y.shape == (2, 3, 6, 5, 3, 2), f"got {tuple(y.shape)}"
    assert torch.equal(y, _ref(x, 3, 2, 1))


def check_matches_manual_dilated(solve):
    torch.manual_seed(1)
    x = torch.randn(1, 2, 9, 9)
    for d in (1, 2, 3):
        assert torch.equal(solve(x, (3, 3), d), _ref(x, 3, 3, d)), f"dilation={d} 不一致"


def test_gapped_sampling(solve):
    # 膨胀必须产生间隔：块内相邻元素在原图相距 dilation
    x = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
    y = solve(x, (2, 2), dilation=3)  # oh=ow=5-3=2
    assert torch.equal(y[0, 0, 0, 0], torch.tensor([[0.0, 3.0], [15.0, 18.0]]))


check_basic.example = True
