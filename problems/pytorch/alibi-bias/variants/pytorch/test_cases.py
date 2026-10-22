import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    y = solve(3, 1)
    # n=1：唯一斜率 2^(-8) = 1/256 = 0.00390625
    expected = torch.tensor([[[-0.0000, -0.00390625, -0.00781250],
                              [-0.00390625, -0.0000, -0.00390625],
                              [-0.00781250, -0.00390625, -0.0000]]])
    assert y.shape == (1, 3, 3)
    assert torch.allclose(y, expected, atol=1e-7), f"got {y.tolist()}"


def check_paper_slopes(solve):
    # 论文 8 头斜率应为 [2^-1, 2^-2, ..., 2^-8]
    y = solve(4, 8)
    first_head_slope = -(y[0, 1, 0])
    assert torch.allclose(first_head_slope, torch.tensor(0.5), atol=1e-6), f"头 0 斜率应为 2^-1, got {first_head_slope}"
    last_head_slope = -(y[7, 1, 0])
    assert torch.allclose(last_head_slope, torch.tensor(2.0 ** -8), atol=1e-7), "头 7 斜率应为 2^-8"
    for h in range(8):
        slope = -(y[h, 1, 0])
        exp = -slope * (torch.arange(4)[:, None] - torch.arange(4)[None, :]).abs().float()
        assert torch.allclose(y[h], exp, atol=1e-6), f"头 {h} 应按斜率×距离构造"


def check_distance_structure(solve):
    y = solve(6, 4)
    # 对角线恒 0；对称；随距离线性递减
    for h in range(4):
        m = y[h]
        assert torch.allclose(torch.diagonal(m), torch.zeros(6), atol=1e-7), "对角线应为 0"
        assert torch.allclose(m, m.T, atol=1e-7), "偏置矩阵应对称"
        assert (m <= 1e-9).all(), "偏置必须非正"
        assert torch.allclose(m[3], (m[3, 0] / 3.0) * torch.tensor([3.0, 2.0, 1.0, 0.0, 1.0, 2.0]), atol=1e-6), "应按距离线性"


def check_heads_monotonic(solve):
    # 头序号越大斜率越小（2^(-8h/n) 几何递减，衰减越慢、看得越远）
    y = solve(5, 6)
    slopes = -y[:, 0, 1]
    assert (slopes[1:] < slopes[:-1]).all(), "斜率应随头递减"


def check_matches_manual(solve):
    torch.manual_seed(0)
    L, H = 7, 3
    y = solve(L, H, dtype=torch.float64)
    idx = torch.arange(L, dtype=torch.float64)
    dist = (idx[:, None] - idx[None, :]).abs()
    slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)], dtype=torch.float64)
    exp = -slopes.view(-1, 1, 1) * dist
    assert torch.allclose(y, exp, atol=1e-12), "与手推公式不一致"


def check_dtype(solve):
    y = solve(4, 2, dtype=torch.float64)
    assert y.dtype == torch.float64


check_basic.example = True
