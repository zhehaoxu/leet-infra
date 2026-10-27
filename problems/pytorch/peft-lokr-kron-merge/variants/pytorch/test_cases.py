import torch

W0 = torch.zeros(4, 4)
W1 = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0]])                          # (2,2)
W2 = torch.tensor([[10.0, 20.0],
                   [30.0, 40.0]])                        # (2,2)


def _oracle(W0, w1, w2, alpha, r):
    """独立重算：显式四重循环按块布局逐位置放置（结构异于参考解的 torch.kron）。"""
    s = float(alpha) / float(r)
    a, b = w1.shape
    c, d = w2.shape
    merged = W0.clone().to(torch.float64)
    for i in range(a):
        for j in range(b):
            for u in range(c):
                for v in range(d):
                    merged[i * c + u, j * d + v] += s * float(w1[i, j]) * float(w2[u, v])
    return merged.to(torch.float32)


def check_example(solve):
    wp = solve(W0, W1, W2, 2.0, 2)      # s = 1
    exp = torch.tensor([[10.0, 20.0, 20.0, 40.0],
                        [30.0, 40.0, 60.0, 80.0],
                        [30.0, 60.0, 40.0, 80.0],
                        [90.0, 120.0, 120.0, 160.0]])
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5), f"got {wp.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(13)
    w1 = torch.randn(2, 3)              # 非方阵因子
    w2 = torch.randn(3, 2)
    w0 = torch.randn(6, 6)
    wp = solve(w0, w1, w2, 6.0, 3)
    assert wp.shape == (6, 6), f"形状错: {tuple(wp.shape)}"
    assert torch.allclose(wp, _oracle(w0, w1, w2, 6.0, 3), atol=1e-5, rtol=1e-5), (
        "与四重循环独立重建不符")


def check_zero_w2_degenerates_to_base(solve):
    """LyCORIS lycoris_way 初始化把 w2 置零：起点适配器不改变基座。"""
    wp = solve(W0, W1, torch.zeros(2, 2), 2.0, 2)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "w2 全零时 W' 应等于 W0"


def check_alpha_zero_noop(solve):
    wp = solve(W0, W1, W2, 0.0, 2)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "alpha=0 应退化为纯基座权重"


def check_rank_one_scaling(solve):
    """r=1 时 s=alpha，缩放直接作用在 Kronecker 积上。"""
    wp = solve(W0, W1, W2, 3.0, 1)
    exp = 3.0 * _oracle(torch.zeros(4, 4), W1, W2, 1.0, 1)
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5)
