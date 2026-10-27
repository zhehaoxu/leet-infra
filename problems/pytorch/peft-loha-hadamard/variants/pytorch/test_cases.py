import torch

W0 = torch.tensor([[0.5, -0.2, 0.1],
                   [0.3, 0.0, -0.4]])                     # (2,3)
W1_A = torch.tensor([[2.0], [-1.0]])                      # (2,1) r=1
W1_B = torch.tensor([[1.0, 0.0, -1.0]])                   # (1,3)
W2_A = torch.tensor([[1.0], [1.0]])                       # (2,1)
W2_B = torch.tensor([[0.0, 1.0, 1.0]])                    # (1,3)


def _oracle(W0, w1_a, w1_b, w2_a, w2_b, alpha, r):
    """独立重算：逐标量三重循环重建 Hadamard 积（结构异于参考解的矩阵乘路径）。"""
    d_out, d_in = W0.shape
    s = float(alpha) / float(r)
    merged = W0.clone().to(torch.float64)
    for i in range(d_out):
        for j in range(d_in):
            b1 = sum(float(w1_a[i, u]) * float(w1_b[u, j]) for u in range(w1_a.shape[1]))
            b2 = sum(float(w2_a[i, u]) * float(w2_b[u, j]) for u in range(w2_a.shape[1]))
            merged[i, j] += s * b1 * b2
    return merged.to(torch.float32)


def check_example(solve):
    wp = solve(W0, W1_A, W1_B, W2_A, W2_B, 2.0, 1)
    exp = torch.tensor([[0.5, -0.2, -3.9],
                        [0.3, 0.0, 1.6]])
    assert torch.allclose(wp, exp, atol=1e-5, rtol=1e-5), f"got {wp.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(11)
    w0 = torch.randn(4, 5)
    w1a, w1b = torch.randn(4, 2), torch.randn(2, 5)
    w2a, w2b = torch.randn(4, 2), torch.randn(2, 5)
    wp = solve(w0, w1a, w1b, w2a, w2b, 6.0, 2)
    assert torch.allclose(wp, _oracle(w0, w1a, w1b, w2a, w2b, 6.0, 2),
                          atol=1e-5, rtol=1e-5), "与逐元素独立重算不符"


def check_zero_branch_degenerates_to_base(solve):
    """LyCORIS 默认初始化把 w2_b 置零：适配器初始应完全不改变基座。"""
    wp = solve(W0, W1_A, W1_B, W2_A, torch.zeros(1, 3), 2.0, 1)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "w2_b 全零时 W' 应等于 W0"


def check_alpha_zero_noop(solve):
    wp = solve(W0, W1_A, W1_B, W2_A, W2_B, 0.0, 1)
    assert torch.allclose(wp, W0, atol=1e-6, rtol=1e-6), "alpha=0 应退化为纯基座权重"


def check_hadamard_not_sum(solve):
    """乘性组合 ≠ 两条 LoRA 分支相加（固定反例断言）。"""
    wp = solve(W0, W1_A, W1_B, W2_A, W2_B, 2.0, 1)
    s = 2.0
    additive = W0 + s * (W1_A @ W1_B + W2_A @ W2_B)
    assert not torch.allclose(wp, additive, atol=1e-3, rtol=1e-3), (
        "LoHa 的 Hadamard 积不应与加法组合一致")
