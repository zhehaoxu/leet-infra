import torch

EPS_C = torch.tensor([[1.0, -0.5, 0.2], [0.3, 0.7, -1.1]])
EPS_U = torch.tensor([[0.4, 0.1, -0.3], [0.6, -0.2, 0.5]])
W = 7.5


def _oracle(eps_c, eps_u, w):
    """独立重算：等价式 (1-w)*eps_u + w*eps_c（结构异于参考解的 uncond+w*(cond-uncond)）。"""
    out = torch.empty_like(eps_c)
    for idx in range(eps_c.numel()):
        c = eps_c.flatten()[idx].item()
        u = eps_u.flatten()[idx].item()
        out.flatten()[idx] = (1 - w) * u + w * c
    return out


def check_example(solve):
    y = solve(EPS_C, EPS_U, W)
    exp = torch.tensor([[4.9, -4.4, 3.45], [-1.65, 6.55, -11.5]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(9)
    c = torch.randn(3, 4)
    u = torch.randn(3, 4)
    for w in [0.0, 1.0, 3.5, 7.5, 12.0]:
        assert torch.allclose(solve(c, u, w), _oracle(c, u, w), atol=1e-4, rtol=1e-4), w


def check_w_zero_is_uncond(solve):
    y = solve(EPS_C, EPS_U, 0.0)
    assert torch.allclose(y, EPS_U, atol=1e-6, rtol=1e-6), "w=0 应等于 eps_uncond"


def check_w_one_is_cond(solve):
    y = solve(EPS_C, EPS_U, 1.0)
    assert torch.allclose(y, EPS_C, atol=1e-6, rtol=1e-6), "w=1 应等于 eps_cond（无引导）"


def check_shape_dtype_and_amplify(solve):
    y = solve(EPS_C, EPS_U, W)
    assert y.shape == EPS_C.shape and y.dtype == torch.float32
    # w>1 沿 cond 方向外推：与 uncond 的距离应比 cond-uncond 更大
    assert torch.all((y - EPS_U).abs() > (EPS_C - EPS_U).abs() - 1e-6), "w>1 应放大条件偏移"
