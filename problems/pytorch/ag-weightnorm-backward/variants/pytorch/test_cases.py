import torch
import torch.nn.functional as F

# 示例：两片都是 ‖v‖ = 5 ⇒ v̂ = [0.6,0.8] 与 [0,1]；g = [2,1]、dW = [[1,0],[0,1]]
# dg = [0.6, 1.0]；dV_0 = (2/5)([1,0] − [0.6,0.8]·0.6) = [0.256, −0.192]；dV_1 = 0
EXAMPLE_V = [[3.0, 4.0], [0.0, 5.0]]
EXAMPLE_G = [2.0, 1.0]
EXAMPLE_DW = [[1.0, 0.0], [0.0, 1.0]]
EXAMPLE_DG = [0.6, 1.0]
EXAMPLE_DV = [[0.256, -0.192], [0.0, 0.0]]


def _t(v):
    return torch.tensor(v, dtype=torch.float64)


def check_example(solve):
    dg, dV = solve(_t(EXAMPLE_V), _t(EXAMPLE_G), _t(EXAMPLE_DW))
    assert torch.allclose(dg, _t(EXAMPLE_DG), atol=1e-9)
    assert torch.allclose(dV, _t(EXAMPLE_DV), atol=1e-9)


check_example.example = True


def _wn_forward(V, g):
    n = V.reshape(V.shape[0], -1).norm(dim=1).view(-1, *([1] * (V.dim() - 1)))
    return g.view(-1, *([1] * (V.dim() - 1))) * V / n


def check_matches_autograd_linear(solve):
    torch.manual_seed(1401)
    V = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    g = torch.randn(5, dtype=torch.float64, requires_grad=True)
    dW = torch.randn(5, 4, dtype=torch.float64)
    W = _wn_forward(V, g)
    refV, refg = torch.autograd.grad((W * dW).sum(), [V, g])
    gotg, gotV = solve(V.detach(), g.detach(), dW)
    assert torch.allclose(gotg, refg, atol=1e-9)
    assert torch.allclose(gotV, refV, atol=1e-9)


def check_matches_autograd_conv4d(solve):
    # 卷积权重 (out, in, kh, kw)：每片折叠 in·kh·kw 三个轴
    torch.manual_seed(1402)
    V = torch.randn(4, 3, 3, 2, dtype=torch.float64, requires_grad=True)
    g = torch.randn(4, dtype=torch.float64, requires_grad=True)
    dW = torch.randn(4, 3, 3, 2, dtype=torch.float64)
    W = _wn_forward(V, g)
    refV, refg = torch.autograd.grad((W * dW).sum(), [V, g])
    gotg, gotV = solve(V.detach(), g.detach(), dW)
    assert gotV.shape == V.shape and gotg.shape == (4,)
    assert torch.allclose(gotg, refg, atol=1e-9)
    assert torch.allclose(gotV, refV, atol=1e-9)


def check_scale_unidentifiability(solve):
    # v 的尺度不可辨识：dV 必须与 v 正交；v 正缩放 c 倍时 W 与 dg 都不变
    torch.manual_seed(1403)
    V = torch.randn(6, 5, dtype=torch.float64)
    g = torch.randn(6, dtype=torch.float64)
    dW = torch.randn(6, 5, dtype=torch.float64)
    dg, dV = solve(V, g, dW)
    proj = (dV * V).sum(dim=1)
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-9)
    for c in (2.0, 0.5, 3.0):
        dg2, dV2 = solve(V * c, g, dW)
        assert torch.allclose(dg2, dg, atol=1e-9)
        assert torch.allclose(_wn_forward(V * c, g), _wn_forward(V, g), atol=1e-9)
        # dV 随 1/c 缩放（方向不变、模长被归一化抵消）
        assert torch.allclose(dV2 * c, dV, atol=1e-9)


def check_g_zero_and_dtype(solve):
    V = _t([[3.0, 4.0]])
    dW = _t([[1.0, 1.0]])
    dg, dV = solve(V, torch.zeros(1, dtype=torch.float64), dW)
    # g=0 时 dV 必为 0，但 dg = ⟨dW, v̂⟩ = 1·0.6 + 1·0.8 仍非 0（g 只缩放 dV）
    assert torch.allclose(dg, torch.tensor([1.4], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(dV, torch.zeros_like(dV), atol=1e-12)
    v32 = torch.randn(3, 4)
    g32 = torch.randn(3)
    a, b = solve(v32, g32, torch.randn(3, 4))
    assert a.dtype == torch.float32 and b.dtype == torch.float32
    assert a.shape == (3,) and b.shape == (3, 4)
    assert solve(v32, torch.zeros(3), torch.randn(3, 4))[1].abs().max() < 1e-12


def check_invalid_args(solve):
    V = torch.randn(3, 4)
    g = torch.randn(3)
    dW = torch.randn(3, 4)
    bad = [
        (V, torch.randn(4), dW),        # g 长度不符
        (V, g.reshape(3, 1), dW),       # g 非一维
        (V, g, torch.randn(4, 3)),      # dW 形状不符
        (torch.zeros(2, 3), torch.randn(2), torch.zeros(2, 3)),   # 零向量切片：方向无定义
        (torch.tensor(1.0), g, dW),     # V 非矩阵
    ]
    for vv, gg, dd in bad:
        try:
            solve(vv, gg, dd)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
