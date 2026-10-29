import torch
import torch.nn.functional as F

# 示例：X(2×2) = [[1,2],[3,4]]，W(3×2) = [[1,0],[2,1],[0,3]]，dY 全 1 (2×3)
# dX = dY @ W ⇒ 每行 [1+2+0, 0+1+3] = [3,4]；dW = ones(3,2) @ X ⇒ 每行 [1+3, 2+4] = [4,6]
EXAMPLE_DY = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
EXAMPLE_X = [[1.0, 2.0], [3.0, 4.0]]
EXAMPLE_W = [[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]]
EXAMPLE_DX = [[3.0, 4.0], [3.0, 4.0]]
EXAMPLE_DW = [[4.0, 6.0], [4.0, 6.0], [4.0, 6.0]]


def check_example(solve):
    dX, dW = solve(torch.tensor(EXAMPLE_DY), torch.tensor(EXAMPLE_X), torch.tensor(EXAMPLE_W))
    assert dX.shape == (2, 2) and dW.shape == (3, 2)
    assert torch.allclose(dX, torch.tensor(EXAMPLE_DX), atol=1e-6)
    assert torch.allclose(dW, torch.tensor(EXAMPLE_DW), atol=1e-6)


check_example.example = True


def check_matches_autograd_2d(solve):
    torch.manual_seed(301)
    X = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    W = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    dY = torch.randn(5, 3, dtype=torch.float64)
    loss = (X @ W.t() * dY).sum()
    refX, refW = torch.autograd.grad(loss, [X, W])
    gotX, gotW = solve(dY, X.detach(), W.detach())
    assert torch.allclose(gotX, refX, atol=1e-9)
    assert torch.allclose(gotW, refW, atol=1e-9)


def check_matches_autograd_batched(solve):
    # 前导两轴 (B, H)：dW 必须把 B、H 全部折叠求和
    torch.manual_seed(302)
    X = torch.randn(2, 3, 5, 4, dtype=torch.float64, requires_grad=True)
    W = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    dY = torch.randn(2, 3, 5, 6, dtype=torch.float64)
    loss = (X @ W.t() * dY).sum()
    refX, refW = torch.autograd.grad(loss, [X, W])
    gotX, gotW = solve(dY, X.detach(), W.detach())
    assert gotX.shape == X.shape and gotW.shape == W.shape
    assert torch.allclose(gotX, refX, atol=1e-9)
    assert torch.allclose(gotW, refW, atol=1e-9)
    # 折叠求和 = 逐 (b, h) 的局部 dW 之和
    parts = sum(solve(dY[b, h], X[b, h].detach(), W.detach())[1]
                for b in range(2) for h in range(3))
    assert torch.allclose(parts, gotW, atol=1e-9)


def check_linear_layer_equivalence(solve):
    # F.linear(X, W) 的前向就是 X @ W.T，梯度应与 autograd 逐位一致
    torch.manual_seed(303)
    for shape in [(7, 8), (2, 7, 8)]:
        X = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        W = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        dY = torch.randn(*(shape[:-1] + (5,)), dtype=torch.float64)
        loss = (F.linear(X, W) * dY).sum()
        refX, refW = torch.autograd.grad(loss, [X, W])
        gotX, gotW = solve(dY, X.detach(), W.detach())
        assert torch.allclose(gotX, refX, atol=1e-9), shape
        assert torch.allclose(gotW, refW, atol=1e-9), shape


def check_m1_and_dtype(solve):
    # M=1（单 token 的 GEMV）与 float32 保形、dW 不做 batch 平均
    torch.manual_seed(304)
    X = torch.randn(1, 6)
    W = torch.randn(4, 6)
    dY = torch.randn(1, 4)
    gotX, gotW = solve(dY, X, W)
    assert gotX.dtype == X.dtype and gotW.dtype == W.dtype
    assert gotX.shape == (1, 6) and gotW.shape == (4, 6)
    assert torch.allclose(gotX, dY @ W, atol=1e-5)
    assert torch.allclose(gotW, dY.t() @ X, atol=1e-5)
    assert not torch.isnan(gotW).any()


def check_invalid_shapes(solve):
    X = torch.randn(4, 5)
    W = torch.randn(3, 5)
    dY = torch.randn(4, 3)
    bad = [
        (torch.randn(4, 2), X, W),          # dY 末轴 != N
        (dY, torch.randn(4, 7), W),         # X 末轴 != K
        (torch.randn(2, 4, 3), X, W),       # 前导轴不一致
        (dY, W, W),                          # M 轴不一致
        (dY, X, W.unsqueeze(0)),            # W 非二维
        (dY[0], X, W),                       # dY 少一维
    ]
    for dy, x, w in bad:
        try:
            solve(dy, x, w)
        except ValueError:
            continue
        raise AssertionError("非法形状应抛 ValueError")
