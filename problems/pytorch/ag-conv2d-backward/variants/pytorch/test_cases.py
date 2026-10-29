import torch
import torch.nn.functional as F

# 示例：1×1×3×3 的 x、1×1×2×2 全 1 核、stride=1、padding=0，dY = [[1,0],[0,1]]
# dx = dY 与翻转核的全卷积 ⇒ [[1,1,0],[1,2,1],[0,1,1]]
# dw[k,l] = Σ_{u,v} dY[u,v]·x[u+k,v+l] ⇒ [[1·1+1·5, 1·2+1·6],[1·4+1·8, 1·5+1·9]]
EXAMPLE_X = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
EXAMPLE_W = [[[[1.0, 1.0], [1.0, 1.0]]]]
EXAMPLE_DY = [[[[1.0, 0.0], [0.0, 1.0]]]]
EXAMPLE_DX = [[[[1.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]]]]
EXAMPLE_DW = [[[[6.0, 8.0], [12.0, 14.0]]]]


def _t(v):
    return torch.tensor(v, dtype=torch.float64)


def check_example(solve):
    dx, dw = solve(_t(EXAMPLE_DY), _t(EXAMPLE_X), _t(EXAMPLE_W))
    assert dx.shape == (1, 1, 3, 3) and dw.shape == (1, 1, 2, 2)
    assert torch.allclose(dx, _t(EXAMPLE_DX), atol=1e-9)
    assert torch.allclose(dw, _t(EXAMPLE_DW), atol=1e-9)


check_example.example = True


def check_matches_autograd_basic(solve):
    torch.manual_seed(1001)
    x = torch.randn(2, 3, 6, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 3, 3, 2, dtype=torch.float64, requires_grad=True)
    y = F.conv2d(x, w)
    dY = torch.randn_like(y)
    refX, refW = torch.autograd.grad((y * dY).sum(), [x, w])
    gX, gW = solve(dY, x.detach(), w.detach())
    assert gX.shape == x.shape and gW.shape == w.shape
    assert torch.allclose(gX, refX, atol=1e-9)
    assert torch.allclose(gW, refW, atol=1e-9)


def check_matches_autograd_stride_padding(solve):
    torch.manual_seed(1002)
    for stride, padding in ((1, 1), (2, 0), (2, 1), (3, 2), (1, 2)):
        x = torch.randn(1, 2, 7, 7, dtype=torch.float64, requires_grad=True)
        w = torch.randn(3, 2, 3, 3, dtype=torch.float64, requires_grad=True)
        y = F.conv2d(x, w, stride=stride, padding=padding)
        dY = torch.randn_like(y)
        refX, refW = torch.autograd.grad((y * dY).sum(), [x, w])
        gX, gW = solve(dY, x.detach(), w.detach(), stride=stride, padding=padding)
        assert torch.allclose(gX, refX, atol=1e-9), (stride, padding, (gX - refX).abs().max())
        assert torch.allclose(gW, refW, atol=1e-9), (stride, padding)


def check_dw_is_batch_sum_and_shapes(solve):
    # dw 折叠 batch；单样本逐个相加与批量结果一致
    torch.manual_seed(1003)
    x = torch.randn(3, 2, 5, 5, dtype=torch.float64)
    w = torch.randn(2, 2, 3, 3, dtype=torch.float64)
    y = F.conv2d(x, w)
    dY = torch.randn_like(y)
    _, dw = solve(dY, x, w)
    acc = torch.zeros_like(w)
    for b in range(3):
        _, part = solve(dY[b:b + 1], x[b:b + 1], w)
        acc = acc + part
    assert torch.allclose(dw, acc, atol=1e-9)
    assert dw.shape == w.shape
    # dx 的形状永远等于 x：padding 不会带来多余的行列
    y_pad = F.conv2d(x, w, padding=1)
    dx, _ = solve(torch.randn_like(y_pad), x, w, padding=1)
    assert dx.shape == x.shape


def check_dtype_and_identity_kernel(solve):
    # 1×1 单位核：dx = dY（通道一致时），dw = Σ dY·x
    x = torch.randn(2, 3, 4, 4)
    w = torch.eye(3).reshape(3, 3, 1, 1).clone()
    dY = torch.randn(2, 3, 4, 4)
    dx, dw = solve(dY, x, w)
    assert dx.dtype == x.dtype and dw.dtype == w.dtype
    assert torch.allclose(dx, dY, atol=1e-6)
    assert torch.allclose(dw.reshape(3, 3), (dY.permute(1, 0, 2, 3)
                                             .reshape(3, -1) @ x.permute(1, 0, 2, 3).reshape(3, -1).t()),
                          atol=1e-4)


def check_invalid_args(solve):
    x = torch.randn(1, 2, 5, 5)
    w = torch.randn(3, 2, 3, 3)
    good = F.conv2d(x, w).detach()
    bad = [
        (good, x, torch.randn(3, 4, 3, 3), 1, 0),   # weight 的 Cin 不符
        (torch.randn(1, 3, 4, 5), x, w, 1, 0),      # dY 空间尺寸不符
        (torch.randn(2, 3, 3, 3), x, w, 1, 0),      # dY 的 batch 不符
        (good, x, w, 0, 0),                          # stride 非正
        (good, x, w, 1, -1),                         # padding 负
        (good, x.unsqueeze(0), w, 1, 0),             # x 非四维
        (good, x, w.unsqueeze(0), 1, 0),             # weight 非四维
    ]
    for dy, xx, ww, s, p in bad:
        try:
            solve(dy, xx, ww, stride=s, padding=p)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
