import torch
import torch.nn.functional as F

# 示例：4×4 递增矩阵做 2×2 max pooling（stride=2），dY = [[1,2],[3,4]]
# 四个窗口的最大值分别在 (1,1)=6、(1,3)=8、(3,1)=14、(3,3)=16 ⇒ 梯度只落这四格
EXAMPLE_X = [[[[1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0],
              [9.0, 10.0, 11.0, 12.0],
              [13.0, 14.0, 15.0, 16.0]]]]
EXAMPLE_DY = [[[[1.0, 2.0], [3.0, 4.0]]]]
EXAMPLE_DX = [[[[0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 2.0],
               [0.0, 0.0, 0.0, 0.0],
               [0.0, 3.0, 0.0, 4.0]]]]


def _t(v):
    return torch.tensor(v, dtype=torch.float64)


def check_example(solve):
    dx = solve(_t(EXAMPLE_X), _t(EXAMPLE_DY), mode='max', kernel=2)
    assert dx.shape == (1, 1, 4, 4)
    assert torch.allclose(dx, _t(EXAMPLE_DX), atol=1e-9)


check_example.example = True


def check_max_matches_autograd(solve):
    torch.manual_seed(1101)
    for k, s, p in ((2, 2, 0), (3, 1, 1), (2, 1, 0), (3, 2, 1)):
        x = torch.randn(2, 3, 8, 7, dtype=torch.float64, requires_grad=True)
        y = F.max_pool2d(x, k, stride=s, padding=p)
        dY = torch.randn_like(y)
        (ref,) = torch.autograd.grad((y * dY).sum(), x)
        got = solve(x.detach(), dY, mode='max', kernel=k, stride=s, padding=p)
        assert got.shape == x.shape, (k, s, p)
        assert torch.allclose(got, ref, atol=1e-9), (k, s, p)


def check_max_tie_and_sparsity(solve):
    # 并列最大值：按声明口径交给行主序最前的一格；且非零元素数恰等于窗口数
    x = torch.zeros(1, 1, 2, 4, dtype=torch.float64)
    dY = torch.tensor([[[[3.0, 5.0]]]], dtype=torch.float64)
    dx = solve(x, dY, mode='max', kernel=2, stride=2)
    assert dx[0, 0, 0, 0] == 3.0 and dx[0, 0, 0, 2] == 5.0
    assert int((dx != 0).sum()) == 2
    # 每格一个窗口时 max 退化为恒等
    x2 = torch.arange(6.0, dtype=torch.float64).reshape(1, 1, 2, 3)
    d2 = torch.ones(1, 1, 2, 3, dtype=torch.float64)
    assert torch.allclose(solve(x2, d2, mode='max', kernel=1), d2, atol=1e-9)


def check_avg_and_adaptive(solve):
    torch.manual_seed(1102)
    x = torch.randn(2, 2, 6, 6, dtype=torch.float64, requires_grad=True)
    y = F.avg_pool2d(x, 3)
    dY = torch.randn_like(y)
    (ref,) = torch.autograd.grad((y * dY).sum(), x)
    got = solve(x.detach(), dY, mode='avg', kernel=3)
    assert torch.allclose(got, ref, atol=1e-9)
    # adaptive_avg：窗口变长且可能重叠，除数必须是实际面积
    for size in ((5, 2), (6, 4), (7, 3), (4, 4)):
        xx = torch.randn(1, 2, size[0], size[0], dtype=torch.float64, requires_grad=True)
        yy = F.adaptive_avg_pool2d(xx, size[1])
        dd = torch.randn_like(yy)
        (r,) = torch.autograd.grad((yy * dd).sum(), xx)
        g = solve(xx.detach(), dd, mode='adaptive_avg')
        assert torch.allclose(g, r, atol=1e-9), size


def check_gradient_conservation(solve):
    # 分摊类反向的守恒式：Σdx == ΣdY（avg 无 padding、adaptive 全覆盖）
    torch.manual_seed(1103)
    x = torch.randn(1, 3, 8, 8, dtype=torch.float64)
    y = F.avg_pool2d(x, 2)
    dY = torch.randn_like(y)
    dx = solve(x, dY, mode='avg', kernel=2)
    assert torch.allclose(dx.sum(), dY.sum(), atol=1e-9)
    ya = F.adaptive_avg_pool2d(x, 3)
    da = torch.randn_like(ya)
    dxa = solve(x, da, mode='adaptive_avg')
    assert torch.allclose(dxa.sum(), da.sum(), atol=1e-9)
    # max 不守恒于元素和，但每窗口恰好一整份 dY 被放置
    ym = F.max_pool2d(x, 2)
    dm = torch.randn_like(ym)
    dxm = solve(x, dm, mode='max', kernel=2)
    assert torch.allclose(dxm.sum(), dm.sum(), atol=1e-9)
    assert int((dxm != 0).sum()) <= int(dm.numel())


def check_dtype_and_batch(solve):
    x = torch.randn(3, 2, 5, 5)
    dY = torch.randn(3, 2, 2, 2)
    dx = solve(x, dY, mode='max', kernel=2)
    assert dx.dtype == x.dtype and dx.shape == x.shape
    dx64 = solve(x.double(), dY.double(), mode='max', kernel=2)
    assert dx64.dtype == torch.float64
    assert torch.allclose(dx64.float(), dx, atol=1e-6)
    # 各 (b,c) 独立选 argmax
    xs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[-1.0, 5.0], [2.0, 0.0]]]])
    ds = torch.ones(2, 1, 1, 1)
    res = solve(xs, ds, mode='max', kernel=2)
    assert res[0, 0, 1, 1] == 1.0 and res[1, 0, 0, 1] == 1.0


def check_invalid_args(solve):
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    dY = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    bad = [
        (x, dY, 'min', 2, None, 0),                 # 未知 mode
        (x, dY, 'avg', 2, None, 1),                 # avg 不允许 padding
        (x, torch.randn(1, 1, 3, 3), 'max', 2, None, 0),   # 输出尺寸不符
        (x, torch.randn(1, 2, 2, 2), 'max', 2, None, 0),   # 通道不符
        (x.squeeze(0), dY, 'max', 2, None, 0),      # 非四维
        (x, dY, 'max', 2, 0, 0),                    # stride 非正
    ]
    for xx, dy, md, kk, st, pp in bad:
        try:
            solve(xx, dy, mode=md, kernel=kk, stride=st, padding=pp)
        except ValueError:
            continue
        raise AssertionError("非法入参应抛 ValueError")
