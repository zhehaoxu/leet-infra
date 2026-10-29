import math

import torch

MODES = ('rne', 'rz', 'rna')


def _same(got, ref):
    """位级一致：NaN 位置必须同为 NaN，其余逐个 bit 相同"""
    gn = torch.isnan(got)
    rn = torch.isnan(ref)
    assert torch.equal(gn, rn)
    m = ~gn
    assert torch.equal(got[m], ref[m])


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    # bf16 在 [1,2) 上的网格步长是 2^-7，这四个数正好都是网格中点（ties）
    x = torch.tensor([1.00390625, 1.01171875, 1.01953125, 1.02734375])
    assert solve(x, 'rne').tolist() == [1.0, 1.015625, 1.015625, 1.03125]
    assert solve(x, 'rz').tolist() == [1.0, 1.0078125, 1.015625, 1.0234375]
    assert solve(x, 'rna').tolist() == [1.0078125, 1.015625, 1.0234375, 1.03125]
    assert solve(-x, 'rne').tolist() == [-1.0, -1.015625, -1.015625, -1.03125]
    assert solve(-x, 'rz').tolist() == [-1.0, -1.0078125, -1.015625, -1.0234375]


def check_matches_hw_cast(solve):
    torch.manual_seed(0)
    pool = [
        torch.randn(4000) * 100.0,
        torch.rand(4000) * 1e38,
        torch.rand(4000) * 1e-38,
        10.0 ** (-torch.rand(4000) * 20.0 - 25.0),
        torch.arange(-2000, 2000).float() * 1e30,
        torch.arange(200).float() * 1e-30,
    ]
    for x in pool:
        x = x.float()
        ref = x.to(torch.bfloat16).to(torch.float32)
        _same(solve(x, 'rne'), ref)
        # 输出必须还在 bf16 网格上（幂等）
        _same(solve(ref, 'rne'), ref)


def check_mode_inequalities(solve):
    torch.manual_seed(1)
    x = (torch.randn(5000) * 10 ** (torch.rand(5000) * 20 - 10)).float()
    rz = solve(x, 'rz')
    rne = solve(x, 'rne')
    rna = solve(x, 'rna')
    ax = x.abs()
    assert torch.all(rz.abs() <= ax)
    assert torch.all((rne - x).abs() <= (rz - x).abs() + 1e-30)
    # rna 的幅值不小于 rz（ties 远离零）
    assert torch.all(rna.abs() >= rz.abs() - 1e-30)
    for y in (rz, rne, rna):
        _same(solve(y, 'rne'), y)
        _same(solve(y, 'rz'), y)
    # 误差上界：半格 = 2^-8 * 2^exponent，相对误差不超过 2^-8
    ok = ax > 1e-30
    rel = ((rne - x).abs() / ax)[ok]
    assert float(rel.max()) < 2 ** -8 + 1e-9


def check_ties_alternate(solve):
    k = torch.arange(8).float()
    x = 1.0 + (2.0 * k + 1.0) * 2.0 ** -8
    assert solve(x, 'rz').tolist() == [(1.0 + i * 2.0 ** -7) for i in range(8)]
    assert solve(x, 'rna').tolist() == [(1.0 + (i + 1) * 2.0 ** -7) for i in range(8)]
    assert solve(x, 'rne').tolist() == [(1.0 + (i + (i % 2)) * 2.0 ** -7) for i in range(8)]
    # 非 ties：低位 > 0x8000 才进位，与奇偶无关
    y = 1.0 + (torch.arange(7).float() + 0.75) * 2.0 ** -7
    assert solve(y, 'rne').tolist() == [(1.0 + (i + 1) * 2.0 ** -7) for i in range(7)]
    z = 1.0 + (torch.arange(7).float() + 0.25) * 2.0 ** -7
    assert solve(z, 'rne').tolist() == [(1.0 + i * 2.0 ** -7) for i in range(7)]


def check_special_values(solve):
    for mode in MODES:
        z = torch.tensor([0.0, -0.0])
        out = solve(z, mode)
        assert out.tolist() == [0.0, -0.0] or (out[0] == 0 and out[1] == 0)
        assert math.copysign(1.0, float(out[0])) > 0
        assert math.copysign(1.0, float(out[1])) < 0
        i = torch.tensor([float('inf'), float('-inf')])
        assert solve(i, mode).tolist() == [float('inf'), float('-inf')]
        # 1e30 不在 bf16 网格上，会真的被舍动；只有 inf 是原样返回的
        g = torch.tensor([1e30, -1e30])
        assert solve(g, 'rne').tolist() != [1e30, -1e30]
        _same(solve(g, 'rne'), g.to(torch.bfloat16).to(torch.float32))
        # f32 最大值：rne/rna 溢出成 inf，rz 截在 bf16 最大有限值
        mx = torch.tensor([torch.finfo(torch.float32).max])
        assert math.isinf(float(solve(mx, 'rne')[0]))
        assert math.isinf(float(solve(mx, 'rna')[0]))
        assert float(solve(mx, 'rz')[0]) == 3.3895313892515355e+38
        # NaN 不能被舍成 inf：含 signaling NaN 的模式 0x7F800001
        # 0x7F800001 / 0xFF800001 是两个 signaling NaN 模式，后者作为 int32 是 -8388607
        raw = torch.tensor([0x7F800001, -8388607], dtype=torch.int32).view(torch.float32)
        assert torch.isnan(raw).tolist() == [True, True]
        out = solve(raw, mode)
        assert math.isnan(float(out[0])) and math.isnan(float(out[1]))
        assert math.copysign(1.0, float(out[0])) > 0
        assert math.copysign(1.0, float(out[1])) < 0
        # bf16 次正规（指数全 0）：网格均匀，舍入仍成立
        sub = torch.tensor([1.4e-45, -2.8e-45, 5e-40])
        _same(solve(sub, 'rne'), sub.to(torch.bfloat16).to(torch.float32))


def check_invalid(solve):
    x = torch.randn(4)
    for bad in (x.double(), x.half(), torch.tensor([1, 2], dtype=torch.int32), [1.0], None):
        try:
            solve(bad, 'rne')
            raise AssertionError('non-float32 input must raise')
        except ValueError:
            pass
    for mode in ('rna ', 'RNA', 'trunc', 'stochastic', '', None, 0):
        try:
            solve(x, mode)
            raise AssertionError('bad rounding must raise')
        except ValueError:
            pass
    # 默认口径必须是 rne
    _same(solve(x), solve(x, 'rne'))
    # 形状与非连续输入不被破坏
    m = torch.randn(2, 3, 4)
    for mode in MODES:
        assert solve(m.transpose(0, 2), mode).shape == torch.Size([4, 3, 2])
        assert solve(torch.tensor(1.00390625), mode).dim() == 0
        assert solve(m, mode).shape == m.shape


check_example.example = True
