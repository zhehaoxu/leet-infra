import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    w = torch.ones(1, 1, 2, 2)
    y = solve(x, w, None, stride=1, padding=0)
    exp = torch.tensor([[[[1.0, 3.0, 2.0], [4.0, 10.0, 6.0], [3.0, 7.0, 4.0]]]])
    assert torch.allclose(y, exp), f"expected {exp.tolist()}, got {y.tolist()}"


def check_stride_padding(solve):
    torch.manual_seed(0)
    x = torch.randn(1, 2, 3, 3)
    w = torch.randn(2, 3, 2, 2)  # (Cin, Cout, kh, kw)
    b = torch.randn(3)
    for st in (1, 2):
        for pd in (0, 1):
            got = solve(x, w, b, stride=st, padding=pd)
            exp = F.conv_transpose2d(x, w, b, stride=st, padding=pd)
            assert torch.allclose(got, exp, atol=1e-5), f"stride={st} pad={pd}"


def check_out_shape(solve):
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4, 4)
    w = torch.randn(3, 5, 3, 3)
    st, pd = 2, 1
    y = solve(x, w, None, stride=st, padding=pd)
    oh = (4 - 1) * st - 2 * pd + 3
    assert y.shape == (2, 5, oh, oh), f"形状错误: {y.shape} vs (2,5,{oh},{oh})"


def check_bias_none(solve):
    torch.manual_seed(2)
    x = torch.randn(1, 1, 3, 3)
    w = torch.randn(1, 2, 2, 2)
    b = torch.randn(2)
    y0 = solve(x, w, None, 1, 0)
    yb = solve(x, w, b, 1, 0)
    assert torch.allclose(yb, y0 + b.view(1, 2, 1, 1), atol=1e-5), "bias 应逐输出通道相加"


check_basic.example = True
