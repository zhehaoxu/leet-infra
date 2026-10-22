import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    # d=2 只有一个频率 theta_0 = 1：位置 1 的 (1,0) 旋到 (cos1, sin1)
    x = torch.tensor([[[1.0, 0.0]]])
    pos = torch.tensor([1])
    y = solve(x, pos)
    expected = torch.tensor([[[0.54030231, 0.84147098]]])
    assert torch.allclose(y, expected, atol=1e-6), f"got {y.tolist()}, expected {expected.tolist()}"


def _manual(x, pos, base=10000.0):
    # 复数实现：相邻两维视作复数，乘 exp(i * pos * theta_i)
    d = x.shape[-1]
    half = d // 2
    theta = torch.tensor([base ** (-2.0 * i / d) for i in range(half)], dtype=x.dtype)
    ang = pos.to(x.dtype)[:, None] * theta[None, :]                  # (L, half)
    rot = torch.complex(torch.cos(ang), torch.sin(ang))              # (L, half) 复数
    xc = torch.view_as_complex(x.reshape(*x.shape[:-1], half, 2).contiguous())
    return torch.view_as_real(xc * rot).flatten(-2, -1)


def check_matches_complex_rotation(solve):
    torch.manual_seed(42)
    x = torch.randn(6, 8)
    pos = torch.arange(6)
    y = solve(x, pos)
    assert torch.allclose(y, _manual(x, pos), atol=1e-5), "与复数旋转实现不一致"
    x3 = torch.randn(2, 5, 8)
    y3 = solve(x3, torch.arange(5))
    assert torch.allclose(y3, _manual(x3, torch.arange(5)), atol=1e-5), "批量维不一致"


def check_position_zero_identity(solve):
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    y = solve(x, torch.zeros(4, dtype=torch.long))
    assert torch.allclose(y, x, atol=1e-6), "位置 0 不应旋转"


def check_norm_preserved(solve):
    # 旋转是正交变换：每对 (2i, 2i+1) 的模长不变，整体范数不变
    torch.manual_seed(1)
    x = torch.randn(3, 10, 32)
    pos = torch.arange(10)
    y = solve(x, pos)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5), "范数应保持"


def check_relative_position_property(solve):
    # RoPE 核心性质：q 在位置 p1、k 在位置 p2 的点积只依赖 p1-p2
    torch.manual_seed(2)
    d = 16
    q = torch.randn(1, d)
    k = torch.randn(1, d)
    attn12 = (solve(q, torch.tensor([3])) * solve(k, torch.tensor([7]))).sum()
    attn04 = (solve(q, torch.tensor([0])) * solve(k, torch.tensor([4]))).sum()
    assert torch.allclose(attn12, attn04, atol=1e-4), "相对位置性质错误"
    attn23 = (solve(q, torch.tensor([3])) * solve(k, torch.tensor([8]))).sum()
    assert not torch.allclose(attn12, attn23, atol=1e-4), "不同相对距离应给出不同点积"


def check_float64(solve):
    x = torch.randn(4, 8, dtype=torch.float64)
    pos = torch.arange(4)
    y = solve(x, pos)
    assert y.dtype == torch.float64
    assert torch.allclose(y, _manual(x, pos), atol=1e-12)


check_basic.example = True
