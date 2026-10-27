import torch
import torch.nn.functional as F

# 示例期望值：x = [[-2.0, -0.5, 0.0, 0.5, 2.0]]，四种 mode 各自的输出（实测回填）
EXAMPLE_X = [[-2.0, -0.5, 0.0, 0.5, 2.0]]
EXAMPLE_SOFTSIGN = [-0.6666666865, -0.3333333433, 0.0, 0.3333333433, 0.6666666865]
EXAMPLE_HARDTANH = [-1.0, -0.5, 0.0, 0.5, 1.0]
EXAMPLE_CELU = [-0.8646647334, -0.3934693336, 0.0, 0.5, 2.0]
EXAMPLE_LOGSIGMOID = [-2.1269280910, -0.9740769863, -0.6931471825,
                      -0.4740769863, -0.1269280165]


def check_example(solve):
    x = torch.tensor(EXAMPLE_X)
    assert torch.allclose(solve(x, mode="softsign"),
                          torch.tensor(EXAMPLE_SOFTSIGN), atol=1e-6)
    assert torch.allclose(solve(x, mode="hardtanh"),
                          torch.tensor(EXAMPLE_HARDTANH), atol=1e-6)
    assert torch.allclose(solve(x, mode="celu"),
                          torch.tensor(EXAMPLE_CELU), atol=1e-6)
    assert torch.allclose(solve(x, mode="logsigmoid"),
                          torch.tensor(EXAMPLE_LOGSIGMOID), atol=1e-6)


check_example.example = True


def check_softsign_matches_torch(solve):
    torch.manual_seed(11)
    x = torch.randn(32, 16) * 3.0
    # 传入非默认的 min/max/alpha：softsign 必须全部忽略
    y = solve(x, mode="softsign", min_value=0.0, max_value=0.5, alpha=2.0)
    assert torch.allclose(y, F.softsign(x), atol=1e-6)
    assert torch.all(y.abs() < 1.0), "softsign 输出必须严格落在 (-1, 1)"
    assert torch.all(y * x >= 0), "softsign 保号"
    assert y.shape == x.shape and y.dtype == x.dtype


def check_softsign_value_range_extremes(solve):
    x = torch.tensor([[-1e4, -100.0, -1.0, 0.0, 1.0, 100.0, 1e4]])
    y = solve(x, mode="softsign")
    assert torch.all(y.abs() < 1.0), y.tolist()
    assert torch.allclose(y, F.softsign(x), atol=1e-6)
    assert abs(y[0, 0].item()) > 0.999 and y[0, -1].item() < 1.0


def check_hardtanh_matches_torch_and_saturates(solve):
    torch.manual_seed(12)
    x = torch.randn(24, 8) * 5.0
    assert torch.allclose(solve(x, mode="hardtanh"), F.hardtanh(x), atol=1e-6)
    # 自定义边界 + alpha 应被忽略
    y = solve(x, mode="hardtanh", min_value=-0.5, max_value=2.0, alpha=3.0)
    assert torch.allclose(y, F.hardtanh(x, min_val=-0.5, max_val=2.0), atol=1e-6)
    # 饱和探针：越界值输出恰等于边界（要求精确相等，而非近似）
    xs = torch.tensor([-7.0, -1.0, 0.0, 1.0, 7.0])
    ys = solve(xs, mode="hardtanh")
    assert torch.equal(ys, torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])), ys.tolist()


def check_celu_matches_torch(solve):
    torch.manual_seed(13)
    x = torch.randn(64, 32)
    for alpha in (0.5, 1.0, 2.5):
        # 传入非默认 min/max：celu 必须忽略
        y = solve(x, mode="celu", alpha=alpha, min_value=-3.0, max_value=3.0)
        assert torch.allclose(y, F.celu(x, alpha=alpha), atol=1e-6), alpha
    x3 = torch.randn(2, 3, 4)  # 形状任意
    assert torch.allclose(solve(x3, mode="celu", alpha=2.0), F.celu(x3, alpha=2.0), atol=1e-6)
    assert y.dtype == x.dtype and y.shape == x.shape


def check_celu_big_alpha_tends_to_identity(solve):
    x = torch.tensor([-5.0, -3.0, -1.0, -0.5, -1e-3, 0.0, 0.5, 1.0, 3.0, 5.0])
    y = solve(x, mode="celu", alpha=1e6)
    # alpha 极大时 exp(x/alpha)-1 ≈ x/alpha，负支退化成 alpha·(x/alpha)=x，整体 ≈ 恒等
    assert torch.allclose(y, x, atol=1e-4), (y - x).tolist()
    assert torch.allclose(y, F.celu(x, alpha=1e6), atol=1e-6)


def check_logsigmoid_stable_form(solve):
    torch.manual_seed(14)
    x = torch.randn(48, 16)
    y = solve(x, mode="logsigmoid")
    assert torch.allclose(y, F.logsigmoid(x), atol=1e-6)
    # 大负值探针：fp32 sigmoid(-120) 下溢为 0，朴素 log(sigmoid) 会产生 -inf；
    # 稳定式必须有限且与 F.logsigmoid 一致
    x2 = torch.tensor([-120.0, -104.0, -50.0, 0.0, 50.0, 104.0, 120.0])
    y2 = solve(x2, mode="logsigmoid")
    assert torch.isfinite(y2).all(), y2.tolist()
    assert torch.allclose(y2, F.logsigmoid(x2), atol=1e-6), y2.tolist()
    # 注释性断言：朴素公式在 -120/-104 处确实非有限——这正是必须用稳定式的原因
    naive = torch.log(torch.sigmoid(x2))
    assert not torch.isfinite(naive[0]) and not torch.isfinite(naive[1])


def check_float64_all_modes(solve):
    x = torch.randn(16, 8, dtype=torch.float64) * 3.0
    assert torch.allclose(solve(x, mode="softsign"), F.softsign(x), atol=1e-9, rtol=1e-9)
    assert torch.allclose(solve(x, mode="hardtanh"), F.hardtanh(x), atol=1e-9, rtol=1e-9)
    assert torch.allclose(solve(x, mode="celu"), F.celu(x), atol=1e-9, rtol=1e-9)
    assert torch.allclose(solve(x, mode="logsigmoid"), F.logsigmoid(x), atol=1e-9, rtol=1e-9)
    assert solve(x, mode="softsign").dtype == torch.float64


def check_invalid_mode(solve):
    try:
        solve(torch.zeros(3), mode="gelu")
    except ValueError:
        return
    raise AssertionError("非法 mode 应抛 ValueError")
