import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([0.2, -1.0, 0.7, -0.3])
    out = solve(x)
    expected = torch.tensor([0.19685, -1.0, 0.70079, -0.29921])
    assert torch.allclose(out, expected, atol=1e-4), f"got {out.tolist()}"


def check_absmax_exact(solve):
    # 最大值命中边界格点，反量化应精确还原
    x = torch.tensor([3.0, -3.0, 1.0])
    out = solve(x)
    assert abs(out[0].item() - 3.0) < 1e-5
    assert abs(out[1].item() + 3.0) < 1e-5


def check_error_bound(solve):
    torch.manual_seed(42)
    x = torch.randn(512) * 2.5
    scale = float(x.abs().max()) / 127.0
    err = (solve(x) - x).abs().max().item()
    assert err <= scale / 2 + 1e-5, f"往返误差 {err} 超过半格 {scale/2}"


def check_matches_fake_quant(solve):
    torch.manual_seed(42)
    for _ in range(50):
        x = torch.randn(16) * (torch.rand(()).item() * 3 + 0.1)
        scale = float(x.abs().max()) / 127.0
        ref = torch.fake_quantize_per_tensor_affine(x, scale, 0, -127, 127)
        assert torch.allclose(solve(x), ref, atol=1e-5)


def check_zero_input(solve):
    out = solve(torch.zeros(5))
    assert torch.isfinite(out).all() and torch.equal(out, torch.zeros(5))


def check_dtype(solve):
    torch.manual_seed(42)
    x = torch.randn(8, dtype=torch.float64)
    assert solve(x).dtype == torch.float64


check_example.example = True
check_absmax_exact.example = True
