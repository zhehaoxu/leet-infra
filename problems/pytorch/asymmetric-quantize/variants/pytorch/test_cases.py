import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_example(solve):
    x = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    out = solve(x)
    expected = torch.tensor([-1.0, 0.0, 0.49412, 2.0])
    assert torch.allclose(out, expected, atol=1e-4), f"got {out.tolist()}"


def check_reconstruct_range(solve):
    # 反量化结果应落在原张量 [min, max] 内（合法格点不会越出量程）
    x = torch.tensor([-3.0, 0.2, 7.0, -1.5])
    out = solve(x)
    assert out.min().item() >= float(x.min()) - 1e-4
    assert out.max().item() <= float(x.max()) + 1e-4


def check_matches_fake_quant(solve):
    torch.manual_seed(42)
    for _ in range(50):
        x = torch.randn(16) * 2.0 - 0.5
        xmin = float(x.min())
        xmax = float(x.max())
        scale = (xmax - xmin) / 255.0
        zp = max(0, min(255, int(round(-xmin / scale))))
        ref = torch.fake_quantize_per_tensor_affine(x, scale, zp, 0, 255)
        assert torch.allclose(solve(x), ref, atol=1e-4)


def check_error_bound(solve):
    torch.manual_seed(42)
    x = torch.randn(512) * 2.0 - 0.5
    scale = (float(x.max()) - float(x.min())) / 255.0
    err = (solve(x) - x).abs().max().item()
    assert err <= scale / 2 + 1e-4, f"往返误差 {err} 超过半格 {scale/2}"


def check_constant_input(solve):
    out = solve(torch.full((5,), 3.0))
    assert torch.isfinite(out).all() and torch.allclose(out, torch.full((5,), 3.0))


def check_dtype(solve):
    torch.manual_seed(42)
    x = torch.randn(8, dtype=torch.float64)
    assert solve(x).dtype == torch.float64


check_example.example = True
check_reconstruct_range.example = True
