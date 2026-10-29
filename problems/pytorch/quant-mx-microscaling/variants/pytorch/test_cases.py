import torch


def _oracle(W):
    """逐块重算 E8M0 编码与归一化。"""
    M, N = W.shape
    assert N % 32 == 0
    codes = torch.zeros(M, N // 32, dtype=torch.int64)
    norm = torch.zeros_like(W, dtype=torch.float64)
    for i in range(M):
        for b in range(N // 32):
            block = W[i, b * 32:(b + 1) * 32]
            m = float(block.abs().max())
            if m == 0:
                code = 0
                scale = 0.0
            else:
                e = int(torch.ceil(torch.log2(torch.tensor(m, dtype=torch.float64))).item())
                code = max(1, min(255, e + 127))
                scale = 2.0 ** (code - 127)
            codes[i, b] = code
            norm[i, b * 32:(b + 1) * 32] = block.to(torch.float64) / scale if scale != 0 else 0.0
    return codes.to(torch.uint8), norm.to(W.dtype)


def check_example(solve):
    W = torch.full((1, 32), 0.5)
    W[0, 0] = 1.5
    W[0, 1] = -1.5
    codes, norm = solve(W)
    assert codes[0, 0].item() == 128, f"code got {codes[0, 0].item()}"
    assert abs(float(norm[0, 0]) - 0.75) < 1e-6
    assert abs(float(norm[0, 2]) - 0.25) < 1e-6


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(19)
    W = torch.randn(3, 64)
    codes, norm = solve(W)
    exp_c, exp_n = _oracle(W)
    assert torch.equal(codes, exp_c), "scale_codes 与逐块重算不符"
    assert torch.allclose(norm, exp_n, atol=1e-5, rtol=1e-5), "normalized 与逐块重算不符"


def check_zero_block(solve):
    W = torch.zeros(2, 32)
    codes, norm = solve(W)
    assert torch.equal(codes, torch.zeros(2, 1, dtype=torch.uint8))
    assert torch.equal(norm, torch.zeros_like(W))


def check_reconstruction(solve):
    """非零块应能通过 normalized * scale 精确还原 W。"""
    torch.manual_seed(23)
    W = torch.randn(4, 128)
    codes, norm = solve(W)
    scales = torch.zeros_like(codes, dtype=torch.float64)
    nonzero = codes.to(torch.int64) > 0
    scales[nonzero] = torch.exp2(codes[nonzero].to(torch.float64) - 127.0)
    blocks = norm.view(W.shape[0], W.shape[1] // 32, 32)
    rec = (blocks.to(torch.float64) * scales.unsqueeze(2)).to(W.dtype).view(W.shape)
    assert torch.allclose(rec, W, atol=1e-5, rtol=1e-5), "重建误差过大"


def check_scaling_linearity(solve):
    """W 乘 2 → code 加 1，normalized 不变。"""
    torch.manual_seed(29)
    W = torch.randn(2, 64).abs() + 0.1  # 保证非零
    c1, n1 = solve(W)
    c2, n2 = solve(W * 2.0)
    assert torch.equal(c2, (c1.to(torch.int64) + 1).to(torch.uint8)), "整体乘 2 应使 code +1"
    assert torch.allclose(n1, n2, atol=1e-5, rtol=1e-5), "normalized 在整体缩放时应不变"


def check_saturation(solve):
    """极大值应被 clip 到 code=255。"""
    W = torch.full((1, 32), torch.finfo(torch.float32).max)
    codes, _ = solve(W)
    assert codes[0, 0].item() == 255, f"极大值应饱和到 255，got {codes[0, 0].item()}"
