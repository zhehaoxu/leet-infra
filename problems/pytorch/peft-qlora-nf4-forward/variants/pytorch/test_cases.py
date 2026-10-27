import torch

NF4 = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
       0.0795, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7226, 1.0]


def _oracle(codes, absmax, x, A, B, alpha, r):
    """独立重算：纯 python 查表反量化 + 逐标量点积（结构异于参考解的矩阵乘路径）。"""
    d_out, d_in = codes.shape
    W = [[NF4[int(codes[i, j])] * float(absmax[i, j // 64]) for j in range(d_in)]
         for i in range(d_out)]
    y = []
    for s in range(x.shape[0]):
        row = []
        for o in range(d_out):
            base = sum(float(x[s, j]) * W[o][j] for j in range(d_in))
            lora = 0.0
            for t in range(r):
                hid = sum(float(x[s, j]) * float(A[t, j]) for j in range(d_in))
                lora += hid * float(B[o, t])
            row.append(base + alpha / r * lora)
        y.append(row)
    n = d_out * d_in
    n_blocks = n // 64
    nbytes = n // 2 + n_blocks + ((n_blocks + 255) // 256) * 4
    return (torch.tensor(y, dtype=torch.float64).to(torch.float32), int(nbytes))


def check_example(solve):
    torch.manual_seed(2)
    codes = torch.full((1, 64), 7, dtype=torch.long)
    codes[0, 0] = 0      # NF4[0] = -1
    codes[0, 32] = 8     # NF4[8] = 0.0795
    codes[0, 63] = 15    # NF4[15] = 1
    absmax = torch.tensor([[2.0]])
    x = torch.zeros(1, 64)
    x[0, 0] = 1.0
    x[0, 32] = 1.0
    x[0, 63] = 1.0
    A = torch.zeros(1, 64)
    B = torch.ones(1, 1)
    y, nbytes = solve(codes, absmax, x, A, B, 0.0, 1)
    assert torch.allclose(y, torch.tensor([[0.159]]), atol=1e-5, rtol=1e-5), (
        f"got {y.tolist()}")          # 2*(-1) + 2*0.0795 + 2*1 = 0.159
    assert nbytes == 37, f"got {nbytes}"   # 32 + 1 + 4


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(41)
    d_out, d_in, n_sam, r = 4, 128, 3, 4
    codes = torch.randint(0, 16, (d_out, d_in))
    absmax = torch.rand(d_out, d_in // 64) + 0.5
    x = torch.randn(n_sam, d_in)
    A = torch.randn(r, d_in)
    B = torch.randn(d_out, r)
    y, nbytes = solve(codes, absmax, x, A, B, 8.0, r)
    exp_y, exp_n = _oracle(codes, absmax, x, A, B, 8.0, r)
    assert torch.allclose(y, exp_y, atol=1e-5, rtol=1e-5), "y 与独立重算不符"
    assert nbytes == exp_n, f"nbytes 应为 {exp_n}，got {nbytes}"


def check_nf4_code_values(solve):
    """码值锚点：0/7/15 -> -absmax/0/+absmax，中间码 8/10 -> 0.0795/0.2461。"""
    A = torch.zeros(1, 64)
    B = torch.ones(1, 1)
    x = torch.zeros(1, 64)
    x[0, 10] = 1.0
    codes0 = torch.full((1, 64), 7, dtype=torch.long)
    codes0[0, 10] = 0
    y, _ = solve(codes0, torch.ones(1, 1), x, A, B, 0.0, 1)
    assert torch.allclose(y, torch.tensor([[-1.0]]), atol=1e-5, rtol=1e-5), f"got {y.tolist()}"
    codes15 = torch.full((1, 64), 7, dtype=torch.long)
    codes15[0, 10] = 15
    y, _ = solve(codes15, torch.ones(1, 1), x, A, B, 0.0, 1)
    assert torch.allclose(y, torch.tensor([[1.0]]), atol=1e-5, rtol=1e-5), f"got {y.tolist()}"
    codes8 = torch.full((1, 64), 7, dtype=torch.long)
    codes8[0, 10] = 8
    y, _ = solve(codes8, torch.ones(1, 1), x, A, B, 0.0, 1)
    assert torch.allclose(y, torch.tensor([[0.0795]]), atol=1e-5, rtol=1e-5), f"got {y.tolist()}"
    codes10 = torch.full((1, 64), 7, dtype=torch.long)
    codes10[0, 10] = 10
    y, _ = solve(codes10, torch.ones(1, 1), x, A, B, 0.0, 1)
    assert torch.allclose(y, torch.tensor([[0.2461]]), atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


def check_bytes_account(solve):
    """nbytes = n/2 + n/64 + ceil(n/64/256)*4：三个尺寸 37 / 136 / 1324（x/A/B 按 d_in 现造）。"""
    for d_out, d_in, exp in ((1, 64, 37), (2, 128, 136), (4, 640, 1324)):
        codes = torch.full((d_out, d_in), 7, dtype=torch.long)
        absmax = torch.ones(d_out, d_in // 64)
        x = torch.zeros(1, d_in)
        A = torch.zeros(1, d_in)
        B = torch.zeros(d_out, 1)
        _, nbytes = solve(codes, absmax, x, A, B, 0.0, 1)
        assert nbytes == exp, f"({d_out},{d_in}) 应为 {exp} 字节，got {nbytes}"


def check_lora_branch_scales(solve):
    """量化基座冻结、LoRA 旁路加性：y(2a)-y(0) == 2*(y(a)-y(0))。"""
    torch.manual_seed(17)
    d_out, d_in, r = 2, 64, 3
    codes = torch.randint(0, 16, (d_out, d_in))
    absmax = torch.rand(d_out, d_in // 64) + 0.5
    x = torch.randn(2, d_in)
    A = torch.randn(r, d_in)
    B = torch.randn(d_out, r)
    y0, _ = solve(codes, absmax, x, A, B, 0.0, r)
    y1, _ = solve(codes, absmax, x, A, B, 4.0, r)
    y2, _ = solve(codes, absmax, x, A, B, 8.0, r)
    lhs = y2 - y0
    rhs = 2.0 * (y1 - y0)
    assert torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5), (
        "alpha 翻倍时旁路增量应严格翻倍")
