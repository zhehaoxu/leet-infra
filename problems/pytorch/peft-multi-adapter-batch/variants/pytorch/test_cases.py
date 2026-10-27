import torch

W0 = torch.eye(2)                                        # (2,2)
A = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])           # (2,1,2)
B = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])       # (2,2,1)
SCALES = torch.tensor([2.0, 3.0])
IDS = torch.tensor([0, 1, 0])
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # (3,2)


def _oracle(W0, A, B, scales, ids, x):
    """独立重算：逐样本逐标量（结构异于参考解的分组矩阵乘路径）。"""
    d_out, d_in = W0.shape
    K, r, _ = A.shape
    y = []
    for s in range(x.shape[0]):
        k = int(ids[s])
        row = []
        for i in range(d_out):
            base = sum(float(x[s, j]) * float(W0[i, j]) for j in range(d_in))
            hid = [sum(float(x[s, j]) * float(A[k, t, j]) for j in range(d_in))
                   for t in range(r)]
            dl = sum(hid[t] * float(B[k, i, t]) for t in range(r))
            row.append(base + float(scales[k]) * dl)
        y.append(row)
    return torch.tensor(y, dtype=torch.float64).to(torch.float32)


def check_example(solve):
    y = solve(W0, A, B, SCALES, IDS, X)
    exp = torch.tensor([[3.0, 0.0], [0.0, 4.0], [3.0, 1.0]])
    assert torch.allclose(y, exp, atol=1e-5, rtol=1e-5), f"got {y.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(171)
    d_in, d_out, r, K, N = 5, 4, 2, 3, 7
    w0 = torch.randn(d_out, d_in)
    a = torch.randn(K, r, d_in)
    b = torch.randn(K, d_out, r)
    scales = torch.randn(K)
    ids = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    x = torch.randn(N, d_in)
    y = solve(w0, a, b, scales, ids, x)
    assert torch.allclose(y, _oracle(w0, a, b, scales, ids, x), atol=1e-5, rtol=1e-5), (
        "y 与逐样本独立重算不符")


def check_zero_scales_identity(solve):
    torch.manual_seed(181)
    d_in, d_out, r, K, N = 4, 3, 2, 2, 5
    w0 = torch.randn(d_out, d_in)
    a = torch.randn(K, r, d_in)
    b = torch.randn(K, d_out, r)
    ids = torch.randint(0, K, (N,))
    x = torch.randn(N, d_in)
    y = solve(w0, a, b, torch.zeros(K), ids, x)
    base = (x.to(torch.float64) @ w0.to(torch.float64).T).to(torch.float32)
    assert torch.allclose(y, base, atol=1e-6, rtol=1e-6), "全零缩放应精确退化为基座输出"


def check_single_adapter_consistency(solve):
    torch.manual_seed(191)
    d_in, d_out, r = 4, 3, 2
    w0 = torch.randn(d_out, d_in)
    a = torch.randn(2, r, d_in)
    b = torch.randn(2, d_out, r)
    scales = torch.randn(2)
    x = torch.randn(4, d_in)
    ids = torch.zeros(4, dtype=torch.long)
    y = solve(w0, a, b, scales, ids, x)
    base = x.to(torch.float64) @ w0.to(torch.float64).T
    lora0 = (x.to(torch.float64) @ a[0].to(torch.float64).T) @ b[0].to(torch.float64).T
    exp = base + scales[0].to(torch.float64) * lora0
    assert torch.allclose(y, exp.to(torch.float32), atol=1e-5, rtol=1e-5), (
        "ids 全同时应与单 adapter 前向一致")


def check_scale_doubling_scope(solve):
    torch.manual_seed(201)
    d_in, d_out, r, K, N = 4, 3, 2, 3, 7
    w0 = torch.randn(d_out, d_in)
    a = torch.randn(K, r, d_in)
    b = torch.randn(K, d_out, r)
    scales = torch.randn(K)
    ids = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    x = torch.randn(N, d_in)
    y0 = solve(w0, a, b, scales, ids, x)
    scales2 = scales.clone()
    scales2[1] = 2 * scales[1]
    y1 = solve(w0, a, b, scales2, ids, x)
    for i in range(N):
        if int(ids[i]) == 1:
            continue
        assert torch.allclose(y1[i], y0[i], atol=1e-6, rtol=1e-6), (
            f"行 {i}（id={int(ids[i])}）不应受 adapter 1 缩放影响")
