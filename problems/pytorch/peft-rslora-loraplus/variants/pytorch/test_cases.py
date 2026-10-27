import torch
import math

A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])   # (4,2) r=4
B = torch.tensor([[1.0, 1.0, 1.0, 1.0]])                             # (1,4)
GA = torch.zeros(4, 2)
GA[0, 0] = 1.0
GB = torch.zeros(1, 4)
GB[0, 1] = 1.0


def _oracle(A, B, gA, gB, lr, lam, alpha, r):
    """独立重算：纯 python 更新因子并逐标量乘加（结构异于参考解的矩阵乘路径）。"""
    d_out, d_in = B.shape[0], A.shape[1]
    A2 = [[float(A[i, j]) - lr * float(gA[i, j]) for j in range(d_in)] for i in range(r)]
    B2 = [[float(B[i, t]) - lam * lr * float(gB[i, t]) for t in range(r)] for i in range(d_out)]
    out = []
    for s in (alpha / math.sqrt(r), alpha / r):
        m = [[sum(B2[i][t] * A2[t][j] for t in range(r)) * s for j in range(d_in)]
             for i in range(d_out)]
        out.append(torch.tensor(m, dtype=torch.float64).to(torch.float32))
    return tuple(out)


def check_example(solve):
    drs, dva = solve(A, B, GA, GB, 0.1, 16.0, 8.0, 4)
    assert torch.allclose(drs, torch.tensor([[15.6, 1.6]]), atol=1e-5, rtol=1e-5), (
        f"got {drs.tolist()}")
    assert torch.allclose(dva, torch.tensor([[7.8, 0.8]]), atol=1e-5, rtol=1e-5), (
        f"got {dva.tolist()}")


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(31)
    r, d_in, d_out = 3, 5, 4
    a, b = torch.randn(r, d_in), torch.randn(d_out, r)
    ga, gb = torch.randn(r, d_in), torch.randn(d_out, r)
    drs, dva = solve(a, b, ga, gb, 0.05, 16.0, 12.0, r)
    e1, e2 = _oracle(a, b, ga, gb, 0.05, 16.0, 12.0, r)
    assert torch.allclose(drs, e1, atol=1e-5, rtol=1e-5), "rsLoRA 缩放与独立重算不符"
    assert torch.allclose(dva, e2, atol=1e-5, rtol=1e-5), "普通缩放与独立重算不符"


def check_ratio_sqrt_r(solve):
    torch.manual_seed(7)
    r = 9
    a, b = torch.randn(r, 4), torch.randn(3, r)
    ga, gb = torch.randn(r, 4), torch.randn(3, r)
    drs, dva = solve(a, b, ga, gb, 0.02, 8.0, 6.0, r)
    mask = dva.abs() > 1e-3
    assert bool(mask.any()), "vanilla 输出不应全零"
    ratio = drs[mask] / dva[mask]
    assert torch.allclose(ratio, torch.full_like(ratio, math.sqrt(r)), atol=1e-4, rtol=1e-4), (
        f"两输出元素比应为 sqrt(r)={math.sqrt(r)}")


def check_zero_grad_no_change(solve):
    """梯度为零时没有更新：输出应等于缩放后的原始 BA 乘积。"""
    torch.manual_seed(3)
    r = 2
    a, b = torch.randn(r, 5), torch.randn(4, r)
    za, zb = torch.zeros_like(a), torch.zeros_like(b)
    drs, dva = solve(a, b, za, zb, 0.1, 16.0, 8.0, r)
    base = b.to(torch.float64) @ a.to(torch.float64)
    assert torch.allclose(drs, (base * (8.0 / math.sqrt(r))).to(torch.float32),
                          atol=1e-5, rtol=1e-5), "零梯度下 rs 输出应为缩放的 BA"
    assert torch.allclose(dva, (base * (8.0 / r)).to(torch.float32),
                          atol=1e-5, rtol=1e-5), "零梯度下 vanilla 输出应为缩放的 BA"


def check_a_side_unaffected_by_lam(solve):
    """λ 只进 B 的更新式：gB=0 时换 λ 输出逐位不变（LoRA+ 的不对称性）。"""
    torch.manual_seed(11)
    r = 3
    a, b = torch.randn(r, 4), torch.randn(2, r)
    ga = torch.randn(r, 4)
    gb = torch.zeros_like(b)
    r1 = solve(a, b, ga, gb, 0.1, 1.0, 8.0, r)
    r16 = solve(a, b, ga, gb, 0.1, 16.0, 8.0, r)
    assert torch.equal(r1[0], r16[0]) and torch.equal(r1[1], r16[1]), (
        "gB=0 时 lambda 不应影响任何输出")
