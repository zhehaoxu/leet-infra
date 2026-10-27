import numpy as np
import torch


def _oracle(V_keep, V_evict):
    """numpy lstsq 独立求最小二乘解（结构异于参考解的 torch.pinv 路径）。"""
    keep = V_keep.numpy().astype(np.float64)
    evict = V_evict.numpy().astype(np.float64)
    coef, *_ = np.linalg.lstsq(keep.T, evict.T, rcond=None)
    merged = keep + coef @ evict
    resid = float(np.linalg.norm(keep.T @ coef - evict.T))
    return (torch.tensor(merged, dtype=torch.float32),
            torch.tensor(coef, dtype=torch.float32),
            torch.tensor(resid, dtype=torch.float32))


def check_example(solve):
    torch.manual_seed(0)
    V_keep = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    V_evict = torch.tensor([[0.5, 0.25]])
    merged, coef, residual = solve(V_keep, V_evict)
    assert coef.tolist() == [[0.5], [0.25]], f"got {coef.tolist()}"
    assert torch.allclose(merged, torch.tensor([[1.25, 0.125], [0.125, 1.0625]]), atol=1e-5, rtol=1e-5), (
        f"got {merged.tolist()}")
    assert float(residual) < 1e-5, f"满基时残差应为 0，got {float(residual)}"


check_example.example = True


def check_exact_span_has_zero_residual(solve):
    """被驱逐值落在存活张成空间内 → 残差 0，且合并后等于 keep + 外积重建。"""
    torch.manual_seed(1)
    V_keep = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
    V_evict = torch.tensor([[2.0, 5.0, 1.0]])          # = 2*row0 + 1*row1
    merged, coef, residual = solve(V_keep, V_evict)
    assert torch.allclose(coef, torch.tensor([[2.0], [1.0]]), atol=1e-5), f"got {coef.tolist()}"
    assert float(residual) < 1e-5, f"落在张成空间内残差应为 0，got {float(residual)}"
    exp = V_keep + coef @ V_evict
    assert torch.allclose(merged, exp, atol=1e-5, rtol=1e-5)


def check_orthogonal_evicted_is_ignored(solve):
    """被子空间正交方向：系数为 0、合并权重不变、残差等于被驱逐值的范数。"""
    torch.manual_seed(2)
    V_keep = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    V_evict = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    merged, coef, residual = solve(V_keep, V_evict)
    assert torch.allclose(coef, torch.zeros(2, 1), atol=1e-6), f"got {coef.tolist()}"
    assert torch.allclose(merged, V_keep, atol=1e-6), "无法表达的方向不应改变存活值"
    assert abs(float(residual) - 5.0) < 1e-5, f"残差应为 5，got {float(residual)}"


def check_rank_deficient_keep(solve):
    """存活行线性相关时用伪逆取最小范数解：不抛错、残差 0、合并结果有限。"""
    torch.manual_seed(3)
    V_keep = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    V_evict = torch.tensor([[2.0, 2.0]])
    merged, coef, residual = solve(V_keep, V_evict)
    assert torch.isfinite(merged).all() and torch.isfinite(coef).all()
    assert float(residual) < 1e-5, f"可达时残差应为 0，got {float(residual)}"
    assert float(torch.linalg.matrix_norm(V_keep.T @ coef - V_evict.T)) < 1e-5


def check_matches_oracle_and_errors(solve):
    """随机用例与 numpy lstsq 对拍 + 线性性 + 非法形状。"""
    torch.manual_seed(21)
    for K, E, d in ((3, 2, 4), (5, 4, 3), (1, 3, 2)):
        V_keep = torch.randn(K, d)
        V_evict = torch.randn(E, d)
        exp = _oracle(V_keep, V_evict)
        got = solve(V_keep, V_evict)
        assert torch.allclose(got[0], exp[0], atol=1e-5, rtol=1e-5), f"{(K, E, d)} merged 不符"
        assert torch.allclose(got[1], exp[1], atol=1e-5, rtol=1e-5), f"{(K, E, d)} coef 不符"
        assert abs(float(got[2]) - float(exp[2])) < 1e-4, f"{(K, E, d)} 残差不符"
    m1, c1, _ = solve(V_keep, V_evict)
    m2, c2, _ = solve(V_keep, 2.0 * V_evict)
    assert torch.allclose(c2, 2.0 * c1, atol=1e-4, rtol=1e-5), "系数应对 V_evict 线性"
    assert torch.allclose(m2 - V_keep, 4.0 * (m1 - V_keep), atol=1e-4, rtol=1e-4), (
        "合并增量含 C@V_evict 的双线性，放大 2 倍后应放大 4 倍")
    for bad in ((torch.zeros(2, 3), torch.zeros(2, 4)), (torch.zeros(0, 3), torch.zeros(1, 3))):
        try:
            solve(*bad)
            raise AssertionError("非法形状应抛 ValueError")
        except ValueError:
            pass
