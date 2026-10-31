import numpy as np
import torch

KEYS = {"u", "coef", "recon", "err"}
TOL = {torch.float32: 3e-5, torch.float64: 1e-9}


def _oracle(k, gs, rank_k):
    """numpy 独立重算：层拼通道 → 精确 SVD → 截断重建 → 逐组 Frobenius 残差（只比重建与残差，不比 u 的列）。"""
    kn = k.detach().numpy().astype(np.float64)
    layers, seq, heads, dim = kn.shape
    rank, recon, err = min(rank_k, seq, gs * heads * dim), [], []
    for g in range(layers // gs):
        mat = kn[g * gs:(g + 1) * gs].transpose(1, 0, 2, 3).reshape(seq, gs * heads * dim)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        rec = (u[:, :rank] * s[:rank]) @ vh[:rank]
        recon.append(rec.reshape(seq, gs, heads, dim).transpose(1, 0, 2, 3))
        err.append(np.linalg.norm(mat - rec))
    return np.concatenate(recon), np.asarray(err)


def _close(a, b, tol):
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _verify(solve, k, gs, rank_k):
    """结构 + gauge + numpy oracle 三件套。"""
    before, got = k.clone(), solve(k, gs, rank_k)
    tol, (layers, seq, heads, dim) = TOL[k.dtype], k.shape
    rank, groups = min(rank_k, seq, gs * heads * dim), layers // gs
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert torch.equal(k, before), "k 被就地修改了"
    assert got["u"].shape == (groups, seq, rank), got["u"].shape
    assert got["coef"].shape == (layers, heads, rank, dim), got["coef"].shape
    assert got["recon"].shape == (layers, seq, heads, dim), got["recon"].shape
    assert got["err"].shape == (groups,), got["err"].shape
    for key in KEYS:
        assert got[key].dtype == k.dtype and torch.isfinite(got[key]).all(), f"{key} dtype/数值不对"
    for g in range(groups):
        u = got["u"][g].double()
        _close(u.T @ u, torch.eye(rank, dtype=torch.float64), tol)      # 共享基必须正交归一
        absu = u.abs()
        pivot = (absu == absu.max(dim=0, keepdim=True).values).double().argmax(dim=0)
        assert (u[pivot, torch.arange(rank)] > 0).all(), f"组 {g} 的 u 符号约定没落地"
    for layer in range(layers):                    # gauge：recon 必须就是 u @ coef
        coef = got["coef"][layer].double().permute(1, 0, 2).reshape(rank, -1)
        _close(got["recon"][layer].double().reshape(seq, -1),
               got["u"][layer // gs].double() @ coef, tol)
    want_recon, want_err = _oracle(k, gs, rank_k)
    _close(got["recon"], torch.tensor(want_recon, dtype=k.dtype), tol)
    _close(got["err"], torch.tensor(want_err, dtype=k.dtype), max(tol, 1e-7))
    return got


def check_example(solve):
    """题面示例：两层各占一个正交方向，rank_k=1 只留强的那个，误差 = 被丢掉的奇异值 1。"""
    k = torch.tensor([[[[2.0, 0.0]], [[0.0, 0.0]]],
                      [[[0.0, 0.0]], [[0.0, 1.0]]]])          # (L=2, S=2, H=1, D=2)
    got = _verify(solve, k, 2, 1)
    assert got["u"].tolist() == [[[1.0], [0.0]]], got["u"].tolist()
    assert got["coef"].tolist() == [[[[2.0, 0.0]]], [[[0.0, 0.0]]]], got["coef"].tolist()
    assert got["recon"][1].abs().sum() == 0.0, "第 1 层没分到秩预算，重建必须是 0"
    assert abs(got["err"][0].item() - 1.0) < 1e-12, got["err"].tolist()
    assert solve(k, 2, 2)["err"].abs().max().item() < 1e-12, "秩 2 时整组应被精确表达"


check_example.example = True


def check_rank_covers_concatenated_width(solve):
    """秩按拼接后的宽度算：rank_k=3 > H*D=2 也用得上；超过 min(S, gs*H*D) 则截断到上限。"""
    torch.manual_seed(1)
    k = torch.randn(2, 3, 1, 2, dtype=torch.float64)             # S=3, C=2, W=4
    got = _verify(solve, k, 2, 3)
    assert got["coef"].shape[2] == 3 and got["err"][0].item() < 1e-12, "3x4 秩 3 应精确重建"
    cut = solve(k, 2, 2)
    assert cut["coef"].shape[2] == 2 and cut["err"][0].item() > 1e-2, "秩 2 截断后必须有残差"
    clamped = _verify(solve, torch.randn(4, 2, 1, 3, dtype=torch.float64), 2, 99)
    assert clamped["u"].shape == (2, 2, 2) and clamped["err"].abs().max().item() < 1e-12


def check_single_position_pins_column_order(solve):
    """S=1 时 rank=1、u=[1]，S·V 退化成拼接后的那一行：coef 必须逐元素等于原 K。"""
    torch.manual_seed(2)
    k = torch.randn(2, 1, 2, 3, dtype=torch.float64)             # 层在前、层内 head-major
    got = _verify(solve, k, 2, 4)
    assert got["u"].tolist() == [[[1.0]]], f"单一位置时 u 必须是 [[1]]，got {got['u'].tolist()}"
    _close(got["coef"], k.permute(0, 2, 1, 3), 1e-9)


def check_groups_are_independent(solve):
    """每组各做一次 SVD：只改第 0 组，第 1 组的 u/coef/recon/err 必须逐位不变。"""
    torch.manual_seed(3)
    k = torch.randn(4, 6, 1, 2, dtype=torch.float64)
    a, k2 = _verify(solve, k, 2, 2), k.clone()
    k2[:2] += 3.0
    b = _verify(solve, k2, 2, 2)
    assert not torch.equal(a["u"][0], b["u"][0]), "第 0 组本该被改动"
    assert torch.equal(a["u"][1], b["u"][1]), "第 1 组的共享 U 串组了"
    assert torch.equal(a["coef"][2:], b["coef"][2:]) and torch.equal(a["err"][1], b["err"][1])


def check_identical_layers_share_coefficients(solve):
    """组内两层相同 → 拼接矩阵是 [B|B]，两块 S·V 逐元素相等、重建相等，残差是 sqrt(2)·尾巴。"""
    torch.manual_seed(4)
    base = torch.randn(5, 2, 3, dtype=torch.float64)
    got = _verify(solve, torch.stack([base, base]).clone(), 2, 2)
    _close(got["coef"][0], got["coef"][1], 1e-9)
    _close(got["recon"][0], got["recon"][1], 1e-9)
    s = np.linalg.svd(base.reshape(5, 6).numpy(), compute_uv=False)
    want = float(np.sqrt(2.0 * (s[2:] ** 2).sum()))
    assert abs(got["err"][0].item() - want) < 1e-9, (got["err"].tolist(), want)


def check_error_is_tail_energy(solve):
    """Eckart–Young：err 等于被截掉的奇异值平方和开根，且随 rank_k 单调不增。"""
    torch.manual_seed(5)
    k = torch.randn(4, 8, 2, 2, dtype=torch.float64)             # gs=2 → W=8
    s = np.linalg.svd(k[:2].permute(1, 0, 2, 3).reshape(8, 8).numpy(), compute_uv=False)
    ranks = (1, 3, 8)
    errs = [_verify(solve, k, 2, r)["err"][0].item() for r in ranks]
    want = [float(np.sqrt((s[r:] ** 2).sum())) for r in ranks]
    assert np.allclose(errs, want, rtol=0, atol=1e-9), (errs, want)
    assert errs[0] >= errs[1] >= errs[2] - 1e-12, f"残差必须随秩单调不增：{errs}"


def check_dtype_sweep(solve):
    """float32 输入四个键都是 float32，数值与 float64 路径在 1e-5 内一致。"""
    torch.manual_seed(6)
    k64 = torch.randn(4, 6, 2, 3, dtype=torch.float64)
    lo, hi = _verify(solve, k64.float(), 2, 2), solve(k64, 2, 2)
    for key in KEYS:
        assert lo[key].dtype == torch.float32, f"{key} 应为 float32"
        _close(lo[key].double(), hi[key], 3e-5)


def check_value_errors(solve):
    """层数不被层组整除（尾组永远等不到 SVD）、参数非法、形状/类型不对，一律 ValueError。"""
    good = torch.randn(4, 5, 2, 2, dtype=torch.float64)
    bad = [(good, 3, 2), (good, 0, 2), (good, 2, 0), (good, 2.0, 2), (good, 2, True),
           (good.reshape(8, 5, 2), 2, 2), (good.to(torch.int64), 2, 2),
           (torch.randn(4, 0, 2, 2), 2, 2)]
    for args in bad:
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"非法输入 {[getattr(a, 'shape', a) for a in args]} 应抛 ValueError")


def check_random_against_oracle(solve):
    """随机形状与 numpy oracle 全键对拍（gs=1/2/4、S<W、S>W），末尾一例是非连续视图。"""
    cases = [(1, 2, 9, 2, 3, 2), (2, 4, 6, 2, 2, 3), (4, 4, 5, 1, 4, 1),
             (2, 6, 12, 2, 3, 5), (4, 8, 4, 2, 2, 8), (2, 2, 3, 1, 2, 3)]
    for seed, (gs, layers, seq, heads, dim, rank) in enumerate(cases):
        torch.manual_seed(100 + seed)
        _verify(solve, torch.randn(layers, seq, heads, dim, dtype=torch.float64), gs, rank)
    k = torch.randn(4, 6, 2, 6, dtype=torch.float64)[..., ::2]    # 步长 2，非连续
    assert not k.is_contiguous()
    _verify(solve, k, 2, 3)
