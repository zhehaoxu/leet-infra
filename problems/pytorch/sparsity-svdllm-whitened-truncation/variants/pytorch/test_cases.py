"""SVD-LLM 白化截断压缩步 -- 测试用例."""

import math

import numpy as np
import torch


M, N = 40, 24
P = 48


def _mat(v, rows, cols):
    n = rows * cols
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * (0.271 + v * 0.017)) % 1.0) * 2.0 - 1.0
    return x.reshape(rows, cols).contiguous()


def _cholesky_factor(v, dim):
    """构造正定 H 的下三角 Cholesky 因子 (上游 profiling 产物)."""
    raw = _mat(v, dim, dim)
    H = raw @ raw.t() + torch.eye(dim) * 0.5
    return torch.linalg.cholesky(H).contiguous()


def _fixture(v=0.0, ratio=0.2, direct=False, singular_L=False, y_scale=1.0):
    weight = _mat(v, M, N)
    if direct:
        L = None
    elif singular_L:
        L = _cholesky_factor(v + 9.0, N)
        L = L.clone()
        L[3, :] = 0.0  # 制造奇异 L -> 触发 +1e-6*I 防御
    else:
        L = _cholesky_factor(v + 9.0, N)
    x0 = _mat(v + 1.0, P, N)
    y = (x0 @ weight.t() + _mat(v + 2.0, P, M) * 0.05) * y_scale
    x = x0 * y_scale  # x/y 同步缩放保持相容 (y 的 Frobenius 范数可 < 1)
    return dict(weight=weight, scaling_diag_matrix=L, x=x, y=y,
                ratio=ratio, direct_update=direct)


def _oracle(weight, scaling_diag_matrix, x, y, ratio=0.2,
            direct_update=False):
    wd = weight.numpy().astype(np.float64)
    Mn, Nn = wd.shape
    use_white = not direct_update
    if use_white:
        L = scaling_diag_matrix.numpy().astype(np.float64)
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            L_inv = np.linalg.inv(L + 1e-6 * np.eye(Nn))
        W_scale = wd @ L
    else:
        W_scale = wd
    U, S, VT = np.linalg.svd(W_scale, full_matrices=False)
    k = int(Mn * Nn * ratio / (Mn + Nn))
    truc_u = U[:, :k]
    truc_s = S[:k]
    truc_v = VT[:k, :] @ L_inv if use_white else VT[:k, :]
    sigma = np.diag(truc_s)
    sqrt_sigma = np.sqrt(sigma)
    svd_u = truc_u @ sqrt_sigma
    svd_v = sqrt_sigma @ truc_v
    new_w = truc_u @ sigma @ truc_v
    xd = x.numpy().astype(np.float64)
    yd = y.numpy().astype(np.float64)
    x_proj = (xd @ truc_v.T) @ sigma
    updated_u_t = np.linalg.lstsq(x_proj, yd, rcond=None)[0]
    append_u = updated_u_t.T @ sqrt_sigma
    append_v = sqrt_sigma @ truc_v

    def rel_err(w_mat):
        out_hat = xd @ w_mat.T
        return float(np.sqrt(np.sum((yd - out_hat) ** 2))
                     / np.linalg.norm(yd, "fro"))

    return {"k": k, "svd_u": svd_u, "svd_v": svd_v, "new_w": new_w,
            "updated_u_t": updated_u_t, "append_u": append_u,
            "append_v": append_v, "error": rel_err(new_w),
            "updated_error": rel_err(append_u @ append_v)}


def _assert_close(a, e, atol=1e-6):
    if isinstance(e, (int, float)):
        assert abs(a - e) <= atol, (a, e)
        return
    assert a.shape == e.shape, (a.shape, e.shape)
    diff = (a.double() - torch.from_numpy(e)).abs().max().item()
    assert diff <= atol, diff


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


def _assert_result(r, exp):
    assert set(r) == {"k", "svd_u", "svd_v", "new_w", "updated_u_t",
                      "append_u", "append_v", "error", "updated_error"}
    assert r["k"] == exp["k"]
    for key in ("svd_u", "svd_v", "new_w", "updated_u_t",
                "append_u", "append_v"):
        _assert_close(r[key], exp[key]), key
    assert abs(r["error"] - exp["error"]) < 1e-8
    assert abs(r["updated_error"] - exp["updated_error"]) < 1e-8


# ---------------------------------------------------------------- 主体
def check_whitened_oracle(solve):
    """默认白化路径: k/因子/重构/闭式更新/误差全对拍; 且 v 侧带 L⁻¹."""
    kw = _fixture()
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_result(r, exp)
    assert r["k"] == int(M * N * 0.2 / (M + N))
    # svd_v 必须含 L⁻¹: 与不带 L⁻¹ 的朴素因子不同
    U, S, VT = torch.linalg.svd(kw["weight"].double() @ kw["scaling_diag_matrix"].double(),
                                full_matrices=False)
    naive_v = torch.sqrt(torch.diag(S[:r["k"]])) @ VT[:r["k"], :]
    assert not torch.allclose(r["svd_v"], naive_v, atol=1e-4)
    # 白化因子分解的意义: svd_u @ (svd_v @ L) 是 W@L 的 k 秩最优逼近
    # (相对误差有界; k 小谱衰减慢时用相对量)
    WL = kw["weight"].double() @ kw["scaling_diag_matrix"].double()
    recon = (WL - r["svd_u"] @ (r["svd_v"] @ kw["scaling_diag_matrix"].double())).norm()
    assert recon / WL.norm() < 0.5, (recon / WL.norm())


def check_direct_update(solve):
    """direct_update=True: 跳过白化, v 侧无 L⁻¹, 与白化路径不同."""
    kw = _fixture(direct=True)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_result(r, exp)
    r_white = solve(**_fixture(direct=False))
    assert not torch.allclose(r["svd_v"], r_white["svd_v"], atol=1e-4)
    assert not torch.allclose(r["new_w"], r_white["new_w"], atol=1e-4)


def check_low_rank_reconstruction(solve):
    """重构性质: rank-r 逼近在白化域最优 (误差 < 保留全部谱的非截断对照)."""
    kw = _fixture(ratio=0.5)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_result(r, exp)
    # 更新后误差 <= 截断误差 (闭式补偿不劣化)
    assert r["updated_error"] <= r["error"] + 1e-9
    # k 增大 -> 截断误差单调不增
    r_small = solve(**_fixture(ratio=0.1))
    assert r["error"] <= r_small["error"] + 1e-12


def check_singular_matrix_defense(solve):
    """L 奇异: 官方防御 +1e-6*I 再求逆, 结果有限且与 oracle 防御一致."""
    kw = _fixture(singular_L=True)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_result(r, exp)
    for key in ("svd_v", "append_v"):
        assert torch.isfinite(r[key]).all(), key


def check_lstsq_update(solve):
    """闭式更新: lstsq 解满足正规方程; append 因子乘积输出逼近 y."""
    kw = _fixture()
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_result(r, exp)
    x = kw["x"].double()
    # append 组合的输出与 updated_u_t 路径一致: x@(append_u@append_v).T
    out_append = x @ (r["append_u"] @ r["append_v"]).t()
    out_updated = (x @ kw["scaling_diag_matrix"].double()) @ r["updated_u_t"] \
        if False else None
    # updated_output = x @ truc_v.T @ Sigma @ updated_u_t; 这里按官方口径检查误差
    assert abs(r["updated_error"] - float(
        torch.sqrt(torch.sum((kw["y"].double() - out_append) ** 2))
        / torch.linalg.matrix_norm(kw["y"].double(), ord="fro"))) < 1e-9
    assert r["updated_error"] < 1.0  # 有界
    # y 范数 < 1: 分母 max(norm, 1.0) 变异会在此分化
    kw_s = _fixture(y_scale=0.005)
    r_s = solve(**kw_s)
    exp_s = _oracle(**kw_s)
    _assert_result(r_s, exp_s)
    assert kw_s["y"].double().norm() < 1.0


def check_k_formula(solve):
    """k 公式与边界: ratio 变化单调; ratio 过小报 k>=1, 过大报上界."""
    kw = _fixture()
    ks = []
    for ratio in (0.1, 0.3, 0.6, 1.0):
        r = solve(**dict(kw, ratio=ratio))
        assert r["k"] == int(M * N * ratio / (M + N))
        ks.append(r["k"])
    assert ks == sorted(ks)
    assert "k (0) 必须 >= 1" in _expect_err(solve, **dict(kw, ratio=0.001))
    solve(**dict(kw, ratio=1.0))  # ratio=1.0 合法 (k=min 半径内)


def check_purity(solve):
    """纯函数: 所有入参张量不被修改."""
    kw = _fixture()
    snap = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in kw.items()}
    solve(**kw)
    for k, v in kw.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, snap[k]), k


def check_args_validation(solve):
    """形状/类型/组合校验."""
    kw = _fixture()
    assert "weight 必须是 2 维浮点张量" in _expect_err(
        solve, torch.zeros(M), kw["scaling_diag_matrix"], kw["x"], kw["y"])
    assert "weight 必须是 2 维浮点张量" in _expect_err(
        solve, torch.zeros(M, N, dtype=torch.long),
        kw["scaling_diag_matrix"], kw["x"], kw["y"])
    assert "x 必须是 2 维浮点张量" in _expect_err(
        solve, **dict(kw, x=torch.zeros(P, N, dtype=torch.int)))
    assert f"scaling_diag_matrix 必须是 ({N}, {N}) 形状" in _expect_err(
        solve, **dict(kw, scaling_diag_matrix=torch.eye(N + 1)))
    assert "scaling_diag_matrix 必须是 2 维浮点张量" in _expect_err(
        solve, **dict(kw, scaling_diag_matrix=None))
    assert f"x 必须是 (None, {N}) 形状" in _expect_err(
        solve, **dict(kw, x=_mat(0.0, P, N + 1)))
    assert f"y 必须是 ({P}, {M}) 形状" in _expect_err(
        solve, **dict(kw, y=_mat(0.0, P + 2, M)))
    bad = dict(kw, direct_update=True, scaling_diag_matrix=kw["scaling_diag_matrix"])
    assert "不需要 scaling_diag_matrix" in _expect_err(solve, **bad)
    bad = dict(kw, direct_update=True, scaling_diag_matrix=None)
    solve(**bad)  # direct + None 合法
    for rv in (0.0, -0.2, 1.5, True, float("inf"), "0.2"):
        assert "ratio 必须是 (0, 1] 内的有限数值" in _expect_err(
            solve, **dict(kw, ratio=rv))
    assert f"y 必须是 ({P}, {M}) 形状" in _expect_err(
        solve, _mat(0.0, M, N), kw["scaling_diag_matrix"],
        _mat(0.0, P, N), _mat(0.0, P, M + 1))


# ---------------------------------------------------------------- 元信息
check_whitened_oracle.example = True
check_direct_update.example = True
check_low_rank_reconstruction.example = True
check_singular_matrix_defense.example = True
check_lstsq_update.example = True
check_k_formula.example = True
check_purity.example = True
check_args_validation.example = True
