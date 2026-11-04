"""LoSparse 低秩+结构化稀疏分解步 -- 测试用例."""

import math

import numpy as np
import torch


H, W, B, L = 12, 8, 2, 5


def _mat(v, rows, cols):
    n = rows * cols
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * (0.271 + v * 0.017)) % 1.0) * 2.0 - 1.0
    return x.reshape(rows, cols).contiguous()


def _oracle(weight, x, parameter_ratio=0.15, rank_ratio=0.1,
            remove_criteria="max_eigenvalue", neuron_mask=None, bias=None):
    """独立 oracle: numpy 记账 + 共享 svd 原语."""
    Hh, Ww = weight.shape
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = int(torch.count_nonzero(S))
    if parameter_ratio is not None:
        k = math.ceil(parameter_ratio * (Hh * Ww) / (Hh + Ww))
    else:
        k = math.ceil(rank * rank_ratio)
    ds = torch.sqrt(torch.diag(S))
    sl = slice(None, k) if remove_criteria == "max_eigenvalue" else slice(-k, None)
    Lm = U @ ds[:, sl]
    Rm = ds[sl, :] @ Vh
    Sres = weight - Lm @ Rm
    out = {}
    if neuron_mask is not None:
        Sm = Sres * neuron_mask.unsqueeze(1)
        idx = torch.nonzero(Sm.sum(dim=1)).flatten()
        sp = Sm[idx, :]
        y = x @ Rm.t() @ Lm.t()
        sc = torch.zeros(*x.shape[:-1], Hh)
        sc[..., idx] = x @ sp.t()
        out["output"] = y + sc if bias is None else y + sc + bias
        out["nonzero_idx"], out["sparse_pruned"] = idx, sp
    else:
        out["output"] = x @ Rm.t() @ Lm.t() + x @ Sres.t()
        out["nonzero_idx"], out["sparse_pruned"] = None, None
    out.update(L=Lm, R=Rm, sparse_residual=Sres, reduced_rank=k, rank=rank,
               is_full_rank=rank == min(Hh, Ww))
    return out


def _fixture(v=0.0, pr=0.15, criteria="max_eigenvalue", mask=True, bias=True):
    weight = _mat(v, H, W)
    x = _mat(v + 1.0, B * L, W).reshape(B, L, W)
    m = (((torch.arange(H, dtype=torch.float32) * 0.37 + v) % 1.0) > 0.35).float() \
        if mask else None
    bias = _mat(v + 2.0, 1, H).squeeze(0) if bias else None
    return dict(weight=weight, x=x, parameter_ratio=pr,
                remove_criteria=criteria, neuron_mask=m, bias=bias)


def _assert_close(a, e, atol=1e-5):
    assert a.shape == e.shape, (a.shape, e.shape)
    diff = (a.double() - e.double()).abs().max().item()
    assert diff <= atol, diff


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


# ---------------------------------------------------------------- 分解
def check_decomposition_oracle(solve):
    """默认 parameter_ratio 路径: L/R/残差/秩元信息全对拍."""
    kw = _fixture()
    r = solve(**kw)
    exp = _oracle(**kw)
    for key in ("L", "R", "sparse_residual", "nonzero_idx", "sparse_pruned"):
        _assert_close(r[key], exp[key])
    assert r["reduced_rank"] == exp["reduced_rank"] == math.ceil(0.15 * H * W / (H + W))
    assert r["rank"] == min(H, W) and r["is_full_rank"] is True
    assert r["L"].shape == (H, r["reduced_rank"]) and r["R"].shape == (r["reduced_rank"], W)


def check_lr_reconstruction(solve):
    """L@R = 前 k 奇异三元组重构 (与全 SVD 重构差 = 尾部能量)."""
    kw = _fixture(mask=False)
    r = solve(**kw)
    k = r["reduced_rank"]
    recon = r["L"] @ r["R"]
    U, S, Vh = torch.linalg.svd(kw["weight"].double(), full_matrices=False)
    full_topk = (U[:, :k] * S[:k]) @ Vh[:k]
    assert torch.allclose(recon.double(), full_topk, atol=1e-4), \
        (recon.double() - full_topk).abs().max()
    # √拆分的均衡性: L 列范数 = sqrt(s_i), R 行范数 = sqrt(s_i)
    norms_l = r["L"].double().norm(dim=0)
    assert torch.allclose(norms_l, S[:k].sqrt(), atol=1e-4), \
        (norms_l - S[:k].sqrt()).abs().max()


def check_min_criteria(solve):
    """min_eigenvalue 取尾部 k 个三元组 (与 max 路径结果不同且等于 oracle)."""
    kw = _fixture(criteria="min_eigenvalue")
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_close(r["L"], exp["L"])
    _assert_close(r["R"], exp["R"])
    kw_max = dict(_fixture(criteria="max_eigenvalue"))
    r_max = solve(**kw_max)
    assert not torch.allclose(r["L"], r_max["L"], atol=1e-3)
    # min 路径重构 = 后 k 三元组
    k = r["reduced_rank"]
    U, S, Vh = torch.linalg.svd(kw["weight"].double(), full_matrices=False)
    tail = (U[:, -k:] * S[-k:]) @ Vh[-k:]
    assert torch.allclose((r["L"] @ r["R"]).double(), tail, atol=1e-4)


def check_rank_ratio_path(solve):
    """parameter_ratio=None 覆盖翻转: k = ceil(rank·rank_ratio), 秩亏权重使 rank < min(H,W)."""
    weight = _mat(0.0, H, W)
    weight = weight.clone()
    weight[4:] = 0.0  # rank = 4 < min(H, W)
    kw = _fixture(mask=False, pr=None)
    kw = dict(kw, weight=weight, rank_ratio=0.5)
    r = solve(**kw)
    exp = _oracle(**kw)
    assert r["reduced_rank"] == exp["reduced_rank"] == math.ceil(4 * 0.5)
    assert r["reduced_rank"] != math.ceil(min(H, W) * 0.5)
    _assert_close(r["L"], exp["L"])


def check_deficient_rank(solve):
    """秩亏权重 (尾部行精确置零 -> 奇异值精确 0): rank=2, is_full_rank=False."""
    weight = _mat(0.0, H, W)
    weight = weight.clone()
    weight[2:] = 0.0
    kw = _fixture(mask=False)
    kw = dict(kw, weight=weight)
    r = solve(**kw)
    assert r["rank"] == 2 and r["is_full_rank"] is False
    exp = _oracle(**kw)
    assert exp["rank"] == r["rank"]
    _assert_close(r["L"], exp["L"])


# ---------------------------------------------------------------- 剪枝与前向
def check_pruned_forward(solve):
    """剪枝路径: nonzero_idx/行提取/scatter 前向全对拍."""
    kw = _fixture(mask=True, bias=True)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_close(r["nonzero_idx"].to(torch.float32),
                  exp["nonzero_idx"].to(torch.float32))
    _assert_close(r["sparse_pruned"], exp["sparse_pruned"])
    _assert_close(r["output"], exp["output"])
    assert r["sparse_pruned"].shape == (len(r["nonzero_idx"]), W)
    # 幸存行 = 掩码为 1 且残差行和非零
    expect_idx = torch.nonzero(
        (r["sparse_residual"] * kw["neuron_mask"].unsqueeze(1)).sum(dim=1)).flatten()
    assert torch.equal(r["nonzero_idx"], expect_idx)


def check_unpruned_forward(solve):
    """未剪枝路径: Y = x@R.T@L.T + x@S.T (+bias); 残差前向是全量."""
    kw = _fixture(mask=False, bias=False)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_close(r["output"], exp["output"])
    assert r["nonzero_idx"] is None and r["sparse_pruned"] is None
    # 恒等性: 未剪枝输出 == x @ weight.T (分解是精确重构 W = LR + S 的线性拆分)
    direct = kw["x"] @ kw["weight"].t()
    assert torch.allclose(r["output"], direct, atol=2e-4), \
        (r["output"] - direct).abs().max()


def check_pruned_equals_masked_full(solve):
    """剪枝前向 == 全量残差前向再按掩码置零 (scatter 语义)."""
    kw = _fixture(mask=True, bias=True)
    r = solve(**kw)
    full = solve(**{**kw, "neuron_mask": None})
    zeroed = full["output"] - kw["x"] @ full["sparse_residual"].t() \
        + kw["x"] @ (full["sparse_residual"] * kw["neuron_mask"].unsqueeze(1)).t()
    assert torch.allclose(r["output"], zeroed, atol=1e-4), \
        (r["output"] - zeroed).abs().max()


def check_bias_none(solve):
    """bias=None: 输出无偏置项."""
    kw = _fixture(mask=True, bias=False)
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_close(r["output"], exp["output"])


# ---------------------------------------------------------------- 校验
def check_args_validation(solve):
    """权重/x/criteria/比例/掩码/bias 校验."""
    kw = _fixture()
    bad = dict(kw, weight=_mat(0.0, H, W).reshape(H, W, 1))
    assert "weight 必须是 2 维" in _expect_err(solve, **bad)
    bad = dict(kw, x=kw["x"].reshape(B * L, W))
    assert "x 必须是 3 维" in _expect_err(solve, **bad)
    bad = dict(kw, x=_mat(0.0, B * L, W + 1).reshape(B, L, W + 1))
    assert "入维必须等于" in _expect_err(solve, **bad)
    bad = dict(kw, remove_criteria="random")
    assert "remove_criteria 必须是" in _expect_err(solve, **bad)
    bad = dict(kw, parameter_ratio=0)
    assert "parameter_ratio 必须是正数或 None" in _expect_err(solve, **bad)
    bad = dict(kw, parameter_ratio=True)
    assert "parameter_ratio 必须是正数或 None" in _expect_err(solve, **bad)
    bad = dict(kw, rank_ratio=-0.1)
    assert "rank_ratio 必须是正数" in _expect_err(solve, **bad)
    bad = dict(kw, neuron_mask=kw["neuron_mask"][: H - 1])
    assert "长度 12 的一维" in _expect_err(solve, **bad)
    bad = dict(kw, neuron_mask=kw["neuron_mask"] * 2)
    assert "0/1 掩码" in _expect_err(solve, **bad)
    bad = dict(kw, bias=_mat(0.0, 1, 3).squeeze(0))
    assert "bias 必须是长度 12" in _expect_err(solve, **bad)
    # k 上界: ratio 过大 -> reduced_rank > min(H, W) -> 报错
    bad = dict(kw, parameter_ratio=5.0)
    assert "reduced_rank" in _expect_err(solve, **bad)
    # 掩码全 0 -> 剪空报错
    bad = dict(kw, neuron_mask=torch.zeros(H))
    assert "剪空了全部神经元" in _expect_err(solve, **bad)


# ---------------------------------------------------------------- 元信息
check_decomposition_oracle.example = True
check_pruned_forward.example = True
