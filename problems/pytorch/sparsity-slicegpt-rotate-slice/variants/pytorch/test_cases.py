"""SliceGPT 旋转-切边步 -- 测试用例."""

import numpy as np
import torch


D, F, S, B = 8, 16, 10, 2


def _sig(v, batch, seq=S, dim=D):
    """确定性 (B, S, d) 信号张量, 值域约 [-1, 1], 谱非退化."""
    n = batch * seq * dim
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * 0.317) % 1.0) * 2.0 - 1.0 + v * 0.037
    return x.reshape(batch, seq, dim).contiguous()


def _w(v, rows, cols, scale=1.0):
    n = rows * cols
    base = torch.arange(n, dtype=torch.float32)
    x = ((base * (0.211 + v * 0.013)) % 1.0) * 2.0 - 1.0
    return (x * scale).reshape(rows, cols).contiguous()


def _mask(v, batch, seq=S):
    n = batch * seq
    base = torch.arange(n, dtype=torch.float32)
    # 乘子 0.37 经扫描保证掩码后谱非退化 (相邻特征值 gap > 1e-3)
    return (((base * 0.37 + v) % 1.0) > 0.35).float().reshape(batch, seq)


def _np_h(batches, masks, damping=True):
    """独立 numpy 复算 H: 掩码置零 -> Σ xᵀx (einsum 逐 token 外积) -> 对角 damping."""
    H = None
    for i, xb in enumerate(batches):
        x = xb.numpy().astype(np.float64).copy()
        if masks is not None:
            x[masks[i].numpy() == 0] = 0.0
        h = np.einsum("bsi,bsj->ij", x, x)
        H = h if H is None else H + h
    if damping:
        damp = 0.01 * float(np.mean(np.diag(H)))
        idx = np.arange(H.shape[0])
        H[idx, idx] += damp
    return H


def _pca(batches, masks):
    """oracle PCA: 独立 H (numpy) + 共享 eigh 原语 (同环境符号确定) + 降序重排."""
    H = _np_h(batches, masks)
    vals, vecs = torch.linalg.eigh(torch.from_numpy(H))
    order = torch.argsort(vals, descending=True)
    return vals[order], vecs[:, order]


def _oracle(layer_inputs, mlp_ln_inputs, layer_outputs, w_qkv, w_o, w1, w2,
            attn_in_dim, attn_out_dim, mlp_out_dim, o_bias=None, ignore_masks=None):
    """独立镜像官方 rotate_and_slice_sequential 记账."""
    q_in = _pca(layer_inputs, ignore_masks)[1]
    attn_sc = q_in.t().clone().to(torch.float32)
    w_qkv_o = [(w.double() @ q_in).to(torch.float32)[:, :attn_in_dim] for w in w_qkv]
    attn_sc = attn_sc[:attn_in_dim, :]

    q_mid = _pca(mlp_ln_inputs, ignore_masks)[1]
    attn_sc = attn_sc @ q_mid.to(torch.float32)[:, :attn_out_dim]
    w_o_o = (q_mid.t() @ w_o.double()).to(torch.float32)[:attn_out_dim, :]
    o_bias_o = None if o_bias is None else \
        (q_mid.t() @ o_bias.double()).to(torch.float32)[:attn_out_dim]
    mlp_sc = q_mid.t().clone().to(torch.float32)[:attn_out_dim, :]
    w1_o = (w1.double() @ q_mid).to(torch.float32)[:, :attn_out_dim]

    q_out = _pca(layer_outputs, ignore_masks)[1]
    mlp_sc = mlp_sc @ q_out.to(torch.float32)
    w2_o = (q_out.t() @ w2.double()).to(torch.float32)[:mlp_out_dim, :]
    mlp_sc = mlp_sc[:, :mlp_out_dim]
    return {"w_qkv": w_qkv_o, "w_o": w_o_o, "o_bias": o_bias_o, "w1": w1_o,
            "w2": w2_o, "attn_shortcut_q": attn_sc, "mlp_shortcut_q": mlp_sc,
            "q_in": q_in, "q_mid": q_mid, "q_out": q_out}


def _fixture(v=0.0, with_bias=True, with_masks=True):
    li = [_sig(v, B), _sig(v + 1.0, B)]
    mi = [_sig(v + 2.0, B), _sig(v + 3.0, B)]
    lo = [_sig(v + 4.0, B), _sig(v + 5.0, B)]
    masks = [_mask(v, B), _mask(v + 1.0, B)] if with_masks else None
    return dict(
        layer_inputs=li, mlp_ln_inputs=mi, layer_outputs=lo,
        w_qkv=[_w(0.1, D, D), _w(0.2, D, D), _w(0.3, D, D)],
        w_o=_w(0.4, D, D), w1=_w(0.5, F, D), w2=_w(0.6, D, F),
        attn_in_dim=5, attn_out_dim=6, mlp_out_dim=7,
        o_bias=_w(0.7, 1, D).squeeze(0) if with_bias else None,
        ignore_masks=masks,
    )


def _assert_close(actual, expected, atol=1e-5):
    if isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_close(a, e, atol)
        return
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    diff = (actual.double() - expected.double()).abs().max().item()
    assert diff <= atol, diff


def _expect_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as e:
        return str(e)
    raise AssertionError("期望 ValueError 但未抛出")


# ---------------------------------------------------------------- 全流水线
def check_full_pipeline_oracle(solve):
    """端到端对拍独立 oracle (掩码 + 偏置 + 三个切边维度全启用)."""
    kw = _fixture()
    r = solve(**kw)
    exp = _oracle(**kw)
    for key in ("w_qkv", "w_o", "o_bias", "w1", "w2",
                "attn_shortcut_q", "mlp_shortcut_q", "q_in", "q_mid", "q_out"):
        _assert_close(r[key], exp[key])
    assert r["o_bias"].shape == (kw["attn_out_dim"],)
    assert r["attn_shortcut_q"].shape == (5, 6)
    assert r["mlp_shortcut_q"].shape == (6, 7)
    assert r["w_qkv"][0].shape == (D, 5) and r["w_o"].shape == (6, D)
    assert r["w1"].shape == (F, 6) and r["w2"].shape == (7, F)


def check_pca_properties(solve):
    """Q 正交、特征对重构、特征值降序且两两可分 (谱非退化)."""
    kw = _fixture()
    r = solve(**kw)
    for name, vals in (("q_in", r["eig_vals"][0]), ("q_mid", r["eig_vals"][1]),
                       ("q_out", r["eig_vals"][2])):
        q = r[name]
        eye = q.t() @ q
        assert torch.allclose(eye, torch.eye(D, dtype=torch.float64), atol=1e-9), \
            (eye - torch.eye(D)).abs().max()
        gaps = vals[:-1] - vals[1:]
        assert (gaps > 1e-3).all(), gaps  # 非退化: 相邻特征值可分
        # 特征对: H_damped @ q[:, i] = vals[i] * q[:, i]
    for name, vals, sigs in (("q_in", r["eig_vals"][0], kw["layer_inputs"]),
                             ("q_mid", r["eig_vals"][1], kw["mlp_ln_inputs"]),
                             ("q_out", r["eig_vals"][2], kw["layer_outputs"])):
        q, H = r[name], torch.from_numpy(_np_h(sigs, kw["ignore_masks"]))
        residual = (H @ q - q * vals).abs().max().item()
        assert residual < 1e-6, residual


def check_eigvals_desc_and_damping(solve):
    """返回特征值 = 无 damping 特征值 + damp (damping 只在对角、可观测)."""
    kw = _fixture()
    r = solve(**kw)
    for i, sigs in enumerate((kw["layer_inputs"], kw["mlp_ln_inputs"], kw["layer_outputs"])):
        H_nodamp = _np_h(sigs, kw["ignore_masks"], damping=False)
        damp = 0.01 * float(np.mean(np.diag(H_nodamp)))
        v_asc = torch.linalg.eigvalsh(torch.from_numpy(H_nodamp))
        expected = torch.sort(v_asc, descending=True).values + damp
        assert torch.allclose(r["eig_vals"][i], expected, atol=1e-8), \
            (r["eig_vals"][i] - expected).abs().max()


def check_mask_semantics(solve):
    """掩码 0 的 token 被置零: 与手工置零 oracle 一致, 与无掩码结果不同."""
    kw = _fixture()
    r = solve(**kw)
    exp = _oracle(**kw)
    _assert_close(r["q_in"], exp["q_in"])
    # 掩码置零后的信号 = 手工置零 (等价无掩码输入)
    li = [x * kw["ignore_masks"][i].unsqueeze(-1) for i, x in enumerate(kw["layer_inputs"])]
    kw2 = dict(kw, layer_inputs=li, ignore_masks=None)
    r2 = solve(**kw2)
    assert torch.allclose(r["q_in"], r2["q_in"], atol=1e-8), \
        (r["q_in"] - r2["q_in"]).abs().max()
    # 掩码确实改变 H
    kw3 = dict(kw, ignore_masks=None)
    r3 = solve(**kw3)
    assert not torch.allclose(r["q_in"], r3["q_in"], atol=1e-4)


def check_batch_accumulation(solve):
    """H 跨 batch 累积: 多 batch 与拼接成单 batch 的谱一致."""
    kw = _fixture()
    r = solve(**kw)
    li_cat = [torch.cat(kw["layer_inputs"], dim=0)]
    mi_cat = [torch.cat(kw["mlp_ln_inputs"], dim=0)]
    lo_cat = [torch.cat(kw["layer_outputs"], dim=0)]
    masks_cat = [torch.cat(kw["ignore_masks"], dim=0)]
    kw2 = dict(kw, layer_inputs=li_cat, mlp_ln_inputs=mi_cat,
               layer_outputs=lo_cat, ignore_masks=masks_cat)
    r2 = solve(**kw2)
    for a, b in zip(r["eig_vals"], r2["eig_vals"]):
        assert torch.allclose(a, b, atol=1e-7), (a - b).abs().max()
    _assert_close(r["q_in"], r2["q_in"], atol=1e-6)


def check_bias_none(solve):
    """o_bias=None 路径: 返回 None, 其余输出不受影响."""
    kw = _fixture(with_bias=False)
    r = solve(**kw)
    assert r["o_bias"] is None
    exp = _oracle(**kw)
    for key in ("w_qkv", "w_o", "w1", "w2", "attn_shortcut_q", "mlp_shortcut_q"):
        _assert_close(r[key], exp[key])


def check_full_dims_no_slice(solve):
    """切边维度 == d: 不切边, W_qkv @ Q @ Q.T ≈ W (旋转保内积, 可逆性)."""
    kw = _fixture()
    kw2 = dict(kw, attn_in_dim=D, attn_out_dim=D, mlp_out_dim=D)
    r = solve(**kw2)
    for w, wo in zip(kw["w_qkv"], r["w_qkv"]):
        recon = (wo.double() @ r["q_in"].t())
        assert torch.allclose(recon, w.double(), atol=1e-5), \
            (recon - w.double()).abs().max()
    recon_o = r["q_mid"].double() @ r["w_o"].double()
    assert torch.allclose(recon_o, kw["w_o"].double(), atol=1e-5)
    assert r["attn_shortcut_q"].shape == (D, D)
    # Q 正交 => shortcut 恒等块: 同一信号三组时 q_in == q_mid == q_out
    same = [_sig(9.0, B), _sig(10.0, B)]
    kw3 = dict(kw, layer_inputs=same, mlp_ln_inputs=same, layer_outputs=same)
    r3 = solve(**kw3)
    assert torch.allclose(r3["q_in"], r3["q_mid"], atol=1e-8)
    assert torch.allclose(r3["q_mid"], r3["q_out"], atol=1e-8)


def check_purity(solve):
    """纯函数: 所有入参 (含掩码置零路径) 不被修改."""
    kw = _fixture()
    snap = {}
    for k, v in kw.items():
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            snap[k] = [t.clone() for t in v]
        elif isinstance(v, torch.Tensor):
            snap[k] = v.clone()
        else:
            snap[k] = v
    solve(**kw)
    for k, v in kw.items():
        s = snap[k]
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            for t, ts in zip(v, s):
                assert torch.equal(t, ts), k
        elif isinstance(v, torch.Tensor):
            assert torch.equal(v, s), k


# ---------------------------------------------------------------- 校验
def check_args_validation(solve):
    """信号组/掩码/权重形状/切边维度校验."""
    kw = _fixture()
    assert "layer_inputs 必须是非空 list" in _expect_err(
        solve, [], **{k: v for k, v in kw.items() if k != "layer_inputs"})
    bad = dict(kw, layer_inputs=[_sig(0.0, B).reshape(B, S * D)])
    assert "3 维 (B, S, d)" in _expect_err(solve, **bad)
    bad = dict(kw, layer_inputs=[_sig(0.0, B), _sig(1.0, B, S, 4)])
    assert "隐维必须一致" in _expect_err(solve, **bad)
    bad = dict(kw, mlp_ln_inputs=[_sig(0.0, B)])
    assert "batch 数必须一致" in _expect_err(solve, **bad)
    bad = dict(kw, ignore_masks="mask")
    assert "ignore_masks 必须是 list 或 None" in _expect_err(solve, **bad)
    bad = dict(kw, ignore_masks=[_mask(0.0, B, 3), _mask(1.0, B)])
    assert "对齐" in _expect_err(solve, **bad)
    bad = dict(kw, ignore_masks=[_mask(0.0, B) * 2, _mask(1.0, B)])
    assert "0/1 掩码" in _expect_err(solve, **bad)
    bad = dict(kw, w_qkv=kw["w_qkv"][0])
    assert "w_qkv 必须是非空 list" in _expect_err(solve, **bad)
    bad = dict(kw, w_qkv=[])
    assert "w_qkv 必须是非空 list" in _expect_err(solve, **bad)
    bad = dict(kw, w_qkv=[_w(0.1, 3, 3), _w(0.2, D, D), _w(0.3, D, D)])
    assert "w_qkv[0] 必须是 (8, 8)" in _expect_err(solve, **bad)
    bad = dict(kw, w1=_w(0.5, F, 4))
    assert "w1 的入维" in _expect_err(solve, **bad)
    bad = dict(kw, w2=_w(0.6, 5, F))
    assert "w2 的出维" in _expect_err(solve, **bad)
    bad = dict(kw, o_bias=_w(0.7, 1, 3).squeeze(0))
    assert "o_bias 必须是长度 8" in _expect_err(solve, **bad)
    for dim_name in ("attn_in_dim", "attn_out_dim", "mlp_out_dim"):
        bad = dict(kw, **{dim_name: 0})
        assert f"{dim_name} 必须是 [1, 8] 内的整数" in _expect_err(solve, **bad)
        bad = dict(kw, **{dim_name: 9})
        assert f"{dim_name} 必须是 [1, 8] 内的整数" in _expect_err(solve, **bad)
        bad = dict(kw, **{dim_name: True})
        assert f"{dim_name} 必须是 [1, 8] 内的整数" in _expect_err(solve, **bad)


# ---------------------------------------------------------------- 元信息
check_full_pipeline_oracle.example = True
check_pca_properties.example = True
