import numpy as np
import torch

KEYS = {"attn_weights", "important_score", "pooled_score", "selected_indices",
        "kept_positions"}
CONF = dict(window_size=2, max_capacity_prompt=5, kernel_size=3, pooling="avgpool",
            metric_mode="after", use_wo=False, use_norm=False, use_ema=False,
            alpha=0.3, tau=1.0)


def _mk(s=7, d=2, h=1, b=1, dtype=torch.float64, seed=None):
    if seed is None:
        q = torch.tensor([[[[0.6, 0.2], [0.1, 0.9], [-0.4, 0.3], [0.8, -0.1],
                            [0.0, 0.5], [-0.7, 0.4], [0.2, 0.2]]]], dtype=dtype)
        k = torch.tensor([[[[0.3, -0.2], [0.7, 0.1], [-0.5, 0.6], [0.2, 0.2],
                            [0.9, -0.3], [-0.1, 0.8], [0.4, 0.0]]]], dtype=dtype)
        v = torch.tensor([[[[0.5, 0.5], [-0.2, 0.7], [0.9, 0.1], [0.0, -0.6],
                            [0.3, 0.3], [-0.8, 0.2], [0.6, -0.4]]]], dtype=dtype)
        return q, k, v, torch.tensor([[0.4, -0.3], [0.2, 0.7]], dtype=dtype)
    rng = np.random.default_rng(seed)
    sh = (b, h, s, d)
    q = torch.from_numpy(rng.standard_normal(sh)).to(dtype)
    k = torch.from_numpy(rng.standard_normal(sh)).to(dtype)
    v = torch.from_numpy(rng.standard_normal(sh)).to(dtype)
    wo = torch.from_numpy(rng.standard_normal((h * 2, h * d))).to(dtype)
    return q, k, v, wo


def _softmax32(x):
    x = x.astype(np.float32)
    e = np.exp(x - x.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)).astype(np.float64)


def _ema_np(x, axis, alpha):
    cur = np.take(x, [0], axis=axis)
    for i in range(1, x.shape[axis]):
        cur = alpha * np.take(x, [i], axis=axis) + (1.0 - alpha) * cur
    return np.squeeze(cur, axis=axis)


def _pool1d(score, ks, kind):
    pad = ks // 2
    xp = np.pad(score, ((0, 0), (0, 0), (pad, pad)),
                constant_values=0.0 if kind == "avgpool" else -np.inf)
    win = np.stack([xp[..., i:i + ks] for i in range(score.shape[-1])], axis=-2)
    return win.mean(-1) if kind == "avgpool" else win.max(-1)


def _oracle(q, k, v, wo, w, cap, ks, pooling, mode, use_wo, use_norm, use_ema, alpha, tau):
    B, H, S, D = q.shape
    L, n = S - w, cap - w
    A = np.matmul(q[:, :, -w:, :], k.transpose(0, 1, 3, 2)) / np.sqrt(D)
    A[:, :, :, -w:] = A[:, :, :, -w:] + np.triu(np.full((w, w), -np.inf), 1)
    A = _softmax32(A)
    proj = wo.reshape(-1, H, D)
    vp, past = v[:, :, :-w], A[:, :, -w:, :-w]
    if mode == "after":
        mean = _ema_np(past, 2, alpha) if use_ema else past.mean(2)
        om = np.einsum("bhl,bhld->bhd", mean, vp)
        score = mean / (1.0 - mean)
        if use_wo:
            po = np.einsum("bhd,chd->bhc", om, proj)
            pv = np.einsum("bhld,chd->bhlc", v, proj)[:, :, :-w]
            score = score * np.linalg.norm(po[:, :, None] - pv, axis=-1) ** tau
        else:
            score = score * np.linalg.norm(om[:, :, None] - vp, axis=-1)
    else:
        outs = np.matmul(A, v)
        sc = A / (1.0 - A)
        if use_wo:
            po, pv = (np.einsum("bhwd,chd->bhwc", outs, proj),
                      np.einsum("bhld,chd->bhlc", v, proj))
            sc = sc * np.linalg.norm(po[:, :, :, None] - pv[:, :, None], axis=-1)
            den = np.linalg.norm(po, axis=-1)[..., None]
        else:
            sc = sc * np.linalg.norm(outs[:, :, :, None] - v[:, :, None], axis=-1)
            den = np.linalg.norm(outs, axis=-1)[..., None]
        if use_norm:
            sc = sc / den
        sc = sc[:, :, -w:, :-w]
        score = _ema_np(sc, 2, alpha) if use_ema else sc.mean(2)
    pooled = score if pooling == "none" else _pool1d(score, ks, pooling)
    if S >= cap:
        sel = np.argsort(-pooled, axis=-1, kind="stable")[..., :n]
        kept = np.concatenate([sel, np.tile(np.arange(S - w, S), (B, H, 1))], axis=-1)
    else:
        sel = np.zeros((B, H, 0), dtype=np.int64)
        kept = np.tile(np.arange(S), (B, H, 1))
    return ({"attn_weights": A, "important_score": score, "pooled_score": pooled,
             "selected_indices": sel.astype(np.int64),
             "kept_positions": kept.astype(np.int64)}, L)


def _verify(solve, q, k, v, wo, **kw):
    got = solve(q, k, v, wo, **kw)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    c = dict(CONF, **kw)
    exp, L = _oracle(q.numpy(), k.numpy(), v.numpy(), wo.numpy(), c["window_size"],
                     c["max_capacity_prompt"], c["kernel_size"], c["pooling"],
                     c["metric_mode"], c["use_wo"], c["use_norm"], c["use_ema"],
                     c["alpha"], c["tau"])
    rt = 1e-5
    assert tuple(got["attn_weights"].shape) == (q.shape[0], q.shape[1], c["window_size"],
                                                q.shape[2]), "attn_weights 形状不对"
    for key in ("attn_weights", "important_score", "pooled_score"):
        t = got[key]
        assert torch.is_tensor(t) and t.dtype == q.dtype, f"{key} dtype 变了"
        assert np.allclose(t.numpy().astype(np.float64), exp[key], rtol=rt, atol=1e-7), \
            f"{key} 对不上：{t.flatten()[:8].tolist()} vs {exp[key].flatten()[:8].tolist()}"
    assert got["important_score"].shape[-1] == L, "分数长度应是 S - window_size"
    for key in ("selected_indices", "kept_positions"):
        assert got[key].dtype == torch.int64, f"{key} 必须是 int64"
        assert got[key].numpy().tolist() == exp[key].tolist(), \
            f"{key} {got[key].tolist()} != {exp[key].tolist()}"
    return got, exp


def check_example(solve):
    """题面四组：after/before 分数不同、alpha=0.5 的 EMA 等于均值、容量不够时不选择。"""
    q, k, v, wo = _mk()
    a, _ = _verify(solve, q, k, v, wo, **CONF)
    r = lambda t, n: [round(x, 4) for x in t.flatten().tolist()][:n]
    assert r(a["attn_weights"], 7) == [0.1375, 0.1228, 0.2562, 0.1618, 0.0993, 0.2224, 0.0]
    assert r(a["important_score"], 5) == [0.0705, 0.1153, 0.1577, 0.1383, 0.0247]
    assert r(a["pooled_score"], 5) == [0.0619, 0.1145, 0.1371, 0.1069, 0.0543]
    assert a["selected_indices"].tolist() == [[[2, 1, 3]]]
    assert a["kept_positions"].tolist() == [[[2, 1, 3, 5, 6]]]
    b, _ = _verify(solve, q, k, v, wo, **dict(CONF, metric_mode="before"))
    assert r(b["important_score"], 5) == [0.0801, 0.1053, 0.191, 0.1352, 0.0314]
    assert not np.allclose(a["important_score"].numpy(), b["important_score"].numpy())
    c, _ = _verify(solve, q, k, v, wo, **dict(CONF, pooling="none", use_ema=True, alpha=0.5))
    assert torch.allclose(c["pooled_score"], a["important_score"], atol=1e-12)
    assert c["selected_indices"].tolist() == [[[2, 3, 1]]]
    d, _ = _verify(solve, q, k, v, wo, **dict(CONF, max_capacity_prompt=9))
    assert d["selected_indices"].shape[-1] == 0
    assert d["kept_positions"].tolist() == [[[0, 1, 2, 3, 4, 5, 6]]]


check_example.example = True


def check_causal_corner_only_zeroes_future(solve):
    """右下角严格上三角被掩成恰好 0，行和仍为 1，窗内每行可见 key 数递增。"""
    q, k, v, wo = _mk(seed=3)
    got, _ = _verify(solve, q, k, v, wo, **dict(CONF, window_size=3, max_capacity_prompt=6))
    aw = got["attn_weights"]
    assert torch.allclose(aw.sum(-1), torch.ones_like(aw.sum(-1)), atol=1e-9)
    corner = aw[0, 0]
    assert corner[0, 5].item() == 0.0 and corner[0, 6].item() == 0.0
    assert corner[1, 6].item() == 0.0 and corner[1, 5].item() > 0.0
    assert corner[2, 4].item() > 0.0 and corner[2, 5].item() > 0.0 and corner[2, 6] > 0.0
    assert not torch.allclose(corner[0], corner[2], atol=1e-9)


def check_dead_switches_per_branch(solve):
    """after 支根本不读 use_norm；before 支根本不读 tau——两个开关各自有一半是死的。"""
    q, k, v, wo = _mk(seed=7)
    for mode, norm_dead, tau_dead in (("after", True, False), ("before", False, True)):
        base = dict(CONF, metric_mode=mode, use_wo=True)
        x, _ = _verify(solve, q, k, v, wo, **base)
        y, _ = _verify(solve, q, k, v, wo, **dict(base, use_norm=True))
        z, _ = _verify(solve, q, k, v, wo, **dict(base, tau=0.25))
        assert torch.equal(x["pooled_score"], y["pooled_score"]) is norm_dead
        assert torch.equal(x["pooled_score"], z["pooled_score"]) is tau_dead
    a, _ = _verify(solve, q, k, v, wo, **dict(CONF, use_wo=True))
    b, _ = _verify(solve, q, k, v, wo, **dict(CONF, use_wo=False))
    assert not torch.allclose(a["important_score"], b["important_score"], atol=1e-9)


def check_identity_wo_matches_value_space(solve):
    """W_o 取单位阵（H=1）时两个空间的差范数相同：两条支路都与不开 Wo 逐位相等。"""
    q, k, v, wo = _mk(seed=5)
    eye = torch.eye(2, dtype=q.dtype)                     # (C_out=H*D, H*D)，H=1、D=2
    for mode in ("after", "before"):
        with_wo, _ = _verify(solve, q, k, v, eye, **dict(CONF, metric_mode=mode, use_wo=True))
        no_wo, _ = _verify(solve, q, k, v, eye, **dict(CONF, metric_mode=mode, use_wo=False))
        assert torch.allclose(with_wo["important_score"], no_wo["important_score"],
                              rtol=1e-7, atol=1e-12), mode
        assert with_wo["selected_indices"].tolist() == no_wo["selected_indices"].tolist()


def check_ema_endpoints(solve):
    """alpha 是最新一项的权重：0.5 等于均值，1.0 只看窗内最后一行，0.0 只看第一行。"""
    q, k, v, wo = _mk()
    mid, _ = _verify(solve, q, k, v, wo, **dict(CONF, use_ema=True, alpha=0.5))
    plain, _ = _verify(solve, q, k, v, wo, **CONF)
    assert torch.allclose(mid["important_score"], plain["important_score"], atol=1e-12)
    v_same = torch.ones(1, 1, 7, 2, dtype=q.dtype) * torch.tensor([0.3, -0.7], dtype=q.dtype)
    opts = dict(CONF, use_ema=True, pooling="none")
    last, _ = _verify(solve, q, k, v_same, wo, **dict(opts, alpha=1.0))
    first, _ = _verify(solve, q, k, v_same, wo, **dict(opts, alpha=0.0))
    norm_u = float(np.linalg.norm([0.3, -0.7]))
    ratios = []
    for got, row in ((last, -1), (first, 0)):
        odds = got["attn_weights"][0, 0, row, :-2]
        ratio = got["important_score"][0, 0] / (odds / (1.0 - odds))
        assert ratio.max() - ratio.min() < 1e-9, "value 全同时分数必须正比于赔率"
        ratios.append(float(ratio.mean()))
    lo = float(last["attn_weights"][0, 0, -1, :-2].sum())
    hi = float(first["attn_weights"][0, 0, 0, :-2].sum())
    assert abs(lo - hi) > 1e-6
    assert abs(ratios[0] - (1.0 - lo) * norm_u) < 1e-9, "alpha=1 必须只取窗内最后一行"
    assert abs(ratios[1] - (1.0 - hi) * norm_u) < 1e-9, "alpha=0 必须只取窗内第一行"


def check_pooling_shapes_and_degeneracies(solve):
    """pooling=none / kernel=1 原样；L=1 时任何核都塌回分数本身；max 与 avg 不等价。"""
    q, k, v, wo = _mk(seed=17)
    n, _ = _verify(solve, q, k, v, wo, **dict(CONF, pooling="none"))
    assert torch.equal(n["pooled_score"], n["important_score"])
    one, _ = _verify(solve, q, k, v, wo, **dict(CONF, kernel_size=1))
    assert torch.equal(one["pooled_score"], n["pooled_score"])
    tail, _ = _verify(solve, q, k, v, wo, **dict(CONF, window_size=6, max_capacity_prompt=7,
                                                 kernel_size=5, pooling="maxpool"))
    assert tail["pooled_score"].shape[-1] == 1
    assert torch.equal(tail["pooled_score"], tail["important_score"])
    mx, _ = _verify(solve, q, k, v, wo, **dict(CONF, pooling="maxpool"))
    assert not torch.allclose(mx["pooled_score"], n["pooled_score"], atol=1e-9)


def check_selection_order_and_ties(solve):
    """选择序是分数降序；完全同分时按下标升序，而不是 torch.topk 的任意序。"""
    q, k, v, wo = _mk(seed=19)
    got, _ = _verify(solve, q, k, v, wo, **dict(CONF, pooling="none"))
    sc, sel = got["pooled_score"][0, 0], got["selected_indices"][0, 0]
    assert sc[sel].tolist() == sorted(sc[sel].tolist(), reverse=True)
    ones = torch.ones(1, 1, 7, 2, dtype=q.dtype)
    ties, _ = _verify(solve, ones, ones.clone(), ones.clone(),
                      torch.ones(1, 2, dtype=q.dtype), **dict(CONF, pooling="none"))
    assert ties["selected_indices"].tolist() == [[[0, 1, 2]]]
    assert ties["kept_positions"].tolist() == [[[0, 1, 2, 5, 6]]]
    edge = _verify(solve, ones, ones.clone(), ones.clone(),
                   torch.ones(1, 2, dtype=q.dtype), **CONF)[0]["pooled_score"][0, 0]
    assert edge[0] == edge[4] < edge[1], "avgpool 的两端把补零算进分母"


def check_no_mutation_and_dtypes(solve):
    """float32 进 float32 出；四个入参都不被就地改写。"""
    q, k, v, wo = _mk(seed=23, dtype=torch.float32)
    cache = [t.clone() for t in (q, k, v, wo)]
    got = solve(q, k, v, wo, **dict(CONF, use_wo=True))
    for t, c in zip((q, k, v, wo), cache):
        assert torch.equal(t, c), "输入被就地改了"
    for key in ("attn_weights", "important_score", "pooled_score"):
        assert got[key].dtype == torch.float32, f"{key} dtype 变成了 {got[key].dtype}"


def check_batch_and_heads_independent(solve):
    """B、H 只做批处理：逐 head 单算与整批算结果一致（W_o 也要按 head 切片）。"""
    q, k, v, wo = _mk(s=9, d=3, h=3, b=2, seed=29)
    opts = dict(CONF, max_capacity_prompt=7, use_wo=True, use_ema=True, pooling="maxpool")
    big, _ = _verify(solve, q, k, v, wo, **opts)
    d = q.shape[-1]
    for b in range(2):
        for h in range(3):
            small, _ = _verify(solve, q[b:b + 1, h:h + 1], k[b:b + 1, h:h + 1],
                               v[b:b + 1, h:h + 1], wo[:, h * d:(h + 1) * d], **opts)
            for key in KEYS:
                assert torch.allclose(big[key][b:b + 1, h:h + 1].float(),
                                      small[key].float(), rtol=1e-6, atol=1e-9), (b, h, key)


def check_against_oracle(solve):
    """随机对拍：两种 metric_mode x 三种池化 x Wo/EMA 开关组合。"""
    for i, (seed, mode, pooling) in enumerate([(41, "after", "avgpool"),
                                               (43, "before", "maxpool"),
                                               (47, "after", "none"),
                                               (53, "before", "avgpool"),
                                               (59, "after", "maxpool"),
                                               (61, "before", "none")]):
        for dtype in (torch.float64, torch.float32):
            q, k, v, wo = _mk(s=11, d=2, h=2, b=2, seed=seed + i, dtype=dtype)
            _verify(solve, q, k, v, wo, **dict(CONF, metric_mode=mode, pooling=pooling,
                                               use_wo=bool(seed % 2),
                                               use_ema=bool(seed % 3),
                                               max_capacity_prompt=8, window_size=3,
                                               kernel_size=5))


def check_value_errors(solve):
    """形状、dtype、开关类型与参数范围的非法口径全部抛 ValueError。"""
    q, k, v, wo = _mk(seed=67)
    bad = [(q[0], k, v, wo, CONF), (q, k[:, :, :, :1], v, wo, CONF),
           (q.float(), k, v, wo, CONF), (q, k, v, wo[0], CONF),
           (q, k, v, wo[:, :1], CONF), (q, k, v, wo, dict(CONF, window_size=7)),
           (q, k, v, wo, dict(CONF, window_size=0)),
           (q, k, v, wo, dict(CONF, max_capacity_prompt=2)),
           (q, k, v, wo, dict(CONF, kernel_size=4)),
           (q, k, v, wo, dict(CONF, kernel_size=True)),
           (q, k, v, wo, dict(CONF, pooling="adaptive")),
           (q, k, v, wo, dict(CONF, metric_mode="both")),
           (q, k, v, wo, dict(CONF, use_ema=1)), (q, k, v, wo, dict(CONF, alpha=1.5)),
           (q, k, v, wo, dict(CONF, alpha=-0.1)), (q, k, v, wo, dict(CONF, tau=-1.0)),
           (q, k, v, wo, dict(CONF, window_size=2.0)), (q, k, v.float(), wo, CONF)]
    for i, (a, b, c, d, opts) in enumerate(bad):
        try:
            solve(a, b, c, d, **opts)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 组非法输入没抛 ValueError")
