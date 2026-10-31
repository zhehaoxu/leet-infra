import numpy as np
import torch

KEYS = {"phi", "delta_t", "score", "normalized", "head_score", "kept"}


def _offsets(max_frames):
    values, step = [], 1
    while step <= max_frames:
        values.append(step)
        step *= 2
    return values or [1]


def _oracle(case, keep, end_frame, frame_len, tcount, omax, agg, norm):
    """numpy 标量重算：反解 RoPE → φ/amp/extra → Σ_f amp·cos(Δt·ω·mask+φ) + Σ_f extra。"""
    qm, qa, kn = (case["q_mean"].numpy().astype(float), case["q_abs"].numpy().astype(float),
                  case["k"].numpy().astype(float))
    rc, rs, omega = (case["rope_cos"].numpy().astype(float), case["rope_sin"].numpy().astype(float),
                     case["omega"].numpy().astype(float))
    layers, heads, freq = qm.shape[:3]
    tokens = kn.shape[1]
    tcount = freq - 2 * (freq // 3) if tcount is None else tcount
    offs = _offsets(omax)
    mask = [1.0 if f < tcount else 0.0 for f in range(freq)]
    delta = np.array([[end_frame - int(case["pos"][t]) // frame_len + o for o in offs]
                      for t in range(tokens)], dtype=float)
    phi, score = np.zeros((layers, heads, tokens, freq)), np.zeros((layers, heads, tokens))
    for l in range(layers):
        for h in range(heads):
            base = np.zeros((tokens, len(offs)))
            for t in range(tokens):
                additive = 0.0
                for f in range(freq):
                    c, s = rc[t, f], rs[t, f]                      # 反解：k_pre = k_post·e^{-iθ}
                    kre = kn[l, t, h, 2 * f] * c + kn[l, t, h, 2 * f + 1] * s
                    kim = kn[l, t, h, 2 * f + 1] * c - kn[l, t, h, 2 * f] * s
                    qr, qi = qm[l, h, f, 0], qm[l, h, f, 1]
                    qmag, kmag = np.hypot(qr, qi), np.hypot(kre, kim)
                    phi[l, h, t, f] = np.arctan2(qi * kre - qr * kim, qr * kre + qi * kim)
                    base[t] += qmag * kmag * np.cos(delta[t] * mask[f] * omega[f] + phi[l, h, t, f])
                    additive += (qa[l, h, f] - qmag) * kmag
                score[l, h, t] = (base[t].max() if agg == "max" else base[t].mean()) + additive
    if norm:
        std = np.maximum(score.std(-1, keepdims=True), 1e-6)         # 总体标准差，下界 1e-6
        normed = (score - score.mean(-1, keepdims=True)) / std
    else:
        normed = score
    head = normed.mean(0)
    kept = np.array([np.sort(np.argsort(-head[h], kind="stable")[:keep]) for h in range(heads)],
                    dtype=np.int64).reshape(heads, keep)
    return {"phi": phi, "delta_t": delta, "score": score, "normalized": normed,
            "head_score": head, "kept": kept}


def _case(seed, layers, heads, freq, tokens, frame_len=1):
    torch.manual_seed(seed)
    q_mean = torch.randn(layers, heads, freq, 2, dtype=torch.float64)
    theta = torch.rand(tokens, freq, dtype=torch.float64) * 3.0
    return {"q_mean": q_mean,
            "q_abs": torch.sqrt(q_mean[..., 0] ** 2 + q_mean[..., 1] ** 2)
            + torch.rand(layers, heads, freq, dtype=torch.float64),
            "k": torch.randn(layers, tokens, heads, 2 * freq, dtype=torch.float64),
            "rope_cos": theta.cos(), "rope_sin": theta.sin(),
            "omega": torch.rand(freq, dtype=torch.float64) * 2.0 + 0.1,
            "pos": torch.arange(tokens, dtype=torch.int64) * frame_len}


def _close(got, want, tol=2e-9):
    torch.testing.assert_close(torch.as_tensor(got, dtype=torch.float64),
                               torch.as_tensor(want, dtype=torch.float64), rtol=tol, atol=tol)


def _verify(solve, case, keep, end_frame, **kw):
    """结构检查（键 / 形状 / dtype / kept 语义）+ numpy oracle 全键对拍。"""
    before = {name: value.clone() for name, value in case.items()}
    got = solve(**case, keep=keep, current_end_frame=end_frame, **kw)
    layers, heads, freq = case["q_mean"].shape[:3]
    tokens, n_off = case["k"].shape[1], len(_offsets(kw.get("offset_max_frames", 128)))
    tol = 3e-5 if case["q_mean"].dtype == torch.float32 else 2e-9
    assert set(got) == KEYS, f"键应为 {sorted(KEYS)}，got {sorted(got)}"
    assert all(torch.equal(case[n], v) for n, v in before.items()), "输入被就地修改了"
    shape = {"phi": (layers, heads, tokens, freq), "delta_t": (tokens, n_off),
             "score": (layers, heads, tokens), "normalized": (layers, heads, tokens),
             "head_score": (heads, tokens), "kept": (heads, keep)}
    for key, want in shape.items():
        assert tuple(got[key].shape) == want, f"{key} 形状应为 {want}，got {tuple(got[key].shape)}"
        assert key == "kept" or got[key].dtype == case["q_mean"].dtype, f"{key} 应沿用输入 dtype"
    assert got["kept"].dtype == torch.int64, "kept 必须是 int64"
    for row in got["kept"].tolist():
        assert len(set(row)) == len(row), f"kept 不能有重复：{row}"
        assert sorted(row) == row, f"kept 每行必须升序：{row}"
    want = _oracle(case, keep, end_frame, kw.get("frame_seq_length", 1), kw.get("temporal_count"),
                   kw.get("offset_max_frames", 128), kw.get("score_aggregation", "mean"),
                   kw.get("normalize_scores", True))
    for key in ("phi", "delta_t", "score", "normalized", "head_score"):
        _close(got[key], want[key], tol)
    assert got["kept"].tolist() == want["kept"].tolist(), (
        f"kept 应是对 head_score 逐 head topk 后升序（同分取小下标）：{got['kept'].tolist()}")
    return got


def check_example(solve):
    """题面示例：L=1、H=1、F=2、T=2、temporal_count=1、offset_max_frames=2。"""
    one, zero = torch.ones(2, 2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64)
    q_mean = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=torch.float64)
    q_abs = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]], [[0.0, 1.0, 1.0, 0.0]]]], dtype=torch.float64)
    got = solve(q_mean, q_abs, k, one, zero, torch.tensor([1.0, 0.0], dtype=torch.float64),
                torch.tensor([0, 1]), 1, current_end_frame=2, temporal_count=1, offset_max_frames=2)
    _close(got["phi"][0, 0, :, 0], [0.0, -np.pi / 2], 1e-6)
    _close(got["delta_t"], [[3.0, 4.0], [2.0, 3.0]])
    _close(got["score"][0, 0], [-0.8218, 0.5252], 1e-4)
    _close(got["normalized"][0, 0], [-1.0, 1.0], 1e-6)
    assert got["kept"].tolist() == [[1]], f"应保留 [1]，got {got['kept'].tolist()}"


check_example.example = True


def check_rope_inversion_is_inverse(solve):
    """反解后 Σ_f Re(q·conj(k_pre)) 必须等于 temporal_count=0 时的 score 主体。"""
    case = _case(11, 1, 2, 3, 4)
    got = _verify(solve, case, keep=2, end_frame=5, temporal_count=0, normalize_scores=False)
    kn, rc, rs = case["k"].numpy(), case["rope_cos"].numpy(), case["rope_sin"].numpy()
    qm, qa = case["q_mean"].numpy(), case["q_abs"].numpy()
    for h in range(2):
        for t in range(4):
            want = 0.0
            for f in range(3):
                kre = kn[0, t, h, 2 * f] * rc[t, f] + kn[0, t, h, 2 * f + 1] * rs[t, f]
                kim = kn[0, t, h, 2 * f + 1] * rc[t, f] - kn[0, t, h, 2 * f] * rs[t, f]
                qmag = np.hypot(qm[0, h, f, 0], qm[0, h, f, 1])
                want += qm[0, h, f, 0] * kre + qm[0, h, f, 1] * kim + (qa[0, h, f] - qmag) * np.hypot(kre, kim)
            _close(got["score"][0, h, t], want, 1e-9)


def check_temporal_mask_freezes_spatial_bins(solve):
    """mask=0 的频率相位不含 Δt：temporal_count=0 时换 current_end_frame 分数不变。"""
    case = _case(12, 1, 1, 4, 5)
    a = _verify(solve, case, keep=2, end_frame=3, temporal_count=0, normalize_scores=False)
    b = _verify(solve, case, keep=2, end_frame=17, temporal_count=0, normalize_scores=False)
    _close(a["score"], b["score"])
    c = solve(**case, keep=2, current_end_frame=3, temporal_count=4, normalize_scores=False)
    d = solve(**case, keep=2, current_end_frame=17, temporal_count=4, normalize_scores=False)
    assert not torch.allclose(c["score"], d["score"]), "全 temporal 时 Δt 变化必须改变分数"


def check_delta_t_walks_powers_of_two(solve):
    """Δt = (current_end_frame − pos//frame_seq_length) + {1,2,4,…}，末项不超过 offset_max_frames。"""
    case = _case(13, 1, 1, 2, 3, frame_len=2)
    for omax, want_cols in ((0, [1.0]), (5, [1.0, 2.0, 4.0]), (6, [1.0, 2.0, 4.0]),
                            (7, [1.0, 2.0, 4.0]), (8, [1.0, 2.0, 4.0, 8.0])):
        got = _verify(solve, case, keep=1, end_frame=9, frame_seq_length=2, offset_max_frames=omax)
        row0 = got["delta_t"][0].tolist()
        assert [v - (9 - int(case["pos"][0]) // 2) for v in row0] == want_cols, (
            f"offsets 应为 {want_cols}，got {row0}")
        row1 = got["delta_t"][1].tolist()
        assert [v - (9 - int(case["pos"][1]) // 2) for v in row1] == want_cols, "Δt 必须减去 token 所在帧"
    assert _verify(solve, case, keep=1, end_frame=9)["delta_t"].shape == (3, 8), "默认 128 帧 → 8 档"


def check_extra_is_an_additive_bias(solve):
    """q_abs 每加 1，score 整体平移 Σ_f |k_pre|：多余项是加性的、与 Δt 无关。"""
    case = _case(14, 1, 2, 3, 4)
    a = _verify(solve, case, keep=2, end_frame=6, normalize_scores=False)
    bumped = dict(case, q_abs=case["q_abs"] + 1.0)
    b = _verify(solve, bumped, keep=2, end_frame=6, normalize_scores=False)
    kn, rc, rs = case["k"].numpy(), case["rope_cos"].numpy(), case["rope_sin"].numpy()
    for h in range(2):
        for t in range(4):
            shift = sum(np.hypot(kn[0, t, h, 2 * f] * rc[t, f] + kn[0, t, h, 2 * f + 1] * rs[t, f],
                                 kn[0, t, h, 2 * f + 1] * rc[t, f] - kn[0, t, h, 2 * f] * rs[t, f])
                        for f in range(3))
            _close(b["score"][0, h, t] - a["score"][0, h, t], shift, 1e-9)


def check_zscore_constant_row_is_zero(solve):
    """z-normalise 的标准差钳到 1e-6：整行常数（T=1 也算）归一化后是 0，不能是 NaN。"""
    case = _case(15, 1, 2, 2, 6)
    case["k"] = case["k"][:, :1].repeat(1, 6, 1, 1)          # 6 条一模一样的 key
    case["pos"] = torch.zeros(6, dtype=torch.int64)
    case["rope_cos"] = case["rope_cos"][:1].repeat(6, 1)
    case["rope_sin"] = case["rope_sin"][:1].repeat(6, 1)
    got = _verify(solve, case, keep=2, end_frame=4, temporal_count=0)
    _close(got["normalized"], torch.zeros(1, 2, 6))
    _close(got["head_score"], torch.zeros(2, 6))
    assert torch.isfinite(got["normalized"]).all(), "标准差为 0 时不能出 NaN/Inf"
    assert got["kept"].tolist() == [[0, 1], [0, 1]], "全同分时取最小下标"


def check_normalize_off_and_layer_mean(solve):
    """normalize_scores=False 时 normalized 就是 score；head_score 是逐层均值。"""
    case = _case(16, 3, 2, 2, 5)
    off = _verify(solve, case, keep=3, end_frame=4, normalize_scores=False)
    _close(off["normalized"], off["score"])
    _close(off["head_score"], off["score"].mean(dim=0))
    on = _verify(solve, case, keep=3, end_frame=4)
    mean = on["normalized"].mean(dim=-1, keepdim=True)
    std = on["normalized"].std(dim=-1, unbiased=False, keepdim=True)
    _close(mean, torch.zeros_like(mean), 1e-9)
    _close(std, torch.ones_like(std), 1e-9)
    assert not torch.allclose(on["head_score"], off["head_score"]), "z-normalise 会改变层间平均"


def check_score_aggregation_max(solve):
    """score_aggregation='max' 逐 token 取 Δt 上的最大值，恒 >= mean。"""
    case = _case(17, 1, 2, 3, 6)
    mean = _verify(solve, case, keep=2, end_frame=5)
    top = _verify(solve, case, keep=2, end_frame=5, score_aggregation="max")
    assert (top["score"] >= mean["score"] - 1e-12).all(), "max 聚合不能低于 mean"


def check_kept_is_per_head_topk(solve):
    """每个 head 独立 topk，再升序返回；head_score 的最高分必然入选。"""
    case = _case(18, 1, 3, 2, 9)
    got = _verify(solve, case, keep=3, end_frame=6)
    for h in range(3):
        best = int(got["head_score"][h].argmax())
        assert best in got["kept"][h].tolist(), f"head {h} 的最高分 token {best} 必须入选"
        assert got["kept"][h].tolist() == sorted(got["kept"][h].tolist()), "kept 行必须升序"


def check_ties_prefer_smaller_index(solve):
    """全同分时按更小的下标取（上游靠 tie_break_noise 抖动，这里钉成确定性）。"""
    case = _case(19, 1, 1, 2, 6)
    for name in ("k", "rope_cos", "rope_sin"):
        case[name] = case[name][:, :1].repeat(1, 6, 1, 1) if name == "k" else case[name][:1].repeat(6, 1)
    case["pos"] = torch.zeros(6, dtype=torch.int64)
    got = _verify(solve, case, keep=4, end_frame=3, temporal_count=0)
    assert got["kept"].tolist() == [[0, 1, 2, 3]], f"同分取小下标，应得 [0,1,2,3]，got {got['kept'].tolist()}"


def check_dtype_and_no_mutation(solve):
    """float32 输入 → float32 输出（kept 仍 int64），且不能改动任何输入。"""
    case = _case(20, 2, 2, 3, 5)
    cpu = {name: value.clone() for name, value in case.items()}
    hi = solve(**case, keep=2, current_end_frame=7)
    assert all(torch.equal(case[n], v) for n, v in cpu.items()), "输入被就地修改了"
    low = {name: value.float() if value.is_floating_point() else value for name, value in case.items()}
    lo = solve(**low, keep=2, current_end_frame=7)
    for key in ("phi", "delta_t", "score", "normalized", "head_score"):
        assert lo[key].dtype == torch.float32, f"{key} 应沿用 float32 输入"
        torch.testing.assert_close(lo[key], hi[key].float(), rtol=1e-5, atol=1e-5)
    assert lo["kept"].dtype == torch.int64 and lo["kept"].tolist() == hi["kept"].tolist()


def check_value_errors(solve):
    """非法入参一律 ValueError；keep=0 这类边界必须返回空选择而不是报错。"""
    case = _case(21, 1, 1, 2, 3)
    empty = solve(**case, keep=0, current_end_frame=2)
    assert empty["kept"].shape == (1, 0), "keep=0 应返回 (H,0) 的 kept"
    base = dict(case, keep=1, current_end_frame=2)
    bad = [dict(base, keep=4),                                     # keep > T
           dict(base, current_end_frame=2.0),                      # 非 int
           dict(base, frame_seq_length=0),                         # 帧长必须 >= 1
           dict(base, temporal_count=3),                           # 超过 F
           dict(base, score_aggregation="sum"),                    # 不支持的聚合
           dict(base, normalize_scores=1),                         # 非 bool
           dict(base, pos=torch.tensor([0.0, 1.0, 2.0])),          # 位置必须是整数
           dict(base, omega=torch.ones(3, dtype=torch.float64)),   # omega 长度 != F
           dict(base, k=torch.randn(1, 3, 2, 4, dtype=torch.float64)),   # head 数不一致
           dict(base, q_abs=torch.ones(1, 1, 3, dtype=torch.float64)),   # F 不一致
           dict(base, rope_cos=torch.ones(2, 2, dtype=torch.float64))]   # T 不一致
    for i, kw in enumerate(bad):
        try:
            solve(**kw)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError：{sorted(kw)}")
