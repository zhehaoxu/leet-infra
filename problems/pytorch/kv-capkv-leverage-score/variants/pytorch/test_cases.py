import numpy as np
import torch

KEYS = {"statistic", "weights", "leverage", "scores", "kept"}
N_SINK, TAU, EPS = 4, 5.0, 1e-6

def _oracle(q, k, v, cos, sin, keep, group_size=1):
    """numpy 独立重算：平均旋转 → 余弦统计量 → 显式外积累加 A → solve。"""
    qn, kn, vn = (x.detach().numpy().astype(np.float64) for x in (q, k, v))
    cn, sn = (x.detach().numpy().astype(np.float64) for x in (cos, sin))
    live_k, live_v = kn[N_SINK:].transpose(1, 0, 2), vn[N_SINK:].transpose(1, 0, 2)
    hkv, n_tok, dim = live_k.shape
    half = dim // 2
    rot = np.zeros((dim, dim))
    rot[half:, :half], rot[:half, half:] = np.eye(half), -np.eye(half)
    anchor = qn.mean(0) @ (np.diag(cn.mean(0)) + rot * sn.mean(0)[None, :]).T
    anchor /= np.maximum(np.linalg.norm(anchor, axis=1, keepdims=True), 1e-12)
    rep_k = np.repeat(live_k, group_size, 0)
    rep_k /= np.maximum(np.linalg.norm(rep_k, axis=2, keepdims=True), 1e-12)
    stat = np.clip(np.einsum("htd,hd->ht", rep_k, anchor), -1.0, 1.0)
    weight = np.exp(TAU * (stat - stat.max(axis=1, keepdims=True)))
    rep_v = np.repeat(live_v, group_size, 0)
    scaled = rep_v * np.sqrt(weight + EPS)[..., None]
    lev = np.zeros_like(weight)
    for g in range(hkv * group_size):
        mat = np.eye(dim)
        for vec in scaled[g]:
            mat = mat + np.outer(vec, vec)                 # 显式逐个外积累加
        for j in range(n_tok):
            lev[g, j] = rep_v[g, j] @ np.linalg.solve(mat, rep_v[g, j])
    grouped = (weight * lev).reshape(hkv, group_size, n_tok).mean(1)
    scores = np.concatenate([np.full((hkv, N_SINK), grouped.max() + 1.0), grouped], 1)
    kept = [sorted(sorted(range(scores.shape[1]), key=lambda c: (-scores[h, c], c))[:keep])
            for h in range(hkv)]
    return {"statistic": torch.tensor(stat), "weights": torch.tensor(weight),
            "leverage": torch.tensor(lev), "scores": torch.tensor(scores),
            "kept": torch.tensor(np.array(kept).reshape(hkv, keep), dtype=torch.int64)}

def _verify(solve, q, k, v, cos, sin, keep, group_size=1):
    before, got = [x.clone() for x in (q, k, v, cos, sin)], solve(q, k, v, cos, sin, keep, group_size)
    want = _oracle(q, k, v, cos, sin, keep, group_size)
    tol = 3e-5 if q.dtype == torch.float32 else 2e-9
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    assert all(torch.equal(a, b) for a, b in zip(before, (q, k, v, cos, sin))), "输入被就地修改了"
    hkv, total, live = int(k.shape[1]), int(k.shape[0]), int(k.shape[0]) - N_SINK
    assert got["kept"].shape == (hkv, keep) and got["kept"].dtype == torch.int64
    shapes = {"statistic": (hkv * group_size, live), "weights": (hkv * group_size, live),
              "leverage": (hkv * group_size, live), "scores": (hkv, total)}
    for name, shape in shapes.items():
        assert got[name].shape == shape, f"{name} 形状应为 {shape}，got {tuple(got[name].shape)}"
        assert got[name].dtype == q.dtype, f"{name} dtype {got[name].dtype} != {q.dtype}"
        assert torch.isfinite(got[name]).all(), f"{name} 出现非有限值"
        torch.testing.assert_close(got[name], want[name].to(q.dtype), rtol=tol, atol=tol)
    assert torch.all(got["statistic"].abs() <= 1.0), "statistic 必须被 clamp 到 [-1, 1]"
    assert torch.all(got["leverage"] >= 0), "v^T A^-1 v 必须非负；出现负数就是 A 不是 SPD"
    assert torch.allclose(got["weights"].amax(-1), torch.ones(hkv * group_size, dtype=q.dtype)), (
        "max-shift 之后每个 head 的权重上确界必须是 1")
    rows = got["kept"].tolist()
    assert rows == want["kept"].tolist(), f"kept：{rows} vs {want['kept'].tolist()}"
    for row in rows:
        assert row == sorted(row) and len(set(row)) == len(row), f"kept 必须升序去重：{row}"
        assert row[:N_SINK] == [0, 1, 2, 3], f"sink 分数严格最高，kept 前 4 位必须是 0..3：{row}"
    return got

def _tables(n_pos, dim, angle=0.0):  # HF 口径全宽 cos/sin 表：同一位置共用一组 (cos, sin)
    return (torch.full((n_pos, dim), float(np.cos(angle)), dtype=torch.float64),
            torch.full((n_pos, dim), float(np.sin(angle)), dtype=torch.float64))

def _rand(seed, total, hkv, group, dim):
    torch.manual_seed(seed)
    return (torch.randn(3, hkv * group, dim, dtype=torch.float64),
            torch.randn(total, hkv, dim, dtype=torch.float64),
            torch.randn(total, hkv, dim, dtype=torch.float64))

def check_example(solve):
    """题面示例：(Lq,H,D)=(2,1,2)、T=6、T'=2、keep=5。"""
    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64)
    k = torch.tensor([[[9.0, 9.0]]] * 4 + [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64)
    v = torch.tensor([[[0.0, 0.0]]] * 4 + [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64)
    cos, sin = (torch.tensor(t, dtype=torch.float64) for t in ([[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]))
    got = _verify(solve, q, k, v, cos, sin, 5)
    assert got["statistic"].tolist() == [[0.0, 1.0]], got["statistic"].tolist()
    exact = torch.tensor([[1 / (1 + np.exp(-5) + EPS), 1 / (2 + EPS)]], dtype=torch.float64)
    torch.testing.assert_close(got["leverage"], exact, rtol=1e-9, atol=1e-12)
    assert got["kept"].tolist() == [[0, 1, 2, 3, 5]], got["kept"].tolist()

check_example.example = True

def check_sink_fill_is_global_max_plus_one(solve):
    """sink 用跨 head 的全局 max + 1，而不是每个 head 自己的 max + 1。"""
    q, k, v = _rand(3, 11, 2, 1, 4)
    v[:, 1, :] = 0.0                                   # head 1 的真实分数全为 0
    got = _verify(solve, q, k, v, *_tables(3, 4, 0.3), 6)
    sink = torch.full((2, N_SINK), got["scores"][:, N_SINK:].max().item() + 1.0, dtype=torch.float64)
    assert torch.allclose(got["scores"][:, :N_SINK], sink), got["scores"].tolist()
    assert got["scores"][1, 0].item() > 1.5, "head 1 的 sink 不能退化成 0 + 1"

def check_zero_columns_are_safe(solve):
    """零列：v_t = 0 → 分数恰为 0；k_t = 0 → 统计量 0；v 全零 → 全并列，余位给更小下标。"""
    q, k, v = _rand(7, 7, 1, 1, 4)
    v[6] = 0.0
    got = _verify(solve, q, k, v, *_tables(2, 4, 0.2), 4)
    assert got["leverage"][0, -1].item() == 0.0 and got["scores"][0, -1].item() == 0.0
    assert got["weights"][0, -1].item() > 0, "权重只由 query/key 决定，与 value 无关"
    q = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    k = torch.tensor([[[0.0, 0.0]]] * 4 + [[[2.0, 0.0]], [[0.0, 0.0]]], dtype=torch.float64)
    got = _verify(solve, q, k, k, *_tables(1, 2), 4)
    assert got["statistic"].tolist() == [[1.0, 0.0]], got["statistic"].tolist()
    assert abs(got["weights"][0, 1].item() - np.exp(-TAU)) < 1e-12
    q, k, v = _rand(19, 9, 1, 1, 4)
    v.zero_()
    got = _verify(solve, q, k, v, *_tables(3, 4, 0.7), 6)
    assert got["kept"].tolist() == [[0, 1, 2, 3, 4, 5]], got["kept"].tolist()

def check_group_average_is_over_contiguous_blocks(solve):
    """GQA：repeat_kv 把同一 kv head 连放 group_size 份，分数对该块取均值而不是取第一份。"""
    q, k, v = _rand(17, 10, 2, 2, 4)
    got = _verify(solve, q, k, v, *_tables(5, 4, 0.4), 5, group_size=2)
    per_head = (got["weights"] * got["leverage"]).reshape(2, 2, -1)
    torch.testing.assert_close(got["scores"][:, N_SINK:], per_head.mean(1), rtol=1e-9, atol=1e-12)
    assert not torch.allclose(got["scores"][:, N_SINK:], per_head[:, 0]), "必须真的对 group 求均值"

def check_rope_average_enters_the_anchor(solve):
    """cos=1/sin=0 时 anchor = mean(q)；cos=0/sin=1 时 anchor = rot @ mean(q)，两半相互抵消。"""
    q = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[3.0, 2.0, 1.0, 0.0]]], dtype=torch.float64)
    sample = [[[1.0, 0.0, 0.0, 0.0]]] * 4 + [[[1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]]
    k = torch.tensor(sample, dtype=torch.float64)
    cos = torch.nn.functional.cosine_similarity
    plain = _verify(solve, q, k, k, *_tables(2, 4), 4)
    torch.testing.assert_close(plain["statistic"][0], cos(q[:, 0].mean(0), k[N_SINK:, 0], dim=-1))
    rot = torch.zeros(4, 4, dtype=torch.float64)
    rot[2:, :2], rot[:2, 2:] = torch.eye(2, dtype=torch.float64), -torch.eye(2, dtype=torch.float64)
    turned = _verify(solve, q, k, k, torch.zeros(2, 4), torch.ones(2, 4), 4)
    torch.testing.assert_close(turned["statistic"][0], cos(rot @ q[:, 0].mean(0), k[N_SINK:, 0], dim=-1))

def check_random_against_oracle(solve):
    """随机形状、多头、GQA 分组、两种 dtype 全键对拍（float32 的输出 dtype 也在这里判）。"""
    cases = [(6, 1, 1, 2, 4), (9, 1, 2, 4, 5), (14, 2, 1, 6, 8), (17, 2, 2, 4, 13), (20, 3, 1, 6, 10)]
    for seed, (total, hkv, group, dim, keep) in enumerate(cases):
        q, k, v = _rand(200 + seed, total, hkv, group, dim)
        cos, sin = _tables(4 + seed, dim, 0.31 * (seed + 1))
        _verify(solve, q, k, v, cos, sin, keep, group)
        _verify(solve, q.float(), k.float(), v.float(), cos.float(), sin.float(), keep, group)

def check_value_errors(solve):
    """非法输入（形状、dtype、keep、cos/sin）必须抛 ValueError。"""
    q, k = torch.randn(2, 2, 4), torch.randn(9, 2, 4)
    cos, sin, odd = torch.ones(2, 4), torch.zeros(2, 4), torch.randn(9, 2, 3)
    bad = [(q, k, k, cos, sin, 6, 4), (q, k, k, cos, sin, 3, 1), (q, k, k, cos, sin, 10, 1),
           (q, k, k, cos, sin, 6.0, 1), (q, torch.randn(4, 2, 4), torch.randn(4, 2, 4), cos, sin, 4, 1),
           (q, k, k[:, :, :3], cos, sin, 6, 1), (q, k, k, torch.ones(1, 1, 4), torch.ones(1, 1, 4), 6, 1),
           (q, k, k, torch.ones(3, 4), sin, 6, 1), (q, k.double(), k, cos, sin, 6, 1),
           (q, torch.randn(0, 2, 4), k, cos, sin, 6, 1), (q.reshape(4, 4), k, k, cos, sin, 6, 1),
           (q, odd, odd, torch.ones(2, 3), torch.zeros(2, 3), 6, 1), (q, k, k, cos, sin, 6, 0)]
    for i, args in enumerate(bad):
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"第 {i} 个非法输入应抛 ValueError")
