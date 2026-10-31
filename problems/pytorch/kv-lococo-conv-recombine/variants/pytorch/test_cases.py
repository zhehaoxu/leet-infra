import numpy as np
import torch

KEYS = {"hh_idx", "res_idx", "conv_prob", "blend", "merged_keys", "merged_values"}


def _oracle(keys, values, scores, conv_weight, normalizer, rate, local_len, mem_size):
    """numpy 独立重算：显式排序选 token，滑窗 conv，逐行 softmax。"""
    kn, vn, sn, w = (t.detach().numpy().astype(np.float64)
                     for t in (keys, values, scores, conv_weight))
    seq, heads, dim = kn.shape
    m = int(mem_size * (1.0 - rate))
    keep, kernel, pad, norm = mem_size - m, w.shape[-1], (w.shape[-1] - 1) // 2, float(normalizer)
    hh, res, prob = [], [], np.zeros((heads, m, seq - keep))
    blend = np.zeros((heads, m, seq - keep))
    mk, mv = np.zeros((heads, mem_size, dim)), np.zeros((heads, mem_size, dim))
    for h in range(heads):
        chosen = set(sorted(range(seq - local_len), key=lambda i: (-sn[h, i], i))[: keep - local_len])
        chosen |= set(range(seq - local_len, seq))
        hh.append(sorted(chosen))
        rest = [i for i in range(seq) if i not in chosen]
        pick = sorted(sorted(range(len(rest)), key=lambda p: (-sn[h, rest[p]], p))[:m])
        res.append([rest[p] for p in pick])
        x = np.pad(np.concatenate([kn[rest, h].T, vn[rest, h].T]), ((0, 0), (pad, pad)))
        win = np.lib.stride_tricks.sliding_window_view(x, kernel, axis=1)      # (2D, n, K)
        out = np.maximum(np.einsum("ocj,ctj->ot", w, win), 0.0)                # 上游 relu
        prob[h] = np.exp(out - out.max(-1, keepdims=True))
        prob[h] /= prob[h].sum(-1, keepdims=True)
        for r, p in enumerate(pick):
            blend[h, r, p] += 1.0 - norm
        blend[h] += norm * prob[h]
        mk[h, :keep], mv[h, :keep] = kn[hh[h], h], vn[hh[h], h]
        mk[h, keep:], mv[h, keep:] = blend[h] @ kn[rest, h], blend[h] @ vn[rest, h]
    raw = {"hh_idx": hh, "res_idx": res, "conv_prob": prob, "blend": blend,
           "merged_keys": mk, "merged_values": mv}
    return {k: torch.tensor(v, dtype=torch.int64 if k.endswith("idx") else keys.dtype)
            for k, v in raw.items()}


def _close(a, b, dtype):
    tol = 1e-5 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _verify(solve, keys, values, scores, conv_weight, normalizer, rate, local_len, mem_size):
    before = [t.clone() for t in (keys, values, scores, conv_weight)]
    got = solve(keys, values, scores, conv_weight, normalizer, rate, local_len, mem_size)
    assert isinstance(got, dict) and set(got) == KEYS, f"返回键不对：{sorted(got)}"
    assert all(torch.equal(a, b) for a, b in zip(before, (keys, values, scores, conv_weight))), "输入被就地修改"
    seq, heads, dim = keys.shape
    m = int(mem_size * (1.0 - rate))
    want_shapes = [(heads, mem_size - m), (heads, m), (heads, m, seq - mem_size + m),
                   (heads, m, seq - mem_size + m), (heads, mem_size, dim), (heads, mem_size, dim)]
    for name, shape in zip(("hh_idx", "res_idx", "conv_prob", "blend", "merged_keys", "merged_values"),
                           want_shapes):
        assert got[name].shape == shape, f"{name} 形状 {tuple(got[name].shape)} != {shape}"
    assert got["hh_idx"].dtype == got["res_idx"].dtype == torch.int64
    for name in ("conv_prob", "blend", "merged_keys", "merged_values"):
        assert got[name].dtype == keys.dtype, f"{name} 应返回输入 dtype"
    assert torch.allclose(got["blend"].sum(-1),
                          torch.ones(heads, m, dtype=got["blend"].dtype)), "blend 每行必须是凸组合"
    want = _oracle(keys, values, scores, conv_weight, normalizer, rate, local_len, mem_size)
    for name in ("hh_idx", "res_idx"):
        assert got[name].tolist() == want[name].tolist(), f"{name}：{got[name].tolist()} vs {want[name].tolist()}"
    for name in ("conv_prob", "blend", "merged_keys", "merged_values"):
        _close(got[name], want[name], keys.dtype)
    return got


def _small(seq=9, rate=0.5, mem=4, local=1, kernel=3):
    torch.manual_seed(3)
    return (torch.randn(seq, 1, 2), torch.randn(seq, 1, 2),
            torch.arange(seq, dtype=torch.float32).unsqueeze(0),
            torch.randn(int(mem * (1.0 - rate)), 4, kernel), rate, local, mem)


def check_example(solve):
    """题面示例：S=6, H=1, D=2, mem_size=4, hh_keep_rate=0.5, local_len=1, K=3。"""
    keys = torch.tensor([[[1., 0.]], [[0., 3.]], [[1., 3.]], [[-1., 3.]], [[0., 3.]], [[2., 2.]]])
    values = torch.tensor([[[1., 2.]], [[2., 5.]], [[0., 5.]], [[1., 5.]], [[-1., 5.]], [[.5, .5]]])
    weight = torch.zeros(2, 4, 3)
    weight[0, 1, 1] = weight[1, 3, 1] = 1.0
    got = _verify(solve, keys, values, torch.tensor([[.9, .1, .7, .2, .6, 5.]]), weight, .5, .5, 1, 4)
    assert got["hh_idx"].tolist() == [[0, 5]] and got["res_idx"].tolist() == [[2, 4]]
    assert got["conv_prob"].tolist() == [[[.25] * 4] * 2], got["conv_prob"].tolist()
    assert got["blend"].tolist() == [[[.125, .625, .125, .125], [.125, .125, .125, .625]]]
    assert got["merged_keys"].tolist() == [[[1., 0.], [2., 2.], [.5, 3.], [0., 3.]]]


check_example.example = True


def check_local_block(solve):
    """末尾 local_len 个 token 必留，且不占前缀 topk 的名额。"""
    keys, values, scores, weight, rate, local, mem = _small()
    scores[0, -local:] = -100.0
    assert _verify(solve, keys, values, scores, weight, .5, rate, local, mem)["hh_idx"].tolist() == [[7, 8]]
    scores[0, -local:] = 100.0
    assert _verify(solve, keys, values, scores, weight, .5, rate, local, mem)["hh_idx"].tolist() == [[7, 8]]


def check_selection(solve):
    """重击按 token 号升序（不是分数序），残差取自补集且升序。"""
    keys, values, scores, weight, rate, local, mem = _small(mem=6)
    scores = torch.tensor([[5., 0., 4., 3., 2., 1., 6., 1.5, 8.]])
    got = _verify(solve, keys, values, scores, weight, .5, rate, local, mem)
    hh, res = got["hh_idx"][0].tolist(), got["res_idx"][0].tolist()
    assert hh == [0, 6, 8], f"应按 token 号升序（分数序是 6,0,8）：{hh}"
    assert res == [2, 3, 4] and not set(hh) & set(res), f"残差应是补集里分数最高的 m 个：{res}"


def check_ties(solve):
    """全同分：重击与残差都取更小的 token 号。"""
    keys, values, scores, weight, rate, local, mem = _small(local=0)
    got = _verify(solve, keys, values, torch.zeros(1, 9), weight, .5, rate, local, mem)
    assert got["hh_idx"].tolist() == [[0, 1]] and got["res_idx"].tolist() == [[2, 3]]


def check_conv_padding(solve):
    """K=21 两侧各补 10 个零，输出长度不变；最右拍在末位读到零填充。"""
    keys, values, scores, weight, rate, local, mem = _small(kernel=21)
    assert _verify(solve, keys, values, scores, weight, .5, rate, local, mem)["conv_prob"].shape == (1, 2, 7)
    keys, values, scores, weight, rate, local, mem = _small()
    weight = torch.zeros(2, 4, 3)
    weight[0, 0, 2] = weight[1, 1, 2] = 1.0      # 只留最右拍：t 处读 x[t+1]
    got = _verify(solve, keys, values, scores, weight, 1.0, rate, local, mem)
    rest = [i for i in range(9) if i not in got["hh_idx"][0].tolist()]
    for o in range(2):
        base = keys[rest, 0, o].double()
        shifted = torch.cat([base[1:], torch.zeros(1, dtype=torch.float64)])
        _close(got["conv_prob"][0, o], torch.softmax(torch.relu(shifted), -1).float(), torch.float32)


def check_relu(solve):
    """卷积全负时 ReLU 归零，softmax 退化成均匀分布。"""
    got = _verify(solve, torch.ones(9, 1, 2), torch.ones(9, 1, 2), torch.arange(9.).reshape(1, 9),
                  -torch.ones(2, 4, 3), 1.0, .5, 1, 4)
    torch.testing.assert_close(got["conv_prob"], torch.full((1, 2, 7), 1 / 7), rtol=1e-6, atol=1e-6)


def check_blend_and_merge(solve):
    """normalizer 两端分别退化成 one-hot / 纯卷积；merged 前 keep 行是原 token。"""
    keys, values, scores, weight, rate, local, mem = _small()
    hard = _verify(solve, keys, values, scores, weight, 0.0, rate, local, mem)
    rest = [i for i in range(9) if i not in hard["hh_idx"][0].tolist()]
    for r, tok in enumerate(hard["res_idx"][0].tolist()):
        assert hard["blend"][0, r, rest.index(tok)].item() == 1.0, "normalizer=0 必须退化成 one-hot"
    soft = _verify(solve, keys, values, scores, weight, 1.0, rate, local, mem)
    _close(soft["blend"], soft["conv_prob"], keys.dtype)
    hh = soft["hh_idx"][0].tolist()
    torch.testing.assert_close(soft["merged_keys"][0, :2], keys[hh, 0], rtol=0, atol=0)
    torch.testing.assert_close(soft["merged_values"][0, :2], values[hh, 0], rtol=0, atol=0)


def check_mem_compress_truncates(solve):
    """m = int(mem_size*(1-rate)) 向零截断：10*0.25=2.5 → 2。"""
    keys, values, scores, weight, rate, local, mem = _small(seq=13, rate=.75, mem=10, local=0)
    got = _verify(solve, keys, values, scores, torch.randn(2, 4, 3), .5, rate, local, mem)
    assert got["hh_idx"].shape == (1, 8) and got["merged_keys"].shape == (1, 10, 2)


def check_dtype_and_random(solve):
    """float32→float32、非连续视图，以及随机形状与 oracle 全键对拍。"""
    torch.manual_seed(11)
    big = torch.randn(20, 2, 4, dtype=torch.float64)
    keys, values, scores = big[::2], big[1::2], torch.randn(2, 10, dtype=torch.float64)
    weight = torch.randn(3, 8, 5, dtype=torch.float64)
    a = solve(keys, values, scores, weight, .5, .5, 2, 6)
    b = solve(keys.contiguous(), values.contiguous(), scores, weight, .5, .5, 2, 6)
    for name in ("conv_prob", "blend", "merged_keys", "merged_values"):
        _close(a[name], b[name], torch.float64)
    lo = solve(keys.float(), values.float(), scores.float(), weight.float(), .5, .5, 2, 6)
    assert all(lo[n].dtype == torch.float32 for n in ("conv_prob", "blend"))
    cases = [(12, 2, 3, 4, .5, 2, 3), (9, 1, 2, 6, .5, 0, 21), (16, 2, 4, 8, .75, 3, 5),
             (7, 3, 2, 4, .5, 1, 7), (20, 2, 3, 10, .6, 4, 21)]
    for seed, (seq, heads, dim, mem, rate, local, kernel) in enumerate(cases):
        torch.manual_seed(200 + seed)
        _verify(solve, torch.randn(seq, heads, dim, dtype=torch.float64),
                torch.randn(seq, heads, dim, dtype=torch.float64),
                torch.randn(heads, seq, dtype=torch.float64),
                torch.randn(int(mem * (1. - rate)), 2 * dim, kernel, dtype=torch.float64),
                .5, rate, local, mem)


def check_value_errors(solve):
    base = [*_small()[:4], .5, .5, 1, 4]
    for i, bad in [(0, base[0][0]), (1, base[1][:-1]), (2, base[2][:, :-1]),
                   (0, base[0].double()), (3, base[3][:, :3]), (3, torch.randn(2, 4, 4)),
                   (6, -1), (7, 4.0), (4, torch.ones(2))]:
        try:
            solve(*(base[:i] + [bad] + base[i + 1:]))
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"第 {i} 个字段非法时应抛 ValueError")
    for rate_bad in (0.0, 1.0, 1.5, True):
        try:
            solve(*base[:4], .5, rate_bad, 1, 4)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"hh_keep_rate={rate_bad} 应被拒绝")
    s = _small(seq=5, mem=6, local=0)
    try:
        solve(s[0], s[1], s[2], s[3], .5, .5, 0, 6)
    except ValueError:
        return
    raise AssertionError("非重击 token 少于 mem_compress 时应抛 ValueError")
