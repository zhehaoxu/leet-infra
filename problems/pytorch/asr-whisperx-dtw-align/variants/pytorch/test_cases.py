import math
import random

import torch


# ---------------- 官方 torch fallback 参考（alignment.py 原样转录） ----------------

def _ref_get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def _ref_backtrack(trellis, emission, tokens, blank_id=0):
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()
    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        path.append((j - 1, t - 1, prob))
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return path[::-1]


def _ref_merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1][0] == path[i2][0]:
            i2 += 1
        score = sum(path[k][2] for k in range(i1, i2)) / (i2 - i1)
        segments.append((transcript[path[i1][0]], path[i1][1],
                         path[i2 - 1][1] + 1, score))
        i1 = i2
    return segments


def _ref_merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2][0] == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg[0] for seg in segs])
                score = sum(seg[3] * (seg[2] - seg[1]) for seg in segs) / \
                    sum(seg[2] - seg[1] for seg in segs)
                words.append((word, segs[0][1], segs[-1][2], score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def _ref(emission, tokens, transcript, separator="|"):
    trellis = _ref_get_trellis(emission, tokens)
    path = _ref_backtrack(trellis, emission, tokens)
    ok = path is not None
    out = {"ok": ok, "path": [], "segments": [], "words": [],
           "trellis_corner": float(trellis[-1, len(tokens)]),
           "col0_inf": int(sum(1 for x in trellis[:, 0].tolist()
                               if math.isinf(x) and x > 0))}
    if ok:
        out["path"] = [[a, b, c] for a, b, c in path]
        segs = _ref_merge_repeats(path, transcript)
        out["segments"] = [list(s) for s in segs]
        out["words"] = [list(w) for w in _ref_merge_words(segs, separator)]
    return out


def _compare(solve, emission, tokens, transcript, separator="|"):
    led = solve(emission, list(tokens), list(transcript), separator)
    ref = _ref(emission, list(tokens), list(transcript), separator)
    assert set(led) == set(ref) == {"ok", "path", "trellis_corner",
                                    "col0_inf", "segments", "words"}
    assert led["ok"] == ref["ok"]
    assert led["col0_inf"] == ref["col0_inf"]
    assert math.isclose(led["trellis_corner"], ref["trellis_corner"],
                        rel_tol=1e-6, abs_tol=1e-9)
    assert len(led["path"]) == len(ref["path"])
    for a, b in zip(led["path"], ref["path"]):
        assert a[0] == b[0] and a[1] == b[1]
        assert math.isclose(a[2], b[2], rel_tol=1e-6, abs_tol=1e-9), (a, b)
    assert len(led["segments"]) == len(ref["segments"])
    for a, b in zip(led["segments"], ref["segments"]):
        assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
        assert math.isclose(a[3], b[3], rel_tol=1e-6, abs_tol=1e-9)
    assert len(led["words"]) == len(ref["words"])
    for a, b in zip(led["words"], ref["words"]):
        assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
        assert math.isclose(a[3], b[3], rel_tol=1e-6, abs_tol=1e-9)
    return led


# ---------------- 手算锚点场景 ----------------
# T=4, V=3 (blank=0, a=1, b=2), tokens=[1, 2]
# emission（log 域，手算 trellis 用）:
#   t0: [-1.0, -2.0, -3.0]   t1: [-2.0, -0.5, -3.0]
#   t2: [-1.5, -3.0, -0.7]   t3: [-1.2, -3.0, -0.9]
# 手算账（见 notes）：
#   列0 = [0, -1, -3, +inf, +inf]（尾部 2 行 +inf -> col0_inf=2）
#   trellis[4,1] = +inf；trellis[4,2] = -3.4（corner）
#   argmax(trellis[:,2]) = 行 3 -> path 仅覆盖帧 1..2
#   path = [(0,1,exp(-0.5)), (1,2,exp(-0.7))]，ok
#   segments = [a:1..2, b:2..3]；words = [ab:1..3 长度加权]
def _anchor_emission():
    return torch.tensor([
        [-1.0, -2.0, -3.0],
        [-2.0, -0.5, -3.0],
        [-1.5, -3.0, -0.7],
        [-1.2, -3.0, -0.9],
    ], dtype=torch.float32)


def check_hand_anchor(solve):
    led = _compare(solve, _anchor_emission(), [1, 2], ["a", "b"])
    assert led["ok"] is True
    assert led["col0_inf"] == 2
    assert math.isclose(led["trellis_corner"], -3.4, rel_tol=1e-6)
    # path 仅两帧：token0@帧1、token1@帧2（帧 0/3 为 blank 不入 path）
    assert led["path"][0][0] == 0 and led["path"][0][1] == 1
    assert math.isclose(led["path"][0][2], math.exp(-0.5), rel_tol=1e-7)
    assert led["path"][1][0] == 1 and led["path"][1][1] == 2
    assert math.isclose(led["path"][1][2], math.exp(-0.7), rel_tol=1e-7)
    assert led["segments"][0][:3] == ["a", 1, 2]
    assert led["segments"][1][:3] == ["b", 2, 3]
    w = led["words"][0]
    assert w[0] == "ab" and w[1] == 1 and w[2] == 3
    mean = (math.exp(-0.5) + math.exp(-0.7)) / 2
    assert math.isclose(w[3], mean, rel_tol=1e-7)


def check_tie_stays_blank(solve):
    """stay 胜 change 的帧：留 blank（不前进），帧概率记 blank 的——
    blank 帧也进 path 且记入段长。"""
    em = torch.tensor([
        [-1.0, -1.0, -5.0],   # t0: a 强（changed=-1.0 必胜 stay=-inf）
        [-0.1, -0.2, -5.0],   # t1: stay(-1.1) > change(-1.2) -> blank@帧1
        [-0.1, -5.0, -0.2],   # t2: b 强
    ], dtype=torch.float32)
    led = _compare(solve, em, [1, 2], ["a", "b"])
    assert led["ok"] is True
    # t0: change(0+e[0,a]=-1.0) 胜 stay(-inf) -> a@帧0
    # t1: stay = trellis[1,1](-1.0)+e[1,0](-0.1) = -1.1
    #     change = trellis[1,0](-1.0)+e[1,a](-0.2) = -1.2 -> stay 胜 -> blank@帧1
    # t2: change = trellis[2,1](-1.1)+e[2,b](-0.2) = -1.3
    #     stay = trellis[2,2](-6.0)+e[2,0] = -6.1 -> change -> b@帧2
    pts = {(p[1]): p for p in led["path"]}
    assert pts[0][0] == 0 and math.isclose(pts[0][2], math.exp(-1.0), rel_tol=1e-7)
    assert pts[1][0] == 0 and math.isclose(pts[1][2], math.exp(-0.1), rel_tol=1e-7)
    assert pts[2][0] == 1 and math.isclose(pts[2][2], math.exp(-0.2), rel_tol=1e-7)
    # a 段含 blank 帧：[0, 2)，分母 2（blank 帧计入段长与均值）
    segs = {s[0]: s for s in led["segments"]}
    assert segs["a"][1:3] == [0, 2]


def check_tie_at_change(solve):
    """真平分场景：stay == change 恰等 -> 严格 > 留 blank/stay。"""
    # trellis[1,1] = e[0,a] = -1.0；t1: stay = -1.0 + e[1,0]，
    # change = e[0,0] + e[1,a] = -0.4 + e[1,a]
    # 取 e[1,0] = -0.4, e[1,a] = -1.0 -> stay = change = -1.4 恰平
    em = torch.tensor([
        [-0.4, -1.0, -5.0],
        [-0.4, -1.0, -5.0],
        [-0.1, -5.0, -0.2],
    ], dtype=torch.float32)
    led = _compare(solve, em, [1, 2], ["a", "b"])
    assert led["ok"] is True
    # t1 恰平 -> 留 blank，帧 1 的 prob = exp(e[1,0]) = exp(-0.4)
    pts = {p[1]: p for p in led["path"]}
    assert pts[1][0] == 0
    assert math.isclose(pts[1][2], math.exp(-0.4), rel_tol=1e-7)
    # token a 被挤到帧 0
    assert pts[0][0] == 0 and math.isclose(pts[0][2], math.exp(-1.0), rel_tol=1e-7)


def check_incomplete(solve):
    """T < N：列 2 全 -inf -> argmax=0 -> 空路径 -> None（for-else quirk）。"""
    em = torch.rand(2, 4) - 5.0
    led = _compare(solve, em, [1, 2, 3], ["a", "b", "c"])
    assert led["ok"] is False
    assert led["path"] == [] and led["segments"] == [] and led["words"] == []
    assert led["col0_inf"] == 3


def check_separator_edges(solve):
    """首/连续/尾 separator 均不产空词；词分 = 长度加权（非帧均）。"""
    # tokens: a | b -> [1, 0, 2]，transcript ["a", "|", "b"]
    em = torch.tensor([
        [-2.0, -0.5, -3.0],
        [-1.0, -3.0, -2.0],   # blank 强 -> "|" 段
        [-1.5, -3.0, -0.6],
    ], dtype=torch.float32)
    led = _compare(solve, em, [1, 0, 2], ["a", "|", "b"])
    assert led["ok"] is True
    ws = led["words"]
    assert len(ws) == 2
    assert ws[0][0] == "a" and ws[1][0] == "b"
    # "|" 段夹在 a/b 之间：a 段右端 = | 段左端（手算：a@0, |@1, b@2）
    segs = {s[0]: s for s in led["segments"]}
    assert segs["a"][1:3] == [0, 1]
    assert segs["|"][1:3] == [1, 2]
    assert segs["b"][1:3] == [2, 3]


def check_merge_aggregation(solve):
    """merge_repeats 帧均 vs merge_words 长度加权：多帧 a 段 + 单帧 b。"""
    em = torch.tensor([
        [-2.0, -0.3, -5.0],
        [-2.0, -0.4, -5.0],
        [-2.0, -0.5, -5.0],
        [-2.0, -5.0, -0.6],
    ], dtype=torch.float32)
    led = _compare(solve, em, [1, 2], ["a", "b"])
    assert led["ok"] is True
    segs = {s[0]: s for s in led["segments"]}
    assert segs["a"][1:3] == [0, 3]          # 三帧 a：[0,3)
    pa = [p[2] for p in led["path"] if p[1] < 3]
    assert math.isclose(segs["a"][3], sum(pa) / len(pa), rel_tol=1e-7)
    w = led["words"][0]
    expect = (segs["a"][3] * 3 + segs["b"][3] * 1) / 4
    assert math.isclose(w[3], expect, rel_tol=1e-7)


def check_forced_completion_inf(solve):
    """列 0 尾部 +inf 填充：col0_inf == N；j==1 回溯到尾行时强制收尾。"""
    em = _anchor_emission()
    led = _compare(solve, em, [1, 2], ["a", "b"])
    assert led["col0_inf"] == 2
    assert math.isinf(led["trellis_corner"]) is False
    # T 恰 = N：尾部 1 行 +inf
    em3 = torch.rand(2, 4) - 5.0
    led2 = _compare(solve, em3, [1, 2], ["a", "b"])
    assert led2["col0_inf"] == 2


def check_randomized(solve):
    """随机对拍 200 组：随机 T/V/emission/tokens（含 separator）。"""
    rng = random.Random(714)
    for _ in range(200):
        T = rng.randint(3, 24)
        V = rng.randint(3, 8)
        N = rng.randint(1, min(T, 6))
        em = torch.randn(T, V).log_softmax(dim=1)
        vocab = list(range(1, V))
        tokens = [rng.choice(vocab) for _ in range(N)]
        alphabet = "abcdefg"
        transcript = [alphabet[t - 1] for t in tokens]
        # 随机把部分 token 换成 separator（下标 0？不行，0 是 blank；
        # 用 V-1 作 separator 词项并在 transcript 对应 "|"）
        sep_tok = V - 1
        for i in range(N):
            if rng.random() < 0.25:
                tokens[i] = sep_tok
                transcript[i] = "|"
        led = solve(em, list(tokens), list(transcript))
        ref = _ref(em, list(tokens), list(transcript))
        assert led["ok"] == ref["ok"] == (T >= N), (T, N)
        assert led["col0_inf"] == ref["col0_inf"] == N
        assert len(led["path"]) == len(ref["path"])
        for a, b in zip(led["path"], ref["path"]):
            assert a[0] == b[0] and a[1] == b[1]
            assert math.isclose(a[2], b[2], rel_tol=1e-6, abs_tol=1e-9)
        for a, b in zip(led["words"], ref["words"]):
            assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
        for a, b in zip(led["segments"], ref["segments"]):
            assert a[:3] == b[:3]
            assert math.isclose(a[3], b[3], rel_tol=1e-6, abs_tol=1e-9)


def check_invalid_inputs(solve):
    em = _anchor_emission()
    bad = [
        lambda: solve(em, [], []),                    # 空 tokens
        lambda: solve(em, [1, 2], ["a"]),             # 长度不齐
        lambda: solve("not a tensor", [1], ["a"]),
        lambda: solve(em, [1], ["a"], separator=None),  # separator 非 str
    ]
    for fn in bad:
        try:
            fn()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("expected error")
