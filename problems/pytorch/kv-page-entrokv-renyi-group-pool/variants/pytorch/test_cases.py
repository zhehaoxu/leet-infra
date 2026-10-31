import math
import random

import torch

KEYS = {"entropies", "group_weights", "page_scores", "retained_pages", "retained_tokens"}
TOL = 1e-5


def _flat(x):
    if isinstance(x, list):
        return [u for v in x for u in _flat(v)]
    return [x]


def _close_all(got, want):
    a, b = _flat(got), _flat(want)
    if len(a) != len(b):
        return False
    for u, v in zip(a, b):
        if abs(float(u) - float(v)) > TOL * max(1.0, abs(float(v))):
            return False
    return True


def _oracle(attn, k, nkv, block, tau, alpha, sink):
    """纯 Python 逐公式实现：不 import 任何解法的中间量，也不调用 torch 算子。

    熵用「隔离 sink + 重归一化」的行，池化用**原始**行；页分数是含零填充的整页 max。
    """
    B, H_Q, T = len(attn), len(attn[0]), len(attn[0][0])
    r = H_Q // nkv
    sink = min(sink, T)
    ent = []
    for b in range(B):
        row = []
        for h in range(H_Q):
            p = [0.0 if t < sink else float(attn[b][h][t]) for t in range(T)]
            s = max(math.fsum(p), 1e-12)
            p = [v / s for v in p]
            if alpha == 1.0:
                e = -math.fsum(v * math.log(v + 1e-12) for v in p)
            else:
                e = math.log(math.fsum(v ** alpha for v in p) + 1e-12) / (1.0 - alpha)
            row.append(e)
        ent.append(row)
    pad = (block - T % block) % block
    nb = (T + pad) // block
    wts, pgs, rpm, rtm = [], [], [], []
    for b in range(B):
        wb, pb, mb, tb = [], [], [], []
        for g in range(nkv):
            z = [-ent[b][g * r + i] / tau for i in range(r)]
            m = max(z)
            ex = [math.exp(v - m) for v in z]
            tot = math.fsum(ex)
            w = [v / tot for v in ex]
            pooled = [math.fsum(w[i] * float(attn[b][g * r + i][t]) for i in range(r))
                      for t in range(T)]
            pooled = pooled + [0.0] * pad
            sc = [max(pooled[bl * block:(bl + 1) * block]) for bl in range(nb)]
            order = sorted(range(nb), key=lambda i: (-sc[i], i))[:k]
            mask = [0] * nb
            for i in order:
                mask[i] = 1
            wb.append(w)
            pb.append(sc)
            mb.append(mask)
            tb.append([mask[t // block] for t in range(T)])
        wts.append(wb)
        pgs.append(pb)
        rpm.append(mb)
        rtm.append(tb)
    return {"entropies": ent, "group_weights": wts, "page_scores": pgs,
            "retained_pages": rpm, "retained_tokens": rtm}


def _verify(solve, attn, k, nkv, block=16, tau=0.5, alpha=2.0, sink=4):
    B, H_Q, T = len(attn), len(attn[0]), len(attn[0][0])
    r = H_Q // nkv
    nb = (T + (block - T % block) % block) // block
    t = torch.tensor(attn, dtype=torch.float32)
    snap = t.clone()
    got = solve(t, k, H_Q, nkv, block, tau, alpha, sink)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    exp = _oracle(attn, k, nkv, block, tau, alpha, sink)
    shapes = {"entropies": (B, H_Q), "group_weights": (B, nkv, r), "page_scores": (B, nkv, nb),
              "retained_pages": (B, nkv, nb), "retained_tokens": (B, nkv, T)}
    for key in ("entropies", "group_weights", "page_scores"):
        v = got[key]
        assert torch.is_tensor(v) and v.is_floating_point(), f"{key} 不是浮点张量"
        assert tuple(v.shape) == shapes[key], f"{key} 形状 {tuple(v.shape)}，期望 {shapes[key]}"
        assert _close_all(v.tolist(), exp[key]), \
            f"{key} = {v.tolist()}，期望 {exp[key]}（k={k} nkv={nkv} block={block}）"
    for key in ("retained_pages", "retained_tokens"):
        v = got[key]
        assert torch.is_tensor(v) and v.dtype == torch.int64, f"{key} 不是 int64：{getattr(v, 'dtype', None)}"
        assert tuple(v.shape) == shapes[key], f"{key} 形状 {tuple(v.shape)}，期望 {shapes[key]}"
        assert _flat(v.tolist()) == _flat(exp[key]), f"{key} = {v.tolist()}，期望 {exp[key]}"
    assert torch.equal(t, snap), "attn 被改写"
    return got


MAIN = [[[0.10, 0.20, 0.05, 0.40, 0.30],
         [0.30, 0.10, 0.20, 0.05, 0.15],
         [0.05, 0.45, 0.10, 0.20, 0.25],
         [0.20, 0.05, 0.35, 0.10, 0.30]]]                    # [B=1, H_Q=4, T=5]

TIE = [[[0.5, 0.5, 0.5, 0.5]]]                              # [B=1, H_Q=1, T=4]


def _rows(B, H_Q, T, seed):
    rng = random.Random(seed)
    return [[[round(rng.random(), 3) for _ in range(T)] for _ in range(H_Q)] for _ in range(B)]


def check_example(solve):
    """题面两个例子：分组 top-1 与页分数平手时的页号优先。"""
    got = _verify(solve, MAIN, 1, 2, block=2, tau=0.5, alpha=2.0, sink=1)
    assert [round(v, 4) for v in got["entropies"][0].tolist()] == [1.1267, 1.2040, 1.1552, 1.0454]
    assert [[round(v, 4) for v in g] for g in got["group_weights"][0].tolist()] == \
        [[0.5386, 0.4614], [0.4453, 0.5547]]
    assert [[round(v, 4) for v in g] for g in got["page_scores"][0].tolist()] == \
        [[0.1923, 0.2385, 0.2308], [0.2281, 0.2387, 0.2777]]
    assert got["retained_pages"].tolist() == [[[0, 1, 0], [0, 0, 1]]]
    assert got["retained_tokens"].tolist() == [[[0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]]

    got = _verify(solve, TIE, 1, 1, block=2, tau=0.5, alpha=2.0, sink=0)
    assert [round(v, 4) for v in got["entropies"][0].tolist()] == [1.3863]
    assert got["group_weights"].tolist() == [[[1.0]]]
    assert got["page_scores"].tolist() == [[[0.5, 0.5]]]
    assert got["retained_pages"].tolist() == [[[1, 0]]]
    assert got["retained_tokens"].tolist() == [[[1, 1, 0, 0]]]


check_example.example = True


def check_sinks_are_isolated_in_entropy_only(solve):
    """熵只看 sink 之后的分布；池化仍带着 sink 的原始权重。"""
    rows = [[[100.0, 100.0, 0.1, 0.2]]]
    got = _verify(solve, rows, 1, 1, block=2, tau=0.5, alpha=2.0, sink=2)
    want = -math.log(1.0 / 9.0 + 4.0 / 9.0)          # p = [1/3, 2/3] 的 Rényi-2 熵
    assert abs(got["entropies"][0][0].item() - want) < 1e-6
    assert got["page_scores"][0][0][0].item() == 100.0   # 页 0 的最大值仍在 sink 上

    same = [[[0.1, 0.2, 0.3, 0.4]]]
    no_sink = _verify(solve, same, 1, 1, block=2, tau=0.5, alpha=2.0, sink=0)
    with_sink = _verify(solve, same, 1, 1, block=2, tau=0.5, alpha=2.0, sink=2)
    assert abs(no_sink["entropies"][0][0].item() - with_sink["entropies"][0][0].item()) > 1e-3


def check_alpha_one_is_shannon(solve):
    """alpha == 1.0（含整数 1）走 Shannon 分支，其余走 Rényi 公式。"""
    rows = [[[0.2, 0.3, 0.5, 0.0, 0.0]]]
    shannon = -(0.2 * math.log(0.2) + 0.3 * math.log(0.3) + 0.5 * math.log(0.5))
    a = _verify(solve, rows, 1, 1, block=1, tau=1.0, alpha=1.0, sink=0)
    b = _verify(solve, rows, 1, 1, block=1, tau=1.0, alpha=1, sink=0)
    assert abs(a["entropies"][0][0].item() - shannon) < 1e-6
    assert a["entropies"].tolist() == b["entropies"].tolist()
    c = _verify(solve, rows, 1, 1, block=1, tau=1.0, alpha=2.0, sink=0)
    renyi2 = -math.log(0.04 + 0.09 + 0.25)
    assert abs(c["entropies"][0][0].item() - renyi2) < 1e-6


def check_group_weights_are_softmax_over_group(solve):
    """softmax 在组内 r 个 head 上做；分组是**连续**切分（h // r），不是交错（h % H_KV）。"""
    rows = [[[1.0, 0.0, 0.0, 0.0],
             [0.25, 0.25, 0.25, 0.25],
             [1.0, 0.0, 0.0, 0.0],
             [0.25, 0.25, 0.25, 0.25]]]
    got = _verify(solve, rows, 1, 2, block=2, tau=0.5, alpha=2.0, sink=0)
    w = got["group_weights"][0].tolist()
    assert all(abs(sum(g) - 1.0) < 1e-9 for g in w), w
    assert w[0][0] > 0.9 and w[1][0] > 0.9, w       # 两组都各自押在低熵的 head 上
    ent = got["entropies"][0].tolist()
    want0 = math.exp(-ent[0] / 0.5) / (math.exp(-ent[0] / 0.5) + math.exp(-ent[1] / 0.5))
    assert abs(w[0][0] - want0) < 1e-6


def check_padding_is_max_neutral(solve):
    """T 不是 block_size 整数倍时零填充，nb 用 ceil；补的 0 不该顶替页内最大值。"""
    rows = [[[0.1, 0.2, 0.3, 0.4, 0.5]]]
    got = _verify(solve, rows, 1, 1, block=3, tau=0.5, alpha=2.0, sink=0)
    assert got["page_scores"].shape == (1, 1, 2)
    assert [round(v, 6) for v in got["page_scores"][0][0].tolist()] == [0.3, 0.5]
    assert got["retained_pages"].tolist() == [[[0, 1]]]
    assert got["retained_tokens"].tolist() == [[[0, 0, 0, 1, 1]]]


def check_ties_break_by_lower_page(solve):
    """页分数平手时选页号小的那个；k == nb 全选、k == 0 全不选。"""
    rows = [[[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]]]
    got = _verify(solve, rows, 1, 1, block=2, tau=0.5, alpha=2.0, sink=0)
    assert got["retained_pages"].tolist() == [[[1, 0, 0]]]
    zero = _verify(solve, rows, 0, 1, block=2, tau=0.5, alpha=2.0, sink=0)
    assert zero["retained_pages"].tolist() == [[[0, 0, 0]]]
    assert zero["retained_tokens"].tolist() == [[[0] * 6]]
    full = _verify(solve, rows, 3, 1, block=2, tau=0.5, alpha=2.0, sink=0)
    assert full["retained_pages"].tolist() == [[[1, 1, 1]]]
    assert full["retained_tokens"].tolist() == [[[1] * 6]]


def check_all_sink_row_stays_finite(solve):
    """整行都是 sink（sink_size > T）时归一化后全 0，clamp 保证熵有限、不出 NaN。"""
    rows = [[[0.0, 0.0, 0.0], [0.4, 0.1, 0.5]]]
    for alpha in (1.0, 2.0, 3.0):
        got = _verify(solve, rows, 1, 1, block=2, tau=0.5, alpha=alpha, sink=9)
        ent = got["entropies"][0].tolist()
        assert all(math.isfinite(v) for v in ent), ent
        assert ent[0] == ent[1], ent        # 两行都被削成全 0，熵相同
        assert got["retained_pages"].tolist() == [[[0, 1]]]   # 池化仍用原始行


def check_batch_and_heads_are_independent(solve):
    """batch 维与 kv head 维互不串：每 (b, g) 只看自己组内的行。"""
    rows = _rows(2, 6, 9, 4)
    for nkv in (1, 2, 3, 6):
        got = _verify(solve, rows, 2, nkv, block=4, tau=0.5, alpha=2.0, sink=1)
        assert tuple(got["page_scores"].shape) == (2, nkv, 3)
    got = _verify(solve, rows, 2, 2, block=4, tau=0.5, alpha=2.0, sink=1)
    assert got["retained_pages"][0].tolist() != got["retained_pages"][1].tolist() or \
        got["page_scores"][0].tolist() != got["page_scores"][1].tolist()
    assert got["group_weights"][0].tolist() != got["group_weights"][1].tolist()


def check_temperature_and_alpha_shift_weights(solve):
    """tau_g 越小组内越「赢者通吃」；alpha 改变熵也就改变权重，但页掩码仍可能稳定。"""
    rows = [[[0.9, 0.1, 0.1, 0.1],
             [0.1, 0.1, 0.1, 0.9],
             [0.4, 0.4, 0.4, 0.4]]]
    cool = _verify(solve, rows, 1, 1, block=2, tau=0.05, alpha=2.0, sink=0)
    warm = _verify(solve, rows, 1, 1, block=2, tau=10.0, alpha=2.0, sink=0)
    gemax = max(cool["group_weights"][0][0].tolist())
    cwmax = max(warm["group_weights"][0][0].tolist())
    assert gemax > cwmax, (gemax, cwmax)


def check_against_oracle(solve):
    """网格遍历：B x 组比 x T x block_size x tau_g x alpha x sink_size x k 全与纯 Python oracle 对拍。"""
    rng = random.Random(20261031)
    for _ in range(80):
        B = rng.choice([1, 2])
        nkv = rng.choice([1, 2, 3])
        r = rng.choice([1, 2, 3])
        H_Q = nkv * r
        T = rng.choice([1, 2, 5, 7, 8, 13, 16, 17, 31])
        block = rng.choice([1, 2, 3, 4, 8, 16])
        tau = rng.choice([0.05, 0.5, 1.0, 4.0])
        alpha = rng.choice([1.0, 1.5, 2.0, 3.0])
        sink = rng.choice([0, 1, 2, 4, 8])
        nb = (T + (block - T % block) % block) // block
        k = rng.randint(0, nb)
        _verify(solve, _rows(B, H_Q, T, rng.randrange(10 ** 6)), k, nkv,
                block=block, tau=tau, alpha=alpha, sink=sink)


def check_value_errors(solve):
    """张量类型/维度/非负、整型参数、组比、k 上界、tau_g/alpha 全部抛 ValueError。"""
    ok = torch.tensor(MAIN, dtype=torch.float32)
    cases = [
        ([[[0.1, 0.2]]], 1, 1, 1),
        (torch.zeros(4), 1, 1, 1),
        (torch.zeros(1, 4), 1, 1, 1),
        (torch.zeros(1, 1, 1, 1), 1, 1, 1),
        (ok.to(torch.int64), 1, 4, 2),
        (torch.full((1, 1, 4), -0.1), 1, 1, 1),
        (torch.zeros(0, 1, 4), 1, 1, 1),
        (torch.zeros(1, 0, 4), 1, 1, 1),
        (torch.zeros(1, 1, 0), 1, 1, 1),
        (ok, 1, 3, 2),          # num_query_heads != H_Q
        (ok, 1, 4, 3),          # H_Q 不能被 num_kv_heads 整除
        (ok, 4, 4, 2),          # k > nb
        (ok, -1, 4, 2),
        (ok, True, 4, 2),
        (ok, 1.0, 4, 2),
        (ok, 1, True, 2),
        (ok, 1, 4, 0),
        (ok, 1, 4, 2.0),
    ]
    for args in cases:
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"非法参数没抛 ValueError：{args}")

    base = (ok, 1, 4, 2)
    for kw in ({"block_size": 0}, {"block_size": True}, {"block_size": 2.0},
               {"sink_size": -1}, {"sink_size": False}, {"tau_g": 0.0}, {"tau_g": -0.5},
               {"tau_g": True}, {"tau_g": "0.5"}, {"alpha": 0}, {"alpha": -1.0},
               {"alpha": False}, {"alpha": None}):
        try:
            solve(*base, **kw)
        except ValueError:
            continue
        raise AssertionError(f"非法关键字参数没抛 ValueError：{kw}")
