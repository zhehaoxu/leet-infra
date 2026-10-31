import random

import torch

KEYS = {"window_sizes", "cache_lens", "evict_steps", "kept_positions"}


def _oracle(init_pos, rows, metric, nl, delta, mcp, rec, dws):
    """按上游字面写法模拟：`cds` / `jump` 都在**每层调用**里 +1，把 num_hidden_layers
    层真的展开来跑，返回 layer 0 的状态。这样参考解里「每 token 推进 nl 格」的折叠
    就不是自证——两条实现只有在 floor((t*nl + l) / (delta*nl)) == floor(t / delta) 成立时同解。
    """
    T, H, P = len(rows), len(rows[0]), len(init_pos)
    cache = [[[int(p) for p in init_pos] for _ in range(H)] for _ in range(nl)]
    cds = jump = jump_layer = 0
    wins, lens, ev = [], [], []
    for t in range(T):
        for l in range(nl):
            for h in range(H):
                cache[l][h].append(P + t)
        step_dw = None
        evicted = False
        for l in range(nl):
            n = len(cache[l][0])
            if metric == "None":
                dw = 0
            elif metric == "slide":
                dw = dws
            else:
                dw = rec + cds // (delta * nl)
                cds += 1
            if l == 0:
                step_dw = dw
            if metric == "None" or n < mcp + dw:
                continue
            if metric == "discontinuous":
                if jump < delta * nl:
                    jump += 1
                    continue
                jump_layer += 1
                if jump_layer == nl:
                    jump = 0
                    jump_layer = 0
            for h in range(H):
                s = cache[l][h]
                region = s[mcp:n - rec]
                ranked = sorted(range(len(region)), key=lambda i: (-rows[t][h][region[i]], i))
                chosen = [region[i] for i in ranked[:dw - rec]]
                cache[l][h] = s[:mcp] + chosen + s[n - rec:]
            evicted = True
        wins.append(step_dw)
        lens.append(len(cache[0][0]))
        if evicted:
            ev.append(t)
    return {"window_sizes": wins, "cache_lens": lens, "evict_steps": ev,
            "kept_positions": [list(s) for s in cache[0]]}


def _mk(init_pos, rows):
    return (torch.tensor(init_pos, dtype=torch.int64),
            torch.tensor(rows, dtype=torch.float32))


def _scores(T, H, W, seed):
    rng = random.Random(seed)
    return [[[round(rng.random(), 3) for _ in range(W)] for _ in range(H)] for _ in range(T)]


def _verify(solve, init_pos, rows, metric, nl, delta, mcp, rec, dws):
    ip, attn = _mk(init_pos, rows)
    snapshot, attn_snapshot = ip.tolist(), attn.clone()
    got = solve(ip, attn, metric, nl, delta, mcp, rec, dws)
    assert isinstance(got, dict) and set(got) == KEYS, f"键不对：{sorted(got)}"
    exp = _oracle(init_pos, rows, metric, nl, delta, mcp, rec, dws)
    T, H = len(rows), len(rows[0])
    shapes = {"window_sizes": (T,), "cache_lens": (T,),
              "evict_steps": (len(exp["evict_steps"]),),
              "kept_positions": (H, len(exp["kept_positions"][0]))}
    for k, v in exp.items():
        t = got[k]
        assert torch.is_tensor(t) and t.dtype == torch.int64, f"{k} 不是 int64：{t.dtype}"
        assert tuple(t.shape) == shapes[k], f"{k} 形状 {tuple(t.shape)}，期望 {shapes[k]}"
        assert t.tolist() == v, f"{metric} nl={nl} d={delta} 的 {k} = {t.tolist()}，期望 {v}"
    assert ip.tolist() == snapshot and attn.shape == attn_snapshot.shape
    assert torch.equal(attn, attn_snapshot), "attn 被改写了"
    return got


MAIN = [[0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
        [0.16, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
        [0.10, 0.11, 0.12, 0.13, 0.20, 0.50, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19],
        [0.19, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.21, 0.22],
        [0.22, 0.19, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20],
        [0.30, 0.31, 0.32, 0.33, 0.34, 0.60, 0.25, 0.70, 0.10, 0.40, 0.20, 0.15]]
MAIN = [[r] for r in MAIN]                    # [T=6, H=1, P+T=12]
TIE = [[[0.40, 0.10, 0.30, 0.20, 0.15, 0.25, 0.35]],
       [[0.25, 0.35, 0.30, 0.30, 0.15, 0.20, 0.40]],
       [[0.05, 0.45, 0.20, 0.30, 0.40, 0.40, 0.10]]]   # [T=3, H=1, P+T=7]


def check_example(solve):
    """题面两个例子：四种 metric 的四个输出键，含突发门、keep == 0 与同分。"""
    r = _verify(solve, [0, 1, 2, 3, 4, 5], MAIN, "discontinuous", 2, 2, 4, 3, 4)
    assert r["window_sizes"].tolist() == [3, 3, 4, 4, 5, 5]
    assert r["cache_lens"].tolist() == [7, 8, 8, 9, 10, 9]
    assert r["evict_steps"].tolist() == [2, 5]
    assert r["kept_positions"].tolist() == [[0, 1, 2, 3, 7, 5, 9, 10, 11]]

    r = _verify(solve, [0, 1, 2, 3, 4, 5], MAIN, "slide", 2, 2, 4, 3, 4)
    assert r["window_sizes"].tolist() == [4] * 6
    assert r["cache_lens"].tolist() == [7, 8, 8, 8, 8, 8]
    assert r["evict_steps"].tolist() == [1, 2, 3, 4, 5]
    assert r["kept_positions"].tolist() == [[0, 1, 2, 3, 7, 9, 10, 11]]

    r = _verify(solve, [0, 1, 2, 3, 4, 5], MAIN, "adaptive", 2, 2, 4, 3, 4)
    assert r["evict_steps"].tolist() == [0, 1, 2, 3, 4, 5]
    assert r["cache_lens"].tolist() == [7, 7, 8, 8, 9, 9]
    assert r["kept_positions"].tolist() == [[0, 1, 2, 3, 7, 6, 9, 10, 11]]

    r = _verify(solve, [0, 1, 2, 3, 4, 5], MAIN, "None", 2, 2, 4, 3, 4)
    assert r["window_sizes"].tolist() == [0] * 6 and r["evict_steps"].numel() == 0
    assert r["cache_lens"].tolist() == [7, 8, 9, 10, 11, 12]
    assert r["kept_positions"].tolist() == [list(range(12))]

    r = _verify(solve, [0, 1, 2, 3], TIE, "adaptive", 1, 1, 2, 1, 3)
    assert r["window_sizes"].tolist() == [1, 2, 3] and r["cache_lens"].tolist() == [3, 4, 5]
    assert r["evict_steps"].tolist() == [0, 1, 2]
    assert r["kept_positions"].tolist() == [[0, 1, 4, 5, 6]]


check_example.example = True


def check_counters_advance_per_layer_call(solve):
    """cds 每层调用 +1 => dw 只由 floor(t / delta) 决定：换 num_hidden_layers 不改窗口序列。"""
    for nl in (1, 2, 4, 8):
        r = _verify(solve, [0, 1, 2, 3, 4, 5], MAIN, "adaptive", nl, 2, 4, 3, 4)
        assert r["window_sizes"].tolist() == [3, 3, 4, 4, 5, 5], (nl, r["window_sizes"].tolist())
        r = _verify(solve, [0, 1, 2, 3, 4, 5], MAIN, "discontinuous", nl, 2, 4, 3, 4)
        assert r["evict_steps"].tolist() == [2, 5], (nl, r["evict_steps"].tolist())
    r = _verify(solve, [0, 1, 2, 3, 4, 5], MAIN, "slide", 4, 2, 4, 3, 4)
    assert r["window_sizes"].tolist() == [4] * 6            # slide 不推进 cds


def check_burst_is_delta_plus_one_periodic(solve):
    """突发门：delta 个 token 只喂计数器，第 delta + 1 个 token 才驱逐。"""
    rows = _scores(18, 1, 10 + 18, 11)
    r = _verify(solve, list(range(10)), rows, "discontinuous", 4, 2, 6, 2, 5)
    assert r["evict_steps"].tolist() == [2, 5, 8, 11, 14, 17]
    r = _verify(solve, list(range(10)), rows, "discontinuous", 4, 1, 6, 2, 5)
    assert r["evict_steps"].tolist() == [1, 3, 5, 7, 9, 11, 13, 15, 17]
    r = _verify(solve, list(range(10)), rows, "adaptive", 4, 2, 6, 2, 5)
    assert r["evict_steps"].tolist() == list(range(18))     # 没有突发门时步步都动


def check_gates_are_in_series(solve):
    """jump 只在预算门已过的调用里推进：预算门卡住时，突发周期整体后移。"""
    rows = _scores(24, 1, 8 + 24, 5)
    ev = _verify(solve, list(range(8)), rows, "discontinuous", 2, 2, 8, 4, 9)["evict_steps"].tolist()
    assert ev[0] == 7 and all(b - a == 3 for a, b in zip(ev, ev[1:])), ev
    ad = _verify(solve, list(range(8)), rows, "adaptive", 2, 2, 8, 4, 9)["evict_steps"].tolist()
    assert ad[0] == 5 and len(ad) > len(ev)


def check_keep_zero_still_evicts(solve):
    """dw == rec 时 keep = 0：这一步仍算驱逐，缓存被砍到 mcp + rec。"""
    r = _verify(solve, [0, 1, 2, 3], TIE, "adaptive", 1, 1, 2, 1, 3)
    assert r["evict_steps"].tolist()[0] == 0 and r["cache_lens"].tolist()[0] == 3
    assert r["kept_positions"].tolist()[0] == [0, 1, 4, 5, 6]
    assert _verify(solve, [0, 1, 2, 3], TIE, "None", 1, 1, 2, 1, 3)["evict_steps"].numel() == 0


def check_prefix_and_tail_survive(solve):
    """前 mcp 个槽位原样冻结、末尾 rec 个无条件保留，中间的才竞争。"""
    for seed in (1, 2, 3):
        rows = _scores(20, 1, 9 + 20, seed)
        for metric in ("slide", "adaptive", "discontinuous"):
            kept = _verify(solve, list(range(9)), rows, metric, 3, 2, 5, 3, 6)
            kept = kept["kept_positions"].tolist()[0]
            assert kept[:5] == list(range(5)), (metric, kept)
            assert kept[-3:] == [26, 27, 28], (metric, kept)
            assert all(5 <= p < 26 for p in kept[5:-3]), (metric, kept)


def check_kept_follow_score_order(solve):
    """选中的槽位按分数降序落位，不排回位置升序；同分时按候选区先后。"""
    kept = _verify(solve, [0, 1, 2, 3, 4, 5], MAIN, "discontinuous",
                   2, 2, 4, 3, 4)["kept_positions"].tolist()[0]
    assert kept[4:6] == [7, 5] and MAIN[5][0][7] > MAIN[5][0][5]
    kept = _verify(solve, [0, 1, 2, 3], TIE, "adaptive", 1, 1, 2, 1, 3)["kept_positions"].tolist()[0]
    assert kept[2:4] == [4, 5] and TIE[2][0][4] == TIE[2][0][5] == 0.40


def check_permuted_init_positions(solve):
    """init_positions 可以无序：冻结的是「前 mcp 个槽位」，不是「位置小于 mcp」。"""
    init = [3, 11, 2, 7]
    rows = _scores(12, 1, 4 + 12, 21)
    for metric in ("adaptive", "discontinuous", "slide"):
        kept = _verify(solve, init, rows, metric, 2, 1, 2, 1, 4)["kept_positions"].tolist()[0]
        assert kept[:2] == [3, 11], (metric, kept)


def check_heads_select_independently(solve):
    """每个 head 用自己的分数行，长度始终相等。"""
    rows = _scores(14, 3, 7 + 14, 33)
    kp = _verify(solve, list(range(7)), rows, "adaptive", 2, 2, 4, 2, 6)
    kp = kp["kept_positions"].tolist()
    assert len(kp) == 3 and len({len(s) for s in kp}) == 1
    assert kp[0] != kp[1] and kp[1] != kp[2] and kp[0] != kp[2], kp
    assert all(p[:4] == [0, 1, 2, 3] for p in kp), kp


def check_cache_len_bounds(solve):
    """slide 封顶在 mcp + decoding_window_size；adaptive 的预算每 delta 个 token 才 +1。"""
    rows = _scores(30, 1, 6 + 30, 7)
    r = _verify(solve, list(range(6)), rows, "slide", 4, 3, 3, 2, 7)
    assert max(r["cache_lens"].tolist()) == 10 == 3 + 7
    assert min(r["cache_lens"].tolist()) == 7 and r["window_sizes"].tolist() == [7] * 30
    r = _verify(solve, list(range(6)), rows, "adaptive", 4, 3, 3, 2, 7)
    assert r["window_sizes"].tolist()[:6] == [2, 2, 2, 3, 3, 3]
    assert r["window_sizes"].tolist()[-1] == 11
    assert all(5 <= c <= w + 5 for c, w in
               zip(r["cache_lens"].tolist(), r["window_sizes"].tolist()))


def check_against_oracle(solve):
    """网格遍历：层数 x delta x 预算 x rec x 四种 metric 全部与逐层 oracle 对拍。"""
    for metric in ("None", "slide", "adaptive", "discontinuous"):
        for nl in (1, 2, 3, 5):
            for delta in (1, 2, 3):
                for P, T in ((3, 12), (7, 9), (11, 16)):
                    rows = _scores(T, 2, P + T, P * 31 + T + nl * 7 + delta)
                    mcp = max(1, P // 3)
                    rec = max(1, P // 2)
                    _verify(solve, list(range(P)), rows, metric, nl, delta, mcp, rec, rec + 3)


def check_value_errors(solve):
    """metric / 参数类型与正负 / 形状 / 取值范围全部抛 ValueError。"""
    ip, attn = _mk([0, 1, 2, 3], TIE)
    for args in [
        (ip, attn, "std", 1, 1, 2, 1, 3), (ip, attn, "Adaptive", 1, 1, 2, 1, 3),
        (ip, attn, "DISCONTINUOUS", 1, 1, 2, 1, 3), (ip, attn, None, 1, 1, 2, 1, 3),
        (ip, attn, "adaptive", 0, 1, 2, 1, 3), (ip, attn, "adaptive", 1, 0, 2, 1, 3),
        (ip, attn, "adaptive", 1, 1, 0, 1, 3), (ip, attn, "adaptive", 1, 1, 2, 0, 3),
        (ip, attn, "adaptive", 1, 1, 2, 1, 0), (ip, attn, "adaptive", -3, 1, 2, 1, 3),
        (ip, attn, "adaptive", True, 1, 2, 1, 3), (ip, attn, "adaptive", 1, 1.0, 2, 1, 3),
        (ip, attn, "adaptive", 1, "1", 2, 1, 3), (ip, attn, "adaptive", 1, 1, 2, 1, 3.0),
        (ip, attn, "slide", 1, 1, 2, 3, 3), (ip, attn, "slide", 1, 1, 2, 2, 2),
        (ip, attn[:, 0], "adaptive", 1, 1, 2, 1, 3),
        (ip, attn.to(torch.int64), "adaptive", 1, 1, 2, 1, 3),
        (ip, torch.zeros(3, 1, 8), "adaptive", 1, 1, 2, 1, 3),
        (ip, attn[:, :, :6], "adaptive", 1, 1, 2, 1, 3),
        (ip, attn[0], "adaptive", 1, 1, 2, 1, 3),
        (ip, torch.zeros(0, 1, 4), "adaptive", 1, 1, 2, 1, 3),
        (ip[1:], attn, "adaptive", 1, 1, 2, 1, 3),
        (ip[:0], attn, "adaptive", 1, 1, 2, 1, 3),
        (ip[0], attn, "adaptive", 1, 1, 2, 1, 3),
        (torch.tensor([[0, 1], [2, 3]]), attn, "adaptive", 1, 1, 2, 1, 3),
        (torch.tensor([0, 1, 1, 3]), attn, "adaptive", 1, 1, 2, 1, 3),
        (torch.tensor([0, 1, 2, 7]), attn, "adaptive", 1, 1, 2, 1, 3),
        (torch.tensor([0, -1, 2, 3]), attn, "adaptive", 1, 1, 2, 1, 3),
        (torch.tensor([0.0, 1.0, 2.0, 3.0]), attn, "adaptive", 1, 1, 2, 1, 3),
        ([0, 1, 2, 3], attn, "adaptive", 1, 1, 2, 1, 3),
        (torch.tensor([0, 1, 2, 3, 4]), attn[:, :, :6], "adaptive", 1, 1, 5, 1, 3),
    ]:
        try:
            solve(*args)
        except ValueError:
            continue
        raise AssertionError(f"非法参数没抛 ValueError：{args[2:]}")
