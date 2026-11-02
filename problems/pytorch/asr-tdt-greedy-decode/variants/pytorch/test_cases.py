import random

import torch


# ---------------- oracle（官方 torch_impl 标量转录） ----------------

def _ref(token_logits, duration_logits, model_durations,
         blank_id=0, max_symbols=None):
    if isinstance(token_logits, torch.Tensor):
        token_logits = token_logits.tolist()
    if isinstance(duration_logits, torch.Tensor):
        duration_logits = duration_logits.tolist()
    T = len(token_logits)

    def amax(row):
        best = 0
        for i in range(1, len(row)):
            if row[i] > row[best]:
                best = i
        return best

    tokens, stamps, durs = [], [], []
    t, last_nb_t, lasts, n_forced = 0, -1, 0, 0
    while t < T:
        tok = amax(token_logits[t])
        dur = model_durations[amax(duration_logits[t])]
        if tok == blank_id:
            if dur == 0:
                dur = 1
        else:
            tokens.append(tok)
            stamps.append(t)
            durs.append(dur)
            if t == last_nb_t:
                lasts += 1
            else:
                lasts, last_nb_t = 1, t
            if max_symbols is not None and lasts >= max_symbols:
                t += 1
                n_forced += 1
                continue
        t += dur
    return {"tokens": tokens, "timestamps": stamps, "durations": durs,
            "final_t": t, "forced_blanks": n_forced}


def _compare(solve, tl, dl, md, **kw):
    led = solve(tl, dl, md, **kw)
    ref = _ref(tl, dl, md, **kw)
    assert set(led) == set(ref) == {"tokens", "timestamps", "durations",
                                    "final_t", "forced_blanks"}
    for k in ref:
        assert led[k] == ref[k], (k, led[k], ref[k])
    return led


V, D = 4, 4
MD = [0, 1, 2, 3]          # duration 下标 -> 帧数
BLANK, A, B, C = 0, 1, 2, 3


def _rows(spec, T, width):
    """spec: list[(idx, val)] 其余 -9；构造每帧 argmax 恰为 idx。"""
    rows = []
    for t in range(T):
        row = [-9.0] * width
        idx, val = spec[t % len(spec)]
        row[idx] = val
        rows.append(row)
    return rows


def check_basic(solve):
    """手算锚点：a@0(d1) -> blank(d1) -> blank(d1) -> b@3(d3 跳出句末)。"""
    tl = [[-9, 1.0, -9, -9],       # t0: a
          [1.0, -9, -9, -9],       # t1: blank
          [1.0, -9, -9, -9],       # t2: blank
          [-9, -9, 1.0, -9],       # t3: b
          [1.0, -9, -9, -9],
          [1.0, -9, -9, -9]]
    dl = [[-9, 1.0, -9, -9],       # t0: dur 1
          [-9, -9, 1.0, -9],       # t1: dur 2
          [-9, 1.0, -9, -9],       # t2: dur 1
          [-9, -9, -9, 1.0],       # t3: dur 3
          [-9, -9, 1.0, -9],
          [-9, -9, 1.0, -9]]
    led = _compare(solve, tl, dl, MD)
    assert led["tokens"] == [A, B]
    assert led["timestamps"] == [0, 3]
    assert led["durations"] == [1, 3]
    assert led["final_t"] == 6
    assert led["forced_blanks"] == 0


def check_blank_zero_forced(solve):
    """blank duration=0 -> 强制 1（masked_fill quirk）。"""
    tl = _rows([(BLANK, 1.0)], 4, V)
    dl = _rows([(0, 1.0)], 4, D)          # 每帧 blank + dur 0 -> 强制 1
    led = _compare(solve, tl, dl, MD)
    assert led["tokens"] == []
    assert led["final_t"] == 4            # 每帧进 1


def check_blank_normal(solve):
    """blank 正常 duration 跳帧不发射。"""
    tl = _rows([(BLANK, 1.0)], 7, V)
    dl = _rows([(2, 1.0)], 7, D)          # dur=2
    led = _compare(solve, tl, dl, MD)
    assert led["tokens"] == [] and led["final_t"] == 8   # 0,2,4,6 -> 8 出界


def check_jump_past_end(solve):
    """非 blank duration 跳出句末：发射仍有效（RNN-T 差异点）。"""
    tl = _rows([(BLANK, 1.0), (C, 1.0)], 6, V)
    dl = _rows([(3, 1.0)], 6, D)          # dur=3
    led = _compare(solve, tl, dl, MD)
    assert led["tokens"] == [C]
    assert led["timestamps"] == [3]
    assert led["final_t"] == 6            # 3 + 3 = 6 出界


def check_label_loop_max_symbols(solve):
    """非 blank dur=0 label loop + max_symbols 强制 blank（+1 非 dur）。"""
    tl = _rows([(A, 1.0)], 5, V)
    dl = _rows([(0, 1.0)], 5, D)          # 每帧 a + dur 0
    led = _compare(solve, tl, dl, MD, max_symbols=2)
    # t0: a@0, lasts=1; a@0, lasts=2 -> 强制 blank t=1
    # t1: a@1 lasts=1; a@1 lasts=2 -> 强制 t=2 ... 共 2*5=10 个 a
    assert led["tokens"] == [A] * 10
    assert led["timestamps"] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert led["durations"] == [0] * 10
    assert led["forced_blanks"] == 5
    assert led["final_t"] == 5


def check_no_max_symbols_error(solve):
    """非 blank dur=0 且 max_symbols=None -> ValueError（域约束）。"""
    tl = _rows([(A, 1.0)], 3, V)
    dl = _rows([(0, 1.0)], 3, D)
    try:
        solve(tl, dl, MD)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def check_argmax_ties(solve):
    """argmax 平分取首个下标（token 与 duration 头同规则）。"""
    tl = [[0.5, 0.5, -9, -9],          # token 0/1 平分 -> blank(0)
          [ -9, 0.7, 0.7, -9]]         # a/b 平分 -> a(1)
    dl = [[0.1, 0.1, -9, -9],          # dur 0/1 平分 -> 0 -> blank 强制 1
          [ -9, 0.9, 0.9, -9]]         # 1/2 平分 -> 1
    led = _compare(solve, tl, dl, MD)
    assert led["tokens"] == [A]
    assert led["timestamps"] == [1]
    assert led["durations"] == [1]


def check_nonblank_zero_dur_normal_path(solve):
    """非 blank dur=0 也有 max_symbols=1 的单发射路径。"""
    # a@0 (dur0) lasts=1 >= 1 -> 强制 blank t=1；b@1 (dur2) -> t=3；
    # t3 blank (dur1) -> t4 出界
    tl = [[-9, 1.0, -9, -9],
          [-9, -9, 1.0, -9],
          [1.0, -9, -9, -9],
          [1.0, -9, -9, -9]]
    dl = [[1.0, -9, -9, -9],       # t0: dur 0
          [-9, -9, 1.0, -9],       # t1: dur 2
          [-9, 1.0, -9, -9],       # t2: dur 1
          [-9, 1.0, -9, -9]]       # t3: dur 1
    led = _compare(solve, tl, dl, MD, max_symbols=1)
    assert led["tokens"] == [A, B]
    assert led["timestamps"] == [0, 1]
    assert led["durations"] == [0, 2]
    # max_symbols=1：每个非 blank 发射后都强制 blank（a@0 后 t=1，
    # b@1 后 t=2）
    assert led["forced_blanks"] == 2
    assert led["final_t"] == 4


def check_randomized(solve):
    """随机对拍 200 组。"""
    rng = random.Random(717)
    for _ in range(200):
        T = rng.randint(1, 30)
        tl = [[rng.uniform(-2, 2) for _ in range(V)] for _ in range(T)]
        dl = [[rng.uniform(-2, 2) for _ in range(D)] for _ in range(T)]
        ms = rng.choice([None, 1, 2, 3])
        if ms is None:
            # 域约束：避免非 blank dur=0 死循环——把 dur0 的 argmax
            # 行的 token 头强行设为 blank
            for t in range(T):
                if max(range(D), key=lambda i: (dl[t][i], -i)) == 0:
                    tl[t] = [2.0] + tl[t][1:]
                    tl[t][0] = 3.0     # blank 最高
        led = solve(tl, dl, MD, max_symbols=ms)
        ref = _ref(tl, dl, MD, max_symbols=ms)
        for k in ref:
            assert led[k] == ref[k], (k, led[k], ref[k])


def check_invalid_inputs(solve):
    tl = _rows([(A, 1.0)], 3, V)
    dl = _rows([(1, 1.0)], 3, D)
    bad = [
        lambda: solve([], [], MD),
        lambda: solve(tl, dl[:-1], MD),                    # 帧数不齐
        lambda: solve(tl, dl, [0, -1, 2, 3]),              # 负 duration
        lambda: solve(tl, [[0.5, 0.5]], MD),               # 宽度不齐
    ]
    for fn in bad:
        try:
            fn()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("expected error")
