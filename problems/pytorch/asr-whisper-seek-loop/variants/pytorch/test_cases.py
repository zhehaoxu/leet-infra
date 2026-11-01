from collections.abc import Callable

TS = 50364
NF = 3000
STRIDE = 2
TP = 0.02


def _oracle(content_frames, windows, no_speech_threshold=0.6, logprob_threshold=-1.0):
    segments, seeks = [], []
    seek = 0
    widx = 0
    while seek < content_frames:
        w = windows[widx]
        widx += 1
        seg_size = min(NF, content_frames - seek)
        off = seek * 0.01
        toks = w["tokens"]
        if (w["no_speech_prob"] > no_speech_threshold
                and w["avg_logprob"] < logprob_threshold):
            seeks.append(seek + seg_size)
            seek += seg_size
            continue
        ts = [t >= TS for t in toks]
        single = ts[-2:] == [False, True]
        cons = [i + 1 for i in range(len(ts) - 1) if ts[i] and ts[i + 1]]
        if cons:
            slices = list(cons) + ([len(toks)] if single else [])
            last = 0
            for cur in slices:
                st = toks[last:cur]
                segments.append((round(off + (st[0] - TS) * TP, 3),
                                 round(off + (st[-1] - TS) * TP, 3)))
                last = cur
            if single:
                seeks.append(seek + seg_size)
                seek += seg_size
            else:
                lp = toks[last - 1] - TS
                seeks.append(seek + lp * STRIDE)
                seek += lp * STRIDE
        else:
            dur = seg_size * 0.01
            tss = [t for t in toks if t >= TS]
            if tss and tss[-1] != TS:
                dur = (tss[-1] - TS) * TP
            segments.append((round(off, 3), round(off + dur, 3)))
            seeks.append(seek + seg_size)
            seek += seg_size
    return {"segments": segments, "seeks": seeks, "final_seek": seek}


def _compare(solve: Callable, *args):
    led = solve(*args)
    exp = _oracle(*args)
    assert led == exp, f"\nled={led}\nexp={exp}"
    return led


def _w(tokens, nsp=0.1, alp=-0.5):
    return {"tokens": tokens, "no_speech_prob": nsp, "avg_logprob": alp}


def check_single_timestamp_ending(solve):
    # [TS, text, TS+1500]：末单时间戳 → 整窗推进
    led = _compare(solve, 3000, [_w([TS, 100, TS + 1500])])
    assert led["segments"] == [(0.0, 30.0)]
    assert led["seeks"] == [3000] and led["final_seek"] == 3000


def check_unfinished_segment_dropped(solve):
    # 连续双时间戳切分、无末单时间戳 → 末时间戳后未完成段丢弃，seek 到最后时间戳
    led = _compare(solve, 6000, [
        _w([TS, 100, TS + 500, TS + 1000]),
        _w([TS, 300, TS + 1500]),
        _w([TS, 9, TS + 1500]),
    ])
    # 相邻对 index 2,3 → consecutive=[3]；single=ts[-2:]=[T,T]=False
    # 段1 tokens[0:3] → (0,10)；TS+1000 起未完成段丢弃；last=tokens[2] → seek+=500*2=1000
    assert led["segments"][0] == (0.0, 10.0)
    assert led["seeks"][0] == 1000
    # 窗2 offset=10s，single ending 整窗
    assert led["segments"][1] == (10.0, 40.0)
    assert led["final_seek"] == 6000


def check_no_timestamp_whole_window(solve):
    # 无时间戳 → 整窗 30s
    led = _compare(solve, 3000, [_w([100, 200])])
    assert led["segments"] == [(0.0, 30.0)]
    assert led["seeks"] == [3000]


def check_no_timestamp_with_ts_only(solve):
    # 有时间戳但无相邻对：末时间戳决定 duration；tss[-1]==TS（0s）时整窗
    led = _compare(solve, 3000, [_w([100, TS, 200])])
    assert led["segments"] == [(0.0, 30.0)]  # 末 ts==begin → duration=整窗
    led2 = _compare(solve, 3000, [_w([100, TS + 750, 200])])
    assert led2["segments"] == [(0.0, 15.0)]


def check_no_speech_skip(solve):
    # no_speech_prob > 0.6 且 avg_logprob < -1.0 → 跳窗 fast-forward
    led = _compare(solve, 6000, [
        _w([100], nsp=0.9, alp=-2.0),
        _w([TS, 50, TS + 1500]),
    ])
    assert led["segments"] == [(30.0, 60.0)]
    assert led["seeks"] == [3000, 6000]


def check_no_speech_and_gate_not_or(solve):
    # 只满足一个条件 → 正常解码（AND 语义）
    led = _compare(solve, 3000, [_w([100, 200], nsp=0.9, alp=-0.5)])
    assert led["segments"] == [(0.0, 30.0)]
    led2 = _compare(solve, 3000, [_w([100, 200], nsp=0.1, alp=-2.0)])
    assert led2["segments"] == [(0.0, 30.0)]


def check_multi_slice_segments(solve):
    # 交替时间戳（无相邻对）→ else 分支整窗，末时间戳定 duration
    led = _compare(solve, 6000, [
        _w([TS, 1, TS + 250, 2, TS + 500, 3, TS + 750]),
        _w([TS, 4, TS + 1500]),
    ])
    assert led["segments"][0] == (0.0, 15.0)
    assert led["seeks"][0] == 3000
    # 真正的连续双时间戳切分：[TS, t, TS+250, TS+500, t, TS+750]
    led2 = _compare(solve, 6000, [
        _w([TS, 1, TS + 250, TS + 500, 3, TS + 750, 200]),
        _w([TS, 5, TS + 1500]),
        _w([TS, 6, TS + 1500]),
    ])
    # 唯一相邻对在 index 2,3（TS+250,TS+500）→ consecutive=[3]；single=False
    # 段1 tokens[0:3] → (0,5)；TS+500 之后未完成段丢弃；last=tokens[2] → seek+=250*2=500
    assert led2["segments"] == [(0.0, 5.0), (5.0, 35.0), (35.0, 65.0)]
    assert led2["seeks"] == [500, 3500, 6000]


def check_short_tail_window(solve):
    # 45s 音频：第二窗只剩 15s（segment_size=min(3000, 1500)）
    led = _compare(solve, 4500, [
        _w([TS, 1, TS + 1500]),
        _w([TS, 2, TS + 750]),
    ])
    assert led["segments"] == [(0.0, 30.0), (30.0, 45.0)]
    assert led["final_seek"] == 4500


def check_seek_jump_gap(solve):
    # 未完成段丢弃产生 seek 跳跃（时间戳中间，非窗口边界）
    led = _compare(solve, 6000, [
        _w([TS, 1, TS + 300, TS + 600]),
        _w([TS, 3, TS + 1500]),
        _w([TS, 9, TS + 1500]),
    ])
    # 相邻对 index 2,3；single=[T,T]=False → 段 (0,6)，seek += 300*2=600（跳跃！）
    assert led["segments"][0] == (0.0, 6.0)
    assert led["seeks"][0] == 600


def check_empty_tokens_invalid(solve):
    # token 序列 < 2 长度时 ts[-2:] 官方为 [:] 全列表（torch 语义）：
    # 单 token [TS] → ts=[True]，single=([True]==[False,True])=False
    led = _compare(solve, 3000, [_w([TS])])
    # consecutive 空；tss=[TS]，末==begin → 整窗
    assert led["segments"] == [(0.0, 30.0)]


def check_random_cross_check(solve):
    import random
    rng = random.Random(700700)
    for _ in range(300):
        total = NF * rng.randint(1, 4) - rng.randint(0, 1500)
        n_windows = 12  # 足够覆盖最坏推进；多余窗口不被消耗
        wins = []
        for _ in range(n_windows):
            toks = []
            for _ in range(rng.randint(2, 10)):
                if rng.random() < 0.4:
                    toks.append(TS + rng.randrange(0, 1501) * 1)
                else:
                    toks.append(rng.randint(0, 50000))
            wins.append(_w(toks, nsp=rng.choice([0.1, 0.9]),
                           alp=rng.choice([-0.5, -2.0])))
        led = solve(total, wins)
        exp = _oracle(total, wins)
        assert led == exp, (total, wins, led, exp)
