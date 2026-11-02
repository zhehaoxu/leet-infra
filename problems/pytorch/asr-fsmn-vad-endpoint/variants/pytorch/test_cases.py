import math


# ---------------- 官方决策核参考（model.py 原样转录，缓冲层剥离同解） ----------------

def _ref(decibels, scores, opts):
    o = {
        "frame_in_ms": 10, "window_size_ms": 200,
        "sil_to_speech_time_thres": 150, "speech_to_sil_time_thres": 150,
        "speech_2_noise_ratio": 1.0, "do_extend": 1,
        "lookback_time_start_point": 200, "lookahead_time_end_point": 100,
        "max_single_segment_time": 60000, "snr_thres": -100.0,
        "noise_frame_num_used_for_snr": 100, "decibel_thres": -100.0,
        "speech_noise_thres": 0.6, "max_end_silence_time": 800,
        "max_start_silence_time": 3000, "detect_mode": "multiple",
    }
    o.update(opts or {})
    frm_ms = o["frame_in_ms"]
    T = len(scores)
    noise_avg_db = -100.0
    max_end_sil_thresh = o["max_end_silence_time"] - o["speech_to_sil_time_thres"]
    win_size = int(o["window_size_ms"] / frm_ms)
    sil2sp = int(o["sil_to_speech_time_thres"] / frm_ms)
    sp2sil = int(o["speech_to_sil_time_thres"] / frm_ms)
    win_state = [0] * win_size
    win_sum, win_pos, pre = 0, 0, 0
    latency = win_size + (int(o["lookback_time_start_point"] / frm_ms)
                          if o["do_extend"] else 0)
    lookahead = int(o["lookahead_time_end_point"] / frm_ms)
    state = 1                        # 1=NotDetected 2=InSpeech 3=EndDetected
    cstart, cend, latest, csil, nends = -1, -1, 0, 0, 0
    max_time_out = False
    frame_states, changes, segs, fakes = [], [], [], []

    def get_fs(t):
        nonlocal noise_avg_db
        if t >= len(decibels):
            return 0
        db = decibels[t]
        if db < o["decibel_thres"]:
            return 0
        s = scores[t]
        npr = math.log(s) * o["speech_2_noise_ratio"]
        spr = math.log(1.0 - s)
        if math.exp(spr) >= math.exp(npr) + o["speech_noise_thres"]:
            if db - noise_avg_db >= o["snr_thres"] and db >= o["decibel_thres"]:
                return 1
            return 0
        if noise_avg_db < -99.9:
            noise_avg_db = db
        else:
            noise_avg_db = (db + noise_avg_db *
                            (o["noise_frame_num_used_for_snr"] - 1)) / \
                o["noise_frame_num_used_for_snr"]
        return 0

    def vstart(f, fake):
        nonlocal cstart
        assert cstart == -1
        cstart = f

    def vend(f, fake):
        nonlocal cend, nends, latest, state
        for t in range(latest + 1, f):
            latest = t
        cend = f
        if not fake:
            nends += 1
            segs.append([cstart * frm_ms, cend * frm_ms])
        state = 3

    def reset():
        nonlocal state, cstart, cend, csil, latest
        nonlocal win_state, win_sum, win_pos, pre
        csil, latest, cstart, cend = 0, 0, -1, -1
        state = 1
        win_pos = win_sum = 0
        win_state = [0] * win_size
        pre = 0

    def maybe_end(is_final, cur):
        if is_final:
            vend(cur, False)

    for t in range(T):
        if state == 3 and o["detect_mode"] != "multiple":
            frame_states.append(-1)
            continue
        fs = get_fs(t)
        frame_states.append(fs)
        # ---- WindowDetector ----
        win_sum -= win_state[win_pos]
        win_sum += fs
        win_state[win_pos] = fs
        win_pos = (win_pos + 1) % win_size
        if pre == 0 and win_sum >= sil2sp:
            pre, ch = 1, 3
        elif pre == 1 and win_sum <= sp2sil:
            pre, ch = 0, 1
        elif pre == 0:
            ch = 2
        else:
            ch = 0
        changes.append(ch)
        # ---- FSM ----
        if ch == 3:
            csil = 0
            if state == 1:
                st = max(0, t - latency)
                vstart(st, False)
                state = 2
                for x in range(st + 1, t + 1):
                    latest = x
            elif state == 2:
                for x in range(latest + 1, t):
                    latest = x
                if t - cstart + 1 > o["max_single_segment_time"] / frm_ms:
                    vend(t, False)
                elif not (t == T - 1):
                    latest = t
                else:
                    maybe_end(True, t)
        elif ch == 1:
            csil = 0
            if state == 2:
                if t - cstart + 1 > o["max_single_segment_time"] / frm_ms:
                    vend(t, False)
                elif not (t == T - 1):
                    latest = t
                else:
                    maybe_end(True, t)
        elif ch == 0:
            csil = 0
            if state == 2:
                if t - cstart + 1 > o["max_single_segment_time"] / frm_ms:
                    max_time_out = True
                    vend(t, False)
                elif not (t == T - 1):
                    latest = t
                else:
                    maybe_end(True, t)
        elif ch == 2:
            csil += 1
            if state == 1:
                if (o["detect_mode"] == "single" and
                    csil * frm_ms > o["max_start_silence_time"]) or \
                        (t == T - 1 and nends == 0):
                    fakes.append([0, 0])
                    vstart(0, True)
                    vend(0, True)
            elif state == 2:
                if csil * frm_ms >= max_end_sil_thresh:
                    lb = int(max_end_sil_thresh / frm_ms)
                    if o["do_extend"]:
                        lb -= lookahead
                        lb -= 1
                        lb = max(0, lb)
                    vend(t - lb, False)
                elif t - cstart + 1 > o["max_single_segment_time"] / frm_ms:
                    vend(t, False)
                elif o["do_extend"] and not (t == T - 1):
                    if csil <= lookahead:
                        latest = t
                else:
                    maybe_end(True, t)
        if state == 3 and o["detect_mode"] == "multiple":
            reset()

    e_lb = int(max_end_sil_thresh / frm_ms)
    if o["do_extend"]:
        e_lb = max(0, e_lb - lookahead - 1)
    return {
        "frame_states": frame_states, "change_states": changes,
        "segments": segs, "fake_segments": fakes,
        "noise_avg_db": noise_avg_db, "max_time_out": max_time_out,
        "latency_frames": latency, "end_lookback": e_lb,
        "max_end_sil_thresh_ms": max_end_sil_thresh,
    }


def _compare(solve, decibels, scores, opts=None):
    led = solve(list(decibels), list(scores), dict(opts) if opts else None)
    ref = _ref(decibels, scores, opts or {})
    assert set(led) == set(ref) == {
        "frame_states", "change_states", "segments", "fake_segments",
        "noise_avg_db", "max_time_out", "latency_frames", "end_lookback",
        "max_end_sil_thresh_ms"}, led.keys()
    for k in ("frame_states", "change_states", "segments",
              "fake_segments", "max_time_out", "latency_frames",
              "end_lookback", "max_end_sil_thresh_ms"):
        assert led[k] == ref[k], (k, led[k], ref[k])
    assert abs(led["noise_avg_db"] - ref["noise_avg_db"]) < 1e-9
    return led


# ---------------- 小参数手算锚点 ----------------
# opts: frm=10ms, win=50ms(5帧), sil2sp=sp2sil=30ms(3帧), lookback=50ms(5帧),
# lookahead=20ms(2帧), max_end_silence=100ms -> thresh=100-30=70ms(7帧),
# max_single=60000, do_extend=1
# -> latency = 5+5=10, end_lookback = max(0, 7-2-1)=4
SMALL = {
    "window_size_ms": 50, "sil_to_speech_time_thres": 30,
    "speech_to_sil_time_thres": 30, "lookback_time_start_point": 50,
    "lookahead_time_end_point": 20, "max_end_silence_time": 100,
}


def _speech_frames(idx):
    """speech 判定的 (db, score)：speech_prob=log(1-0.01)=log(0.99)，
    noise_prob=log(0.01)；exp(log0.99)=0.99 >= 0.01+0.6 ✓"""
    return (-30.0, 0.01)


def _sil_frames(idx):
    """sil 判定：score=0.9 -> speech=log(0.1), noise=log(0.9)；
    exp(-2.3)=0.1 < 0.9+0.6 ✓"""
    return (-30.0, 0.9)


def check_constants(solve):
    led = _compare(solve, [], [], SMALL)
    assert led["latency_frames"] == 10
    assert led["max_end_sil_thresh_ms"] == 70
    assert led["end_lookback"] == 4
    led2 = _compare(solve, [], [], {})
    assert led2["latency_frames"] == 40      # 20 + 200/10
    assert led2["max_end_sil_thresh_ms"] == 650   # 800-150
    assert led2["end_lookback"] == 65 - 10 - 1    # 54
    # do_extend=0：latency 仅窗长，end 回退不减 lookahead
    led3 = _compare(solve, [], [], {**SMALL, "do_extend": 0})
    assert led3["latency_frames"] == 5
    assert led3["end_lookback"] == 7


def check_frame_states(solve):
    """帧判定账：score 门 + 噪声 EMA 首帧直赋 + decibel 门。"""
    db = [-30.0] * 6
    sc = [0.01, 0.9, 0.9, 0.9, 0.9, 0.9]
    led = _compare(solve, db, sc, SMALL)
    # 帧 0 speech（0.99 >= 0.01+0.6）；帧 1-5 sil；EMA 首个 sil = -30
    assert led["frame_states"] == [1, 0, 0, 0, 0, 0]
    assert abs(led["noise_avg_db"] - (-30.0)) < 1e-9
    # decibel 门：低于 decibel_thres 直接 Sil 且不更新 EMA
    db2 = [-30.0, -200.0, -30.0]
    sc2 = [0.9, 0.9, 0.9]
    led2 = _compare(solve, db2, sc2, {**SMALL, "decibel_thres": -100.0})
    assert led2["frame_states"] == [0, 0, 0]
    assert abs(led2["noise_avg_db"] - (-30.0)) < 1e-9  # 帧1 不更新


def check_window_detector(solve):
    """窗机：3/20 阈值翻转 + 环形缓冲遗忘。"""
    # 用 speech/sil 交替序列核对 change_states
    db, sc = [], []
    seq = [1] * 4 + [0] * 4 + [1] * 4      # 12 帧
    for s in seq:
        d, p = (_speech_frames(0) if s else _sil_frames(0))
        db.append(d)
        sc.append(p)
    led = _compare(solve, db, sc, SMALL)
    cs = led["change_states"]
    # 手算（win=5, th=3；环形缓冲未满不遗忘）：
    #   t0-2: sum=1,2,3 -> 2,2,sil2speech(3)
    #   t3: sum=4 -> speech2speech(0)
    #   t4(fs=0): pos=4 未满, sum=4-0+0=4 -> 0（窗内旧 speech 帧未出窗）
    #   t5: pos=0 遗忘帧0, sum=4-1+0=3 -> 3<=3 -> speech2sil(1)
    #   t6-7: sum=2,1 -> sil2sil(2)
    #   t8-10: sum=1+1=2? t8: pos=3 遗忘帧3(speech), sum=1-1+1=1 -> 2
    #     t9: pos=4 遗忘帧4(sil), sum=1-0+1=2 -> 2
    #     t10: pos=0 遗忘帧5(sil), sum=2-0+1=3 -> sil2speech(3)
    #   t11: sum=4 -> 0
    assert cs == [2, 2, 3, 0, 0, 1, 2, 2, 2, 2, 3, 0]


def check_basic_segment(solve):
    """基础段：起点回退 + 尾延展 + 静音收尾。"""
    # 60 帧：0-19 sil，20-39 speech，40-59 sil（末帧 is_final）
    db = []
    sc = []
    for t in range(60):
        if 20 <= t < 40:
            d, p = _speech_frames(t)
        else:
            d, p = _sil_frames(t)
        db.append(d)
        sc.append(p)
    led = _compare(solve, db, sc, SMALL)
    # 手算：0-19 全 sil -> 窗内无 speech 残留；t20 sum=1, t21 sum=2,
    #   t22 sum=3 -> sil2speech 于 t=22；start = max(0, 22-10) = 12
    assert len(led["segments"]) == 1
    assert led["segments"][0][0] == 120
    # 收尾：静音 sil2sil 自 t=40 起需窗内 speech 出窗后计数，
    #   continous*10 >= 70 在 t=48 触发（continous=7），end = 48-4 = 44
    s, e = led["segments"][0]
    assert e == 440


def check_fake_segment(solve):
    """纯静音：末帧兜底假段（fake，不产真段）。"""
    db = [-30.0] * 10
    sc = [0.9] * 10
    led = _compare(solve, db, sc, SMALL)
    assert led["segments"] == []
    assert led["fake_segments"] == [[0, 0]]


def check_single_mode_timeout(solve):
    """single 模式：静音超时假段 + 终态后跳帧。"""
    db = [-30.0] * 30
    sc = [0.9] * 30
    led = _compare(solve, db, sc, {**SMALL, "detect_mode": "single",
                                   "max_start_silence_time": 100})
    # continous*10 > 100 在帧 11 触发（continous=11）——注意 Sil2Sil 分支
    # 在窗机 change=2 时才走；帧 11 时 csil=11 -> 假段
    assert led["fake_segments"] == [[0, 0]]
    assert led["segments"] == []
    assert led["frame_states"][12] == -1     # 终态后跳帧


def check_max_single_segment(solve):
    """max_single_segment_time 强切（Speech2Speech 分支置位）。"""
    # 长语音 + 小 max_single：段长 > 阈即切
    n = 40
    db = [-30.0] * n
    sc = [0.01] * n
    led = _compare(solve, db, sc,
                   {**SMALL, "max_single_segment_time": 100})
    assert led["max_time_out"] is True
    assert len(led["segments"]) >= 1
    # 段长 40-起点+1 > 100/10=10 -> 起点后 10 帧即切
    assert led["segments"][0][1] - led["segments"][0][0] == 100


def check_multi_segment_reset(solve):
    """多段：ResetDetection 清窗机但不清噪声 EMA。"""
    # 两段噪声 db 不同（-30 / -40），使 EMA 是否随段重置可观测：
    # 官方 ResetDetection 不清 EMA -> 第二段 sil 帧 EMA 从 -30 缓变
    db = []
    sc = []
    for t in range(120):
        if 10 <= t < 25 or 60 <= t < 80:
            d, p = _speech_frames(t)
        else:
            d, p = (-40.0, 0.9) if t >= 40 else (-30.0, 0.9)
        db.append(d)
        sc.append(p)
    led = _compare(solve, db, sc, SMALL)
    assert len(led["segments"]) == 2
    s0, e0 = led["segments"][0]
    s1, e1 = led["segments"][1]
    assert s0 < e0 <= s1 < e1
    # EMA 不清：末值从 -30 向 -40 缓变（未到 -40）；若被重置则首 sil
    # 帧直赋 -40，可据此区分
    assert led["noise_avg_db"] > -40.0
    assert led["noise_avg_db"] < -30.0


def check_randomized(solve):
    """随机对拍 200 组：随机 speech/sil 帧序列。"""
    import random
    rng = random.Random(716)
    for _ in range(200):
        T = rng.randint(20, 120)
        db, sc = [], []
        for _ in range(T):
            if rng.random() < 0.4:
                d, p = _speech_frames(0)
            else:
                d, p = _sil_frames(0)
            db.append(d + rng.uniform(-0.01, 0.01))
            sc.append(p)
        mode = rng.choice(["multiple", "single"])
        opts = {**SMALL, "detect_mode": mode}
        led = solve(list(db), list(sc), dict(opts))
        ref = _ref(db, sc, opts)
        for k in ("frame_states", "change_states", "segments",
                  "fake_segments", "max_time_out", "latency_frames",
                  "end_lookback"):
            assert led[k] == ref[k], (k, led[k], ref[k])


def check_snr_gate(solve):
    """SNR 门：speech 概率门过但 cur_snr < snr_thres -> Sil 且
    **不更新噪声 EMA**（官方 else 分支才更新）。"""
    # EMA 初始 -100：首 sil 帧（score=0.9, db=-30）直赋 EMA=-30
    # 帧 1：speech 概率门过（0.99>=0.01+0.6）但 snr=-31-(-30)=-1 < 0
    #   -> 门失败 Sil，EMA 保持 -30
    # 帧 2：同帧 1（db=-29 -> snr=1>=0 -> Speech）
    db = [-30.0, -31.0, -29.0]
    sc = [0.9, 0.01, 0.01]
    led = _compare(solve, db, sc,
                   {**SMALL, "snr_thres": 0})
    assert led["frame_states"] == [0, 0, 1]
    assert abs(led["noise_avg_db"] - (-30.0)) < 1e-9  # 帧1 门失败不更新


def check_invalid_inputs(solve):
    bad = [
        lambda: solve([-30.0], [0.9, 0.9], None),        # 长度不齐
        lambda: solve([-30.0], [0.9], {"bad_key": 1}),   # 未知 opt
        lambda: solve([-30.0], [0.0], None),             # score=0 域外
    ]
    for fn in bad:
        try:
            fn()
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("expected error")
    # score=0 会让 math.log(0) 崩（官方同），不入域
    try:
        solve([-30.0], [0.0], None)
    except (ValueError, KeyError, math域 := ValueError):
        pass
