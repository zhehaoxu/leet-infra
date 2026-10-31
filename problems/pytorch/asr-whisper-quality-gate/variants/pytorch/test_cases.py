import zlib


def _cr(text):
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b))


def _py(windows, temperatures, crt, lpt, nst):
    """oracle：独立实现（以 while 逐温度推进、先算 skip 再合并输出）。"""
    out = []
    for w in windows:
        crs = []
        for s in w["texts"]:
            raw = s.encode("utf-8")
            packed = zlib.compress(raw)
            crs.append(len(raw) / len(packed))
        i = 0
        chain = []
        sel = len(temperatures) - 1
        covered = False
        while i < len(temperatures):
            chain.append(i)
            too_rep = crs[i] > crt
            too_low = w["avg_logprob"][i] < lpt
            is_silence = w["no_speech_prob"][i] > nst and too_low
            nf = too_rep or too_low
            if is_silence:
                nf = False
                covered = True
            if not nf:
                sel = i
                break
            i += 1
            covered = False
        skip = w["no_speech_prob"][sel] > nst and not w["avg_logprob"][sel] > lpt
        out.append({"compression_ratios": crs, "selected_temp_index": sel,
                    "fallback_chain": chain, "silence_covered": covered,
                    "action": "skip" if skip else "transcribe"})
    return out


def _compare(solve, windows, temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
             crt=2.4, lpt=-1.0, nst=0.6):
    led = solve(windows, temperatures, crt, lpt, nst)
    ref = _py(windows, temperatures, crt, lpt, nst)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


GOOD = {"texts": ["the quick brown fox jumps over the lazy dog near the river bank today."] * 6,
        "avg_logprob": [-0.2] * 6, "no_speech_prob": [0.1] * 6}


def _win(n, texts=None, alp=-0.2, nsp=0.1):
    return {"texts": (texts or ["hello world"]) * n if texts is None else texts,
            "avg_logprob": [alp] * n, "no_speech_prob": [nsp] * n}


def check_anchor_accept_first(solve):
    led = _compare(solve, [GOOD])
    assert len(led) == 1
    r = led[0]
    assert r["selected_temp_index"] == 0
    assert r["fallback_chain"] == [0]
    assert r["silence_covered"] is False
    assert r["action"] == "transcribe"


def check_fallback_on_compression(solve):
    # 温度 0 重复文本 → cr 高回退；温度 1 正常
    w = _win(6, texts=["abc" * 20, "hello world", "hello world", "hi", "hi", "hi"])
    led = _compare(solve, [w])
    r = led[0]
    assert r["compression_ratios"][0] > 2.4
    assert r["selected_temp_index"] == 1
    assert r["fallback_chain"] == [0, 1]
    assert r["action"] == "transcribe"


def check_silence_cover(solve):
    # nsp 高且 alp 低：每温度都被 silence 覆盖 → 温度 0 就"通过"，
    # 但段级因 nsp 高且 alp 不高于阈值 → skip
    w = {"texts": ["hello world"] * 6,
         "avg_logprob": [-2.0] * 6, "no_speech_prob": [0.9] * 6}
    led = _compare(solve, [w])
    r = led[0]
    assert r["selected_temp_index"] == 0
    assert r["fallback_chain"] == [0]
    assert r["silence_covered"] is True
    assert r["action"] == "skip"


def check_skip_exemption(solve):
    # nsp 高但 alp 高（> -1.0）→ 豁免，不 skip
    w = {"texts": ["hello"] * 6, "avg_logprob": [0.5] * 6, "no_speech_prob": [0.9] * 6}
    led = _compare(solve, [w])
    assert led[0]["action"] == "transcribe"
    assert led[0]["silence_covered"] is False  # alp 不 < 阈值，覆盖不成立
    # nsp 低 → 不 skip
    w2 = {"texts": ["hello"] * 6, "avg_logprob": [-2.0] * 6, "no_speech_prob": [0.3] * 6}
    led2 = _compare(solve, [w2])
    assert led2[0]["action"] == "transcribe"


def check_all_temps_exhausted(solve):
    # 每温度都 cr 高且 alp 低（nsp 低，无覆盖）→ 用尽全部温度选最后一个
    rep = "xyzxyzxyzxyzxyzxyz"
    w = {"texts": [rep] * 6, "avg_logprob": [-2.0] * 6, "no_speech_prob": [0.1] * 6}
    led = _compare(solve, [w])
    r = led[0]
    assert r["fallback_chain"] == [0, 1, 2, 3, 4, 5]
    assert r["selected_temp_index"] == 5
    assert r["silence_covered"] is False
    assert r["action"] == "transcribe"  # nsp 低不 skip（官方无 fail 概念）


def check_boundary_values(solve):
    # cr == 阈值恰等：不触发回退（严格 >）
    hello_bytes = "hello".encode("utf-8")
    hello_cr = len(hello_bytes) / len(zlib.compress(hello_bytes))
    w = {"texts": ["hello"] * 6, "avg_logprob": [0.0] * 6, "no_speech_prob": [0.0] * 6}
    led = _compare(solve, [_win(2, texts=["hello", "hello"])], temperatures=(0.0, 1.0),
                   crt=hello_cr, lpt=-1.0, nst=0.6)
    assert led[0]["selected_temp_index"] == 0  # cr == crt 恰等不回退（严格 >）
    # alp == -1.0 恰等：不 < 阈值 → 无覆盖、无回退；段级豁免要 alp > -1.0 不成立 → nsp 高则 skip
    w2 = {"texts": ["hello"] * 6, "avg_logprob": [-1.0] * 6, "no_speech_prob": [0.9] * 6}
    led2 = _compare(solve, [w2])
    assert led2[0]["silence_covered"] is False
    assert led2[0]["action"] == "skip"
    # nsp == 0.6 恰等：不 > 阈值 → 不 skip
    w3 = {"texts": ["hello"] * 6, "avg_logprob": [-2.0] * 6, "no_speech_prob": [0.6] * 6}
    led3 = _compare(solve, [w3])
    assert led3[0]["action"] == "transcribe"


def check_custom_thresholds_and_temps(solve):
    led = _compare(solve, [_win(2, nsp=0.0)], temperatures=(0.0, 0.5), crt=1.0, lpt=-0.1, nst=0.05)
    # alp=-0.2 < -0.1 → 回退；温度 1 同样 → 用尽选 1
    assert led[0]["selected_temp_index"] == 1
    assert led[0]["fallback_chain"] == [0, 1]


def check_empty_text_and_multibyte(solve):
    # 空文本：cr = 0 / len(zlib.compress(b''))——zlib 有头开销，cr < 1
    w = {"texts": ["", "hello"] + ["hello"] * 4,
         "avg_logprob": [0.0] * 6, "no_speech_prob": [0.0] * 6}
    led = _compare(solve, [w])
    assert led[0]["compression_ratios"][0] < 1.0
    # 多字节 utf-8 按字节数计
    w2 = {"texts": ["你好世界"] * 6, "avg_logprob": [0.0] * 6, "no_speech_prob": [0.0] * 6}
    led2 = _compare(solve, [w2])
    b = "你好世界".encode("utf-8")
    assert abs(led2[0]["compression_ratios"][0] - len(b) / len(zlib.compress(b))) < 1e-12


def check_invalid_inputs(solve):
    def ve(*args):
        try:
            solve(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")
    ve([], (0.0,))
    ve([GOOD], ())                            # 空温度序列
    ve([GOOD], "ab")                          # temperatures 非 list
    ve([GOOD], (0.0, -0.5))                   # 负温度
    ve([{"texts": ["a"] * 2, "avg_logprob": [0.0] * 2, "no_speech_prob": [0.0] * 2}], (0.0,))  # 长度不齐
    ve([{"texts": ["a"], "avg_logprob": [0.0], "no_speech_prob": [0.0]}], (0.0, 1.0))
    ve([{"texts": [1], "avg_logprob": [0.0], "no_speech_prob": [0.0]}], (0.0,))
    ve([{"texts": ["a"], "avg_logprob": ["x"], "no_speech_prob": [0.0]}], (0.0,))
    ve([{"texts": ["a"], "avg_logprob": [float("nan")], "no_speech_prob": [0.0]}], (0.0,))
    ve([{"texts": ["a"], "avg_logprob": [0.0], "no_speech_prob": [True]}], (0.0,))
    ve([{"texts": ["a"], "avg_logprob": [0.0], "no_speech_prob": [0.0], "extra": 1}], (0.0,))
    try:
        solve([_win(1)], (0.0,), True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    ve("nope", (0.0,))


def check_oracle_random(solve):
    import random
    rng = random.Random(20260671)
    words = ["a", "hello", "world", "xyz", "foo", "the"]
    for _ in range(120):
        n_t = rng.randint(1, 4)
        temps = [rng.choice([0.0, 0.2, 0.5]) for _ in range(n_t)]
        n_w = rng.randint(1, 5)
        wins = []
        for _ in range(n_w):
            texts, alps, nsps = [], [], []
            for _i in range(n_t):
                # 一半概率造重复文本（高 cr）
                if rng.random() < 0.5:
                    unit = rng.choice(words)
                    texts.append(unit * rng.randint(1, 12))
                else:
                    texts.append(" ".join(rng.choice(words) for _ in range(rng.randint(1, 8))))
                alps.append(rng.choice([-2.5, -1.5, -1.0, -0.5, 0.0, 0.5]))
                nsps.append(rng.choice([0.1, 0.59, 0.6, 0.61, 0.9]))
            wins.append({"texts": texts, "avg_logprob": alps, "no_speech_prob": nsps})
        led = solve(wins, temps)
        ref = _py(wins, temps, 2.4, -1.0, 0.6)
        assert led == ref, "random mismatch"
