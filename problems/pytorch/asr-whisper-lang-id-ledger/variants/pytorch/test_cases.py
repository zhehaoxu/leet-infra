import random

import torch


# 语言表（缩微版：token id 与 code 同序，code 允许乱序排列以测 zip 对齐）
LANG_TOKENS = [7, 3, 11, 5, 9]
LANG_CODES = ["zh", "en", "fr", "de", "es"]


def _make_logits(n_audio, V, seed, tie=False, single=False):
    g = torch.Generator().manual_seed(seed)
    if single:
        lg = torch.randn(V, generator=g) * 3
        if tie:
            lg[LANG_TOKENS[0]] = lg[LANG_TOKENS[1]] = 2.0
        return lg
    lg = torch.randn(n_audio, V, generator=g) * 3
    if tie:
        lg[:, LANG_TOKENS[0]] = lg[:, LANG_TOKENS[1]] = 2.0
    return lg


def _py(logits, lang_tokens, lang_codes, language=None):
    """oracle：decoding.py::detect_language 官方逻辑原样实现。"""
    if not lang_tokens or not lang_codes:
        raise ValueError("This model doesn't have language tokens "
                         "so it can't perform lang id")
    out = {"detected": language is None}
    if language is not None:
        out.update({"language": language, "windows_consumed": [],
                    "tokens": None, "probs": None, "prob_sum": None,
                    "top": None, "raw_top_prob": None})
        return out
    single = logits.ndim == 1
    if single:
        logits = logits.unsqueeze(0)
    n_audio = logits.shape[0]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(lang_tokens)] = False
    sup = logits.double().clone()
    sup[:, mask] = -float("inf")
    language_tokens = sup.argmax(dim=-1)
    probs_t = torch.softmax(sup, dim=-1)
    raw_t = torch.softmax(logits.double(), dim=-1)
    all_probs = []
    for i in range(n_audio):
        d = {c: float(probs_t[i, j])
             for j, c in zip(lang_tokens, lang_codes)}
        all_probs.append(d)
    tops = [max(d, key=d.get) for d in all_probs]
    raws = []
    for i, top in enumerate(tops):
        j = lang_codes.index(top)
        raws.append(float(raw_t[i, lang_tokens[j]]))
    toks = language_tokens.tolist()
    out.update({
        "language": tops[0],
        "windows_consumed": [0],
        "tokens": toks[0] if single else toks,
        "probs": all_probs[0] if single else all_probs,
        "prob_sum": ([sum(d.values()) for d in all_probs] if not single
                     else sum(all_probs[0].values())),
        "top": tops[0] if single else tops,
        "raw_top_prob": raws[0] if single else raws,
    })
    return out


def _compare(solve, logits, language=None):
    led = solve(logits, LANG_TOKENS, LANG_CODES, language)
    ref = _py(logits, LANG_TOKENS, LANG_CODES, language)
    assert set(led) == set(ref) == {"detected", "language",
                                    "windows_consumed", "tokens", "probs",
                                    "prob_sum", "top", "raw_top_prob"}
    assert led["detected"] == ref["detected"]
    assert led["language"] == ref["language"]
    assert led["windows_consumed"] == ref["windows_consumed"]
    if language is not None:
        assert led["tokens"] is None and led["probs"] is None
        return led
    assert led["tokens"] == ref["tokens"]
    assert led["top"] == ref["top"]
    ref_probs = ref["probs"]
    led_probs = led["probs"]
    led_sums = led["prob_sum"]
    ref_sums = ref["prob_sum"]
    led_raws = led["raw_top_prob"]
    ref_raws = ref["raw_top_prob"]
    if isinstance(ref_probs, dict):
        ref_probs, led_probs = [ref_probs], [led_probs]
        ref_sums, led_sums = [ref_sums], [led_sums]
        ref_raws, led_raws = [ref_raws], [led_raws]
    for rp, lp in zip(ref_probs, led_probs):
        assert list(rp.keys()) == list(lp.keys())
        for k in rp:
            assert abs(rp[k] - lp[k]) < 1e-9, k
        assert abs(sum(lp.values()) - 1.0) < 1e-9
    for a, b in zip(led_sums, ref_sums):
        assert abs(a - b) < 1e-9
    for a, b in zip(led_raws, ref_raws):
        assert abs(a - b) < 1e-9
    return led


def check_renormalization(solve):
    """抑制后 softmax = 语言集合上重归一化；raw 概率作对照必不同。"""
    logits = _make_logits(1, 16, 715, single=True)
    led = _compare(solve, logits)
    assert led["detected"] is True and led["windows_consumed"] == [0]
    assert abs(led["prob_sum"] - 1.0) < 1e-9
    assert set(led["probs"].keys()) == set(LANG_CODES)
    # raw 与重归一化不同（非语言质量被剔除）
    assert led["raw_top_prob"] < led["probs"][led["top"]]
    # 手算核对：重归一化 = raw / sum(raw over languages)
    raw_all = torch.softmax(logits.double(), dim=-1)
    denom = sum(float(raw_all[j]) for j in LANG_TOKENS)
    top_idx = LANG_CODES.index(led["top"])
    expect = float(raw_all[LANG_TOKENS[top_idx]]) / denom
    assert abs(led["probs"][led["top"]] - expect) < 1e-9


def check_argmax_and_max_tie(solve):
    """argmax 平分取首个下标；max(probs) 平分取首个插入键。"""
    logits = _make_logits(1, 16, 1, tie=True, single=True)
    led = _compare(solve, logits)
    # argmax 平分取"下标较小"者（torch.argmax 首个最大）：token 3 < 7
    assert led["tokens"] == min(LANG_TOKENS[0], LANG_TOKENS[1])
    assert led["top"] == LANG_CODES[0]              # max 平分取首个插入键
    # 平分时二语言重归一化概率相等
    assert abs(led["probs"]["zh"] - led["probs"]["en"]) < 1e-12


def check_batch_and_single(solve):
    """1D 输入 -> 标量 + 单 dict；2D 输入 -> 列表 + dict 列表。"""
    lg1 = _make_logits(1, 16, 3, single=True)
    led1 = _compare(solve, lg1)
    assert isinstance(led1["tokens"], int)
    assert isinstance(led1["probs"], dict)
    lg2 = _make_logits(3, 16, 4)
    led2 = _compare(solve, lg2)
    assert isinstance(led2["tokens"], list) and len(led2["tokens"]) == 3
    assert isinstance(led2["probs"], list) and len(led2["probs"]) == 3
    # 单条与批内同数据同结果（unsqueeze 语义）
    led1b = _compare(solve, lg1.unsqueeze(0))
    assert led1b["tokens"] == [led1["tokens"]]
    assert list(led1b["probs"][0].items()) == list(led1["probs"].items())


def check_language_skip(solve):
    """transcribe 显式 language -> 跳过检测，首窗策略账。"""
    led = _compare(solve, _make_logits(1, 16, 5), language="fr")
    assert led["detected"] is False
    assert led["language"] == "fr" and led["windows_consumed"] == []
    assert led["tokens"] is None and led["probs"] is None


def check_randomized(solve):
    """随机对拍 200 组。"""
    rng = random.Random(715)
    for t in range(200):
        n = rng.choice([1, 2, 4])
        V = rng.choice([128, 16, 1024])
        lg = _make_logits(n, V, 1000 + t, single=(n == 1 and rng.random() < 0.5))
        led = solve(lg, LANG_TOKENS, LANG_CODES)
        ref = _py(lg, LANG_TOKENS, LANG_CODES)
        assert led["tokens"] == ref["tokens"]
        assert led["top"] == ref["top"]
        rp = ref["probs"] if isinstance(ref["probs"], list) else [ref["probs"]]
        lp = led["probs"] if isinstance(led["probs"], list) else [led["probs"]]
        for rd, ld in zip(rp, lp):
            for k in rd:
                assert abs(rd[k] - ld[k]) < 1e-9
        for k in ("prob_sum", "raw_top_prob"):
            rv = ref[k]
            lv = led[k]
            if not isinstance(rv, list):
                rv, lv = [rv], [lv]
            for a, b in zip(rv, lv):
                assert abs(a - b) < 1e-9


def check_invalid_inputs(solve):
    lg = _make_logits(1, 16, 6)
    bad = [
        lambda: solve(lg, [], []),                       # 空语言表
        lambda: solve(lg, [50259], ["zh", "en"]),        # 长度不齐
        lambda: solve(lg.unsqueeze(0).unsqueeze(0),
                      LANG_TOKENS, LANG_CODES),          # 3D
    ]
    for fn in bad:
        try:
            fn()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("expected error")
