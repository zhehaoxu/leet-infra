import re

from copy import deepcopy


def _dc(x):
    return deepcopy(x)


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution）：手扫权重表 + finditer 去重

_WSUM = 3.0   # 1+0.8+0.6+0.4+0.2


def _oracle(text_with_calls, positions, cands, tau_s=0.05, tau_f=1.0,
            k=5, m=5):
    # 采样
    tau_dropped, passed = [], []
    for i, p in positions:
        if p > tau_s:
            passed.append((i, p))
        else:
            tau_dropped.append(i)
    ranked = sorted(passed, key=lambda ip: (-ip[1], ip[0]))
    kept = [i for i, _ in ranked[:k]]
    topk_dropped = [i for i, _ in ranked[k:]]
    kept.sort()                                  # 原序输出

    # 评估
    by_pos = {}
    for ci, cd in enumerate(cands):
        for key in ("pos", "call", "lp_plus", "lp_none", "lp_call_only"):
            if key not in cd:
                raise ValueError("missing " + key)
        if not (len(cd["lp_plus"]) == len(cd["lp_none"])
                == len(cd["lp_call_only"])):
            raise ValueError("len mismatch")
        if cd["pos"] not in {i for i, _ in positions}:
            raise ValueError("bad pos")
        by_pos.setdefault(cd["pos"], []).append(ci)
    evaluated = []
    for pos in kept:
        evaluated.extend(by_pos.get(pos, [])[:m])
    ev_set = set(evaluated)
    skipped = [ci for ci in range(len(cands)) if ci not in ev_set]

    scores, kept_calls = [], []
    for ci in evaluated:
        cd = cands[ci]
        lp_ = _loss2(cd["lp_plus"])
        ln_ = _loss2(cd["lp_none"])
        lc_ = _loss2(cd["lp_call_only"])
        l_minus = min(ln_, lc_)
        score = l_minus - lp_
        ok = score >= tau_f
        scores.append({"cand": ci, "pos": cd["pos"], "L_plus": lp_,
                       "L_none": ln_, "L_call_only": lc_,
                       "L_minus": l_minus, "score": score, "kept": ok})
        if ok:
            kept_calls.append(cd["call"])

    # 去重：finditer 手扫
    pat = re.compile(r'(\ \[(\w+)\(([^)]*)\))(\])')
    out, removed, pos_end = [], [], 0
    n = 0
    for mo in pat.finditer(text_with_calls):
        n += 1
        if n == 1:
            continue
        removed.append(mo.group(0))
    # kept_text：删掉第 2..n 个匹配
    pieces, last = [], 0
    n = 0
    for mo in pat.finditer(text_with_calls):
        n += 1
        if n == 1:
            continue
        pieces.append(text_with_calls[last:mo.start()])
        last = mo.end()
    pieces.append(text_with_calls[last:])
    kept_text = "".join(pieces)
    return {"kept_positions": kept,
            "tau_dropped": tau_dropped,
            "topk_dropped": topk_dropped,
            "evaluated": evaluated,
            "skipped_cands": skipped,
            "scores": scores,
            "kept_calls": kept_calls,
            "has_calls": n > 0,
            "kept_text": kept_text,
            "removed": removed}


def _loss2(lps):
    # 又一种写法：直接按 t 查表
    tbl = {}
    for t in range(0, 5):
        tbl[t] = (1 - 0.2 * t) / _WSUM
    tot = 0.0
    for t, lp in enumerate(lps):
        tot += tbl.get(t, 0.0) * lp
    return -tot


# ---------------------------------------------------------------- compare

def _eq(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), (len(a), len(b))
        for x, y in zip(a, b):
            _eq(x, y)
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), \
            (a.keys() ^ b.keys())
        for k in a:
            _eq(a[k], b[k])
        return
    if isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) < 1e-9, (a, b)
        return
    assert a == b, (a, b)


def _compare(solve, *args, **kw):
    got = solve(*[_dc(x) for x in args], **deepcopy(kw))
    exp = _oracle(*[_dc(x) for x in args], **deepcopy(kw))
    _eq(got, exp)
    return got


# ---------------------------------------------------------------- checks

def _cand(pos, call, lpp, lpn, lpc):
    return {"pos": pos, "call": call, "lp_plus": lpp, "lp_none": lpn,
            "lp_call_only": lpc}


def check_sampling(solve):
    """τ_s 严格 >（恰等丢弃）、top-k 按 p 降序稳定（平分保原序）、
    kept_positions 输出保持原序。"""
    positions = [(0, 0.05), (1, 0.06), (2, 0.5), (3, 0.5), (4, 0.4),
                 (5, 0.3), (6, 0.3)]
    led = _compare(solve, "t", positions, [], tau_s=0.05, k=4)
    assert led["tau_dropped"] == [0]            # 0.05 恰等 → 丢
    # 过 τ_s：1(0.06),2(0.5),3(0.5),4(0.4),5(0.3),6(0.3)
    # top-4 by p desc stable: 2(0.5),3(0.5),4(0.4),5(0.3)
    assert led["kept_positions"] == [2, 3, 4, 5]
    assert led["topk_dropped"] == [6, 1]
    # 无 τ_s 淘汰时只截 top-k
    led = _compare(solve, "t", positions[1:], [], tau_s=0.05, k=2)
    assert led["kept_positions"] == [2, 3]
    assert led["topk_dropped"] == [4, 5, 6, 1]


def check_weights_and_loss(solve):
    """w̃_t = max(0, 1-0.2t)、全局归一 /3.0（不按后缀长度重归一）、
    t>=5 权重 0；恰等门用精确可表值（w_2=0.2，0.2*5=1.0 exact）。"""
    # 精确构造：lp 只在 t=2 非零 → L = 0.2*|lp2| 精确
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c0", [0, 0, -5.0, 0, 0],
                          [0, 0, -10.0, 0, 0], [0, 0, -10.0, 0, 0])],
                   tau_s=0.05, tau_f=1.0)
    s = led["scores"][0]
    assert abs(s["L_plus"] - 1.0) < 1e-9
    assert abs(s["L_none"] - 2.0) < 1e-9
    # 浮点 quirk：score = 0.9999999999999999 < 1.0 → 恰等边界被
    # (w̃/3)*lp 的舍入吃掉，tau_f=1.0 时判丢（钉死该行为）
    assert s["kept"] is False
    # 含等门的证据改走精确路：三表全同 → L+=L- → score 恰 0.0
    # （0.0 >= 0 True；`>=`→`>` 变异在此被抓）
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c0c", [-1.0] * 5, [-1.0] * 5, [-1.0] * 5)],
                   tau_f=0.0)
    assert led["scores"][0]["score"] == 0.0
    assert led["scores"][0]["kept"] is True
    # 全 -1 的 5 token 后缀：L = (1+0.8+0.6+0.4+0.2)/3 ≈ 1.0（浮点）
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c0b", [-1.0] * 5, [-2.0] * 5, [-2.0] * 5)])
    assert abs(led["scores"][0]["L_plus"] - 1.0) < 1e-9
    # 2 token 全 -1：L = (1+0.8)/3 = 0.6（不重归一为 1）
    led = _compare(solve, "t", [(3, 1.0)],
                   [_cand(3, "c1", [-1.0] * 2, [-1.0] * 2, [-1.0] * 2)])
    assert abs(led["scores"][0]["L_plus"] - 0.6) < 1e-9
    assert abs(led["scores"][0]["score"] - 0.0) < 1e-9
    # t>=5 权重 0：极端 logp 不影响 L
    lpp = [0.0] * 5 + [-100.0] * 3
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c2", lpp, [0.0] * 8, [0.0] * 8)])
    assert abs(led["scores"][0]["L_plus"] - 0.0) < 1e-9
    # t=1..4 的不同 logp 权重核算：lp=[-3,-2,0,0,0] →
    # L = (1*3 + 0.8*2 + 0)/3 = 4.6/3
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c3", [-3.0, -2.0, 0.0, 0.0, 0.0],
                          [0.0] * 5, [0.0] * 5)])
    assert abs(led["scores"][0]["L_plus"] - 4.6 / 3) < 1e-9


def check_min_branch(solve):
    """L⁻ = min(L(ε), L(e(c,ε))) 双对照：构造 call-only 优于 none
    （call 文本帮了一点、响应是垃圾）与 none 优于 call-only 两路。"""
    # call-only 更小：L_none=2.0、L_call=1.5 → L⁻=1.5；L⁺=1.0
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c", [-1.0] * 5, [-2.0] * 5,
                          [-1.5] * 5)])
    assert abs(led["scores"][0]["L_minus"] - 1.5) < 1e-9
    assert abs(led["scores"][0]["score"] - 0.5) < 1e-9
    # none 更小：L_none=1.2、L_call=2.0 → L⁻=1.2
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c", [-1.0] * 5, [-1.2] * 5, [-2.0] * 5)])
    assert abs(led["scores"][0]["L_minus"] - 1.2) < 1e-9
    # 门下方但 tau_f 放宽到 0.4 → 留（score≈0.5，浮点两侧皆 > 0.4）
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c", [-1.0] * 5, [-2.0] * 5, [-1.5] * 5)],
                   tau_f=0.4)
    assert led["scores"][0]["kept"] is True
    assert led["kept_calls"] == ["c"]
    # score 恰 0.0（三表全同）+ tau_f=0 → 含等门留
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c", [-1.0] * 5, [-1.0] * 5, [-1.0] * 5)],
                   tau_f=0.0)
    assert led["scores"][0]["kept"] is True
    # 门下方：score < tau_f → 丢
    led = _compare(solve, "t", [(0, 1.0)],
                   [_cand(0, "c", [0, 0, -5.01, 0, 0],
                          [0, 0, -10.0, 0, 0], [0, 0, -10.0, 0, 0])],
                   tau_f=1.0)
    assert led["scores"][0]["kept"] is False
    assert led["kept_calls"] == []


def check_m_cap_and_skips(solve):
    """每位置前 m 个候选（输入序）、其余进 skipped；位置未过采样的
    候选不评估；缺键/长度不一致/坏 pos → ValueError。"""
    positions = [(0, 0.9), (1, 0.1)]
    cands = [_cand(0, "a", [-1.0] * 5, [-2.0] * 5, [-2.0] * 5),
             _cand(0, "b", [-0.5] * 5, [-2.0] * 5, [-2.0] * 5),
             _cand(0, "c", [-0.1] * 5, [-2.0] * 5, [-2.0] * 5),
             _cand(1, "d", [-3.0] * 5, [0.0] * 5, [0.0] * 5)]
    led = _compare(solve, "t", positions, cands, tau_s=0.05, k=5, m=2)
    assert led["evaluated"] == [0, 1, 3]        # 位置 0 前 m=2 + 位置 1 的 d
    assert led["skipped_cands"] == [2]          # c 超 m 上限
    assert led["topk_dropped"] == []            # k=5 两位置全保
    assert [s["cand"] for s in led["scores"]] == [0, 1, 3]
    # m=5 时 d 也不评估（位置 1 已被 k=1 截）
    led = _compare(solve, "t", positions, cands, k=1, m=5)
    assert led["evaluated"] == [0, 1, 2]
    assert led["skipped_cands"] == [3]
    assert led["topk_dropped"] == [1]
    try:
        solve("t", positions, [{"pos": 0, "call": "x",
                                "lp_plus": [-1.0] * 5,
                                "lp_none": [-1.0] * 5}],
              tau_s=0.05)
        assert False
    except ValueError:
        pass
    try:
        solve("t", positions, [_cand(0, "x", [-1.0] * 5,
                                     [-1.0] * 4, [-1.0] * 5)])
        assert False
    except ValueError:
        pass
    try:
        solve("t", positions, [_cand(9, "x", [-1.0] * 5,
                                     [-1.0] * 5, [-1.0] * 5)])
        assert False
    except ValueError:
        pass


def check_dedup(solve):
    """去重：保第一个完整匹配、其余替换 ''；removed 记完整匹配；
    正则 quirk——前导空格必须、\\w+ 工具名、args [^)]* 遇 ')' 截断
    （含括号 args 不匹配）。"""
    txt = ("A [calc(2+3)] -> 5 B [calc(4*4)] -> 16 C"
           " [calc(1)] -> 1")
    led = _compare(solve, txt, [], [], tau_s=1.0)
    assert led["has_calls"] is True
    assert led["kept_text"] == ("A [calc(2+3)] -> 5 B"
                                " -> 16 C -> 1")
    assert led["removed"] == [" [calc(4*4)]", " [calc(1)]"]
    # 前导空格 quirk：行首无空格的调用不匹配、不计数
    txt = "[qa(x)] start [qa(y)] end"
    led = _compare(solve, txt, [], [], tau_s=1.0)
    assert led["has_calls"] is True             # ' [qa(y)]' 命中
    assert led["removed"] == []                 # 仅 1 个匹配 → 不去重
    assert led["kept_text"] == "[qa(x)] start [qa(y)] end"
    # 无前导空格的孤立调用：不匹配 → has_calls False
    led = _compare(solve, "[only(1)]", [], [], tau_s=1.0)
    assert led["has_calls"] is False
    assert led["removed"] == []
    # args 含 ')' → [^)]* 截断后不接 ']' → 整个不匹配
    led = _compare(solve, "x [f((1+2)*3)] y", [], [], tau_s=1.0)
    assert led["has_calls"] is False
    # 空工具名：\w+ 不匹配 ' [(2)]'（\w* 变异会把它当第二个匹配删掉）
    led = _compare(solve, "a [f(1)] b [(2)] c", [], [], tau_s=1.0)
    assert led["has_calls"] is True
    assert led["removed"] == []                 # 只匹配 1 个 → 不删
    assert led["kept_text"] == "a [f(1)] b [(2)] c"
    # 多参数正常匹配
    led = _compare(solve, "v [mt(hi, de)] v", [], [], tau_s=1.0)
    assert led["has_calls"] is True
    # 单匹配不去重（kept_text 原样）
    led = _compare(solve, "a [f(1)] b", [], [], tau_s=1.0)
    assert led["kept_text"] == "a [f(1)] b"
    assert led["removed"] == []


def check_random_cross(solve):
    """随机交叉验证 oracle（固定种子可复现）。"""
    import random
    rng = random.Random(20261102)
    tools = ["calc", "qa", "mt", "wiki"]
    for _ in range(120):
        positions = [(i, round(rng.uniform(0, 1), 3))
                     for i in range(rng.randint(0, 8))]
        cands = []
        for _ in range(rng.randint(0, 10) if positions else 0):
            pos = rng.choice(positions)[0]
            n_tok = rng.randint(1, 9)
            lp = [round(rng.uniform(-4, 0), 3) for _ in range(n_tok)]
            cands.append(_cand(pos, "c%d" % rng.randint(0, 99),
                               lp, lp[:], lp[:]))
        # 随机增强文本
        parts = []
        for _ in range(rng.randint(0, 4)):
            parts.append(" ")
            parts.append("[%s(%s)]" % (rng.choice(tools),
                                       "x,y" * rng.randint(1, 2)))
            parts.append(" -> %d " % rng.randint(0, 9))
        txt = "".join(parts) or "plain"
        led = _compare(solve, txt, positions, cands,
                       tau_s=0.05, tau_f=1.0,
                       k=rng.randint(1, 5), m=rng.randint(1, 4))
        assert led["has_calls"] == (" [" in txt and "]" in txt
                                    and "(" in txt)


def check_example(solve):
    """run 模式样例：采样 + 加权 loss + 去重一条龙。"""
    positions = [(0, 0.05), (1, 0.6), (2, 0.7)]
    cands = [_cand(1, "qa1", [0, 0, -5.0, 0, 0],
                   [0, 0, -10.0, 0, 0], [0, 0, -10.0, 0, 0]),
             _cand(2, "qa2", [0, 0, -5.0, 0, 0],
                   [0, 0, -6.0, 0, 0], [0, 0, -6.0, 0, 0])]
    txt = "a [calc(1)] b [calc(2)] c"
    led = _compare(solve, txt, positions, cands, tau_s=0.05,
                   tau_f=0.9, k=2, m=1)
    assert led["tau_dropped"] == [0]
    assert led["kept_positions"] == [1, 2]
    assert led["evaluated"] == [0, 1]           # m=1 每位置取 1
    assert abs(led["scores"][0]["score"] - 1.0) < 1e-9
    assert led["scores"][0]["kept"] is True     # ≈1.0 > 0.9
    assert abs(led["scores"][1]["score"] - 0.2) < 1e-9
    assert led["kept_calls"] == ["qa1"]
    assert led["removed"] == [" [calc(2)]"]


check_example.example = True
