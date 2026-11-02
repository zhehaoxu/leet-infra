import math

from copy import deepcopy


def _dc(x):
    return deepcopy(x)


# ---------------------------------------------------------------- oracle
# 独立实现（不 import solution）：logprob 表直接用 dict.get 兜底 -100、
# 索引扫描用 next()、聚合用 dict 逐项累加。

RET = ("[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]")
REL = ("[Irrelevant]", "[Relevant]")
GRD = ("[Fully supported]", "[Partially supported]",
       "[No support / Contradictory]")
UT = ("[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]",
      "[Utility:5]")
UT_W = (-1.0, -0.5, 0.0, 0.5, 1.0)
CTRL = ("[Fully supported]", "[Partially supported]",
        "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
        "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>",
        "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]",
        "[Utility:5]")


def _prob(table, tok):
    return math.exp(float(table.get(tok, -100.0)))


def _first_pos(tokens, names):
    for i in range(len(tokens)):
        if tokens[i] in names:
            return i
    return -1


def _oracle(no_retr, evidences, second_text, mode="adaptive_retrieval",
            threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5,
            use_seqscore=False, use_groundness=False, use_utility=False,
            closed=False):
    for e in evidences:
        if len(e["tokens"]) != len(e["lps"]):
            raise ValueError("length mismatch")
        if not e["lps"]:
            raise ValueError("empty generation")

    tbl = no_retr["lp0"]
    ratio = None
    if mode == "always_retrieve":
        go = True
    elif mode == "no_retrieval":
        go = False
    elif threshold is not None:
        vals = {}
        for tok in RET:
            if tok not in tbl:
                raise ValueError("missing gate token")
            vals[tok] = float(tbl[tok])
        ratio = vals["[Retrieval]"] / (vals["[Retrieval]"]
                                      + vals["[No Retrieval]"])
        go = ratio > threshold
    else:
        go = "[Retrieval]" in no_retr["text"]

    res = {}
    if mode != "always_retrieve":
        res["no_retrieval"] = (no_retr["text"], None)

    sc = []
    ans_pred = None
    if go:
        if not evidences:
            raise ValueError("no evidence")
        for i, e in enumerate(evidences):
            rel = {t: _prob(e["lps"][0], t) for t in REL}
            r = rel["[Relevant]"] / sum(rel.values())

            g = 0.0
            if use_groundness:
                p = _first_pos(e["tokens"], GRD)
                if p >= 0:
                    d = {t: _prob(e["lps"][p], t) for t in GRD}
                    tot = sum(d.values())
                    g = d["[Fully supported]"] / tot \
                        + 0.5 * d["[Partially supported]"] / tot

            u = 0.0
            if use_utility:
                p = _first_pos(e["tokens"], UT)
                if p >= 0:
                    d = {t: _prob(e["lps"][p], t) for t in UT}
                    tot = sum(d.values())
                    u = sum(UT_W[k] * d["[Utility:%d]" % (k + 1)] / tot
                            for k in range(5))

            sx = math.exp(float(e["cumulative_logprob"])
                          / max(len(e["tokens"]), 1))
            fin = w_rel * r + w_sup * g + w_use * u
            if use_seqscore:
                fin = fin + sx
            sc.append({"idx": i, "relevance": r, "ground": g,
                       "utility": u, "seq_exp": sx, "final": fin})
            res["retrieval_%d" % i] = (e["text"], fin)
            ans_pred = e["text"]
    else:
        ans_pred = second_text

    a2s = None
    path = None
    ascore = None
    if len(res) == 1:
        s = ans_pred
        for t in CTRL:
            s = s.replace(t, "")
        s = s.replace("</s>", "")
        if len(s) == 0:
            answer = ""
        else:
            answer = s[1:] if s[0] == " " else s
    elif closed:
        a2s = {}
        for k in res:
            if k == "no_retrieval":
                continue
            txt = res[k][0]
            for t in CTRL:
                txt = txt.replace(t, "")
            txt = txt.replace("</s>", "").replace("\n", "")
            txt = txt.replace("<|endoftext|>", "")
            a2s[txt] = a2s.get(txt, 0) + res[k][1]
        order = sorted(a2s, key=lambda a: -a2s[a])
        answer = order[0]
        ascore = a2s[order[0]]
    else:
        cand = [(k, res[k][1]) for k in res if k != "no_retrieval"]
        best = None
        for k, v in cand:
            if best is None or v > best[1]:
                best = (k, v)
        path = best[0]
        answer = res[path][0]
        ascore = best[1]

    return {"do_retrieve": go,
            "retrieval_ratio": ratio,
            "result_keys": list(res.keys()),
            "evidence_scores": sc,
            "answer": answer,
            "best_path": path,
            "answer2score": a2s,
            "answer_score": ascore,
            "n_evidence": len(evidences)}


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

_L = math.log
_LP_SYM = {"[No Retrieval]": -1.0, "[Retrieval]": -1.0,
           "[Continue to Use Evidence]": -1.0}


def _ev(text, toks, lps, cum=0.0):
    return {"text": text, "tokens": toks, "lps": lps,
            "cumulative_logprob": cum}


def _rel(p_rel):
    return {"[Relevant]": _L(p_rel), "[Irrelevant]": _L(1 - p_rel)}


def check_retrieval_gate(solve):
    """检索门用**原始 log-prob 相除**（不是 exp 后的概率比）：负数除法
    把方向翻过来——越不想要检索，比值越接近 1，反而越触发。"""
    ev = [_ev("ans", ["ans"], [_rel(0.6)], 0.0)]
    # lp 偏 No Retrieval（-0.1）而 Retrieval 只有 -2.0 → 比值 0.952
    lp = {"[No Retrieval]": -0.1, "[Retrieval]": -2.0,
          "[Continue to Use Evidence]": -3.0}
    led = _compare(solve, {"text": "x [No Retrieval]", "lp0": lp}, ev,
                   "y", threshold=0.5)
    assert led["do_retrieve"] is True
    assert abs(led["retrieval_ratio"] - (-2.0 / -2.1)) < 1e-12
    # 反过来（模型更想检索）→ 比值 0.0476 → 不触发
    lp2 = {"[No Retrieval]": -2.0, "[Retrieval]": -0.1,
           "[Continue to Use Evidence]": -3.0}
    led = _compare(solve, {"text": "x [Retrieval]", "lp0": lp2}, ev,
                   "y", threshold=0.5)
    assert led["do_retrieve"] is False
    assert abs(led["retrieval_ratio"] - (-0.1 / -2.1)) < 1e-12
    assert led["result_keys"] == ["no_retrieval"]
    # 门槛严格大于：恰等 0.5 不触发
    led = _compare(solve, {"text": "x", "lp0": _LP_SYM}, ev, "y",
                   threshold=0.5)
    assert abs(led["retrieval_ratio"] - 0.5) < 1e-12
    assert led["do_retrieve"] is False
    # threshold=None → 退化成字符串子串判定（"[No Retrieval]" 不含
    # "[Retrieval]" 子串）+ ratio 不下发
    led = _compare(solve, {"text": "x [No Retrieval]", "lp0": _LP_SYM},
                   ev, "y")
    assert led["do_retrieve"] is False
    assert led["retrieval_ratio"] is None
    led = _compare(solve, {"text": "x [Retrieval]", "lp0": _LP_SYM}, ev,
                   "y [No Retrieval]")
    assert led["do_retrieve"] is True
    # mode 短路：always_retrieve / no_retrieval 都不看门
    led = _compare(solve, {"text": "x [No Retrieval]", "lp0": _LP_SYM}, ev,
                   "y", mode="always_retrieve", threshold=0.99)
    assert led["do_retrieve"] is True
    assert led["retrieval_ratio"] is None
    a = _compare(solve, {"text": "x [Retrieval]", "lp0": _LP_SYM}, ev, "y",
                 mode="no_retrieval")
    assert a["do_retrieve"] is False
    assert a["result_keys"] == ["no_retrieval"]


def check_second_generation(solve):
    """do_retrieve=False：results 里存的仍是**第一次**生成，但最终答案
    用的是**第二次**生成（追加 [No Retrieval]）并走 postprocess
    （删控制 token、删 </s>、且只剥一个前导空格）。"""
    led = _compare(solve, {"text": "first [Relevant]", "lp0": _LP_SYM},
                   [], " second [Irrelevant]", mode="no_retrieval")
    assert led["result_keys"] == ["no_retrieval"]
    assert led["answer"] == "second "      # 第一次的 first 不出现；尾空格留着
    assert led["answer_score"] is None
    # 空串与纯 token 文本
    led = _compare(solve, {"text": "a", "lp0": _LP_SYM}, [], "[Relevant]",
                   mode="no_retrieval")
    assert led["answer"] == ""
    # 只剥一个前导空格（两个空格只掉一个）
    led = _compare(solve, {"text": "a", "lp0": _LP_SYM}, [], "  x",
                   mode="no_retrieval")
    assert led["answer"] == " x"


def check_segment_scores(solve):
    """relevance = P(Relevant)/两者之和；缺失反思 token 兜底 exp(-100)
    （次正规数，不是 0）——两者都缺时恰为 0.5。"""
    lp0 = _LP_SYM
    led = _compare(solve, {"text": "x", "lp0": lp0},
                   [_ev("ans", ["ans"], [_rel(0.6)], -2.0)], "y",
                   mode="always_retrieve")
    s = led["evidence_scores"][0]
    assert abs(s["relevance"] - 0.6) < 1e-12
    assert abs(s["seq_exp"] - math.exp(-2.0)) < 1e-12
    assert abs(s["final"] - 0.6) < 1e-12      # ground/utility 默认关
    # 两个 rel token 都缺 → exp(-100) 相等 → 恰 0.5
    led = _compare(solve, {"text": "x", "lp0": lp0},
                   [_ev("ans", ["ans"], [{}], 0.0)], "y",
                   mode="always_retrieve")
    assert led["evidence_scores"][0]["relevance"] == 0.5
    # 只给一个 → 归一化后≈1
    led = _compare(solve, {"text": "x", "lp0": lp0},
                   [_ev("ans", ["ans"], [{"[Relevant]": 0.0}], 0.0)], "y",
                   mode="always_retrieve")
    assert abs(led["evidence_scores"][0]["relevance"] - 1.0) < 1e-12
    # 权重：w_rel/w_sup/w_use 全上
    led = _compare(solve, {"text": "x", "lp0": lp0},
                   [_ev("ans", ["ans"], [_rel(0.8)], 0.0)], "y",
                   mode="always_retrieve", w_rel=2.0)
    assert abs(led["evidence_scores"][0]["final"] - 1.6) < 1e-12


def check_ground_score(solve):
    """ground = P(Fully)/Σ + 0.5·P(Partially)/Σ（第三个 token 权重 0），
    只在**第一个** ground token 出现的位置读 logprob；没出现则 0.0。"""
    lp0 = _LP_SYM
    lps = [{}, {"[Fully supported]": _L(0.6),
                "[Partially supported]": _L(0.3),
                "[No support / Contradictory]": _L(0.1)}, {}]
    ev = [_ev("a [Fully supported] b", ["a", "[Fully supported]", "b"],
              lps, 0.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev, "y",
                   mode="always_retrieve", use_groundness=True)
    assert abs(led["evidence_scores"][0]["ground"] - (0.6 + 0.5 * 0.3)) \
        < 1e-12
    # 出现两次 ground token → 只认第一个位置
    lps2 = [{}, {"[Partially supported]": _L(0.2),
                 "[Fully supported]": _L(0.7),
                 "[No support / Contradictory]": _L(0.1)}, {},
            {"[Fully supported]": _L(0.99),
             "[Partially supported]": _L(0.005),
             "[No support / Contradictory]": _L(0.005)}]
    ev2 = [_ev("a [Partially supported] b [Fully supported]",
               ["a", "[Partially supported]", "b", "[Fully supported]"],
               lps2, 0.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev2, "y",
                   mode="always_retrieve", use_groundness=True)
    assert abs(led["evidence_scores"][0]["ground"]
               - (0.7 + 0.5 * 0.2)) < 1e-12
    # 该位置一个 ground token 都没给 → 三个 exp(-100) 相等 → 1/3+1/6=0.5
    ev3 = [_ev("a [Fully supported]", ["a", "[Fully supported]"], [{}, {}],
               0.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev3, "y",
                   mode="always_retrieve", use_groundness=True)
    assert abs(led["evidence_scores"][0]["ground"] - 0.5) < 1e-12
    # 没出现 ground token → 0.0（不是 0.5）
    led = _compare(solve, {"text": "x", "lp0": lp0},
                   [_ev("ans", ["ans"], [_rel(0.5)], 0.0)], "y",
                   mode="always_retrieve", use_groundness=True)
    assert led["evidence_scores"][0]["ground"] == 0.0


def check_utility_score(solve):
    """utility 权重表 [-1,-0.5,0,0.5,1]：均匀分布时恰为 0，偏好
    [Utility:5] 时为正、偏好 [Utility:1] 时为负。"""
    lp0 = _LP_SYM
    u5 = {"[Utility:1]": _L(0.125), "[Utility:2]": _L(0.125),
          "[Utility:3]": _L(0.125), "[Utility:4]": _L(0.125),
          "[Utility:5]": _L(0.5)}
    ev = [_ev("a [Utility:5] b", ["a", "[Utility:5]", "b"], [{}, u5, {}],
              0.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev, "y",
                   mode="always_retrieve", use_utility=True)
    assert abs(led["evidence_scores"][0]["utility"] - 0.375) < 1e-12
    # 均匀 → 0
    uu = {("[Utility:%d]" % i): _L(0.2) for i in range(1, 6)}
    led = _compare(solve, {"text": "x", "lp0": lp0},
                   [_ev("a [Utility:3] b", ["a", "[Utility:3]", "b"],
                        [{}, uu, {}], 0.0)], "y",
                   mode="always_retrieve", use_utility=True)
    assert abs(led["evidence_scores"][0]["utility"]) < 1e-12
    # 偏好 [Utility:1] → 负分（w_use 默认 0.5 也不改变符号）
    u1 = {"[Utility:1]": _L(0.9), "[Utility:2]": _L(0.025),
          "[Utility:3]": _L(0.025), "[Utility:4]": _L(0.025),
          "[Utility:5]": _L(0.025)}
    led = _compare(solve, {"text": "x", "lp0": lp0},
                   [_ev("a [Utility:1] b", ["a", "[Utility:1]", "b"],
                        [{}, u1, {}], 0.0)], "y",
                   mode="always_retrieve", use_utility=True, w_use=1.0)
    assert abs(led["evidence_scores"][0]["utility"] - (-0.875)) < 1e-12
    assert led["evidence_scores"][0]["final"] < 0


def check_default_flags(solve):
    """use_groundness/use_utility 默认关：证据里出现对应反思 token 也
    不计分（对应 CLI 的 --use_groundness / --use_utility 开关）。"""
    lp0 = _LP_SYM
    grd = {"[Fully supported]": _L(0.9), "[Partially supported]": _L(0.05),
           "[No support / Contradictory]": _L(0.05)}
    ut = {"[Utility:1]": _L(0.05), "[Utility:2]": _L(0.05),
          "[Utility:3]": _L(0.05), "[Utility:4]": _L(0.05),
          "[Utility:5]": _L(0.8)}
    ev = [_ev("a [Fully supported] b [Utility:5]",
              ["a", "[Fully supported]", "b", "[Utility:5]"],
              [{}, grd, {}, ut], 0.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev, "y",
                   mode="always_retrieve")
    sc = led["evidence_scores"][0]
    assert sc["ground"] == 0.0 and sc["utility"] == 0.0
    assert abs(sc["final"] - 0.5) < 1e-12      # 只剩 relevance(两 rel 全缺)
    # 打开后才有分
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev, "y",
                   mode="always_retrieve", use_groundness=True,
                   use_utility=True)
    sc = led["evidence_scores"][0]
    assert abs(sc["ground"] - 0.925) < 1e-12
    assert sc["utility"] > 0


def check_seqscore(solve):
    """use_seqscore 追加 exp(cumulative_logprob / max(len(tokens), 1))：
    长度归一，空 token 序列按 1 归一。"""
    lp0 = _LP_SYM
    ev = [_ev("a", ["a"], [_rel(0.9)], -3.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev, "y",
                   mode="always_retrieve", use_seqscore=True)
    assert abs(led["evidence_scores"][0]["seq_exp"] - math.exp(-3.0)) \
        < 1e-12
    assert abs(led["evidence_scores"][0]["final"]
               - (0.9 + math.exp(-3.0))) < 1e-12
    # 两条 token、cum=-3 → exp(-1.5)
    ev2 = [_ev("ab", ["a", "b"],
               [{"[Relevant]": 0.0}, {"[Relevant]": 0.0}], -3.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev2, "y",
                   mode="always_retrieve", use_seqscore=True)
    assert abs(led["evidence_scores"][0]["seq_exp"]
               - math.exp(-1.5)) < 1e-12
    # 单 token、cum=0 → exp(0)=1
    ev3 = [_ev("z", ["z"], [_rel(0.5)], 0.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev3, "y",
                   mode="always_retrieve", use_seqscore=True)
    assert led["evidence_scores"][0]["seq_exp"] == 1.0


def check_aggregation(solve):
    """closed=True 按答案文本累加分数再取最高（同文本合并）；closed=False
    取最高分路径并返回**未 postprocess** 的原文。"""
    lp0 = _LP_SYM
    ev = [_ev("ans [Relevant] 1", ["ans"], [_rel(0.9)], 0.0),
          _ev("ans [Relevant] 2", ["ans"], [_rel(0.2)], 0.0)]
    led = _compare(solve, {"text": "x [No Retrieval]", "lp0": lp0}, ev,
                   "y", threshold=0.4)
    assert led["result_keys"] == ["no_retrieval", "retrieval_0",
                                  "retrieval_1"]
    assert led["best_path"] == "retrieval_0"
    assert led["answer"] == "ans [Relevant] 1"    # 原文，不去控制 token
    assert abs(led["answer_score"] - 0.9) < 1e-12
    led = _compare(solve, {"text": "x [No Retrieval]", "lp0": lp0}, ev,
                   "y", threshold=0.4, closed=True)
    assert led["best_path"] is None
    assert led["answer"] == "ans  1"              # 去掉 token 后双空格
    assert abs(led["answer2score"]["ans  1"] - 0.9) < 1e-12
    assert abs(led["answer2score"]["ans  2"] - 0.2) < 1e-12
    # 同答案合并（两条证据文本相同 → 分数相加）+ 换行/endoftext 清理
    ev2 = [_ev("yes", ["y"], [_rel(0.5)], 0.0),
           _ev("yes\n", ["y"], [_rel(0.6)], 0.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev2, "y",
                   threshold=0.4, closed=True)
    assert abs(led["answer2score"]["yes"] - 1.1) < 1e-12
    # 平局：closed 稳定取先出现的答案；closed=False 取先出现的路径
    ev3 = [_ev("b1", ["b"], [_rel(0.5)], 0.0),
           _ev("b2", ["b"], [_rel(0.5)], 0.0)]
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev3, "y",
                   threshold=0.4, closed=True)
    assert led["answer"] == "b1"
    led = _compare(solve, {"text": "x", "lp0": lp0}, ev3, "y",
                   threshold=0.4)
    assert led["best_path"] == "retrieval_0"


def check_single_result(solve):
    """len(results)==1 分支：always_retrieve 单证据时也用 postprocess
    （删控制 token、剥一个前导空格），且不下发 best_path/answer_score；
    no_retrieval 路径的 answer 来自第二次生成。"""
    lp0 = _LP_SYM
    led = _compare(solve, {"text": "first", "lp0": lp0},
                   [_ev("ev [Relevant] ans", ["e"], [{}], 0.0)], "second",
                   mode="always_retrieve")
    assert led["result_keys"] == ["retrieval_0"]
    assert led["answer"] == "ev  ans"
    assert led["best_path"] is None
    assert led["answer_score"] is None
    assert led["evidence_scores"][0]["idx"] == 0


def check_errors(solve):
    """域约束：threshold 路径缺反思 token / tokens-lps 长度不一致 /
    要检索却没证据。"""
    lp0 = {"[Retrieval]": -1.0}
    ev = [_ev("a", ["a"], [_rel(0.5)], 0.0)]
    try:
        solve({"text": "x", "lp0": lp0}, ev, "y", threshold=0.5)
        assert False
    except ValueError:
        pass
    try:
        solve({"text": "x", "lp0": _LP_SYM},
              [{"text": "a", "tokens": ["a"], "lps": [], 
                "cumulative_logprob": 0.0}], "y", threshold=0.5)
        assert False
    except ValueError:
        pass
    try:
        solve({"text": "x", "lp0": _LP_SYM},
              [{"text": "a", "tokens": [], "lps": [],
                "cumulative_logprob": 0.0}], "y", threshold=0.5)
        assert False
    except ValueError:
        pass
    try:
        solve({"text": "x [Retrieval]", "lp0": _LP_SYM}, [], "y")
        assert False
    except ValueError:
        pass
    try:
        solve({"text": "x", "lp0": _LP_SYM}, [], "y",
              mode="always_retrieve")
        assert False
    except ValueError:
        pass


def check_random_cross(solve):
    """随机反思 token 表与证据集对拍 oracle（固定种子）。"""
    import random
    rng = random.Random(20261102)
    toks = ["ans", "[Relevant]", "[Irrelevant]", "[Fully supported]",
            "[Partially supported]", "[Utility:1]", "[Utility:5]",
            "x"]
    for _ in range(120):
        n_ev = rng.randint(1, 3)
        evs = []
        for _ in range(n_ev):
            n_tok = rng.randint(1, 5)
            seq = [rng.choice(toks) for _ in range(n_tok)]
            lps = []
            for _ in range(n_tok):
                d = {}
                for t in ("[Relevant]", "[Irrelevant]", "[Fully supported]",
                          "[Partially supported]", "[No support / "
                          "Contradictory]", "[Utility:1]", "[Utility:5]"):
                    if rng.random() < 0.7:
                        d[t] = round(rng.uniform(-4, 0), 3)
                lps.append(d)
            text = " ".join(rng.choice(["yes", "no", "[Relevant]",
                                        "y\n"]) for _ in range(2))
            evs.append(_ev(text, seq, lps, round(rng.uniform(-6, 0), 3)))
        lp0 = {t: round(rng.uniform(-4, -0.01), 3) for t in RET}
        kw = dict(mode=rng.choice(["adaptive_retrieval", "always_retrieve",
                                   "no_retrieval"]),
                  threshold=rng.choice([None, 0.3, 0.5]),
                  w_rel=rng.choice([1.0, 0.5, 2.0]),
                  w_sup=rng.choice([1.0, 0.0]),
                  w_use=rng.choice([0.5, 1.0]),
                  use_seqscore=rng.random() < 0.5,
                  use_groundness=rng.random() < 0.5,
                  use_utility=rng.random() < 0.5,
                  closed=rng.random() < 0.5)
        _compare(solve, {"text": rng.choice(["a [Retrieval]", "a"]), "lp0":
                         lp0}, evs, "second " + rng.choice(["a", "[Relevant]"]),
                 **kw)


def check_example(solve):
    """run 模式样例：门 → 两条证据打分 → 取最高分路径。"""
    lp0 = {"[No Retrieval]": -0.5, "[Retrieval]": -1.5,
           "[Continue to Use Evidence]": -2.0}
    ev = [_ev("paris [Relevant]", ["paris"], [_rel(0.7)], -1.0),
          _ev("lyon [Relevant]", ["lyon"], [_rel(0.2)], -1.0)]
    led = _compare(solve, {"text": "answer? [No Retrieval]", "lp0": lp0},
                   ev, "no", threshold=0.5)
    assert led["do_retrieve"] is True          # -1.5 / -2.0 = 0.75 > 0.5
    assert led["best_path"] == "retrieval_0"
    assert led["answer"] == "paris [Relevant]"
    assert led["n_evidence"] == 2


check_example.example = True
