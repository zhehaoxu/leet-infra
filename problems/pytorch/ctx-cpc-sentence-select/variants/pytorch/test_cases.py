"""CPC 句级压缩选择规则锚点。编码器打分离线入参化。"""
import numpy as np

EXPECTED = {"F1": {"segs": ["The fox runs fast.", "Birds fly high.", "It was raining.", "Photosynthesis makes energy.", "END matches pattern", "Question?"], "scores": [0.82, 0.35, 0.91, 0.12, 0.6, 0.7], "lengths": [5, 4, 4, 4, 4, 2], "target": 20, "boosts": [False, False, False, False, True, False], "order": [4, 2, 0, 5, 1, 3], "idxs": [0, 1, 2, 4, 5], "taken": 23, "sel": ["The fox runs fast.", "Birds fly high.", "It was raining.", "END matches pattern", "Question?"]}, "F2": {"idxs2": [4], "taken2": 4, "sel3": [], "taken3": -1}, "F3": {"selA": ["aa", "bbb"], "takenA": 6, "idxB": [1], "takenB": 3}, "F4": {"segs4": ["xEND tail", "END head", "end lower"], "pattern": "END", "boosts4": [False, True, False]}, "F5": {"order5": [1, 2, 0, 3], "sel5": ["q", "r"], "idx5": [1, 2]}}

CHECKS = [
    "check_boost_flags",
    "check_rank_key",
    "check_select_basic",
    "check_select_budget",
    "check_select_stable",
]


def _l(x):
    return np.asarray(x).tolist()


def check_boost_flags(solve):
    f = EXPECTED["F4"]
    got = solve(mode="boost_flags", segments=list(f["segs4"]), pattern=f["pattern"])
    assert _l(got) == list(f["boosts4"]), got
    # re.match 仅句首: 句中出现不算
    got2 = solve(mode="boost_flags", segments=["mid PATTERN word"], pattern="PATTERN")
    assert _l(got2) == [False]
    # pattern None → 全 False
    got3 = solve(mode="boost_flags", segments=["a", "b"], pattern=None)
    assert _l(got3) == [False, False]


def check_rank_key(solve):
    f = EXPECTED["F1"]
    order = solve(mode="rank_key", boosts=list(f["boosts"]), scores=list(f["scores"]))
    assert _l(order) == list(f["order"]), order
    # boost 句最先、boost 组内按分降序
    order2 = solve(mode="rank_key", boosts=[False, True, True],
                   scores=[0.9, 0.1, 0.5])
    assert _l(order2) == [2, 1, 0], order2


def check_select_basic(solve):
    f = EXPECTED["F1"]
    sel, idxs, taken = solve(mode="select_segments", segments=list(f["segs"]),
                             scores=list(f["scores"]), boosts=list(f["boosts"]),
                             fakes=[False] * len(f["segs"]),
                             lengths=list(f["lengths"]), target=f["target"])
    assert _l(idxs) == list(f["idxs"]), idxs
    assert taken == f["taken"], taken
    assert sel == list(f["sel"]), sel


def check_select_budget(solve):
    f = EXPECTED["F2"]
    f3 = EXPECTED["F3"]
    # fake 句跳过
    base = EXPECTED["F1"]
    fakes = [False] * len(base["segs"])
    fakes[-1] = True  # 末句 "Question?" 是 question echo
    sel2, idxs2, taken2 = solve(mode="select_segments", segments=list(base["segs"]),
                                scores=list(base["scores"]), boosts=list(base["boosts"]),
                                fakes=fakes, lengths=list(base["lengths"]), target=3)
    assert _l(idxs2) == list(f["idxs2"]), idxs2
    assert taken2 == f["taken2"], taken2
    # 全 fake → 空且 taken 保持 -1
    sel3, idxs3, taken3 = solve(mode="select_segments", segments=list(base["segs"]),
                                scores=list(base["scores"]), boosts=list(base["boosts"]),
                                fakes=[True] * len(base["segs"]),
                                lengths=list(base["lengths"]), target=100)
    assert sel3 == [] and _l(idxs3) == [] and taken3 == f["taken3"], (taken3,)
    # 等额=预算继续（> 才断）
    selA, _, takenA = solve(mode="select_segments", segments=["aa", "bbb"],
                            scores=[0.5, 0.9], boosts=[False, False],
                            fakes=[False, False], lengths=[2, 3], target=3)
    assert selA == list(f3["selA"]) and takenA == f3["takenA"], (selA, takenA)
    # 溢出句先收再 break（add-then-break 怪癖）
    selB, idxB, takenB = solve(mode="select_segments", segments=["aa", "bbb"],
                               scores=[0.5, 0.9], boosts=[False, False],
                               fakes=[False, False], lengths=[2, 3], target=1)
    assert _l(idxB) == list(f3["idxB"]) and takenB == f3["takenB"], (idxB, takenB)


def check_select_stable(solve):
    f = EXPECTED["F5"]
    order5 = solve(mode="rank_key", boosts=[False] * 4, scores=[0.5, 0.9, 0.9, 0.5])
    assert _l(order5) == list(f["order5"]), order5
    sel5, idx5, _ = solve(mode="select_segments", segments=["p", "q", "r", "s"],
                          scores=[0.5, 0.9, 0.9, 0.5], boosts=[False] * 4,
                          fakes=[False] * 4, lengths=[1, 1, 1, 1], target=2)
    assert _l(idx5) == list(f["idx5"]), idx5
    assert sel5 == list(f["sel5"]), sel5
    # 输出保原序而非分数序
    sel6, idx6, _ = solve(mode="select_segments", segments=["m", "n"],
                          scores=[0.3, 0.9], boosts=[False, False],
                          fakes=[False, False], lengths=[3, 3], target=10)
    assert sel6 == ["m", "n"] and _l(idx6) == [0, 1], (sel6, idx6)
