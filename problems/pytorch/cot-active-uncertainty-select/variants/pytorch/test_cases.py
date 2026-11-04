"""Active-Prompt 不确定度选样 -- 测试用例."""

import math
from collections import Counter

NO_SOLUTION = "-10086"


def _entropy(freq):
    n = sum(freq)
    if n == 0:
        return float("nan")
    return -sum((c / n) * math.log(c / n) for c in freq if c > 0)


def _oracle(samples_per_question, sort_by="disagreement",
            dataset_kind="generic", qes_limit=0):
    """独立复刻官方账本（Counter 路径，不 import solution）。"""
    n = len(samples_per_question)
    limit = n if (qes_limit == 0 or qes_limit > n) else qes_limit
    records = []
    for idx in range(limit):
        counts = Counter()
        if dataset_kind == "strategyqa":
            counts = Counter({"yes": 0, "no": 0})
        for ans in samples_per_question[idx]:
            key = ans if ans != "" else NO_SOLUTION
            counts[key] += 1
        occurrence = dict(counts)
        record = {"dataset_idx": idx, "occurrence": occurrence}
        if dataset_kind == "numerical":
            vals = []
            for ans, c in occurrence.items():
                vals.extend([float(ans)] * c)
            record["variance"] = (
                sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
                / len(vals)) if vals else float("nan")
        record["entropy"] = _entropy(list(occurrence.values()))
        record["disagreement"] = len(occurrence)
        records.append(record)
    if sort_by == "disagreement":
        if dataset_kind == "strategyqa":
            try:
                records.sort(key=lambda r: abs(r["occurrence"]["yes"]
                                              - r["occurrence"]["no"]))
            except Exception:
                records.sort(key=lambda r: -len(r["occurrence"]))
        else:
            records.sort(key=lambda r: -len(r["occurrence"]))
    elif sort_by == "variance" and dataset_kind == "numerical":
        records.sort(key=lambda r: -r["variance"])
    elif sort_by == "entropy":
        records.sort(key=lambda r: -r["entropy"])
    return records


def _gen_fixture():
    return [
        ["5", "5", "7", "", "5"],
        ["3", "3", "3", "3", "3"],
        ["1", "2", "3", "4", "5"],
    ]


def _num_fixture():
    return [
        ["10", "10", "10", "10", "10"],
        ["0", "0", "0", "0", ""],
        ["8", "8", "8", "8", "8"],
    ]


def _sqa_fixture():
    return [
        ["yes", "yes", "no", "yes", "no"],
        ["yes", "yes", "yes", "yes", ""],
        ["no", "no", "yes", "no", "no"],
    ]


def _approx(a, b):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) < 1e-9


def _records_close(got, exp):
    if len(got) != len(exp):
        return False
    for g, e in zip(got, exp):
        if set(g) != set(e) or g["dataset_idx"] != e["dataset_idx"]:
            return False
        for k in g:
            if k in ("variance", "entropy"):
                if not _approx(g[k], e[k]):
                    return False
            elif g[k] != e[k]:
                return False
    return True


def check_generic_oracle(solve):
    fx = _gen_fixture()
    for sb in ("disagreement", "entropy"):
        got = solve(fx, sort_by=sb)
        exp = _oracle(fx, sort_by=sb)
        assert _records_close(got, exp), (sb, got, exp)
    got = solve(fx)
    assert [r["dataset_idx"] for r in got] == [2, 0, 1], got
    assert got[0]["disagreement"] == 5
    assert got[1]["occurrence"] == {"5": 3, "7": 1, "-10086": 1}
    assert got[2]["entropy"] == 0.0
    assert got[2]["occurrence"] == {"3": 5}


def check_numerical_variance(solve):
    fx = _num_fixture()
    got = solve(fx, sort_by="variance", dataset_kind="numerical")
    assert [r["dataset_idx"] for r in got] == [1, 0, 2]
    # NO_SOLUTION = -10086 也进方差：ans_list = [0,0,0,0,-10086]
    vals = [0.0, 0.0, 0.0, 0.0, -10086.0]
    mean = sum(vals) / 5
    exp_var = sum((v - mean) ** 2 for v in vals) / 5
    assert abs(got[0]["variance"] - exp_var) < 1e-6
    assert got[0]["variance"] > 1.6e7
    assert got[1]["variance"] == 0.0 and got[1]["dataset_idx"] == 0
    assert got[2]["variance"] == 0.0
    # occurrence 仍以原串计数
    assert got[0]["occurrence"] == {"0": 4, "-10086": 1}
    assert _records_close(got, _oracle(fx, "variance", "numerical"))


def check_strategyqa_sort(solve):
    fx = _sqa_fixture()
    got = solve(fx, sort_by="disagreement", dataset_kind="strategyqa")
    # abs(yes-no) 升序：q0 差 1、q2 差 3、q1 差 4
    assert [r["dataset_idx"] for r in got] == [0, 2, 1], got
    assert got[0]["occurrence"] == {"yes": 3, "no": 2}
    assert got[1]["occurrence"] == {"no": 4, "yes": 1}
    assert got[2]["occurrence"] == {"yes": 4, "no": 0, "-10086": 1}
    # 预置零频不进 entropy：-(4/5 ln 4/5 + 1/5 ln 1/5)
    assert abs(got[1]["entropy"] - (-(0.8 * math.log(0.8)
                                     + 0.2 * math.log(0.2)))) < 1e-9
    assert got[2]["entropy"] == got[1]["entropy"]
    assert got[2]["disagreement"] == 3  # yes/no/-10086 三个键


def check_entropy_ledger(solve):
    fx = _gen_fixture()
    got = solve(fx, sort_by="entropy")
    # 降序：q2（ln5）> q0（-0.6ln0.6-0.4ln0.2）> q1（0）
    exp = [math.log(5),
           -(0.6 * math.log(0.6) + 0.4 * math.log(0.2)),
           0.0]
    for r, e in zip(got, exp):
        assert abs(r["entropy"] - e) < 1e-9, (r, e)
    # entropy 与 occurrence 键序无关（yes/no 预置零频贡献 0 已在 strategyqa 验）


def check_sort_variants(solve):
    fx = _gen_fixture()
    # variance + generic：官方 elif 不匹配 → 静默不排序，保持输入序
    got = solve(fx, sort_by="variance")
    assert [r["dataset_idx"] for r in got] == [0, 1, 2]
    # entropy 降序：q2（ln5）> q0 > q1（0）
    got_e = solve(fx, sort_by="entropy")
    assert [r["dataset_idx"] for r in got_e] == [2, 0, 1]
    # numerical + entropy
    got_n = solve(_num_fixture(), sort_by="entropy", dataset_kind="numerical")
    assert _records_close(got_n, _oracle(_num_fixture(), "entropy",
                                         "numerical"))


def check_qes_limit(solve):
    fx = _gen_fixture()
    # limit=2：只算前两题、子集内排序
    got = solve(fx, qes_limit=2)
    assert [r["dataset_idx"] for r in got] == [0, 1]
    # limit 超过题数 = 全部
    got_all = solve(fx, qes_limit=99)
    assert [r["dataset_idx"] for r in got_all] == [2, 0, 1]
    # limit=0 = 全部
    assert [r["dataset_idx"] for r in solve(fx)] == [2, 0, 1]
    # limit=1：单元素
    got1 = solve(fx, qes_limit=1)
    assert [r["dataset_idx"] for r in got1] == [0]


def check_tie_stability(solve):
    # disagreement 平局保持输入序
    fx = [
        ["a", "b"],
        ["c", "d"],
        ["e", "e"],
        ["f", "g"],
    ]
    got = solve(fx)
    assert [r["dataset_idx"] for r in got] == [0, 1, 3, 2], got
    # entropy 平局：全同票熵 0，两个全同票题保持输入序
    fx2 = [["x", "x"], ["y", "y"], ["p", "q"]]
    got2 = solve(fx2, sort_by="entropy")
    assert [r["dataset_idx"] for r in got2] == [2, 0, 1], got2


def check_purity(solve):
    import copy
    fx = _num_fixture()
    snap = copy.deepcopy(fx)
    solve(fx, sort_by="variance", dataset_kind="numerical")
    solve(_sqa_fixture(), dataset_kind="strategyqa")
    assert fx == snap


def check_args_validation(solve):
    fx = _gen_fixture()
    for bad in (
        lambda: solve("nope"),
        lambda: solve([]),
        lambda: solve([[1, 2]]),
        lambda: solve([["a"], "nope"]),
        lambda: solve([[]]),
        lambda: solve(fx, sort_by="none"),
        lambda: solve(fx, sort_by=None),
        lambda: solve(fx, dataset_kind="gsm8k"),
        lambda: solve(fx, dataset_kind=None),
        lambda: solve(fx, qes_limit=-1),
        lambda: solve(fx, qes_limit=True),
        lambda: solve(fx, qes_limit=1.5),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    try:
        solve([[]])
    except ValueError as e:
        assert "num_trails" in str(e), e
    try:
        solve(fx, dataset_kind="gsm8k")
    except ValueError as e:
        assert "dataset_kind" in str(e), e


check_generic_oracle.example = True
check_numerical_variance.example = True
check_strategyqa_sort.example = True
check_entropy_ledger.example = True
check_sort_variants.example = True
check_qes_limit.example = True
check_tie_stability.example = True
check_purity.example = True
check_args_validation.example = True
