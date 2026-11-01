import functools
import string


def _py(pairs, ignore_space=True, is_filter=False):
    """oracle：递归 memo 独立实现 Levenshtein + 官方语义复刻。"""
    digits_letters = string.digits + string.ascii_letters

    def normalize(t):
        return "".join(filter(lambda x: x in digits_letters, t)).lower()

    def lev(a, b):
        @functools.lru_cache(maxsize=None)
        def d(i, j):
            if i == 0:
                return j
            if j == 0:
                return i
            return min(
                d(i - 1, j) + 1,
                d(i, j - 1) + 1,
                d(i - 1, j - 1) + (a[i - 1] != b[j - 1]),
            )

        return d(len(a), len(b))

    rows = []
    sum_ned = 0.0
    exact_cnt = 0
    for pred_raw, target_raw in pairs:
        pred = pred_raw.replace(" ", "") if ignore_space else pred_raw
        target = target_raw.replace(" ", "") if ignore_space else target_raw
        if is_filter:
            pred = normalize(pred)
            target = normalize(target)
        dist = lev(pred, target)
        maximum = max(len(pred), len(target))
        ned = dist / maximum if maximum else 0
        exact = pred == target
        if exact:
            exact_cnt += 1
        sum_ned += ned
        rows.append(
            {"pred": pred, "target": target, "dist": dist, "ned": ned, "exact": exact}
        )
    n = len(rows)
    eps = 1e-5
    return {
        "rows": rows,
        "acc": exact_cnt / (n + eps),
        "norm_edit_dis": 1 - sum_ned / (n + eps),
    }


def _compare(solve, pairs, ignore_space=True, is_filter=False):
    led = solve(pairs, ignore_space, is_filter)
    ref = _py(pairs, ignore_space, is_filter)
    assert led == ref, f"solve vs oracle:\n{led}\nvs\n{ref}"
    return led


def check_anchor_basic(solve):
    led = _compare(
        solve,
        [["kitten", "sitting"], ["abc", "abc"], ["x", "y"]],
    )
    r = led["rows"]
    # kitten→sitting：3 次替换（k→s, e→i, e→i 与插入 g）= dist 3，分母 max=7
    assert r[0]["dist"] == 3
    assert abs(r[0]["ned"] - 3 / 7) < 1e-12
    assert r[0]["exact"] is False
    assert r[1]["dist"] == 0 and r[1]["ned"] == 0.0 and r[1]["exact"] is True
    assert r[2]["dist"] == 1 and abs(r[2]["ned"] - 1.0) < 1e-12
    # acc = 1/(3+eps) ≈ 0.33333
    assert abs(led["acc"] - 1 / 3) < 1e-4 and led["acc"] < 1 / 3 + 1e-9


def check_ned_denominator_is_max(solve):
    # 官方 rapidfuzz 语义：分母是 max(len)，不是 len(target)
    # "abcd" vs "ab"：dist=2, max=4 → ned=0.5（按 len(target)=2 会错成 1.0）
    led = _compare(solve, [["abcd", "ab"]])
    assert led["rows"][0]["dist"] == 2
    assert abs(led["rows"][0]["ned"] - 0.5) < 1e-12


def check_empty_string_pairs(solve):
    # 空-空：maximum=0 → ned=0 且 exact=True；空-非空：ned=1.0
    led = _compare(solve, [["", ""], ["", "abc"], ["def", ""]])
    r = led["rows"]
    assert r[0]["dist"] == 0 and r[0]["ned"] == 0.0 and r[0]["exact"] is True
    assert r[1]["dist"] == 3 and r[1]["ned"] == 1.0 and r[1]["exact"] is False
    assert r[2]["dist"] == 3 and r[2]["ned"] == 1.0
    # 汇总：acc = 1/(3+eps)，ned 指标 = 1 - (0+1+1)/(3+eps)
    assert abs(led["acc"] - 1 / 3) < 1e-4
    assert abs(led["norm_edit_dis"] - (1 - 2 / 3)) < 1e-4


def check_no_damerau(solve):
    # 经典 Levenshtein 不含相邻换位："ab"→"ba" 需要 2 步，不是 1
    led = _compare(solve, [["ab", "ba"]])
    assert led["rows"][0]["dist"] == 2
    assert abs(led["rows"][0]["ned"] - 1.0) < 1e-12


def check_ignore_space(solve):
    # 默认去空格："a b" 与 "ab" 预处理后相等
    led = _compare(solve, [["a b", "ab"], ["he llo", "hello world"]])
    r = led["rows"]
    assert r[0]["pred"] == "ab" and r[0]["target"] == "ab" and r[0]["exact"] is True
    assert r[0]["ned"] == 0.0
    assert r[1]["pred"] == "hello" and r[1]["target"] == "helloworld"
    assert r[1]["dist"] == 5
    # 关掉开关后空格参与比较
    led2 = _compare(solve, [["a b", "ab"]], ignore_space=False)
    assert led2["rows"][0]["pred"] == "a b"
    assert led2["rows"][0]["dist"] == 1 and led2["rows"][0]["exact"] is False


def check_is_filter(solve):
    # filter 只保留 digits+letters 再 lower；空格先被 replace 也会被 filter 删
    led = _compare(solve, [["A-1 x!", "a1X"], ["你好abc", "ABC"]], is_filter=True)
    r = led["rows"]
    assert r[0]["pred"] == "a1x" and r[0]["target"] == "a1x" and r[0]["exact"] is True
    # 中文字符被 filter 剔除
    assert r[1]["pred"] == "abc" and r[1]["target"] == "abc" and r[1]["exact"] is True
    # is_filter=False 时同样输入不等
    led2 = _compare(solve, [["A-1 x!", "a1X"]], is_filter=False)
    assert led2["rows"][0]["exact"] is False


def check_eps_summary(solve):
    # 全对批：acc = n/(n+eps) < 1（eps 分母 quirk）
    led = _compare(solve, [["aa", "aa"], ["bb", "bb"]])
    assert led["acc"] < 1.0 and abs(led["acc"] - 1.0) < 1e-4
    assert abs(led["norm_edit_dis"] - 1.0) < 1e-12
    # 全错批：ned 指标 = 1 - 均值（非均值本身）
    led2 = _compare(solve, [["a", "b"], ["cd", "ef"]])
    # 均值 = (1+1)/2 = 1 → 指标 0（含 eps 略正）
    assert led2["norm_edit_dis"] > 0.0 and led2["norm_edit_dis"] < 1e-4


def check_empty_batch(solve):
    # 官方公式对空批的直接行为（反直觉锚点）：acc=0.0、norm_edit_dis=1.0
    led = _compare(solve, [])
    assert led["rows"] == []
    assert led["acc"] == 0.0
    assert led["norm_edit_dis"] == 1.0


def check_invalid_inputs(solve):
    def ve(*args, **kw):
        try:
            solve(*args, **kw)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    ve("nope")                        # 非 list
    ve([[1, 2]])                      # 非 str 元素
    ve([["ab"]])                      # 长度不足
    ve([["ab", "cd", "ef"]])          # 长度超出
    ve([["ab", 3]])                   # target 非 str
    ve([["ab", "cd"]], ignore_space=1)    # 非 bool
    ve([["ab", "cd"]], is_filter="y")     # 非 bool


def check_oracle_random(solve):
    import random

    rng = random.Random(20260687)
    alphabet = "abcde AB1!，é"
    for _ in range(200):
        n = rng.randint(0, 12)
        pairs = []
        for _i in range(n):
            pa = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 16)))
            ta = list(pa)
            for _k in range(rng.randint(0, 3)):
                op = rng.randint(0, 2)
                if op == 0 and ta:  # 删
                    del ta[rng.randrange(len(ta))]
                elif op == 1:  # 插
                    ta.insert(rng.randint(0, len(ta)), rng.choice(alphabet))
                elif ta:  # 替换
                    ta[rng.randrange(len(ta))] = rng.choice(alphabet)
            pairs.append([pa, "".join(ta)])
        ig = rng.random() < 0.7
        fl = rng.random() < 0.4
        led = solve(pairs, ig, fl)
        ref = _py(pairs, ig, fl)
        assert led == ref, "random mismatch"
