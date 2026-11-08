"""851 判题用例：Captum NoiseTunnel 加噪平滑账。

五个 check_*（每个都接收 solve 参数，由 judge 注入）：
  check_noise_layout    —— repeat_interleave 加噪布局 / stdevs 类型分支
  check_batch_schedule  —— nt_samples_batch_size 的 min / 整除 / remaining
  check_aggregate       —— smoothgrad / smoothgrad_sq / vargrad 三种聚合口径
  check_baseline_expand —— baseline 的 draw 抽样 / repeat_interleave 条件
  check_noise_tunnel    —— 完整编排（per-partition perturbed / delta cat）

每个 check 都带独立 oracle（按官方算式重写的纯 Python），并对关键结论做
**双向断言**（等于 oracle + 必须不等于那个「看起来也合理」的错误候选）；
域约束不仅断言「抛 ValueError」，还断言消息里点名了 violated constraint。
"""

import math


# ----------------------------------------------------------------- 工具
def _get(res, key, default=None):
    if isinstance(res, dict):
        return res.get(key, default)
    return default


def _flat(x):
    out = []
    stack = [x]
    while stack:
        node = stack.pop(0)
        if isinstance(node, (list, tuple)):
            stack = list(node) + stack
        else:
            out.append(float(node))
    return out


def _close(a, b, tol=1e-9):
    # 字符串 / None / bool 直接比较，不能走 _flat 的 float() 转换
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    fa, fb = _flat(a), _flat(b)
    if len(fa) != len(fb):
        return False
    for u, v in zip(fa, fb):
        if abs(u - v) > tol:
            return False
    return True


def _must_differ(got, wrong, msg):
    """双向断言：got 必须不等于那个错误候选。"""
    if _close(got, wrong, tol=1e-9):
        raise AssertionError(msg)


class _LCG:
    """确定性伪随机，避免判题用例依赖任何随机源。"""

    def __init__(self, seed):
        self.s = seed & 0x7FFFFFFF

    def __call__(self):
        self.s = (1103515245 * self.s + 12345) % (1 << 31)
        return (self.s / float(1 << 31)) * 2.0 - 1.0


def _repeat_interleave(rows, n):
    out = []
    for row in rows:
        for _ in range(n):
            out.append(list(row))
    return out


def _repeat(rows, n):
    out = []
    for _ in range(n):
        for row in rows:
            out.append(list(row))
    return out


# --------------------------------------------------------------- oracle 层
def _o_batch(nt, bs):
    eff = nt if bs is None else min(nt, bs)
    part = nt // eff
    rem = nt - part * eff
    sizes = [eff] * part + ([rem] if rem > 0 else [])
    return {"effective_batch_size": eff, "partition": part, "remaining": rem,
            "batch_sizes": sizes, "total": sum(sizes)}


def _o_acc(attrs, sizes, nt, d):
    """独立累加：按 partition 分段，每段 [b*k][d]，第 bi 个占 [bi*k:(bi+1)*k]。"""
    b = len(attrs) // nt
    s = [[0.0] * d for _ in range(b)]
    s2 = [[0.0] * d for _ in range(b)]
    off = 0
    for k in sizes:
        for bi in range(b):
            for si in range(k):
                row = attrs[off + bi * k + si]
                for di in range(d):
                    v = float(row[di])
                    s[bi][di] += v
                    s2[bi][di] += v * v
        off += b * k
    E = [[s[bi][di] / nt for di in range(d)] for bi in range(b)]
    E2 = [[s2[bi][di] / nt for di in range(d)] for bi in range(b)]
    V = [[E2[bi][di] - E[bi][di] * E[bi][di] for di in range(d)]
         for bi in range(b)]
    return {"sum": s, "sum_sq": s2, "expected": E, "expected_sq": E2,
            "smoothgrad": E, "smoothgrad_sq": E2, "vargrad": V,
            "bsz": b, "d": d, "nt_samples": nt}


def _o_noise_layout(inputs, stdev, k, noise):
    b = len(inputs)
    d = len(inputs[0])
    rep = _repeat_interleave(inputs, k)
    alt = _repeat(inputs, k)
    pert = [[rep[i][j] + noise[i][j] for j in range(d)]
            for i in range(b * k)]
    return {
        "expanded_size": [b * k, d], "stdevs_": [stdev],
        "stdev_expanded": [[stdev] * d for _ in range(b * k)],
        "repeat_interleave": rep, "perturbed": pert,
        "source_index": [i // k for i in range(b * k)],
        "repeat_alt": alt, "repeat_alt_source": [i % b for i in range(b * k)],
        "n_inputs": 1, "bsz": b, "nt_samples": k, "d": d,
    }


def _o_baseline(x_list, bl_list, k, draw, idx):
    b = len(x_list)
    d = len(x_list[0])
    num_ref = len(bl_list)
    if draw:
        exp = [list(bl_list[v]) for v in idx]
        return {"mode": "draw", "expanded_baselines": exp,
                "num_drawn": k * b, "indices": idx, "num_ref": num_ref,
                "shape": [k * b, d]}
    if num_ref == b and b > 1:
        exp = _repeat_interleave(bl_list, k)
        return {"mode": "repeat_interleave", "expanded_baselines": exp,
                "num_drawn": 0, "indices": None, "num_ref": num_ref,
                "shape": [len(exp), d]}
    return {"mode": "unchanged",
            "expanded_baselines": [list(r) for r in bl_list],
            "num_drawn": 0, "indices": None, "num_ref": num_ref,
            "shape": [num_ref, d]}


def _mk_layout(b, d, k, seed, stdev=0.5):
    rnd = _LCG(seed)
    inputs = [[rnd() for _ in range(d)] for _ in range(b)]
    noise = [[rnd() * stdev for _ in range(d)] for _ in range(b * k)]
    return inputs, noise, stdev


def _mk_agg(b, d, nt, bs, seed):
    rnd = _LCG(seed)
    x = [[rnd() for _ in range(d)] for _ in range(b)]
    sizes = _o_batch(nt, bs)["batch_sizes"]
    noise = []
    pert = []
    for k in sizes:
        rep = _repeat_interleave(x, k)
        for i in range(b * k):
            nz = [rnd() for _ in range(d)]
            noise.append(nz)
            pert.append([rep[i][j] + nz[j] for j in range(d)])
    # attributions 与 perturbed 相关但不等同，且方差非零
    attrs = [[pert[i][j] * (1.0 + 0.1 * j) + 0.05 for j in range(d)]
             for i in range(len(pert))]
    delta_parts = [[float(i + 1) + 0.5 * p for i in range(b * k)]
                   for p, k in enumerate(sizes)]
    return {"x": x, "noise": noise, "pert": pert, "attrs": attrs,
            "sizes": sizes, "delta_parts": delta_parts,
            "b": b, "d": d, "nt": nt, "bs": bs}


# =============================================================== check 1
def check_noise_layout(solve):
    cases = [
        (2, 2, 3, 0.5, 7),
        (1, 3, 1, 0.0, 9),
        (3, 2, 4, 1.0, 13),
        (2, 4, 2, 0.25, 21),
        (4, 3, 5, 0.75, 33),
    ]
    for b, d, k, stdev, seed in cases:
        x, noise, _ = _mk_layout(b, d, k, seed, stdev)
        res = solve("noise_layout", inputs=x, stdevs=stdev,
                    nt_samples=k, noise=noise)
        ora = _o_noise_layout(x, stdev, k, noise)
        for key in ("expanded_size", "stdevs_", "stdev_expanded",
                    "repeat_interleave", "perturbed", "source_index",
                    "repeat_alt", "repeat_alt_source", "n_inputs", "bsz",
                    "nt_samples", "d"):
            assert _close(_get(res, key), ora[key]), \
                "noise_layout.%s must equal the oracle" % key

        # ★ 双向断言 1：repeat_interleave 是连续重复，不是 repeat 的交替
        if b > 1 and k > 1:
            _must_differ(_get(res, "repeat_interleave"),
                         _get(res, "repeat_alt"),
                         "repeat_interleave must be [x0]*k,[x1]*k,... not "
                         "the alternating torch.repeat layout")
            _must_differ(_get(res, "source_index"),
                         _get(res, "repeat_alt_source"),
                         "source_index must be i//k (block layout), not i%b")
        # ★ 双向断言 2：perturbed 基于 repeat_interleave，不是 repeat
        #   （b=1 或 k=1 时两种布局天然相同，必须挑 b>1 且 k>1 的数据）
        if b > 1 and k > 1:
            wrong_pert = [[_get(res, "repeat_alt")[i][j] + noise[i][j]
                           for j in range(d)] for i in range(b * k)]
            _must_differ(_get(res, "perturbed"), wrong_pert,
                         "perturbed must be repeat_interleave(inputs,k) + noise")
        # ★ 双向断言 3：expanded_size 是 [bsz*k, d]，不是 [bsz, k*d] / [k, bsz*d]
        assert _get(res, "expanded_size") == [b * k, d], \
            "expanded_size must be (bsz*nt_samples,) + input.shape[1:]"
        # ★ 双向断言 4：stdev_expanded 每个元素都等于 stdev（不是 stdev^2）
        se = _get(res, "stdev_expanded")
        for row in se:
            for v in row:
                assert abs(v - stdev) < 1e-12, \
                    "stdev_expanded must be a broadcast of the scalar stdev"
        # ★ 双向断言 5：n_inputs == 1（单输入展开成 1 个 stdev）
        assert _get(res, "n_inputs") == 1, "n_inputs must be 1"

    # ---- 域约束：stdevs 类型分支（官方 assert / 官方真 bug）----
    x = [[1.0, 2.0], [3.0, 4.0]]
    noise = [[0.0, 0.0]] * 6
    for bad, kw in (
        (1, "float"),                      # int 1：官方 assert isinstance(float)
        ((0.5,), "official bug"),         # tuple 长度匹配：官方 UnboundLocalError
        ((0.5, 0.7), "must equal"),       # tuple 长度不匹配：官方 assert
        (True, "bool"),                   # bool 不是 float
    ):
        try:
            solve("noise_layout", inputs=x, stdevs=bad, nt_samples=3,
                  noise=noise)
        except ValueError as exc:
            assert kw in str(exc), \
                "stdevs=%r must be rejected with a message naming %r, got %r" \
                % (bad, kw, str(exc))
        else:
            raise AssertionError("noise_layout must reject stdevs=%r" % (bad,))

    # float 分支必须可用
    r = solve("noise_layout", inputs=x, stdevs=0.5, nt_samples=3, noise=noise)
    assert _close(_get(r, "stdevs_"), [0.5]), "float stdevs must be accepted"
    # nt_samples 非正
    for bad in (0, -1):
        try:
            solve("noise_layout", inputs=x, stdevs=0.5,
                  nt_samples=bad, noise=[])
        except ValueError as exc:
            assert "positive integer" in str(exc), \
                "nt_samples must be rejected with 'positive integer'"
        else:
            raise AssertionError("noise_layout must reject nt_samples=%r" % bad)


# =============================================================== check 2
def check_batch_schedule(solve):
    combos = [(10, 3), (10, None), (4, 10), (7, 1), (1, 5),
              (6, 2), (5, 3), (3, 7), (8, 4), (2, 1), (9, 9), (13, 5),
              (1, None), (11, 4), (12, 5)]
    for nt, bs in combos:
        res = solve("batch_schedule", nt_samples=nt,
                    nt_samples_batch_size=bs)
        ora = _o_batch(nt, bs)
        for key in ("effective_batch_size", "partition", "remaining",
                    "batch_sizes", "total"):
            assert _close(_get(res, key), ora[key]), \
                "batch_schedule.%s must equal the oracle (nt=%d, bs=%s)" \
                % (key, nt, bs)
        # ★ 双向断言 1：effective_batch_size 是 min(nt, bs)，不是 bs 原值
        want_eff = nt if bs is None else min(nt, bs)
        assert _get(res, "effective_batch_size") == want_eff, \
            "effective_batch_size must be min(nt_samples, bs)"
        # ★ 双向断言 2：partition 是整除（向下取整），不是 ceil
        ceil_part = (nt + want_eff - 1) // want_eff
        if ceil_part != ora["partition"]:
            assert _get(res, "partition") != ceil_part, \
                "partition must be floor division, not ceil"
        # ★ 双向断言 3：remaining 单独成一批，且 batch_sizes 之和 == nt
        sizes = _get(res, "batch_sizes")
        assert sum(sizes) == nt, "batch_sizes must sum to nt_samples"
        if ora["remaining"] > 0:
            assert sizes[-1] == ora["remaining"], \
                "the remaining samples must form a final smaller partition"
            assert len(sizes) == ora["partition"] + 1, \
                "there must be exactly partition+1 batches when remaining>0"
        else:
            assert len(sizes) == ora["partition"], \
                "there must be exactly partition batches when remaining==0"
        # ★ 双向断言 4：bs>nt 被 min 压到 nt（partition=1，不会为 0）
        if bs is not None and bs > nt:
            assert sizes == [nt], \
                "bs>nt_samples must be clamped by min to a single partition"

    # ---- 域约束 ----
    for bad, kw in ((0, "positive integer"), (-3, "positive integer")):
        try:
            solve("batch_schedule", nt_samples=bad)
        except ValueError as exc:
            assert kw in str(exc), "nt_samples must be a positive integer"
        else:
            raise AssertionError("batch_schedule must reject nt_samples=%r" % bad)
    for bad in (0, -2):
        try:
            solve("batch_schedule", nt_samples=5, nt_samples_batch_size=bad)
        except ValueError as exc:
            assert "positive integer" in str(exc), \
                "nt_samples_batch_size must be a positive integer or None"
        else:
            raise AssertionError("batch_schedule must reject bs=%r" % bad)


# =============================================================== check 3
def check_aggregate(solve):
    cases = [
        (1, 2, 6, 2, 3), (2, 2, 10, 3, 5), (3, 3, 5, None, 7),
        (2, 4, 4, 10, 11), (3, 2, 7, 2, 13), (1, 3, 12, 5, 17),
    ]
    for b, d, nt, bs, seed in cases:
        c = _mk_agg(b, d, nt, bs, seed)
        attrs = c["attrs"]
        sizes = c["sizes"]
        for nt_type in ("smoothgrad", "smoothgrad_sq", "vargrad"):
            res = solve("aggregate", attributions=attrs,
                        nt_samples=nt, nt_type=nt_type,
                        nt_samples_batch_size=bs)
            ora = _o_acc(attrs, sizes, nt, d)
            for key in ("attributions", "smoothgrad", "smoothgrad_sq",
                        "vargrad", "sum", "sum_sq", "expected", "expected_sq",
                        "effective_batch_size", "partition", "remaining",
                        "batch_sizes", "bsz", "d", "nt_samples", "nt_type"):
                want = ora[nt_type if key == "attributions" else key] \
                    if key in ("attributions", "smoothgrad", "smoothgrad_sq",
                               "vargrad", "sum", "sum_sq", "expected",
                               "expected_sq", "bsz", "d", "nt_samples") \
                    else (_o_batch(nt, bs)[key] if key in (
                        "effective_batch_size", "partition", "remaining",
                        "batch_sizes") else nt_type)
                assert _close(_get(res, key), want), \
                    "aggregate.%s must equal the oracle (%s)" % (key, nt_type)

            # ★ 双向断言 1：sum_sq 是逐元素平方求和，不是 (sum)^2
            sq_wrong = [[ora["sum"][bi][di] ** 2 for di in range(d)]
                        for bi in range(b)]
            _must_differ(_get(res, "sum_sq"), sq_wrong,
                         "sum_sq must be sum(X^2) per element, not (sum X)^2")
            # ★ 双向断言 2：smoothgrad_sq = E[X^2]，不是 (E[X])^2
            e_sq_wrong = [[ora["expected"][bi][di] ** 2 for di in range(d)]
                          for bi in range(b)]
            _must_differ(_get(res, "smoothgrad_sq"), e_sq_wrong,
                         "smoothgrad_sq must be E[X^2], not E[X]^2")
            # ★ 双向断言 3：vargrad = E[X^2]-E[X]^2（总体方差，除以 n）
            #   —— 不是 E[X^2]，也不是带 Bessel 校正的样本方差 (除以 n-1)
            _must_differ(_get(res, "vargrad"), _get(res, "smoothgrad_sq"),
                         "vargrad must subtract E[X]^2 from E[X^2]")
            sample_var = [
                [ora["sum_sq"][bi][di] / (nt - 1)
                 - (ora["sum"][bi][di] / nt) ** 2 * nt / (nt - 1)
                 for di in range(d)] for bi in range(b)]
            _must_differ(_get(res, "vargrad"), sample_var,
                         "vargrad must be the population variance (divide by n, "
                         "not n-1 Bessel correction)")
            # ★ 双向断言 4：除以 nt_samples，不是 partition 数或 eff
            if ora["sum"][0][0] != 0.0:
                wrong_div = [[ora["sum"][bi][di] / len(sizes)
                              for di in range(d)] for bi in range(b)]
                _must_differ(_get(res, "smoothgrad"), wrong_div,
                             "the normalizer must be nt_samples")

        # ★ 双向断言 5：三种口径必须两两不同（方差非零数据）
        r1 = solve("aggregate", attributions=attrs, nt_samples=nt,
                   nt_type="smoothgrad", nt_samples_batch_size=bs)
        r2 = solve("aggregate", attributions=attrs, nt_samples=nt,
                   nt_type="smoothgrad_sq", nt_samples_batch_size=bs)
        r3 = solve("aggregate", attributions=attrs, nt_samples=nt,
                   nt_type="vargrad", nt_samples_batch_size=bs)
        _must_differ(_get(r1, "attributions"), _get(r2, "attributions"),
                     "smoothgrad and smoothgrad_sq must differ")
        _must_differ(_get(r2, "attributions"), _get(r3, "attributions"),
                     "smoothgrad_sq and vargrad must differ")

    # ---- 域约束 ----
    good = [[1.0, 2.0]] * 6
    for bad, kw in (("smoothgrad_sq2", "must be either"),
                    ("bogus", "must be either")):
        try:
            solve("aggregate", attributions=good, nt_samples=3, nt_type=bad)
        except ValueError as exc:
            assert kw in str(exc), \
                "unknown nt_type must be rejected with 'must be either'"
        else:
            raise AssertionError("aggregate must reject nt_type=%r" % bad)
    for bad, kw in ((0, "positive integer"), (-1, "positive integer")):
        try:
            solve("aggregate", attributions=good, nt_samples=bad)
        except ValueError as exc:
            assert kw in str(exc), "nt_samples must be a positive integer"
        else:
            raise AssertionError("aggregate must reject nt_samples=%r" % bad)
    try:
        solve("aggregate", attributions=[[1.0, 2.0]] * 5, nt_samples=3)
    except ValueError as exc:
        assert "divisible" in str(exc), \
            "attributions row count must be divisible by nt_samples"
    else:
        raise AssertionError("aggregate must reject a non-divisible row count")


# =============================================================== check 4
def check_baseline_expand(solve):
    cases = [
        # (x, baselines, k, draw, indices or None)
        ([[1.0, 2.0], [3.0, 4.0]],
         [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
         3, True, [0, 2, 4, 1, 3, 0]),
        ([[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5]], 4, True,
         [0, 0, 0, 0, 0, 0, 0, 0]),
        ([[1.0, 2.0], [3.0, 4.0]], [[0.1, 0.2], [0.3, 0.4]],
         3, False, None),
        ([[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5]], 3, False, None),
        ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
         [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
         2, True, [1, 3, 0, 4, 2, 1]),
        ([[1.0, 2.0]], [[0.1, 0.2]], 5, False, None),
    ]
    for x, bl, k, draw, idx in cases:
        res = solve("baseline_expand", baselines=bl, inputs=x,
                    n_samples=k, draw_baseline_from_distrib=draw,
                    baseline_indices=idx)
        ora = _o_baseline(x, bl, k, draw, idx)
        for key in ("mode", "expanded_baselines", "num_drawn", "indices",
                    "num_ref", "shape"):
            assert _close(_get(res, key), ora[key]), \
                "baseline_expand.%s must equal the oracle" % key
        # ★ 双向断言 1：draw 分支展开的是 baselines[idx]，不能是原样
        if draw:
            assert _get(res, "num_drawn") == k * len(x), \
                "num_drawn must be n_samples * bsz"
            assert len(_get(res, "indices")) == k * len(x), \
                "the number of drawn indices must be n_samples * bsz"
            _must_differ(_get(res, "expanded_baselines"), bl,
                         "draw=True must expand baselines by the drawn indices")
            # 错误候选：只抽 k 个（漏乘 bsz）
            assert _get(res, "num_drawn") != k, \
                "num_drawn must not be just n_samples"
        else:
            assert _get(res, "num_drawn") == 0, \
                "the default branch must not draw indices"
            if len(bl) == len(x) and len(x) > 1:
                # ★ 双向断言 2：默认扩展是 repeat_interleave，不是 repeat 交替
                exp = _get(res, "expanded_baselines")
                alt = _repeat(bl, k)
                _must_differ(exp, alt,
                             "default baseline expansion must be "
                             "repeat_interleave, not alternating repeat")
                assert _get(res, "shape") == [len(x) * k, len(x[0])], \
                    "expanded baseline shape must be [bsz*k, d]"
            else:
                assert _get(res, "mode") == "unchanged", \
                    "a scalar/single baseline must be kept unchanged"
                assert _close(_get(res, "expanded_baselines"), bl), \
                    "unchanged branch must return the original baselines"

    # ---- 域约束 ----
    x = [[1.0, 2.0], [3.0, 4.0]]
    bl = [[0.5, 0.5]]
    for bad_kw, kw_payload in (
        ("no-indices", dict(draw_baseline_from_distrib=True,
                            baseline_indices=None)),
        ("count", dict(draw_baseline_from_distrib=True,
                       baseline_indices=[0, 0, 0])),
        ("oob", dict(draw_baseline_from_distrib=True,
                     baseline_indices=[0, 0, 0, 0, 0, 5])),
    ):
        try:
            solve("baseline_expand", baselines=bl, inputs=x, n_samples=3,
                  **kw_payload)
        except ValueError as exc:
            msg = str(exc)
            want = {"no-indices": "must be provided", "count": "6 elements",
                    "oob": "out of range"}[bad_kw]
            assert want in msg, \
                "baseline_expand must reject %s with %r, got %r" \
                % (bad_kw, want, msg)
        else:
            raise AssertionError("baseline_expand must reject %s" % bad_kw)
    try:
        solve("baseline_expand", baselines=bl, inputs=x, n_samples=0)
    except ValueError as exc:
        assert "positive integer" in str(exc), "n_samples must be positive"
    else:
        raise AssertionError("baseline_expand must reject n_samples=0")


# =============================================================== check 5
def check_noise_tunnel(solve):
    cases = [
        (2, 2, 6, 2, 3), (2, 3, 10, 3, 5), (3, 2, 5, None, 7),
        (1, 3, 7, 2, 11), (3, 3, 4, 10, 13), (2, 4, 12, 5, 17),
    ]
    for b, d, nt, bs, seed in cases:
        c = _mk_agg(b, d, nt, bs, seed)
        x, noise, attrs = c["x"], c["noise"], c["attrs"]
        sizes, delta_parts = c["sizes"], c["delta_parts"]
        flat_pert = c["pert"]
        for nt_type in ("smoothgrad", "smoothgrad_sq", "vargrad"):
            for with_delta in (False, True):
                kw = dict(inputs=x, noise=noise, attributions=attrs,
                          nt_type=nt_type, nt_samples=nt,
                          nt_samples_batch_size=bs, stdevs=0.5)
                if with_delta:
                    kw["return_convergence_delta"] = True
                    kw["delta_parts"] = delta_parts
                res = solve("noise_tunnel", **kw)
                ora = _o_acc(attrs, sizes, nt, d)
                batch = _o_batch(nt, bs)
                for key in ("attributions", "smoothgrad", "smoothgrad_sq",
                            "vargrad", "sum", "sum_sq", "expected",
                            "expected_sq", "stdevs_",
                            "effective_batch_size", "partition", "remaining",
                            "batch_sizes", "bsz", "d", "nt_samples",
                            "nt_type"):
                    if key in ("attributions",):
                        want = ora[nt_type]
                    elif key in ("smoothgrad", "smoothgrad_sq", "vargrad",
                                 "sum", "sum_sq", "expected", "expected_sq",
                                 "bsz", "d", "nt_samples"):
                        want = ora[key]
                    elif key in ("effective_batch_size", "partition",
                                 "remaining", "batch_sizes"):
                        want = batch[key]
                    elif key == "stdevs_":
                        want = [0.5]
                    else:
                        want = nt_type
                    assert _close(_get(res, key), want), \
                        "noise_tunnel.%s must equal the oracle (%s, delta=%s)" \
                        % (key, nt_type, with_delta)

                # ★ 双向断言 1：perturbed 是 per-partition 拼接，不是一次性
                #   repeat_interleave(inputs, nt_samples)
                assert _close(_get(res, "perturbed"), flat_pert), \
                    "perturbed must be concat_p(repeat_interleave(inputs,k_p)) + noise"
                rep_all = _repeat_interleave(x, nt)
                wrong_pert = [[rep_all[i][j] + noise[i][j] for j in range(d)]
                              for i in range(b * nt)]
                if len(sizes) > 1 and b > 1:
                    _must_differ(_get(res, "perturbed"), wrong_pert,
                                 "perturbed must be built per partition, not "
                                 "as one repeat_interleave(inputs, nt_samples)")

                # ★ 双向断言 2：delta 是 cat 拼接，不是文档声称的算术平均
                if with_delta:
                    cat = [v for part in delta_parts for v in part]
                    assert _close(_get(res, "delta"), cat), \
                        "delta must be torch.cat(delta_partial_list, dim=0)"
                    assert _get(res, "delta_shape") == [b * nt], \
                        "delta_shape must be [bsz*nt_samples]"
                    # 错误候选：每个 bi 对 partition 做平均，再广播到 b*nt
                    mean_delta = []
                    for bi in range(b):
                        vals = [delta_parts[p][bi] for p in range(len(sizes))]
                        mean_delta.append(sum(vals) / len(vals))
                    wrong_delta = [mean_delta[i % b] for i in range(b * nt)]
                    _must_differ(_get(res, "delta"), wrong_delta,
                                 "delta must be cat, not arithmetic mean across "
                                 "nt_samples")
                else:
                    assert _get(res, "delta") is None, \
                        "delta must be None when return_convergence_delta=False"
                    assert _get(res, "delta_shape") is None, \
                        "delta_shape must be None when delta is not requested"

        # ★ 双向断言 3：stdevs=0 时 smoothgrad=x, smoothgrad_sq=x^2, vargrad=0
        zero_noise = [[0.0] * d for _ in range(b * nt)]
        zero_pert = []
        for k in sizes:
            zero_pert.extend(_repeat_interleave(x, k))
        for nt_type, want in (
            ("smoothgrad", x),
            ("smoothgrad_sq", [[v * v for v in row] for row in x]),
            ("vargrad", [[0.0] * d for _ in range(b)]),
        ):
            r = solve("noise_tunnel", inputs=x, noise=zero_noise,
                      attributions=zero_pert, nt_type=nt_type,
                      nt_samples=nt, nt_samples_batch_size=bs, stdevs=0.0)
            assert _close(_get(r, "attributions"), want), \
                "stdevs=0 must give %s = %r" % (nt_type, want)

    # ---- 域约束 ----
    x = [[1.0, 2.0], [3.0, 4.0]]
    noise = [[0.0, 0.0]] * 6
    attrs = [[1.0, 2.0]] * 6
    for bad, kw in (
        (1, "float"), ((0.5,), "official bug"),
        ((0.5, 0.7), "must equal"),
    ):
        try:
            solve("noise_tunnel", inputs=x, noise=noise, attributions=attrs,
                  nt_samples=3, stdevs=bad)
        except ValueError as exc:
            assert kw in str(exc), \
                "noise_tunnel must reject stdevs=%r with %r" % (bad, kw)
        else:
            raise AssertionError("noise_tunnel must reject stdevs=%r" % (bad,))
    for bad, kw in (("smoothgrad_sq2", "must be either"),):
        try:
            solve("noise_tunnel", inputs=x, noise=noise, attributions=attrs,
                  nt_samples=3, nt_type=bad)
        except ValueError as exc:
            assert kw in str(exc), "unknown nt_type must be rejected"
        else:
            raise AssertionError("noise_tunnel must reject nt_type=%r" % bad)
    # delta_parts 必给 / 数量 / 长度
    for label, payload, want in (
        ("missing", dict(return_convergence_delta=True), "must be provided"),
        ("count", dict(return_convergence_delta=True,
                       delta_parts=[[1.0] * 2],
                       nt_samples_batch_size=1), "one entry per batch partition"),
    ):
        try:
            solve("noise_tunnel", inputs=x, noise=noise, attributions=attrs,
                  nt_samples=3, **payload)
        except ValueError as exc:
            assert want in str(exc), \
                "noise_tunnel delta %s must be rejected with %r, got %r" \
                % (label, want, str(exc))
        else:
            raise AssertionError("noise_tunnel must reject delta %s" % label)
    try:
        solve("bogus_mode")
    except ValueError as exc:
        assert "unknown mode" in str(exc), "unknown mode must be rejected"
    else:
        raise AssertionError("solve must reject an unknown mode")
