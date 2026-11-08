"""order 838 判题用例（暂存）：librosa YIN 的 CMND / 抛物线插值 / 绝对阈值账。

独立 oracle（不 import solution），每个 check_* 含网格对拍 + 手算锚点。
"""

import math

TINY = 2.2250738585072014e-308


# ------------------------------------------------------------------ 独立 oracle
def oracle_cmnd(sq, acf, min_period, max_period, tiny):
    sq = list(sq)
    acf = list(acf)
    # E = cumsum(sq)，然后官方把 E[0] 抹零
    e = []
    run = 0.0
    for v in sq:
        run += v
        e.append(run)
    e[0] = 0.0
    # d[k] = 2*(r0-rk) - E[k-1]
    d = [0.0] * (max_period + 1)
    for k in range(1, max_period + 1):
        d[k] = 2.0 * (acf[0] - acf[k]) - e[k - 1]
    # cummean[j] = (d[1]+...+d[j+1]) / (j+1)
    cm = []
    run = 0.0
    for j in range(max_period):
        run += d[j + 1]
        cm.append(run / float(j + 1))
    den = cm[min_period - 1:max_period]
    num = d[min_period:max_period + 1]
    out = [num[i] / (den[i] + tiny) for i in range(len(num))]
    return {
        "difference": d,
        "cumulative_mean": cm,
        "denominator": den,
        "cmnd": out,
    }


def oracle_parabolic(x):
    x = list(x)
    n = len(x)
    a = [0.0] * n
    b = [0.0] * n
    s = [0.0] * n
    for i in range(1, n - 1):
        a[i] = x[i + 1] + x[i - 1] - 2.0 * x[i]
        b[i] = (x[i + 1] - x[i - 1]) / 2.0
        s[i] = 0.0 if abs(b[i]) >= abs(a[i]) else (-b[i]) / a[i]
    s[0] = 0.0
    s[n - 1] = 0.0
    return {"a": a, "b": b, "shifts": s}


def oracle_trough(x, min_period, sr, trough_threshold):
    x = list(x)
    n = len(x)
    par = oracle_parabolic(x)
    sh = par["shifts"]
    trough = [False] * n
    for i in range(1, n - 1):
        trough[i] = (x[i] < x[i - 1]) and (x[i] <= x[i + 1])
    trough[n - 1] = x[n - 1] < x[n - 2]
    trough[0] = x[0] < x[1]
    below = [bool(trough[i]) and bool(x[i] < trough_threshold) for i in range(n)]
    gmin = 0
    for i in range(1, n):
        if x[i] < x[gmin]:
            gmin = i
    period_idx = -1
    for i in range(n):
        if below[i]:
            period_idx = i
            break
    no_trough = period_idx < 0
    if no_trough:
        period_idx = gmin
    shift = sh[period_idx]
    period = min_period + period_idx + shift
    return {
        "is_trough": trough,
        "is_threshold_trough": below,
        "global_min": gmin,
        "yin_period": period_idx,
        "no_trough_below_threshold": no_trough,
        "shift": shift,
        "period": period,
        "f0": sr / period,
    }


# ------------------------------------------------------------------ 工具
def _close(actual, expect, atol=1e-9, rtol=1e-9, tag=""):
    if isinstance(expect, (list, tuple)):
        assert isinstance(actual, (list, tuple)), "%s: expected list, got %r" % (tag, type(actual))
        assert len(actual) == len(expect), "%s: length %d != %d" % (tag, len(actual), len(expect))
        for i, item in enumerate(expect):
            _close(actual[i], item, atol=atol, rtol=rtol, tag="%s[%d]" % (tag, i))
        return
    if isinstance(expect, bool) or isinstance(actual, bool):
        assert bool(actual) == bool(expect), "%s: got %r expected %r" % (tag, actual, expect)
        return
    if isinstance(expect, int):
        assert int(actual) == int(expect), "%s: got %r expected %r" % (tag, actual, expect)
        return
    a = float(actual)
    e = float(expect)
    if e != e:  # NaN
        assert a != a, "%s: got %r expected NaN" % (tag, actual)
        return
    assert abs(a - e) <= atol + rtol * abs(e), "%s: got %r expected %r" % (tag, actual, expect)


def _cmp(got, exp, tag):
    assert isinstance(got, dict), "%s: expected dict, got %r" % (tag, type(got))
    assert set(got.keys()) == set(exp.keys()), "%s: keys %r != %r" % (
        tag, sorted(got.keys()), sorted(exp.keys()))
    for k in exp:
        _close(got[k], exp[k], tag=tag + "." + k)


class _Rng:
    """确定性 LCG，避免依赖 random/numpy。"""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / float(0x7FFFFFFF)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.next()

    def randint(self, lo, hi):
        return lo + int(self.next() * (hi - lo + 1)) % (hi - lo + 1)

    def pick(self, seq):
        return seq[self.randint(0, len(seq) - 1)]


# ------------------------------------------------------------------ check_*
def check_cmnd(solve):
    # 手算锚点 1：sq=[1,4,9,16,25] acf=[55,34,20,10,4] min=2 max=4
    # E = [1,5,14,30,55] -> E[0]=0
    # d[1]=2*(55-34)-E[0]=42
    # d[2]=2*(55-20)-E[1]=70-5=65
    # d[3]=2*(55-10)-E[2]=90-14=76
    # d[4]=2*(55-4)-E[3]=102-30=72
    # cummean = [42/1, (42+65)/2, (42+65+76)/3, (42+65+76+72)/4]
    #          = [42, 53.5, 61, 63.75]
    # den = cummean[1:4] = [53.5, 61, 63.75]
    # cmnd = [65/53.5, 76/61, 72/63.75]
    r = solve(
        "cmnd",
        sq=[1.0, 4.0, 9.0, 16.0, 25.0],
        acf=[55.0, 34.0, 20.0, 10.0, 4.0],
        min_period=2,
        max_period=4,
    )
    exp = {
        "difference": [0.0, 42.0, 65.0, 76.0, 72.0],
        "cumulative_mean": [42.0, 53.5, 61.0, 63.75],
        "denominator": [53.5, 61.0, 63.75],
        "cmnd": [65.0 / 53.5, 76.0 / 61.0, 72.0 / 63.75],
    }
    _cmp(r, exp, "cmnd.anchor1")

    # 手算锚点 2：E[0] quirk —— acf 全常数时 d[1] 必须只依赖 acf 差
    # sq=[4,4,4,4], acf=[10,7,5,3], min=1, max=3
    # E = [4,8,12,16] -> E[0]=0
    # d[1]=2*(10-7)-0=6 ; d[2]=2*(10-5)-E[1]=10-8=2 ; d[3]=2*(10-3)-E[2]=14-12=2
    # cummean=[6, 4, 10/3]
    # den = cummean[0:3] = [6,4,10/3]
    # cmnd = [d1/6, d2/4, d3/(10/3)] = [1, 0.5, 0.6]
    r = solve(
        "cmnd",
        sq=[4.0, 4.0, 4.0, 4.0],
        acf=[10.0, 7.0, 5.0, 3.0],
        min_period=1,
        max_period=3,
    )
    exp = {
        "difference": [0.0, 6.0, 2.0, 2.0],
        "cumulative_mean": [6.0, 4.0, 10.0 / 3.0],
        "denominator": [6.0, 4.0, 10.0 / 3.0],
        "cmnd": [1.0, 0.5, 0.6],
    }
    _cmp(r, exp, "cmnd.anchor2")

    # 手算锚点 3：den 切片偏移 min_period-1 —— min_period=2 时 den[0] 应是 cummean[1]
    # sq=[1,1,1,1,1], acf=[5,3,2,1,0], min=2, max=4
    # E=[1,2,3,4,5] -> E[0]=0
    # d[1]=2*(5-3)-0=4 ; d[2]=2*(5-2)-E[1]=6-2=4 ; d[3]=2*(5-1)-E[2]=8-3=5 ; d[4]=2*(5-0)-E[3]=10-4=6
    # cummean=[4,4,13/3,19/4]
    # den=cummean[1:4]=[4,13/3,19/4]
    # cmnd=[d2/4, d3/(13/3), d4/(19/4)]=[1, 15/13, 24/19]
    r = solve(
        "cmnd",
        sq=[1.0, 1.0, 1.0, 1.0, 1.0],
        acf=[5.0, 3.0, 2.0, 1.0, 0.0],
        min_period=2,
        max_period=4,
    )
    exp = {
        "difference": [0.0, 4.0, 4.0, 5.0, 6.0],
        "cumulative_mean": [4.0, 4.0, 13.0 / 3.0, 19.0 / 4.0],
        "denominator": [4.0, 13.0 / 3.0, 19.0 / 4.0],
        "cmnd": [1.0, 15.0 / 13.0, 24.0 / 19.0],
    }
    _cmp(r, exp, "cmnd.anchor3")

    # 手算锚点 4：tiny 加在分母上 —— den=0 时不能 ZeroDivisionError
    # sq=[0,0,0], acf=[0,0,0], min=1, max=2: E=[0,0,0]->E[0]=0
    # d[1]=0, d[2]=0 ; cummean=[0,0] ; den=[0,0] ; cmnd=[0/tiny, 0/tiny]=0
    r = solve(
        "cmnd",
        sq=[0.0, 0.0, 0.0],
        acf=[0.0, 0.0, 0.0],
        min_period=1,
        max_period=2,
    )
    exp = {
        "difference": [0.0, 0.0, 0.0],
        "cumulative_mean": [0.0, 0.0],
        "denominator": [0.0, 0.0],
        "cmnd": [0.0, 0.0],
    }
    _cmp(r, exp, "cmnd.anchor4")

    # 网格对拍
    rng = _Rng(20261106)
    for case in range(120):
        max_period = rng.randint(2, 8)
        min_period = rng.randint(1, max_period)
        length = max_period + rng.randint(1, 4)
        sq = [rng.uniform(0.0, 4.0) for _ in range(length)]
        acf = [rng.uniform(-1.0, 6.0) for _ in range(max_period + 1)]
        acf[0] = max(acf) + rng.uniform(0.0, 2.0)
        tiny = TINY
        got = solve(
            "cmnd",
            sq=sq,
            acf=acf,
            min_period=min_period,
            max_period=max_period,
            tiny=tiny,
        )
        exp = oracle_cmnd(sq, acf, min_period, max_period, tiny)
        _cmp(got, exp, "cmnd.grid%d" % case)
    print("cmnd OK")


def check_parabolic_shifts(solve):
    # 手算锚点 1：x=[1.0,0.2,0.5]
    # i=1: a=0.5+1.0-0.4=1.1, b=(0.5-1.0)/2=-0.25, |b|<|a| -> shift=0.25/1.1
    # 首尾强制 0
    r = solve("parabolic_shifts", x=[1.0, 0.2, 0.5])
    exp = {
        "a": [0.0, 1.1, 0.0],
        "b": [0.0, -0.25, 0.0],
        "shifts": [0.0, 0.25 / 1.1, 0.0],
    }
    _cmp(r, exp, "par.anchor1")

    # 手算锚点 2：a=0 分支 —— x=[0,1,2] 时 a=2+0-2=0, b=(2-0)/2=1, |b|>=|a| -> shift=0
    r = solve("parabolic_shifts", x=[0.0, 1.0, 2.0])
    exp = {
        "a": [0.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "shifts": [0.0, 0.0, 0.0],
    }
    _cmp(r, exp, "par.anchor2")

    # 手算锚点 3：|b| >= |a| 抑制 —— x=[0,1,3] a=3+0-2=1, b=(3-0)/2=1.5 -> shift=0
    r = solve("parabolic_shifts", x=[0.0, 1.0, 3.0])
    exp = {
        "a": [0.0, 1.0, 0.0],
        "b": [0.0, 1.5, 0.0],
        "shifts": [0.0, 0.0, 0.0],
    }
    _cmp(r, exp, "par.anchor3")

    # 手算锚点 3b：a<0 且 b<0 -> shift=-b/a 为负
    # x=[1,1.5,0]: a=0+1-3=-2, b=(0-1)/2=-0.5 -> shift = 0.5/-2 = -0.25
    r = solve("parabolic_shifts", x=[1.0, 1.5, 0.0])
    exp = {
        "a": [0.0, -2.0, 0.0],
        "b": [0.0, -0.5, 0.0],
        "shifts": [0.0, -0.25, 0.0],
    }
    _cmp(r, exp, "par.anchor3b")

    # 手算锚点 4：五点，验证中间两 bin 各自独立算
    # x=[1,0,1,0,1]
    # i=1: a=1+1-0=2, b=(1-1)/2=0 -> |b|<|a| -> shift=0
    # i=2: a=0+0-2=-2, b=(0-0)/2=0 -> shift=0
    # i=3: a=1+1-0=2, b=(1-1)/2=0 -> shift=0
    r = solve("parabolic_shifts", x=[1.0, 0.0, 1.0, 0.0, 1.0])
    exp = {
        "a": [0.0, 2.0, -2.0, 2.0, 0.0],
        "b": [0.0, 0.0, 0.0, 0.0, 0.0],
        "shifts": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    _cmp(r, exp, "par.anchor4")

    # 手算锚点 5：|b| == |a| 的相等边界 —— 官方是 >= 抑制，必须得 0
    # x=[-0.5, 0.0, 1.5]
    # i=1: a = 1.5 + (-0.5) - 0 = 1.0, b = (1.5 - (-0.5)) / 2 = 1.0
    # |b| >= |a| -> shift = 0（若改成 > 则会得到 -b/a = -1.0）
    r = solve("parabolic_shifts", x=[-0.5, 0.0, 1.5])
    exp = {
        "a": [0.0, 1.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "shifts": [0.0, 0.0, 0.0],
    }
    _cmp(r, exp, "par.anchor5")

    # 手算锚点 6：首尾 bin 必须为 0（numba stencil 默认边界输出 0，
    # librosa 又显式置 0）—— x=[0.0, 1.0, 0.0] 若误把首尾也算进去会得到非 0
    r = solve("parabolic_shifts", x=[0.0, 1.0, 0.0])
    exp = {
        "a": [0.0, -2.0, 0.0],
        "b": [0.0, 0.0, 0.0],
        "shifts": [0.0, 0.0, 0.0],
    }
    _cmp(r, exp, "par.anchor6")

    # 网格对拍
    rng = _Rng(83811)
    for case in range(120):
        n = rng.randint(3, 12)
        style = case % 3
        if style == 0:
            x = [rng.uniform(-3.0, 3.0) for _ in range(n)]
        elif style == 1:
            x = [rng.uniform(0.0, 1.0) for _ in range(n)]
        else:
            # 量化到 0.25，制造并列与等差段
            x = [round(rng.uniform(0.0, 3.0) * 4.0) / 4.0 for _ in range(n)]
        got = solve("parabolic_shifts", x=x)
        exp = oracle_parabolic(x)
        _cmp(got, exp, "par.grid%d" % case)
    print("parabolic_shifts OK")


def check_trough_selection(solve):
    # 手算锚点 1：x=[0.5,0.05,0.3,0.02,0.4] min=2 sr=100 thr=0.1
    # localmin: i=1 (0.05<0.5 && 0.05<=0.3)=T ; i=2 F ; i=3 (0.02<0.3 && 0.02<=0.4)=T
    # 末元素 0.4<0.02 F ; 第0列覆盖 0.5<0.05 F
    # below: i=1 (0.05<0.1) T ; i=3 (0.02<0.1) T
    # argmax -> 1 ; global_min=3 ; no_trough=False
    # shifts: i=1 a=0.3+0.5-0.1=0.7, b=(0.3-0.5)/2=-0.1 -> 0.1/0.7
    # period = 2+1+0.1/0.7 ; f0 = 100/period
    r = solve(
        "trough_selection",
        x=[0.5, 0.05, 0.3, 0.02, 0.4],
        min_period=2,
        sr=100.0,
        trough_threshold=0.1,
    )
    exp = {
        "is_trough": [False, True, False, True, False],
        "is_threshold_trough": [False, True, False, True, False],
        "global_min": 3,
        "yin_period": 1,
        "no_trough_below_threshold": False,
        "shift": 0.1 / 0.7,
        "period": 3.0 + 0.1 / 0.7,
        "f0": 100.0 / (3.0 + 0.1 / 0.7),
    }
    _cmp(r, exp, "trough.anchor1")

    # 手算锚点 2：左严格右非严格 —— x=[1,1,0,1] i=1: x1<x0? 1<1 F -> 非谷
    # i=2: 0<1 T 且 0<=1 T -> T ; 末 1<0 F ; 第0列 1<1 F
    r = solve(
        "trough_selection",
        x=[1.0, 1.0, 0.0, 1.0],
        min_period=1,
        sr=50.0,
        trough_threshold=0.5,
    )
    exp = {
        "is_trough": [False, False, True, False],
        "is_threshold_trough": [False, False, True, False],
        "global_min": 2,
        "yin_period": 2,
        "no_trough_below_threshold": False,
        "shift": 0.0,
        "period": 3.0,
        "f0": 50.0 / 3.0,
    }
    _cmp(r, exp, "trough.anchor2")

    # 手算锚点 3：全 False 回退 global_min
    # x=[0.5,0.4,0.6,0.7] thr=0.05 -> 无谷值低于阈值
    # localmin: i=1 (0.4<0.5 && 0.4<=0.6) T ; i=2 (0.6<0.4) F ; 末 0.7<0.6 F ; 第0列 0.5<0.4 F
    # below 全 F -> no_trough=True, yin_period=global_min=1
    # shifts i=1: a=0.6+0.5-0.8=0.3, b=(0.6-0.5)/2=0.05 -> |b|<|a| -> -0.05/0.3
    # period = 1+1-0.05/0.3 ; f0 = 50/period
    r = solve(
        "trough_selection",
        x=[0.5, 0.4, 0.6, 0.7],
        min_period=1,
        sr=50.0,
        trough_threshold=0.05,
    )
    exp = {
        "is_trough": [False, True, False, False],
        "is_threshold_trough": [False, False, False, False],
        "global_min": 1,
        "yin_period": 1,
        "no_trough_below_threshold": True,
        "shift": -0.05 / 0.3,
        "period": 2.0 - 0.05 / 0.3,
        "f0": 50.0 / (2.0 - 0.05 / 0.3),
    }
    _cmp(r, exp, "trough.anchor3")

    # 手算锚点 4：阈值严格小于 —— 等于阈值不算 trough
    # x=[0.5,0.1,0.3] thr=0.1
    # localmin i=1 (0.1<0.5 && 0.1<=0.3) T ; 末 0.3<0.1 F ; 第0列 0.5<0.1 F
    # below: 0.1<0.1 F -> 全 F -> 回退 global_min=1
    # shifts i=1: a=0.3+0.5-0.2=0.6, b=(0.3-0.5)/2=-0.1 -> 0.1/0.6
    # period = 1+1+0.1/0.6 ; f0=80/period
    r = solve(
        "trough_selection",
        x=[0.5, 0.1, 0.3],
        min_period=1,
        sr=80.0,
        trough_threshold=0.1,
    )
    exp = {
        "is_trough": [False, True, False],
        "is_threshold_trough": [False, False, False],
        "global_min": 1,
        "yin_period": 1,
        "no_trough_below_threshold": True,
        "shift": 0.1 / 0.6,
        "period": 2.0 + 0.1 / 0.6,
        "f0": 80.0 / (2.0 + 0.1 / 0.6),
    }
    _cmp(r, exp, "trough.anchor4")

    # 手算锚点 5：min_period 偏移 + 并列 argmin 取第一个
    # x=[0.2,0.05,0.3,0.05,0.4] min=3 sr=100 thr=0.1
    # localmin: i=1 (0.05<0.2 && 0.05<=0.3) T ; i=2 (0.3<0.05) F ;
    #   i=3 (0.05<0.3 && 0.05<=0.4) T ; 末 0.4<0.05 F ; 第0列 0.2<0.05 F
    # below: i=1 T, i=3 T -> argmax 取第一个 True -> 1
    # global_min: 0.05 出现在 1 和 3 -> 取第一个 -> 1
    # shifts i=1: a=0.3+0.2-0.1=0.4, b=(0.3-0.2)/2=0.05 -> -0.05/0.4
    # period = 3+1-0.05/0.4 ; f0=100/period
    r = solve(
        "trough_selection",
        x=[0.2, 0.05, 0.3, 0.05, 0.4],
        min_period=3,
        sr=100.0,
        trough_threshold=0.1,
    )
    exp = {
        "is_trough": [False, True, False, True, False],
        "is_threshold_trough": [False, True, False, True, False],
        "global_min": 1,
        "yin_period": 1,
        "no_trough_below_threshold": False,
        "shift": -0.05 / 0.4,
        "period": 4.0 - 0.05 / 0.4,
        "f0": 100.0 / (4.0 - 0.05 / 0.4),
    }
    _cmp(r, exp, "trough.anchor5")

    # 手算锚点 6：yin_period 落在首 bin —— 边缘 shift 必须为 0
    # x=[0.05, 0.5, 0.3], min=1, sr=100, thr=0.1
    # localmin: i=1 (0.5<0.05) F ; 末 0.3<0.5 T ; 第0列覆盖 0.05<0.5 T
    # is_trough=[T,F,T] ; below: 0.05<0.1 T, 0.3<0.1 F -> argmax 首个 True = 0
    # global_min=0 ; no_trough=False ; shift=shifts[0]=0（边缘强制 0）
    # period = 1+0+0 = 1 ; f0 = 100/1
    r = solve(
        "trough_selection",
        x=[0.05, 0.5, 0.3],
        min_period=1,
        sr=100.0,
        trough_threshold=0.1,
    )
    exp = {
        "is_trough": [True, False, True],
        "is_threshold_trough": [True, False, False],
        "global_min": 0,
        "yin_period": 0,
        "no_trough_below_threshold": False,
        "shift": 0.0,
        "period": 1.0,
        "f0": 100.0,
    }
    _cmp(r, exp, "trough.anchor6")

    # 手算锚点 7：yin_period 落在末 bin —— 边缘 shift 必须为 0
    # x=[0.5, 0.3, 0.05], min=2, sr=100, thr=0.1
    # localmin: i=1 (0.3<0.5 且 0.3<=0.05) F ; 末 0.05<0.3 T ; 第0列 0.5<0.3 F
    # is_trough=[F,F,T] ; below: 0.05<0.1 T -> yin_period=2 ; global_min=2
    # shift=shifts[2]=0 ; period = 2+2+0 = 4 ; f0 = 100/4 = 25
    r = solve(
        "trough_selection",
        x=[0.5, 0.3, 0.05],
        min_period=2,
        sr=100.0,
        trough_threshold=0.1,
    )
    exp = {
        "is_trough": [False, False, True],
        "is_threshold_trough": [False, False, True],
        "global_min": 2,
        "yin_period": 2,
        "no_trough_below_threshold": False,
        "shift": 0.0,
        "period": 4.0,
        "f0": 25.0,
    }
    _cmp(r, exp, "trough.anchor7")

    # 网格对拍（含并列值风格：量化网格才能区分 localmin 的严格/非严格边界）
    rng = _Rng(83812)
    for case in range(120):
        n = rng.randint(3, 12)
        style = case % 4
        if style == 0:
            x = [rng.uniform(0.0, 1.0) for _ in range(n)]
        elif style == 1:
            x = [rng.uniform(0.0, 0.3) for _ in range(n)]
        elif style == 2:
            x = [float(rng.randint(0, 3)) / 3.0 for _ in range(n)]
        else:
            x = [float(rng.randint(0, 1)) for _ in range(n)]
        min_period = rng.randint(1, 5)
        sr = rng.uniform(50.0, 4000.0)
        thr = rng.uniform(0.01, 0.9)
        got = solve(
            "trough_selection",
            x=x,
            min_period=min_period,
            sr=sr,
            trough_threshold=thr,
        )
        exp = oracle_trough(x, min_period, sr, thr)
        _cmp(got, exp, "trough.grid%d" % case)
    print("trough_selection OK")
