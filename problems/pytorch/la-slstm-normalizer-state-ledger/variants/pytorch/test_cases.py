"""order 839 判题用例（暂存）：xLSTM sLSTM 的 max 稳定器、normalizer state 与点式递推账。

独立 oracle（不 import solution），每个 check_* 含网格对拍 + 手算锚点。
"""

import math

NAN = float("nan")


# ------------------------------------------------------------------ 独立 oracle
def oracle_logsigmoid(x):
    if x >= 0.0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def oracle_sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def oracle_max_stabilizer(iraw, fraw, m_prev, n_prev):
    h = len(iraw)
    logfplusm = []
    for i in range(h):
        logfplusm.append(m_prev[i] + oracle_logsigmoid(fraw[i]))
    zero_n = True
    for v in n_prev:
        if v != 0.0:
            zero_n = False
            break
    if zero_n:
        m_new = list(iraw)
    else:
        m_new = []
        for i in range(h):
            m_new.append(iraw[i] if iraw[i] > logfplusm[i] else logfplusm[i])
    igate = []
    fgate = []
    for i in range(h):
        gi = math.exp(iraw[i] - m_new[i])
        gf = math.exp(logfplusm[i] - m_new[i])
        igate.append(1.0 if gi > 1.0 else gi)
        fgate.append(1.0 if gf > 1.0 else gf)
    le = []
    for i in range(h):
        le.append(igate[i] <= 1.0 and fgate[i] <= 1.0)
    return {
        "logfplusm": logfplusm,
        "m_new": m_new,
        "zero_n_branch": zero_n,
        "igate": igate,
        "fgate": fgate,
        "gate_le_one": le,
    }


def oracle_pointwise(wx, ry, bias, states, h):
    raw = []
    for i in range(4 * h):
        raw.append(wx[i] + ry[i] + bias[i])
    iraw = raw[0:h]
    fraw = raw[h:2 * h]
    zraw = raw[2 * h:3 * h]
    oraw = raw[3 * h:4 * h]
    stab = oracle_max_stabilizer(iraw, fraw, states[3], states[2])
    ogate = [oracle_sigmoid(oraw[i]) for i in range(h)]
    zval = [math.tanh(zraw[i]) for i in range(h)]
    c_new = [stab["fgate"][i] * states[1][i] + stab["igate"][i] * zval[i]
             for i in range(h)]
    n_new = [stab["fgate"][i] * states[2][i] + stab["igate"][i] for i in range(h)]
    y_new = []
    for i in range(h):
        if n_new[i] == 0.0:
            y_new.append(NAN)
        else:
            y_new.append(ogate[i] * c_new[i] / n_new[i])
    return {
        "raw_gates": [iraw, fraw, zraw, oraw],
        "logfplusm": stab["logfplusm"],
        "m_new": stab["m_new"],
        "zero_n_branch": stab["zero_n_branch"],
        "igate": stab["igate"],
        "fgate": stab["fgate"],
        "ogate": ogate,
        "zval": zval,
        "c_new": c_new,
        "n_new": n_new,
        "y_new": y_new,
    }


def oracle_bias(head_dim, block_idx, num_blocks, mode):
    if num_blocks > 1:
        ratio = block_idx / float(num_blocks - 1)
    else:
        ratio = 0.0
    if mode == "powerlaw":
        exponent = 0.3 + 1.3 * ratio
        bias = []
        for d in range(head_dim):
            base = NAN if head_dim == 1 else d / float(head_dim - 1)
            bias.append(-(-5.0 + 12.0 * (base ** exponent)))
        return {"bias": bias, "exponent": exponent, "ratio": ratio}
    exponent = 0.0
    bias = []
    for d in range(head_dim):
        bias.append(3.0 if head_dim == 1 else 3.0 + 3.0 * d / float(head_dim - 1))
    return {"bias": bias, "exponent": exponent, "ratio": ratio}


# ------------------------------------------------------------------ 工具
def _close(actual, expect, atol=1e-9, rtol=1e-9, tag=""):
    if isinstance(expect, (list, tuple)):
        assert isinstance(actual, (list, tuple)), "%s: expected list got %r" % (tag, type(actual))
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
    if e != e:
        assert a != a, "%s: got %r expected NaN" % (tag, actual)
        return
    if a != a:
        raise AssertionError("%s: got NaN expected %r" % (tag, e))
    assert abs(a - e) <= atol + rtol * abs(e), "%s: got %r expected %r" % (tag, actual, expect)


def _cmp(got, exp, tag):
    assert isinstance(got, dict), "%s: expected dict got %r" % (tag, type(got))
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


# ------------------------------------------------------------------ check_*
def check_max_stabilizer(solve):
    # 手算锚点 1：n 全零 -> 整帧走 torch.all 分支，mnew = iraw
    # iraw=[1.0, -2.0], fraw=[0.5, 0.25], m_prev=[0.0, 0.0], n_prev=[0.0, 0.0]
    # logfplusm = 0 + logsigmoid(0.5), logsigmoid(0.25)
    # logsigmoid(0.5) = -log(1+exp(-0.5)) ; logsigmoid(0.25) = -log(1+exp(-0.25))
    # m_new = iraw = [1.0, -2.0]
    # igate = min(exp(iraw - m_new),1) = [1.0, 1.0]
    # fgate = min(exp(logfplusm - m_new),1)
    #   通道 0: exp(-0.4741 - 1) = 0.2050
    #   通道 1: exp(-0.5759 + 2) = 4.154 > 1 -> 被 minimum 截断为 1.0
    ls05 = -math.log1p(math.exp(-0.5))
    ls025 = -math.log1p(math.exp(-0.25))
    r = solve("max_stabilizer", iraw=[1.0, -2.0], fraw=[0.5, 0.25],
              m_prev=[0.0, 0.0], n_prev=[0.0, 0.0])
    exp = {
        "logfplusm": [ls05, ls025],
        "m_new": [1.0, -2.0],
        "zero_n_branch": True,
        "igate": [1.0, 1.0],
        "fgate": [math.exp(ls05 - 1.0), 1.0],
        "gate_le_one": [True, True],
    }
    _cmp(r, exp, "ms.anchor1")

    # 手算锚点 2：n 只有一个非零 -> torch.all 为 False，整帧走 max 分支
    # iraw=[0.0, 5.0], fraw=[0.0, 0.0], m_prev=[0.0, 0.0], n_prev=[0.0, 1.0]
    # logfplusm = [logsigmoid(0), logsigmoid(0)] = [-log2, -log2]
    # m_new = max(iraw, logfplusm) = [max(0,-log2), max(5,-log2)] = [0, 5]
    # igate = [min(exp(0-0),1), min(exp(5-5),1)] = [1,1]
    # fgate = [min(exp(-log2-0),1), min(exp(-log2-5),1)] = [0.5, exp(-log2-5)]
    ls0 = -math.log(2.0)
    r = solve("max_stabilizer", iraw=[0.0, 5.0], fraw=[0.0, 0.0],
              m_prev=[0.0, 0.0], n_prev=[0.0, 1.0])
    exp = {
        "logfplusm": [ls0, ls0],
        "m_new": [0.0, 5.0],
        "zero_n_branch": False,
        "igate": [1.0, 1.0],
        "fgate": [0.5, math.exp(ls0 - 5.0)],
        "gate_le_one": [True, True],
    }
    _cmp(r, exp, "ms.anchor2")

    # 手算锚点 3：minimum(...,1) 硬上界 —— iraw 远大于 logfplusm 时 igate 应被截到 1
    # 情形：n 非零 -> 走 max 分支，m_new = max(iraw, logfplusm)，
    # 当 iraw >= logfplusm 时 igate = exp(iraw - m_new) = 1（自然 <=1）
    # 反过来 fgate = exp(logfplusm - m_new)；若 m_new 由 iraw 决定且 iraw 远小，
    # fgate 会 > 1，必须被截断。
    # iraw=[-10.0], fraw=[10.0], m_prev=[0.0], n_prev=[1.0]
    # logfplusm = logsigmoid(10) = -log(1+exp(-10)) ≈ -4.54e-5
    # m_new = max(-10, -4.54e-5) = -4.54e-5
    # igate = exp(-10 + 4.54e-5) ≈ 4.54e-5
    # fgate = exp(0) = 1
    ls10 = -math.log1p(math.exp(-10.0))
    r = solve("max_stabilizer", iraw=[-10.0], fraw=[10.0],
              m_prev=[0.0], n_prev=[1.0])
    exp = {
        "logfplusm": [ls10],
        "m_new": [ls10],
        "zero_n_branch": False,
        "igate": [math.exp(-10.0 - ls10)],
        "fgate": [1.0],
        "gate_le_one": [True],
    }
    _cmp(r, exp, "ms.anchor3")

    # 手算锚点 4：igate 超过 1 的情形 —— 走 zero_n 分支时 igate = exp(0) = 1，
    # 但若 iraw 远大于 logfplusm 且 m_new 取 iraw，igate 仍是 1；
    # 真正会 >1 的是 fgate：zero_n 分支下 m_new = iraw，
    # fgate = exp(logfplusm - iraw)，当 logfplusm > iraw 时 > 1，必须截到 1。
    # iraw=[-5.0], fraw=[5.0], m_prev=[10.0], n_prev=[0.0]
    # logfplusm = 10 + logsigmoid(5) = 10 - 0.006715 = 9.993285
    # m_new = iraw = -5.0
    # fgate 原始 exp(9.993285 + 5) = exp(14.99) >> 1 -> 截断为 1
    # igate = exp(-5 + 5) = 1
    ls5 = -math.log1p(math.exp(-5.0))
    r = solve("max_stabilizer", iraw=[-5.0], fraw=[5.0],
              m_prev=[10.0], n_prev=[0.0])
    exp = {
        "logfplusm": [10.0 + ls5],
        "m_new": [-5.0],
        "zero_n_branch": True,
        "igate": [1.0],
        "fgate": [1.0],
        "gate_le_one": [True],
    }
    _cmp(r, exp, "ms.anchor4")

    # 手算锚点 5：m_prev 参与 logfplusm —— m_prev=0 与 m_prev=3 结果不同
    r1 = solve("max_stabilizer", iraw=[0.0], fraw=[0.0], m_prev=[0.0],
               n_prev=[1.0])
    r2 = solve("max_stabilizer", iraw=[0.0], fraw=[0.0], m_prev=[3.0],
               n_prev=[1.0])
    ls_zero = -math.log(2.0)
    _cmp(r1, {
        "logfplusm": [ls_zero],
        "m_new": [0.0],
        "zero_n_branch": False,
        "igate": [1.0],
        "fgate": [math.exp(ls_zero)],
        "gate_le_one": [True],
    }, "ms.anchor5a")
    _cmp(r2, {
        "logfplusm": [3.0 + ls_zero],
        "m_new": [3.0 + ls_zero],
        "zero_n_branch": False,
        "igate": [math.exp(-(3.0 + ls_zero))],
        "fgate": [1.0],
        "gate_le_one": [True],
    }, "ms.anchor5b")

    # 网格对拍
    rng = _Rng(83911)
    for case in range(120):
        h = rng.randint(1, 8)
        style = case % 4
        if style == 0:
            iraw = [rng.uniform(-3.0, 3.0) for _ in range(h)]
            fraw = [rng.uniform(-3.0, 3.0) for _ in range(h)]
        elif style == 1:
            iraw = [rng.uniform(-20.0, 20.0) for _ in range(h)]
            fraw = [rng.uniform(-20.0, 20.0) for _ in range(h)]
        elif style == 2:
            iraw = [rng.uniform(-60.0, 0.0) for _ in range(h)]
            fraw = [rng.uniform(-60.0, 0.0) for _ in range(h)]
        else:
            iraw = [rng.uniform(-3.0, 3.0) for _ in range(h)]
            fraw = [rng.uniform(-3.0, 3.0) for _ in range(h)]
        m_prev = [rng.uniform(-5.0, 5.0) for _ in range(h)]
        if case % 2 == 0:
            n_prev = [0.0] * h
        elif case % 4 == 1:
            n_prev = [0.0 if i % 2 == 0 else rng.uniform(0.1, 3.0) for i in range(h)]
        else:
            n_prev = [rng.uniform(0.0, 3.0) for _ in range(h)]
        got = solve("max_stabilizer", iraw=iraw, fraw=fraw,
                    m_prev=m_prev, n_prev=n_prev)
        exp = oracle_max_stabilizer(iraw, fraw, m_prev, n_prev)
        _cmp(got, exp, "ms.grid%d" % case)
    print("max_stabilizer OK")


def check_pointwise_step(solve):
    # 手算锚点 1：全零 n -> torch.all 分支；h=1
    # wx=[1.0, 0.0, 0.0, 0.0], ry=[0.0]*4, bias=[0.0]*4
    # raw = [1,0,0,0] -> iraw=1, fraw=0, zraw=0, oraw=0
    # states: y=0, c=2.0, n=0.0, m=0.0
    # logfplusm = 0 + logsigmoid(0) = -log2
    # zero_n_branch=True -> m_new = iraw = 1.0
    # igate = min(exp(1-1),1) = 1 ; fgate = min(exp(-log2-1),1) = exp(-log2-1)
    # ogate = sigmoid(0) = 0.5 ; zval = tanh(0) = 0
    # c_new = fgate*2 + 1*0 = 2*exp(-log2-1)
    # n_new = fgate*0 + 1 = 1
    # y_new = 0.5 * c_new / 1
    ls0 = -math.log(2.0)
    fg = math.exp(ls0 - 1.0)
    r = solve("pointwise_step", wx=[1.0, 0.0, 0.0, 0.0],
              ry=[0.0, 0.0, 0.0, 0.0], bias=[0.0, 0.0, 0.0, 0.0],
              states=[[0.0], [2.0], [0.0], [0.0]], head_dim=1)
    exp = {
        "raw_gates": [[1.0], [0.0], [0.0], [0.0]],
        "logfplusm": [ls0],
        "m_new": [1.0],
        "zero_n_branch": True,
        "igate": [1.0],
        "fgate": [fg],
        "ogate": [0.5],
        "zval": [0.0],
        "c_new": [2.0 * fg],
        "n_new": [1.0],
        "y_new": [0.5 * 2.0 * fg],
    }
    _cmp(r, exp, "pw.anchor1")

    # 手算锚点 2：n 非零 -> max 分支；且 ogate/zval 参与 c_new
    # wx=[0.0, 0.0, 2.0, 0.0], ry=[0]*4, bias=[0]*4
    # raw=[0,0,2,0] -> iraw=0, fraw=0, zraw=2, oraw=0
    # states: c=1.0, n=2.0, m=0.0
    # logfplusm = logsigmoid(0) = -log2
    # m_new = max(0, -log2) = 0
    # igate = min(exp(0),1) = 1 ; fgate = min(exp(-log2),1) = 0.5
    # ogate = sigmoid(0) = 0.5 ; zval = tanh(2)
    # c_new = 0.5*1 + 1*tanh(2)
    # n_new = 0.5*2 + 1 = 2
    # y_new = 0.5 * c_new / 2
    t2 = math.tanh(2.0)
    cnew = 0.5 * 1.0 + t2
    r = solve("pointwise_step", wx=[0.0, 0.0, 2.0, 0.0],
              ry=[0.0, 0.0, 0.0, 0.0], bias=[0.0, 0.0, 0.0, 0.0],
              states=[[0.0], [1.0], [2.0], [0.0]], head_dim=1)
    exp = {
        "raw_gates": [[0.0], [0.0], [2.0], [0.0]],
        "logfplusm": [ls0],
        "m_new": [0.0],
        "zero_n_branch": False,
        "igate": [1.0],
        "fgate": [0.5],
        "ogate": [0.5],
        "zval": [t2],
        "c_new": [cnew],
        "n_new": [2.0],
        "y_new": [0.5 * cnew / 2.0],
    }
    _cmp(r, exp, "pw.anchor2")

    # 手算锚点 3：部分 n 为零时整帧仍走 max 分支（torch.all 语义，不是逐元素）
    # h=2, n_prev=[0.0, 1.0] -> torch.all 为 False -> 整帧 max 分支
    # wx=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 其余 0
    # raw 全 0 -> iraw=0, fraw=0
    # logfplusm = [-log2, -log2]
    # m_new = max(0, -log2) = [0, 0]
    # igate = [1,1] ; fgate = [0.5, 0.5]
    # c_prev = [1.0, 1.0] -> c_new = 0.5*1 + 1*0 = [0.5, 0.5]
    # n_prev = [0.0, 1.0] -> n_new = 0.5*n + 1 = [1.0, 1.5]
    # ogate = [0.5, 0.5] ; zval = [0,0]
    # y_new = [0.5*0.5/1.0, 0.5*0.5/1.5] = [0.25, 1/6]
    r = solve("pointwise_step", wx=[0.0] * 8, ry=[0.0] * 8, bias=[0.0] * 8,
              states=[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], head_dim=2)
    exp = {
        "raw_gates": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "logfplusm": [ls0, ls0],
        "m_new": [0.0, 0.0],
        "zero_n_branch": False,
        "igate": [1.0, 1.0],
        "fgate": [0.5, 0.5],
        "ogate": [0.5, 0.5],
        "zval": [0.0, 0.0],
        "c_new": [0.5, 0.5],
        "n_new": [1.0, 1.5],
        "y_new": [0.25, 0.5 * 0.5 / 1.5],
    }
    _cmp(r, exp, "pw.anchor3")

    # 手算锚点 4：y_new 的分母是 n_new 而不是 c_new
    # wx=[0,0,0,0]（全 0）, states c=1.0, n=3.0, m=0.0
    # iraw=0,fraw=0 -> logfplusm=-log2 ; n 非零 -> m_new = 0
    # igate=1, fgate=0.5 ; ogate=0.5 ; zval=0
    # c_new = 0.5*1 + 0 = 0.5 ; n_new = 0.5*3 + 1 = 2.5
    # y_new = 0.5*0.5/2.5 = 0.1（若误用 c_new 当分母会得到 0.5）
    r = solve("pointwise_step", wx=[0.0, 0.0, 0.0, 0.0], ry=[0.0] * 4,
              bias=[0.0] * 4,
              states=[[0.0], [1.0], [3.0], [0.0]], head_dim=1)
    exp = {
        "raw_gates": [[0.0], [0.0], [0.0], [0.0]],
        "logfplusm": [ls0],
        "m_new": [0.0],
        "zero_n_branch": False,
        "igate": [1.0],
        "fgate": [0.5],
        "ogate": [0.5],
        "zval": [0.0],
        "c_new": [0.5],
        "n_new": [2.5],
        "y_new": [0.1],
    }
    _cmp(r, exp, "pw.anchor4")

    # 手算锚点 5：tanh 只作用在 zraw 上（不是 c_new 上）
    # wx z 门给 1.0, c_prev=0.0, n=1.0, m=0.0
    # raw=[0,0,1,0] -> iraw=0, fraw=0, zraw=1, oraw=0
    # m_new=0, igate=1, fgate=0.5, ogate=0.5, zval=tanh(1)
    # c_new = 0.5*0 + 1*tanh(1) = tanh(1)   （若误用 tanh(c_new) 会得 0）
    # n_new = 0.5*1 + 1 = 1.5
    # y_new = 0.5*tanh(1)/1.5
    t1 = math.tanh(1.0)
    r = solve("pointwise_step", wx=[0.0, 0.0, 1.0, 0.0], ry=[0.0] * 4,
              bias=[0.0] * 4,
              states=[[0.0], [0.0], [1.0], [0.0]], head_dim=1)
    exp = {
        "raw_gates": [[0.0], [0.0], [1.0], [0.0]],
        "logfplusm": [ls0],
        "m_new": [0.0],
        "zero_n_branch": False,
        "igate": [1.0],
        "fgate": [0.5],
        "ogate": [0.5],
        "zval": [t1],
        "c_new": [t1],
        "n_new": [1.5],
        "y_new": [0.5 * t1 / 1.5],
    }
    _cmp(r, exp, "pw.anchor5")

    # 网格对拍
    rng = _Rng(83912)
    for case in range(120):
        h = rng.randint(1, 6)
        style = case % 4
        if style == 0:
            wx = [rng.uniform(-2.0, 2.0) for _ in range(4 * h)]
            ry = [rng.uniform(-2.0, 2.0) for _ in range(4 * h)]
            bias = [rng.uniform(-1.0, 1.0) for _ in range(4 * h)]
        elif style == 1:
            wx = [rng.uniform(-1.0, 1.0) for _ in range(4 * h)]
            ry = [rng.uniform(-1.0, 1.0) for _ in range(4 * h)]
            bias = [0.0] * (4 * h)
        elif style == 2:
            wx = [rng.uniform(-30.0, -5.0) for _ in range(4 * h)]
            ry = [rng.uniform(-30.0, -5.0) for _ in range(4 * h)]
            bias = [rng.uniform(-5.0, 5.0) for _ in range(4 * h)]
        else:
            wx = [rng.uniform(-8.0, 8.0) for _ in range(4 * h)]
            ry = [rng.uniform(-8.0, 8.0) for _ in range(4 * h)]
            bias = [rng.uniform(-3.0, 3.0) for _ in range(4 * h)]
        if case % 3 == 0:
            n_prev = [0.0] * h
        elif case % 3 == 1:
            n_prev = [0.0 if i % 2 == 0 else rng.uniform(0.1, 3.0) for i in range(h)]
        else:
            n_prev = [rng.uniform(0.0, 3.0) for _ in range(h)]
        c_prev = [rng.uniform(-2.0, 2.0) for _ in range(h)]
        m_prev = [rng.uniform(-3.0, 3.0) for _ in range(h)]
        y_prev = [rng.uniform(-2.0, 2.0) for _ in range(h)]
        states = [y_prev, c_prev, n_prev, m_prev]
        got = solve("pointwise_step", wx=wx, ry=ry, bias=bias,
                    states=states, head_dim=h)
        exp = oracle_pointwise(wx, ry, bias, states, h)
        _cmp(got, exp, "pw.grid%d" % case)
    print("pointwise_step OK")


def check_bias_powerlaw(solve):
    # 手算锚点 1：powerlaw，head_dim=4, block_idx=0, num_blocks=1
    # ratio = 0.0（num_blocks==1）, exponent = 0.3
    # base = [0, 1/3, 2/3, 1]
    # bias = -( -5 + 12 * base^0.3 )
    exp_bias = [-( -5.0 + 12.0 * (d / 3.0) ** 0.3) for d in range(4)]
    r = solve("bias_powerlaw", head_dim=4, block_idx=0, num_blocks=1)
    _cmp(r, {"bias": exp_bias, "exponent": 0.3, "ratio": 0.0}, "bias.anchor1")

    # 手算锚点 2：num_blocks=3, block_idx=2 -> ratio = 1.0, exponent = 1.6
    exp_bias = [-( -5.0 + 12.0 * (d / 3.0) ** 1.6) for d in range(4)]
    r = solve("bias_powerlaw", head_dim=4, block_idx=2, num_blocks=3)
    _cmp(r, {"bias": exp_bias, "exponent": 1.6, "ratio": 1.0}, "bias.anchor2")

    # 手算锚点 3：num_blocks=4, block_idx=1 -> ratio = 1/3, exponent = 0.3+1.3/3
    ratio = 1.0 / 3.0
    exponent = 0.3 + 1.3 * ratio
    exp_bias = [-( -5.0 + 12.0 * (d / 2.0) ** exponent) for d in range(3)]
    r = solve("bias_powerlaw", head_dim=3, block_idx=1, num_blocks=4)
    _cmp(r, {"bias": exp_bias, "exponent": exponent, "ratio": ratio}, "bias.anchor3")

    # 手算锚点 4：负号在括号外 —— base=0 时 bias = -( -5 + 0 ) = 5
    r = solve("bias_powerlaw", head_dim=2, block_idx=0, num_blocks=1)
    exp_bias = [5.0, -( -5.0 + 12.0 * 1.0 ** 0.3)]
    _cmp(r, {"bias": exp_bias, "exponent": 0.3, "ratio": 0.0}, "bias.anchor4")

    # 手算锚点 5：head_dim == 1 时 0/0 -> NaN
    r = solve("bias_powerlaw", head_dim=1, block_idx=0, num_blocks=1)
    _cmp(r, {"bias": [NAN], "exponent": 0.3, "ratio": 0.0}, "bias.anchor5")

    # 手算锚点 6：small_init -> linspace(3.0, 6.0, head_dim)
    r = solve("bias_powerlaw", head_dim=4, block_idx=0, num_blocks=1,
              bias_mode="small_init")
    _cmp(r, {"bias": [3.0, 4.0, 5.0, 6.0], "exponent": 0.0, "ratio": 0.0},
         "bias.anchor6")

    # 手算锚点 7：small_init 且 head_dim == 1 -> linspace(3,6,1) = [3.0]
    r = solve("bias_powerlaw", head_dim=1, block_idx=0, num_blocks=1,
              bias_mode="small_init")
    _cmp(r, {"bias": [3.0], "exponent": 0.0, "ratio": 0.0}, "bias.anchor7")

    # 网格对拍：全部 head_dim / num_blocks / block_idx 组合
    for head_dim in range(1, 9):
        for num_blocks in range(1, 6):
            for block_idx in range(num_blocks):
                for mode in ("powerlaw", "small_init"):
                    got = solve("bias_powerlaw", head_dim=head_dim,
                                block_idx=block_idx, num_blocks=num_blocks,
                                bias_mode=mode)
                    exp = oracle_bias(head_dim, block_idx, num_blocks, mode)
                    _cmp(got, exp, "bias.grid[%d,%d,%d,%s]" % (
                        head_dim, num_blocks, block_idx, mode))
    print("bias_powerlaw OK")
