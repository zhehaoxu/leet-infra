"""order 843 判题用例（暂存）：Captum LRP 的 z-rule relevance 传播账。

独立 oracle（不 import solution），每个 check_* 含手算锚点 + 网格对拍。

口径：
  - 所有输入量都是整数；eps 取精确有理数 1/10^9（官方是 float64 的 1e-9）。
  - 全链路精确有理数比较（fractions.Fraction），输出转 float 后按相对容差 1e-12 比对。
  - 官方 LRP 走两次前向：第一次原权重前向存激活，第二次「改写权重 × 原激活」。
  - sign(0) 强制为 +1；delta = output - sum(attr)（output 在前）。
"""

from fractions import Fraction

EPS = Fraction(1, 10 ** 9)


# ------------------------------------------------------------------ 独立 oracle
def _f(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    return Fraction(str(x))


def _sign(v):
    """官方 `torch.sign` 后 `sign[sign == 0] = 1`。"""
    if v == 0:
        return Fraction(1)
    return Fraction(1) if v > 0 else Fraction(-1)


def oracle_rewrite(rule, weights, bias, gamma, set_bias_to_zero):
    """官方 `_manipulate_weights` 的精确语义。"""
    W = [[_f(v) for v in row] for row in weights]
    B = [_f(v) for v in bias]
    if rule == "gamma":
        g = _f(gamma)
        W = [[W[i][j] + g * (W[i][j] if W[i][j] > 0 else Fraction(0))
              for j in range(len(W[i]))] for i in range(len(W))]
    elif rule == "alpha1_beta0":
        W = [[W[i][j] if W[i][j] > 0 else Fraction(0) for j in range(len(W[i]))]
             for i in range(len(W))]
    if set_bias_to_zero:
        B = [Fraction(0)] * len(B)
    return W, B


def oracle_stack(layers, inputs, target):
    """官方 LRP 的 z-rule 传播账（精确有理数）。

    返回的 activations / outputs / relevance_out / relevance_in 都按**前向顺序**
    （下标 0 = 靠输入层的第一层）。
    """
    parsed = []
    for lay in layers:
        W = [[_f(v) for v in row] for row in lay["weights"]]
        B = [_f(v) for v in lay["bias"]]
        parsed.append((W, B, lay.get("rule", "epsilon"),
                       lay.get("gamma", Fraction(1, 4)),
                       bool(lay.get("set_bias_to_zero", False))))

    # 第一次前向：原权重链式
    acts = [[_f(v) for v in inputs]]
    for (W, B, _r, _g, _s) in parsed:
        prev = acts[-1]
        acts.append([sum(W[i][j] * prev[j] for j in range(len(prev))) + B[i]
                     for i in range(len(W))])

    rw = [oracle_rewrite(r, W, B, g, s) for (W, B, r, g, s) in parsed]

    # 第二次前向：改写权重 × 对应层的原激活
    outs = []
    for i in range(len(parsed)):
        Wp, Bp = rw[i]
        a = acts[i]
        outs.append([sum(Wp[i2][j] * a[j] for j in range(len(a))) + Bp[i2]
                     for i2 in range(len(Wp))])

    # 反向
    rel_outs, rel_ins = [], []
    upstream = [Fraction(0)] * len(acts[-1])
    upstream[target] = Fraction(1)
    for i in range(len(parsed) - 1, -1, -1):
        Wp, _Bp = rw[i]
        y = outs[i]
        ro = [upstream[k] / (y[k] + _sign(y[k]) * EPS) for k in range(len(y))]
        a = acts[i]
        tmp = [sum(ro[i2] * Wp[i2][j] for i2 in range(len(Wp))) for j in range(len(a))]
        ri = [tmp[j] * a[j] for j in range(len(a))]
        rel_outs.append(ro)
        rel_ins.append(ri)
        upstream = ri
    rel_outs.reverse()
    rel_ins.reverse()

    out_target = acts[-1][target]
    return {
        "activations": acts[1:],
        "outputs": outs,
        "relevance_out": rel_outs,
        "relevance_in": rel_ins,
        "attribution": [v * out_target for v in rel_ins[0]],
        "output_target": out_target,
    }


# ------------------------------------------------------------------ 断言工具
def _cmp(got, exp, tag):
    """逐位相等断言（有理数分支）。"""
    assert got == exp, "%s: got %r expected %r" % (tag, got, exp)


def _cmp_close(got, exp, tag, tol=1e-12):
    """相对容差断言（float 输出）。"""
    assert len(got) == len(exp), "%s: length %d != %d" % (tag, len(got), len(exp))
    for i, (g, e) in enumerate(zip(got, exp)):
        ef = float(e)
        if ef == 0.0:
            assert abs(g) <= tol, "%s[%d]: got %r expected %r" % (tag, i, g, ef)
        else:
            assert abs(g - ef) <= tol * abs(ef), \
                "%s[%d]: got %r expected %r (diff %g)" % (tag, i, g, ef, abs(g - ef))


def _cmp_close2d(got, exp, tag, tol=1e-12):
    """二维（按层）的相对容差断言。"""
    assert len(got) == len(exp), "%s: layers %d != %d" % (tag, len(got), len(exp))
    for i, (gr, er) in enumerate(zip(got, exp)):
        _cmp_close(gr, er, "%s[%d]" % (tag, i), tol)


def _cmp_scalar(got, exp, tag, tol=1e-12):
    """标量相对容差断言。"""
    if exp == 0:
        assert abs(got) <= tol, "%s: got %r expected %r" % (tag, got, exp)
    else:
        assert abs(got - exp) <= tol * abs(exp), \
            "%s: got %r expected %r (diff %g)" % (tag, got, exp, abs(got - exp))


# ------------------------------------------------------------------ 随机源
class _Rng:
    """确定性 LCG，保证判题可复现。"""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s

    def randint(self, lo, hi):
        """闭区间 [lo, hi]。"""
        span = hi - lo + 1
        return lo + (self.next() % span)

    def choice(self, seq):
        return seq[self.next() % len(seq)]


def _rand_matrix(rng, rows, cols, lo=-6, hi=6):
    return [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def _rand_vector(rng, n, lo=-6, hi=6):
    return [rng.randint(lo, hi) for _ in range(n)]


# ------------------------------------------------------------------ check 1
def check_rewrite_rule(solve):
    """官方 `_manipulate_weights` 的权重改写账。"""
    # ---- 手算锚点 1：gamma，权重含正/负/零
    # W = [[-3, 5, -7, 2], [1, -4, 3, 6], [-2, 8, -1, 4]], gamma = 1/4
    # 官方语义：W' = W + gamma * max(W, 0)
    # row0: -3 + 0 = -3 ; 5 + 5/4 = 25/4 ; -7 + 0 = -7 ; 2 + 1/2 = 5/2
    # row1: 1 + 1/4 = 5/4 ; -4 ; 3 + 3/4 = 15/4 ; 6 + 3/2 = 15/2
    # row2: -2 ; 8 + 2 = 10 ; -1 ; 4 + 1 = 5
    W = [[-3, 5, -7, 2], [1, -4, 3, 6], [-2, 8, -1, 4]]
    B = [-1, 2, -3]
    r = solve("rewrite_rule", rule="gamma", weights=W, bias=B, gamma="1/4")
    exp = [[-3.0, 6.25, -7.0, 2.5], [1.25, -4.0, 3.75, 7.5], [-2.0, 10.0, -1.0, 5.0]]
    _cmp_close([v for row in r["weights"] for v in row],
               [v for row in exp for v in row], "rw.anchor1.w")
    _cmp_close(r["bias"], [-1.0, 2.0, -3.0], "rw.anchor1.b")

    # ---- 手算锚点 2：alpha1_beta0 —— 负权重清零，正权重保留
    r = solve("rewrite_rule", rule="alpha1_beta0", weights=W, bias=B)
    exp = [[0.0, 5.0, 0.0, 2.0], [1.0, 0.0, 3.0, 6.0], [0.0, 8.0, 0.0, 4.0]]
    _cmp_close([v for row in r["weights"] for v in row],
               [v for row in exp for v in row], "rw.anchor2.w")

    # ---- 手算锚点 3：epsilon —— 权重与偏置原样（官方 `_manipulate_weights` 是 pass）
    r = solve("rewrite_rule", rule="epsilon", weights=W, bias=B)
    _cmp_close([v for row in r["weights"] for v in row],
               [float(v) for row in W for v in row], "rw.anchor3.w")
    _cmp_close(r["bias"], [float(v) for v in B], "rw.anchor3.b")

    # ---- 手算锚点 4：set_bias_to_zero —— 偏置全 0，权重照常改写
    r = solve("rewrite_rule", rule="gamma", weights=W, bias=B, gamma="1/4",
              set_bias_to_zero=True)
    _cmp_close(r["bias"], [0.0, 0.0, 0.0], "rw.anchor4.b")
    _cmp_close([v for row in r["weights"] for v in row],
               [v for row in exp for v in row] if False else
               [-3.0, 6.25, -7.0, 2.5, 1.25, -4.0, 3.75, 7.5, -2.0, 10.0, -1.0, 5.0],
               "rw.anchor4.w")

    # ---- 手算锚点 5：gamma = 0 时等价于不改权重（但**不等于** epsilon 的 bias 处理）
    r = solve("rewrite_rule", rule="gamma", weights=W, bias=B, gamma="0")
    _cmp_close([v for row in r["weights"] for v in row],
               [float(v) for row in W for v in row], "rw.anchor5.w")

    # ---- 网格对拍：三种 rule × 多个 gamma × 两种 bias 开关
    rng = _Rng(8431)
    for trial in range(40):
        rows = rng.randint(1, 4)
        cols = rng.randint(1, 4)
        Wg = _rand_matrix(rng, rows, cols)
        Bg = _rand_vector(rng, rows)
        for rule in ("epsilon", "gamma", "alpha1_beta0"):
            for gamma in ("0", "1/4", "1/2", "3/4", "1", "1/10"):
                for sbz in (False, True):
                    got = solve("rewrite_rule", rule=rule, weights=Wg, bias=Bg,
                                gamma=gamma, set_bias_to_zero=sbz)
                    Wf, Bf = oracle_rewrite(rule, Wg, Bg, Fraction(gamma), sbz)
                    tag = "rw.g[%d.%s.%s.%s]" % (trial, rule, gamma, sbz)
                    _cmp_close([v for row in got["weights"] for v in row],
                               [v for row in Wf for v in row], tag + ".w")
                    _cmp_close(got["bias"], Bf, tag + ".b")

    # ---- 非法 rule
    try:
        solve("rewrite_rule", rule="identity", weights=W, bias=B)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown rule")


# ------------------------------------------------------------------ check 2
def check_z_rule_layer(solve):
    """单层线性层的 z-rule relevance 分解。"""
    # ---- 手算锚点 1：W=[[1,2],[3,4]], b=[1,-1], inputs=[2,3], target=0
    # outputs = [1*2+2*3+1, 3*2+4*3-1] = [9, 17]
    # sign(9)=+1, sign(17)=+1
    # grad = one-hot(0) = [1, 0]
    # relevance_out = [1/(9 + 1e-9), 0]
    # relevance_in = (rel_out @ W) * inputs
    #   rel_out @ W = [1*1+0*3, 1*2+0*4] = [1, 2]
    #   * inputs = [2, 6]
    # attribution = relevance_in * outputs[target] = [18, 54]
    r = solve("z_rule_layer", weights=[[1, 2], [3, 4]], bias=[1, -1],
              inputs=[2, 3], target=0)
    _cmp_close(r["outputs"], [9.0, 17.0], "zl.anchor1.outputs")
    _cmp_close(r["relevance_out"], [float(Fraction(1) / (9 + EPS)), 0.0],
               "zl.anchor1.rel_out")
    _cmp_close(r["relevance_in"], [2.0 / (9 + EPS), 6.0 / (9 + EPS)], "zl.anchor1.rel_in")
    _cmp_close(r["attribution"], [18.0 / (9 + EPS), 54.0 / (9 + EPS)], "zl.anchor1.attr")

    # ---- 手算锚点 2：**sign(0) = +1 的官方 quirk**
    # W=[[1,1]], b=[-2], inputs=[1,1] -> outputs = [1+1-2] = [0]
    # sign(0) 强制为 +1 -> 分母 = 0 + 1e-9
    # grad = one-hot(0) = [1] -> relevance_out = [1/(0+1e-9)] = [1e9]
    # relevance_in = (rel_out @ W) * inputs = [1e9 * 1 * 1] = [1e9]
    # attribution = relevance_in * outputs[target] = 1e9 * 0 = 0
    r = solve("z_rule_layer", weights=[[1, 1]], bias=[-2], inputs=[1, 1], target=0)
    _cmp_close(r["outputs"], [0.0], "zl.anchor2.outputs")
    _cmp_close(r["relevance_out"], [float(Fraction(1) / EPS)], "zl.anchor2.rel_out")
    # relevance_in = (rel_out @ W) * inputs = ([1e9*1] + [1e9*1]) * [1, 1]
    _cmp_close(r["relevance_in"], [float(Fraction(1) / EPS)] * 2, "zl.anchor2.rel_in")
    _cmp_close(r["attribution"], [0.0, 0.0], "zl.anchor2.attr")

    # ---- 手算锚点 3：负输出 -> 分母是 outputs - eps（sign = -1）
    # W=[[1]], b=[-3], inputs=[1] -> outputs = [-2]
    # sign(-2) = -1 -> 分母 = -2 - 1e-9
    # relevance_out = 1/(-2 - eps)
    # relevance_in = rel_out * 1 * 1
    # attribution = relevance_in * outputs[target] = rel_out * (-2)
    r = solve("z_rule_layer", weights=[[1]], bias=[-3], inputs=[1], target=0)
    den = Fraction(-2) - EPS
    _cmp_close(r["outputs"], [-2.0], "zl.anchor3.outputs")
    _cmp_close(r["relevance_out"], [float(Fraction(1) / den)], "zl.anchor3.rel_out")
    _cmp_close(r["attribution"], [float(Fraction(1) / den * Fraction(-2))],
               "zl.anchor3.attr")

    # ---- 网格对拍
    rng = _Rng(8432)
    for trial in range(60):
        rows = rng.randint(1, 4)
        cols = rng.randint(1, 5)
        Wg = _rand_matrix(rng, rows, cols)
        Bg = _rand_vector(rng, rows)
        xg = _rand_vector(rng, cols)
        target = rng.randint(0, rows - 1)
        got = solve("z_rule_layer", weights=Wg, bias=Bg, inputs=xg, target=target)
        orc = oracle_stack([{"rule": "epsilon", "weights": Wg, "bias": Bg}], xg, target)
        tag = "zl.g[%d]" % trial
        _cmp_close(got["outputs"], orc["outputs"][0], tag + ".outputs")
        _cmp_close(got["relevance_out"], orc["relevance_out"][0], tag + ".rel_out")
        _cmp_close(got["relevance_in"], orc["relevance_in"][0], tag + ".rel_in")
        _cmp_close(got["attribution"], orc["attribution"], tag + ".attr")

    # ---- target 越界
    try:
        solve("z_rule_layer", weights=[[1, 2], [3, 4]], bias=[1, -1],
              inputs=[2, 3], target=5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range target")


# ------------------------------------------------------------------ check 3
def check_stack_relevance(solve):
    """多层栈的 z-rule 传播账（核心考点：官方的两次前向语义）。"""
    # ---- 手算锚点 1：两层，layer0 = gamma(1/4)，layer1 = epsilon
    # W0 = [[-3,5,-7,2],[1,-4,3,6],[-2,8,-1,4]], b0 = [-1,2,-3]
    # W1 = [[2,-1,3],[-4,5,1],[1,2,-3]],          b1 = [4,-2,1]
    # inputs = [1,2,0,-1], target = 1
    #
    # 第一次前向（原权重链式）：
    #   a1 = W0 @ x + b0 = [-3+10+0-2-1, 1-8+0-6+2, -2+16+0-4-3] = [4, -11, 7]
    #   a2 = W1 @ a1 + b1 = [8+11+21+4, -16-55+7-2, 4-22-21+1] = [44, -66, -38]
    #
    # 改写 layer0 权重（gamma=1/4，只放大正权重）：
    #   W0' = [[-3, 25/4, -7, 5/2], [5/4, -4, 15/4, 15/2], [-2, 10, -1, 5]]
    # layer1 是 epsilon，权重不变。
    #
    # 第二次前向（**改写权重 × 原激活**）：
    #   y1 = W0' @ x + b0 = [-3 + 25/2 - 5/2 - 1, 5/4 - 8 - 15/2 + 2, -2 + 20 - 5 - 3]
    #      = [6, -49/4, 10]
    #   y2 = W1 @ a1 + b1 = [44, -66, -38]     ← 注意用的是**原激活 a1**，不是 y1
    #
    # 反向：
    #   layer1: grad = one-hot(1) = [0,1,0]
    #     rel_out1 = [0, 1/(-66 - eps), 0]
    #     rel_in1 = (rel_out1 @ W1) * a1
    #            = [0*2+1*(-4), 0*(-1)+1*5, 0*3+1*1] * [4,-11,7] / (-66-eps)
    #            = [-4, 5, 1] * [4,-11,7] / (-66-eps)
    #            = [-16, -55, 7] / (-66-eps)
    #   layer0: upstream = rel_in1
    #     rel_out0 = rel_in1 / (y1 + sign(y1)*eps)
    #     rel_in0 = (rel_out0 @ W0') * x
    #   attribution = rel_in0 * a2[target] = rel_in0 * (-66)
    W0 = [[-3, 5, -7, 2], [1, -4, 3, 6], [-2, 8, -1, 4]]
    B0 = [-1, 2, -3]
    W1 = [[2, -1, 3], [-4, 5, 1], [1, 2, -3]]
    B1 = [4, -2, 1]
    layers = [{"rule": "gamma", "weights": W0, "bias": B0, "gamma": "1/4"},
              {"rule": "epsilon", "weights": W1, "bias": B1}]
    r = solve("stack_relevance", layers=layers, inputs=[1, 2, 0, -1], target=1)
    orc = oracle_stack(layers, [1, 2, 0, -1], 1)

    _cmp_close(r["activations"][0], [4.0, -11.0, 7.0], "sr.anchor1.act0")
    _cmp_close(r["activations"][1], [44.0, -66.0, -38.0], "sr.anchor1.act1")
    _cmp_close(r["outputs"][0], [6.0, -12.25, 10.0], "sr.anchor1.out0")
    _cmp_close(r["outputs"][1], [44.0, -66.0, -38.0], "sr.anchor1.out1")
    _cmp_close2d(r["relevance_out"], orc["relevance_out"], "sr.anchor1.rel_out")
    _cmp_close2d(r["relevance_in"], orc["relevance_in"], "sr.anchor1.rel_in")
    _cmp_close(r["attribution"], orc["attribution"], "sr.anchor1.attr")

    # ---- 手算锚点 2：**两次前向的关键差异**
    # 若误把第二次前向写成「改写权重链式」，layer1 的 outputs 会是
    #   W1 @ y1 + b1 = [2*6 + (-1)*(-49/4) + 3*10 + 4, ...] = [193/4? ...]
    # 而官方（正确）是 [44, -66, -38]。这里显式钉死正确值。
    _cmp(r["outputs"][1], [44.0, -66.0, -38.0], "sr.anchor2.twopass")
    # 错版（改写权重链式）：y1 = [6, -49/4, 10]
    #   row0: 2*6 + (-1)*(-49/4) + 3*10 + 4 = 12 + 49/4 + 30 + 4 = 233/4 = 58.25
    #   row1: -4*6 + 5*(-49/4) + 1*10 - 2 = -24 - 245/4 + 10 - 2 = -373/4
    #   row2: 1*6 + 2*(-49/4) + (-3)*10 + 1 = 6 - 49/2 - 30 + 1 = -115/2
    assert r["outputs"][1] != [58.25, -93.25, -57.5], \
        "second forward must use original activations, not rewritten chain"

    # ---- 手算锚点 3：单层栈等价于 z_rule_layer
    one = solve("stack_relevance",
                layers=[{"rule": "epsilon", "weights": W0, "bias": B0}],
                inputs=[1, 2, 0, -1], target=1)
    single = solve("z_rule_layer", weights=W0, bias=B0, inputs=[1, 2, 0, -1], target=1)
    _cmp_close(one["attribution"], single["attribution"], "sr.anchor3.attr")
    _cmp_close(one["outputs"][0], single["outputs"], "sr.anchor3.outputs")

    # ---- 手算锚点 4：set_bias_to_zero 只影响 bias，权重照常改写
    r4 = solve("stack_relevance",
               layers=[{"rule": "gamma", "weights": W0, "bias": B0, "gamma": "1/4",
                        "set_bias_to_zero": True}],
               inputs=[1, 2, 0, -1], target=1)
    orc4 = oracle_stack([{"rule": "gamma", "weights": W0, "bias": B0, "gamma": "1/4",
                           "set_bias_to_zero": True}], [1, 2, 0, -1], 1)
    _cmp_close(r4["outputs"][0], orc4["outputs"][0], "sr.anchor4.out0")
    _cmp_close(r4["attribution"], orc4["attribution"], "sr.anchor4.attr")
    # bias 置零后第一次前向的激活也变了（原权重链式里的 bias 被清零）
    _cmp_close(r4["activations"][0], orc4["activations"][0], "sr.anchor4.act0")

    # ---- 网格对拍：随机层数 × 随机 rule × 随机 gamma
    rng = _Rng(8433)
    for trial in range(50):
        n_layers = rng.randint(1, 3)
        width = rng.randint(2, 4)
        layers_g = []
        cur = width
        for li in range(n_layers):
            nxt = rng.randint(1, max(1, width))
            if li == n_layers - 1:
                nxt = rng.randint(1, 3)
            else:
                nxt = rng.randint(1, 4)
            Wg = _rand_matrix(rng, nxt, cur)
            Bg = _rand_vector(rng, nxt)
            rule = rng.choice(("epsilon", "gamma", "alpha1_beta0"))
            lay = {"rule": rule, "weights": Wg, "bias": Bg}
            if rule == "gamma":
                lay["gamma"] = rng.choice(("0", "1/4", "1/2", "1"))
            if rng.next() % 2 == 0:
                lay["set_bias_to_zero"] = True
            layers_g.append(lay)
            cur = nxt
        xg = _rand_vector(rng, width)
        target = rng.randint(0, len(layers_g[-1]["weights"]) - 1)
        got = solve("stack_relevance", layers=layers_g, inputs=xg, target=target)
        orc = oracle_stack(layers_g, xg, target)
        tag = "sr.g[%d]" % trial
        _cmp_close2d(got["activations"], orc["activations"], tag + ".act")
        _cmp_close2d(got["outputs"], orc["outputs"], tag + ".out")
        _cmp_close2d(got["relevance_out"], orc["relevance_out"], tag + ".rel_out")
        _cmp_close2d(got["relevance_in"], orc["relevance_in"], tag + ".rel_in")
        _cmp_close(got["attribution"], orc["attribution"], tag + ".attr")

    # ---- 非法 rule
    try:
        solve("stack_relevance", layers=[{"rule": "identity", "weights": W0, "bias": B0}],
              inputs=[1, 2, 0, -1], target=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown rule")

    # ---- 空层栈必须报错（不是静默返回空结果）
    try:
        solve("stack_relevance", layers=[], inputs=[1, 2, 0, -1], target=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty layers")

    # ---- target 越界必须报错（不是静默取模）
    try:
        solve("stack_relevance", layers=[{"rule": "epsilon", "weights": W0, "bias": B0}],
              inputs=[1, 2, 0, -1], target=9)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range target")


# ------------------------------------------------------------------ check 4
def check_completeness(solve):
    """完备性 delta：官方 LRP 的符号是 output 在前。"""
    # ---- 手算锚点 1：attributions=[1,2,3], output=10
    # attr_sum = 6 ; delta = 10 - 6 = 4
    r = solve("completeness", attributions=[1, 2, 3], output=10)
    _cmp_scalar(r["attr_sum"], 6.0, "cp.anchor1.attr_sum")
    _cmp_scalar(r["delta"], 4.0, "cp.anchor1.delta")

    # ---- 手算锚点 2：负数，delta 仍为 output - attr_sum
    r = solve("completeness", attributions=[-4, 6], output=-2)
    _cmp_scalar(r["attr_sum"], 2.0, "cp.anchor2.attr_sum")
    _cmp_scalar(r["delta"], -4.0, "cp.anchor2.delta")

    # ---- 手算锚点 3：**符号与 IG 相反**
    # IG 的 delta = attr_sum - (end_out - start_out)；LRP 是 output - attr_sum。
    # 这里 attr_sum=6, output=10：IG 式会得到 6 - 10 = -4，LRP 式是 10 - 6 = +4。
    r = solve("completeness", attributions=[1, 2, 3], output=10)
    assert r["delta"] > 0, "LRP delta must be output - attr_sum (output first)"
    _cmp(r["delta"], 4.0, "cp.anchor3.delta")

    # ---- 手算锚点 4：全零
    r = solve("completeness", attributions=[0, 0], output=0)
    _cmp_scalar(r["attr_sum"], 0.0, "cp.anchor4.attr_sum")
    _cmp_scalar(r["delta"], 0.0, "cp.anchor4.delta")

    # ---- 网格对拍
    rng = _Rng(8434)
    for trial in range(40):
        n = rng.randint(1, 6)
        attrs = _rand_vector(rng, n, -9, 9)
        out = rng.randint(-20, 20)
        got = solve("completeness", attributions=attrs, output=out)
        exp_sum = float(sum(attrs))
        exp_delta = float(out - sum(attrs))
        tag = "cp.g[%d]" % trial
        _cmp_close([got["attr_sum"]], [exp_sum], tag + ".attr_sum")
        _cmp_close([got["delta"]], [exp_delta], tag + ".delta")
