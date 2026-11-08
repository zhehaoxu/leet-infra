"""order 842 判题用例（暂存）：Captum Integrated Gradients 求积与归因账。

独立 oracle（不 import solution），每个 check_* 含手算锚点 + 网格对拍。

口径：
  * Riemann 四变体（left/right/middle/trapezoid）一律**精确有理数**：
    step_sizes 与 alphas 用 fractions.Fraction 表达，输出转 float 后
    **逐位相等**判定。
  * Gauss-Legendre 含无理数根，判定用相对容差 1e-12。
  * 布局：scaled_features 与 grads 都是官方 cat 后的 **step-major**
    （第 s*B + b 行属于第 s 个 alpha 的第 b 个样本）。
  * delta 符号顺序：attr_sum - (end_out_sum - start_out_sum)。
"""

import math
from fractions import Fraction


# ------------------------------------------------------------------ 独立 oracle
def _linspace_frac(start, end, n):
    """torch.linspace 的精确有理等价（官方四变体都走它）。

    torch.linspace(start, end, n) 的第 i 个点 = start + i*(end-start)/(n-1)。
    """
    if n == 1:
        return [Fraction(start)]
    step = (Fraction(end) - Fraction(start)) / (n - 1)
    return [Fraction(start) + i * step for i in range(n)]


def oracle_riemann_params(method, n):
    """官方 riemann_builders 的精确有理口径。

    返回 (step_sizes, alphas)，元素都是 Fraction。
    """
    # step_sizes：官方 deltas = [1/n]*n；trapezoid 时首尾再 /2
    deltas = [Fraction(1, n)] * n
    if method == "riemann_trapezoid":
        deltas[0] = deltas[0] / 2
        deltas[-1] = deltas[-1] / 2
    # alphas：官方四种 linspace 端点约定
    if method == "riemann_trapezoid":
        alphas = _linspace_frac(0, 1, n)
    elif method == "riemann_left":
        alphas = _linspace_frac(0, Fraction(1) - Fraction(1, n), n)
    elif method == "riemann_middle":
        alphas = _linspace_frac(Fraction(1, 2 * n),
                                Fraction(1) - Fraction(1, 2 * n), n)
    elif method == "riemann_right":
        alphas = _linspace_frac(Fraction(1, n), 1, n)
    else:
        raise AssertionError("bad method %r" % method)
    return deltas, alphas


# Gauss-Legendre 官方 numpy 真值（captum 的 gauss_legendre_builders 直接调 leggauss）。
# 键：n -> (nodes 升序, weights 升序对应)
GL_TABLE = {
    1: (
        (0.0,),
        (2.0,),
    ),
    2: (
        (-0.5773502691896257, 0.5773502691896257),
        (1.0, 1.0),
    ),
    3: (
        (-0.7745966692414834, 0.0, 0.7745966692414834),
        (0.5555555555555555, 0.888888888888889, 0.5555555555555555),
    ),
    4: (
        (-0.8611363115940526, -0.33998104358485626, 0.33998104358485626, 0.8611363115940526),
        (0.34785484513745346, 0.6521451548625465, 0.6521451548625465, 0.34785484513745346),
    ),
    5: (
        (-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664),
        (0.2369268850561894, 0.47862867049936625, 0.5688888888888886, 0.47862867049936625, 0.2369268850561894),
    ),
    6: (
        (-0.932469514203152, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.932469514203152),
        (0.17132449237917038, 0.3607615730481387, 0.46791393457269104, 0.46791393457269104, 0.3607615730481387, 0.17132449237917038),
    ),
    7: (
        (-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585),
        (0.12948496616887065, 0.2797053914892766, 0.3818300505051185, 0.41795918367346885, 0.3818300505051185, 0.2797053914892766, 0.12948496616887065),
    ),
    8: (
        (-0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564984, 0.18343464249564984, 0.525532409916329, 0.7966664774136267, 0.9602898564975362),
        (0.10122853629037652, 0.22238103445337445, 0.31370664587788716, 0.3626837833783618, 0.3626837833783618, 0.31370664587788716, 0.22238103445337445, 0.10122853629037652),
    ),
    9: (
        (-0.9681602395076261, -0.8360311073266358, -0.6133714327005904, -0.3242534234038089, 0.0, 0.3242534234038089, 0.6133714327005904, 0.8360311073266358, 0.9681602395076261),
        (0.08127438836157368, 0.18064816069485742, 0.26061069640293555, 0.3123470770400032, 0.3302393550012601, 0.3123470770400032, 0.26061069640293555, 0.18064816069485742, 0.08127438836157368),
    ),
    10: (
        (-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.43339539412924716, -0.14887433898163122, 0.14887433898163122, 0.43339539412924716, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717),
        (0.06667134430868729, 0.1494513491505805, 0.21908636251598226, 0.26926671930999685, 0.29552422471475304, 0.29552422471475304, 0.26926671930999685, 0.21908636251598226, 0.1494513491505805, 0.06667134430868729),
    ),
    11: (
        (-0.978228658146057, -0.8870625997680953, -0.7301520055740494, -0.5190961292068118, -0.26954315595234496, 0.0, 0.26954315595234496, 0.5190961292068118, 0.7301520055740494, 0.8870625997680953, 0.978228658146057),
        (0.05566856711617405, 0.12558036946490395, 0.18629021092773437, 0.23319376459199057, 0.26280454451024676, 0.27292508677790067, 0.26280454451024676, 0.23319376459199057, 0.18629021092773437, 0.12558036946490395, 0.05566856711617405),
    ),
    12: (
        (-0.9815606342467192, -0.9041172563704748, -0.7699026741943047, -0.5873179542866175, -0.3678314989981802, -0.12523340851146894, 0.12523340851146894, 0.3678314989981802, 0.5873179542866175, 0.7699026741943047, 0.9041172563704748, 0.9815606342467192),
        (0.04717533638651128, 0.10693932599531876, 0.16007832854334636, 0.2031674267230659, 0.2334925365383549, 0.24914704581340297, 0.24914704581340297, 0.2334925365383549, 0.2031674267230659, 0.16007832854334636, 0.10693932599531876, 0.04717533638651128),
    ),
}


def oracle_gl_params(n):
    """官方 gauss_legendre_builders：step_sizes = 0.5*weights，alphas = 0.5*(1+nodes)。"""
    nodes, weights = GL_TABLE[n]
    step_sizes = [0.5 * w for w in weights]
    alphas = [0.5 * (1.0 + x) for x in nodes]
    return step_sizes, alphas


def oracle_params(method, n):
    if method == "gausslegendre":
        return oracle_gl_params(n)
    return oracle_riemann_params(method, n)


def _is_gl(method):
    return method == "gausslegendre"


def oracle_scaled_features(input_, baseline, method, n):
    """官方 scaled_features：cat([baseline + alpha*(input-baseline) for alpha in alphas], dim=0)。

    布局 step-major：第 s*B + b 行。
    """
    step_sizes, alphas = oracle_params(method, n)
    bsz = len(input_)
    feat = len(input_[0])
    out = []
    for s in range(n):
        alpha = alphas[s]
        for b in range(bsz):
            row = []
            for f in range(feat):
                if _is_gl(method):
                    row.append(baseline[b][f] + alpha * (input_[b][f] - baseline[b][f]))
                else:
                    row.append(float(Fraction(baseline[b][f])
                                     + alpha * (Fraction(input_[b][f]) - Fraction(baseline[b][f]))))
            out.append(row)
    return out


def oracle_ig_aggregate(input_, baseline, grads, method, n, multiply_by_inputs=True):
    """官方 _attribute 的聚合段。

    scaled_grads = grad.view(S, -1) * step_sizes.view(S, 1)
    total_grads  = reshape((S, B) + (F,)).sum(dim=0)
    attributions = total_grads * (input - baseline)
    """
    step_sizes, alphas = oracle_params(method, n)
    bsz = len(input_)
    feat = len(input_[0])
    gl = _is_gl(method)

    scaled = []
    for s in range(n):
        w = step_sizes[s]
        for b in range(bsz):
            row = grads[s * bsz + b]
            if gl:
                scaled.append([row[f] * w for f in range(feat)])
            else:
                scaled.append([Fraction(row[f]) * w for f in range(feat)])

    total = []
    for b in range(bsz):
        row = []
        for f in range(feat):
            acc = 0
            for s in range(n):
                acc = acc + scaled[s * bsz + b][f]
            row.append(acc)
        total.append(row)

    if multiply_by_inputs:
        attributions = []
        for b in range(bsz):
            row = []
            for f in range(feat):
                if gl:
                    row.append(total[b][f] * (input_[b][f] - baseline[b][f]))
                else:
                    row.append(total[b][f] * (Fraction(input_[b][f]) - Fraction(baseline[b][f])))
            attributions.append(row)
    else:
        attributions = [list(r) for r in total]

    def conv(mat):
        if gl:
            return [[float(v) for v in r] for r in mat]
        return [[float(v) for v in r] for r in mat]

    return {"total_grads": conv(total), "attributions": conv(attributions)}


def oracle_ig_completeness(attributions, start_out, end_out):
    """官方 compute_convergence_delta。

    attr_sum[b] = Σ_f attributions[b][f]
    delta[b]    = attr_sum[b] - (end_out_sum[b] - start_out_sum[b])
    """
    bsz = len(attributions)
    attr_sum = [sum(Fraction(v) for v in attributions[b]) for b in range(bsz)]
    end_out_sum = [sum(Fraction(v) for v in end_out[b]) for b in range(bsz)]
    start_out_sum = [sum(Fraction(v) for v in start_out[b]) for b in range(bsz)]
    delta = [attr_sum[b] - (end_out_sum[b] - start_out_sum[b]) for b in range(bsz)]
    return {
        "attr_sum": [float(v) for v in attr_sum],
        "end_out_sum": [float(v) for v in end_out_sum],
        "start_out_sum": [float(v) for v in start_out_sum],
        "delta": [float(v) for v in delta],
    }


# ------------------------------------------------------------------ 断言工具
def _cmp(got, exp, tag):
    """逐位相等断言（用于有理数分支）。"""
    assert got == exp, "%s: got %r expected %r" % (tag, got, exp)


def _cmp_close(got, exp, tag, tol=1e-12):
    """相对容差断言（用于 Gauss-Legendre 分支）。"""
    assert len(got) == len(exp), "%s: length %d != %d" % (tag, len(got), len(exp))
    for i, (g, e) in enumerate(zip(got, exp)):
        if e == 0:
            assert abs(g) <= tol, "%s[%d]: got %r expected %r" % (tag, i, g, e)
        else:
            assert abs(g - e) <= tol * abs(e), "%s[%d]: got %r expected %r" % (tag, i, g, e)


def _cmp_scalar(got, exp, tag, tol=1e-12):
    """标量相对容差断言。"""
    if exp == 0:
        assert abs(got) <= tol, "%s: got %r expected %r" % (tag, got, exp)
    else:
        assert abs(got - exp) <= tol * abs(exp), "%s: got %r expected %r" % (tag, got, exp)


def _cmp_mat(got, exp, tag, close=False):
    assert len(got) == len(exp), "%s: rows %d != %d" % (tag, len(got), len(exp))
    for i, (gr, er) in enumerate(zip(got, exp)):
        if close:
            _cmp_close(gr, er, "%s[%d]" % (tag, i))
        else:
            _cmp(gr, er, "%s[%d]" % (tag, i))


# ------------------------------------------------------------------ 随机源
class _Rng:
    """确定性 LCG，保证判题可复现。"""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s

    def randint(self, a, b):
        return a + self.next() % (b - a + 1)

    def choice(self, seq):
        return seq[self.next() % len(seq)]

    def rand(self):
        return self.next() / float(0x7FFFFFFF)


RIEMANN = ("riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid")


# ------------------------------------------------------------------ check 1
def check_quadrature_params(solve):
    """求积参数：四种 Riemann 的 step_sizes / alphas，以及 Gauss-Legendre。"""

    # ---- 手算锚点 1：riemann_left n=4
    # step_sizes = [1/4]*4 = [0.25,0.25,0.25,0.25]
    # alphas = linspace(0, 3/4, 4) = [0, 1/4, 1/2, 3/4]
    r = solve("quadrature_params", method="riemann_left", n_steps=4)
    _cmp(r["step_sizes"], [0.25, 0.25, 0.25, 0.25], "qp.left4.step")
    _cmp(r["alphas"], [0.0, 0.25, 0.5, 0.75], "qp.left4.alpha")

    # ---- 手算锚点 2：riemann_right n=4
    # alphas = linspace(1/4, 1, 4) = [1/4, 1/2, 3/4, 1]
    r = solve("quadrature_params", method="riemann_right", n_steps=4)
    _cmp(r["step_sizes"], [0.25, 0.25, 0.25, 0.25], "qp.right4.step")
    _cmp(r["alphas"], [0.25, 0.5, 0.75, 1.0], "qp.right4.alpha")

    # ---- 手算锚点 3：riemann_middle n=4
    # alphas = linspace(1/8, 7/8, 4) = [1/8, 3/8, 5/8, 7/8]
    r = solve("quadrature_params", method="riemann_middle", n_steps=4)
    _cmp(r["step_sizes"], [0.25, 0.25, 0.25, 0.25], "qp.middle4.step")
    _cmp(r["alphas"], [0.125, 0.375, 0.625, 0.875], "qp.middle4.alpha")

    # ---- 手算锚点 4：riemann_trapezoid n=4（关键 quirk）
    # step_sizes = [1/8, 1/4, 1/4, 1/8]，**和为 3/4 而不是 1**
    # alphas = linspace(0, 1, 4) = [0, 1/3, 2/3, 1]
    r = solve("quadrature_params", method="riemann_trapezoid", n_steps=4)
    _cmp(r["step_sizes"], [0.125, 0.25, 0.25, 0.125], "qp.trap4.step")
    _cmp(r["alphas"], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], "qp.trap4.alpha")
    _cmp(sum(r["step_sizes"]), 0.75, "qp.trap4.step_sum")

    # ---- 手算锚点 5：trapezoid n=3 首尾都减半
    # step_sizes = [1/6, 1/3, 1/6]，和 = 2/3
    r = solve("quadrature_params", method="riemann_trapezoid", n_steps=3)
    _cmp(r["step_sizes"], [1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0], "qp.trap3.step")
    _cmp(r["alphas"], [0.0, 0.5, 1.0], "qp.trap3.alpha")

    # ---- 手算锚点 6：riemann_left n=2
    # step_sizes = [1/2, 1/2]；alphas = linspace(0, 1/2, 2) = [0, 1/2]
    r = solve("quadrature_params", method="riemann_left", n_steps=2)
    _cmp(r["step_sizes"], [0.5, 0.5], "qp.left2.step")
    _cmp(r["alphas"], [0.0, 0.5], "qp.left2.alpha")

    # ---- 手算锚点 7：gausslegendre n=1（唯一解析可手算的）
    # leggauss(1) = node 0, weight 2 -> step_sizes = [1.0], alphas = [0.5]
    r = solve("quadrature_params", method="gausslegendre", n_steps=1)
    _cmp_close(r["step_sizes"], [1.0], "qp.gl1.step")
    _cmp_close(r["alphas"], [0.5], "qp.gl1.alpha")

    # ---- 手算锚点 8：gausslegendre n=2
    # nodes = ±1/√3, weights = [1, 1]
    # step_sizes = [0.5, 0.5]（**权重不是等宽步长**）
    # alphas = 0.5*(1 + node) = [0.5 - 1/(2√3), 0.5 + 1/(2√3)]
    r = solve("quadrature_params", method="gausslegendre", n_steps=2)
    _cmp_close(r["step_sizes"], [0.5, 0.5], "qp.gl2.step")
    root3 = math.sqrt(3.0)
    _cmp_close(r["alphas"], [0.5 - 1.0 / (2.0 * root3), 0.5 + 1.0 / (2.0 * root3)],
               "qp.gl2.alpha")

    # ---- 手算锚点 9：gausslegendre n=3 的对称性与和
    # nodes = [-√(3/5), 0, √(3/5)]，weights = [5/9, 8/9, 5/9]
    # step_sizes = [5/18, 4/9, 5/18]，和 = 1
    # alphas = [0.5 - √(3/5)/2, 0.5, 0.5 + √(3/5)/2]
    r = solve("quadrature_params", method="gausslegendre", n_steps=3)
    _cmp_close(r["step_sizes"], [5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0], "qp.gl3.step")
    _cmp_scalar(sum(r["step_sizes"]), 1.0, "qp.gl3.step_sum")
    s35 = math.sqrt(3.0 / 5.0)
    _cmp_close(r["alphas"], [0.5 - s35 / 2.0, 0.5, 0.5 + s35 / 2.0], "qp.gl3.alpha")

    # ---- 网格对拍：四种 Riemann × 多个 n_steps
    for method in RIEMANN:
        for n in (2, 3, 4, 5, 8, 13, 21):
            exp_steps, exp_alphas = oracle_riemann_params(method, n)
            r = solve("quadrature_params", method=method, n_steps=n)
            _cmp(r["step_sizes"], [float(v) for v in exp_steps],
                 "qp.grid.%s.%d.step" % (method, n))
            _cmp(r["alphas"], [float(v) for v in exp_alphas],
                 "qp.grid.%s.%d.alpha" % (method, n))
            # 结构性质：alphas 单调不减且全在 [0,1]
            _cmp(all(0.0 <= v <= 1.0 for v in r["alphas"]),
                 True, "qp.grid.%s.%d.range" % (method, n))

    # ---- 网格对拍：Gauss-Legendre × 官方 numpy 真值
    for n in range(1, 13):
        exp_steps, exp_alphas = oracle_gl_params(n)
        r = solve("quadrature_params", method="gausslegendre", n_steps=n)
        _cmp_close(r["step_sizes"], exp_steps, "qp.glgrid.%d.step" % n)
        _cmp_close(r["alphas"], exp_alphas, "qp.glgrid.%d.alpha" % n)
        # 结构性质：权重和 = 1（缩放自 2），alphas 关于 0.5 对称
        _cmp_scalar(sum(r["step_sizes"]), 1.0, "qp.glgrid.%d.step_sum" % n)
        for i in range(n):
            _cmp_scalar(r["alphas"][i] + r["alphas"][n - 1 - i], 1.0,
                        "qp.glgrid.%d.sym[%d]" % (n, i))

    # ---- 非法输入：Riemann 要求 n > 1，gausslegendre 要求 n > 0
    for method in RIEMANN:
        for n in (0, 1):
            try:
                solve("quadrature_params", method=method, n_steps=n)
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError for %s n_steps=%d" % (method, n))
    try:
        solve("quadrature_params", method="gausslegendre", n_steps=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for gausslegendre n_steps=0")

    # ---- 非法 method
    try:
        solve("quadrature_params", method="riemann_bogus", n_steps=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown method")


# ------------------------------------------------------------------ check 2
def check_scaled_features(solve):
    """scaled_features 的官方 step-major 布局。"""

    # ---- 手算锚点 1：B=1, F=2, left n=2
    # alphas = [0, 1/2]
    # baseline=[1,2], input=[3,6]
    # row0 = baseline + 0*(input-baseline) = [1,2]
    # row1 = baseline + 0.5*(input-baseline) = [2,4]
    r = solve("scaled_features", input=[[3, 6]], baseline=[[1, 2]],
              method="riemann_left", n_steps=2)
    _cmp(r["scaled_features"], [[1.0, 2.0], [2.0, 4.0]], "sf.anchor1")

    # ---- 手算锚点 2：B=2, F=1, left n=2（**step-major**：row1 是 alpha=1/2 的样本 0）
    # baseline=[[0],[10]], input=[[4],[20]]
    # alpha=0   -> [[0],[10]]
    # alpha=1/2 -> [[2],[15]]
    # step-major 拼接 -> [[0],[10],[2],[15]]
    # batch-major 拼接 -> [[0],[2],[10],[15]]（错解）
    r = solve("scaled_features", input=[[4], [20]], baseline=[[0], [10]],
              method="riemann_left", n_steps=2)
    _cmp(r["scaled_features"], [[0.0], [10.0], [2.0], [15.0]], "sf.anchor2")

    # ---- 手算锚点 3：trapezoid n=3 的 alpha = [0, 1/2, 1]
    # baseline=[[2]], input=[[6]]
    # -> [2, 4, 6]
    r = solve("scaled_features", input=[[6]], baseline=[[2]],
              method="riemann_trapezoid", n_steps=3)
    _cmp(r["scaled_features"], [[2.0], [4.0], [6.0]], "sf.anchor3")

    # ---- 手算锚点 4：middle n=3 的 alpha = [1/6, 1/2, 5/6]
    # baseline=[[0]], input=[[6]] -> [1, 3, 5]
    r = solve("scaled_features", input=[[6]], baseline=[[0]],
              method="riemann_middle", n_steps=3)
    _cmp(r["scaled_features"], [[1.0], [3.0], [5.0]], "sf.anchor4")

    # ---- 网格对拍
    rng = _Rng(842104)
    for case in range(120):
        method = RIEMANN[case % 4]
        n = 2 + (case * 5) % 6
        bsz = 1 + (case * 3) % 3
        feat = 1 + (case * 7) % 4
        baseline = [[rng.randint(-9, 9) for _ in range(feat)] for _ in range(bsz)]
        input_ = [[rng.randint(-9, 9) for _ in range(feat)] for _ in range(bsz)]
        exp = oracle_scaled_features(input_, baseline, method, n)
        r = solve("scaled_features", input=input_, baseline=baseline,
                  method=method, n_steps=n)
        _cmp(r["scaled_features"], exp,
             "sf.grid[%d].%s.n%d.b%d.f%d" % (case, method, n, bsz, feat))


# ------------------------------------------------------------------ check 3
def check_ig_aggregate(solve):
    """IG 聚合：step_sizes 加权 + 沿 step 维求和 + multiply_by_inputs。"""

    # ---- 手算锚点 1：B=1, F=1, left n=2
    # step_sizes = [1/2, 1/2]，alphas = [0, 1/2]
    # baseline=[1], input=[3] -> scaled = [[1],[2]]
    # grads（step-major 长度 S*B=2）= [[5],[7]]
    # scaled_grads = [[5*1/2],[7*1/2]] = [[5/2],[7/2]]
    # total_grads（沿 step 维求和）= 5/2 + 7/2 = 6
    # attributions = 6 * (3-1) = 12
    r = solve("ig_aggregate", input=[[3]], baseline=[[1]], grads=[[5], [7]],
              method="riemann_left", n_steps=2)
    _cmp(r["total_grads"], [[6.0]], "ig.anchor1.total")
    _cmp(r["attributions"], [[12.0]], "ig.anchor1.attr")

    # ---- 手算锚点 2：multiply_by_inputs=False 时 attributions == total_grads
    r = solve("ig_aggregate", input=[[3]], baseline=[[1]], grads=[[5], [7]],
              method="riemann_left", n_steps=2, multiply_by_inputs=False)
    _cmp(r["total_grads"], [[6.0]], "ig.anchor2.total")
    _cmp(r["attributions"], [[6.0]], "ig.anchor2.attr")

    # ---- 手算锚点 3：B=2, F=1, left n=2（**step-major 的 grads**）
    # step_sizes = [1/2, 1/2]
    # baseline=[[0],[0]], input=[[2],[4]]
    # grads = [[10],[20],[30],[40]]  -> 第 s*B+b 行
    #   alpha=0.5 段: sample0 grad=10, sample1 grad=20
    #   alpha=1.0 段: sample0 grad=30, sample1 grad=40
    # scaled_grads = [[5],[10],[15],[20]]
    # total_grads: sample0 = 5+15 = 20 ; sample1 = 10+20 = 30
    # attributions: sample0 = 20*(2-0) = 40 ; sample1 = 30*(4-0) = 120
    r = solve("ig_aggregate", input=[[2], [4]], baseline=[[0], [0]],
              grads=[[10], [20], [30], [40]],
              method="riemann_left", n_steps=2)
    _cmp(r["total_grads"], [[20.0], [30.0]], "ig.anchor3.total")
    _cmp(r["attributions"], [[40.0], [120.0]], "ig.anchor3.attr")

    # ---- 手算锚点 4：trapezoid n=3（步长首尾减半 -> 总和 2/3）
    # step_sizes = [1/6, 1/3, 1/6]
    # baseline=[[0]], input=[[3]], grads=[[3],[3],[3]]
    # scaled_grads = [3/6, 3/3, 3/6] = [1/2, 1, 1/2]
    # total = 2 ; attributions = 2 * 3 = 6
    r = solve("ig_aggregate", input=[[3]], baseline=[[0]], grads=[[3], [3], [3]],
              method="riemann_trapezoid", n_steps=3)
    _cmp(r["total_grads"], [[2.0]], "ig.anchor4.total")
    _cmp(r["attributions"], [[6.0]], "ig.anchor4.attr")

    # ---- 手算锚点 5：多特征
    # B=1, F=2, left n=2, step_sizes=[1/2,1/2]
    # baseline=[[0,1]], input=[[2,5]], grads=[[4,10],[8,20]]
    # scaled_grads = [[2,5],[4,10]]
    # total = [6, 15]
    # attributions = [6*(2-0), 15*(5-1)] = [12, 60]
    r = solve("ig_aggregate", input=[[2, 5]], baseline=[[0, 1]],
              grads=[[4, 10], [8, 20]], method="riemann_left", n_steps=2)
    _cmp(r["total_grads"], [[6.0, 15.0]], "ig.anchor5.total")
    _cmp(r["attributions"], [[12.0, 60.0]], "ig.anchor5.attr")

    # ---- 网格对拍
    rng = _Rng(842205)
    for case in range(140):
        method = RIEMANN[case % 4]
        n = 2 + (case * 3) % 6
        bsz = 1 + (case * 5) % 3
        feat = 1 + (case * 11) % 4
        baseline = [[rng.randint(-8, 8) for _ in range(feat)] for _ in range(bsz)]
        input_ = [[rng.randint(-8, 8) for _ in range(feat)] for _ in range(bsz)]
        # grads 是官方 flatten 后的 step-major 布局，长度 S*B
        grads = [[rng.randint(-15, 15) for _ in range(feat)] for _ in range(n * bsz)]
        exp = oracle_ig_aggregate(input_, baseline, grads, method, n)
        r = solve("ig_aggregate", input=input_, baseline=baseline, grads=grads,
                  method=method, n_steps=n)
        _cmp_mat(r["total_grads"], exp["total_grads"],
                 "ig.grid[%d].total.%s.n%d.b%d.f%d" % (case, method, n, bsz, feat))
        _cmp_mat(r["attributions"], exp["attributions"],
                 "ig.grid[%d].attr.%s.n%d.b%d.f%d" % (case, method, n, bsz, feat))

    # ---- 网格对拍：multiply_by_inputs=False
    for case in range(40):
        method = RIEMANN[case % 4]
        n = 2 + (case % 5)
        bsz = 1 + (case % 3)
        feat = 1 + (case % 4)
        baseline = [[rng.randint(-8, 8) for _ in range(feat)] for _ in range(bsz)]
        input_ = [[rng.randint(-8, 8) for _ in range(feat)] for _ in range(bsz)]
        grads = [[rng.randint(-15, 15) for _ in range(feat)] for _ in range(n * bsz)]
        exp = oracle_ig_aggregate(input_, baseline, grads, method, n,
                                  multiply_by_inputs=False)
        r = solve("ig_aggregate", input=input_, baseline=baseline, grads=grads,
                  method=method, n_steps=n, multiply_by_inputs=False)
        _cmp_mat(r["total_grads"], exp["total_grads"],
                 "ig.nomult.grid[%d].total" % case)
        _cmp_mat(r["attributions"], exp["attributions"],
                 "ig.nomult.grid[%d].attr" % case)


# ------------------------------------------------------------------ check 4
def check_ig_completeness(solve):
    """完备性公理 delta：attr_sum - (end_out_sum - start_out_sum)。"""

    # ---- 手算锚点 1：B=1, F=2
    # attributions = [[1, 2]] -> attr_sum = 3
    # start_out = [[5]] -> start_out_sum = 5
    # end_out = [[11]] -> end_out_sum = 11
    # delta = 3 - (11 - 5) = -3
    r = solve("ig_completeness", attributions=[[1, 2]], start_out=[[5]], end_out=[[11]])
    _cmp(r["attr_sum"], [3.0], "ic.anchor1.attr_sum")
    _cmp(r["start_out_sum"], [5.0], "ic.anchor1.start")
    _cmp(r["end_out_sum"], [11.0], "ic.anchor1.end")
    _cmp(r["delta"], [-3.0], "ic.anchor1.delta")

    # ---- 手算锚点 2：符号顺序（attr_sum 在前）
    # attributions=[[4]], start_out=[[1]], end_out=[[1]]
    # delta = 4 - (1 - 1) = 4（若写反则是 -4）
    r = solve("ig_completeness", attributions=[[4]], start_out=[[1]], end_out=[[1]])
    _cmp(r["delta"], [4.0], "ic.anchor2.delta")

    # ---- 手算锚点 3：完备性成立时 delta = 0
    # attributions=[[3]], start_out=[[2]], end_out=[[5]] -> 3 - (5-2) = 0
    r = solve("ig_completeness", attributions=[[3]], start_out=[[2]], end_out=[[5]])
    _cmp(r["delta"], [0.0], "ic.anchor3.delta")

    # ---- 手算锚点 4：B=2, 多特征
    # attributions=[[1,2],[3,4]] -> attr_sum = [3, 7]
    # start_out=[[1],[2]] -> [1, 2]
    # end_out=[[4],[5]]   -> [4, 5]
    # delta = [3 - 3, 7 - 3] = [0, 4]
    r = solve("ig_completeness", attributions=[[1, 2], [3, 4]],
              start_out=[[1], [2]], end_out=[[4], [5]])
    _cmp(r["attr_sum"], [3.0, 7.0], "ic.anchor4.attr_sum")
    _cmp(r["delta"], [0.0, 4.0], "ic.anchor4.delta")

    # ---- 手算锚点 5：负数与浮点输入
    # attributions=[[-1.5, 2.5]] -> attr_sum = 1.0
    # start_out=[[0.25]], end_out=[[1.25]] -> delta = 1 - (1.25-0.25) = 0.0
    r = solve("ig_completeness", attributions=[[-1.5, 2.5]],
              start_out=[[0.25]], end_out=[[1.25]])
    _cmp(r["attr_sum"], [1.0], "ic.anchor5.attr_sum")
    _cmp(r["delta"], [0.0], "ic.anchor5.delta")

    # ---- 网格对拍
    rng = _Rng(842306)
    for case in range(120):
        bsz = 1 + (case * 5) % 3
        feat = 1 + (case * 3) % 5
        attributions = [[rng.randint(-20, 20) + rng.choice([0, 0.5, -0.25])
                         for _ in range(feat)] for _ in range(bsz)]
        start_out = [[rng.randint(-20, 20) + rng.choice([0, 0.5])
                      for _ in range(feat)] for _ in range(bsz)]
        end_out = [[rng.randint(-20, 20) + rng.choice([0, 0.5])
                    for _ in range(feat)] for _ in range(bsz)]
        exp = oracle_ig_completeness(attributions, start_out, end_out)
        r = solve("ig_completeness", attributions=attributions,
                  start_out=start_out, end_out=end_out)
        _cmp(r["attr_sum"], exp["attr_sum"], "ic.grid[%d].attr_sum" % case)
        _cmp(r["start_out_sum"], exp["start_out_sum"], "ic.grid[%d].start" % case)
        _cmp(r["end_out_sum"], exp["end_out_sum"], "ic.grid[%d].end" % case)
        _cmp(r["delta"], exp["delta"], "ic.grid[%d].delta" % case)


CHECKS = [
    check_quadrature_params,
    check_scaled_features,
    check_ig_aggregate,
    check_ig_completeness,
]
