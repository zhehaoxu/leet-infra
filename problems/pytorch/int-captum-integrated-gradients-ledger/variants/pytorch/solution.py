"""order 842 参考解（暂存）：Captum Integrated Gradients 的求积与归因账。

官方回源：pytorch/captum (main)
  captum/attr/_utils/approximation_methods.py::approximation_parameters,
      riemann_builders, gauss_legendre_builders, Riemann, SUPPORTED_METHODS
  captum/attr/_core/integrated_gradients.py::IntegratedGradients._attribute
  captum/attr/_utils/common.py::_reshape_and_sum, _sum_rows
  captum/attr/_utils/attribution.py::Attribution.compute_convergence_delta

四段纯数值账，不做前向传播、不做自动微分、不依赖 torch/numpy。

**为什么 Riemann 四变体用精确有理数口径**
官方 `riemann_builders` 的 step_sizes 是 `1/n`（trapezoid 首尾再减半），
alphas 走 `torch.linspace` 的四种不同端点约定。这两组数在**数学上**全都是
精确有理数：

    left      step_sizes = [1/n]*n
              alphas     = [i/n        for i in 0..n-1]     # linspace(0, 1-1/n, n)
    right     step_sizes = [1/n]*n
              alphas     = [(i+1)/n    for i in 0..n-1]     # linspace(1/n, 1, n)
    middle    step_sizes = [1/n]*n
              alphas     = [(2i+1)/(2n) for i in 0..n-1]    # linspace(1/(2n), 1-1/(2n), n)
    trapezoid step_sizes = [1/(2n)] + [1/n]*(n-2) + [1/(2n)]
              alphas     = [i/(n-1)    for i in 0..n-1]     # linspace(0, 1, n)

但官方 `torch.linspace` 默认走 **float32**，会引入 ~1e-8 量级的尾差
（例：trapezoid n=4 的第二个 alpha，float32 是 0.3333333432674408，
精确值是 1/3 = 0.3333333333333333）。为了让账本可判定，本题把
  * step_sizes 与 alphas 一律按**精确有理数**计算（用 fractions.Fraction）；
  * 输出统一转成 float（Fraction -> float 是正确舍入，唯一确定）。

于是「trapezoid 的 step_sizes 之和是 1 - 1/n 而不是 1」这类官方 quirk
能被逐位判定，而不是被浮点噪声淹没。

**Gauss-Legendre 单独处理**
`gauss_legendre_builders` 用 numpy `leggauss(n)`：
  * step_sizes = 0.5 * weights      —— 是**求积权重**，不是等宽步长！
  * alphas     = 0.5 * (1 + nodes)  —— 是**节点**从 [-1,1] 缩放到 [0,1]
节点是 Legendre 多项式的无理数根，无法用有理数表达，因此：
  * 断言 `n > 0`（Riemann 是 `n > 1`）；
  * 本题用纯 Python Newton 法求 Legendre 根（与 numpy leggauss 实测
    n=50 时节点差 <= 1.11e-16、权重差 <= 2.67e-15），
  * 判定用相对容差 1e-12（irrational 分支无法逐位判定）。

**聚合与完备性**
官方 `_attribute` 的关键布局：
    scaled_features = cat([baseline + alpha*(input-baseline) for alpha in alphas], dim=0)
  -> 形状 (S*B, F)，**step-major**：第 s 段的 B 行是第 s 个 alpha 的整批样本。
    scaled_grads = grad.view(S, -1) * step_sizes.view(S, 1)
  -> 注意官方这里 `torch.tensor(step_sizes).float()` 把步长**降到 float32**。
    total_grads = _reshape_and_sum(scaled_grad, S, B, (F,))
  -> reshape((S,B)+F).sum(dim=0)：沿 **step** 维求和，不是沿 batch 维。
    attributions = total_grads * (input - baseline)   # multiply_by_inputs=True

官方 `compute_convergence_delta`：
    row_sums   = [_sum_rows(attribution) for attribution in attributions]
    attr_sum   = stack([sum(row_sum) for row_sum in zip(*row_sums)])
  -> `zip(*row_sums)` 沿 **sample** 维 zip：单输入张量时 attr_sum[b] = Σ_f a[b][f]
    _delta = attr_sum - (end_out_sum - start_out_sum)
  -> **符号顺序：attr_sum 在前**，end_out_sum - start_out_sum 在后。

本题全部 Riemann 分支用 Fraction 全程精确（grads / input / baseline 都是
有理数），输出转 float；Gauss-Legendre 分支用 float 容差判定。
"""

import math
from fractions import Fraction

# ---------------------------------------------------------------- 官方常量
SUPPORTED_RIEMANN_METHODS = (
    "riemann_left",
    "riemann_right",
    "riemann_middle",
    "riemann_trapezoid",
)
SUPPORTED_METHODS = SUPPORTED_RIEMANN_METHODS + ("gausslegendre",)


# ---------------------------------------------------------------- Riemann 参数
def _require_steps(n, minimum, message):
    """官方断言的统一入口。

    官方在 `step_sizes` 与 `alphas` 两个闭包里各写了一次 assert（语义相同），
    这里收敛成单一 guard，便于判定「n 的下界」这一个语义点。
    """
    if n < minimum:
        raise ValueError(message)


def _riemann_step_sizes(method, n):
    """精确有理 step_sizes（Riemann 四变体）。

    官方：`deltas = [1/n]*n`；trapezoid 时 `deltas[0] /= 2` 且 `deltas[-1] /= 2`。
    """
    if method == "riemann_trapezoid":
        return [Fraction(1, 2 * n)] + [Fraction(1, n)] * (n - 2) + [Fraction(1, 2 * n)]
    return [Fraction(1, n)] * n


def _riemann_alphas(method, n):
    """精确有理 alphas（Riemann 四变体）。

    官方四种 linspace 端点约定在数学上等价于下式（步长都是 1/n 或 1/(n-1)）。
    """
    if method == "riemann_left":
        # linspace(0, 1-1/n, n) -> 步长 (1-1/n)/(n-1) = 1/n
        return [Fraction(i, n) for i in range(n)]
    if method == "riemann_right":
        # linspace(1/n, 1, n) -> 步长 (1-1/n)/(n-1) = 1/n
        return [Fraction(i + 1, n) for i in range(n)]
    if method == "riemann_middle":
        # linspace(1/(2n), 1-1/(2n), n) -> 步长 (1-1/n)/(n-1) = 1/n
        return [Fraction(2 * i + 1, 2 * n) for i in range(n)]
    if method == "riemann_trapezoid":
        # linspace(0, 1, n) -> 步长 1/(n-1)
        return [Fraction(i, n - 1) for i in range(n)]
    raise ValueError("Provided Reimann approximation method is not valid.")


# ---------------------------------------------------------------- Gauss-Legendre
def _legendre(n, x):
    """返回 (P_n(x), P'_n(x))，Legendre 三项递推。"""
    if n == 0:
        return 1.0, 0.0
    p0, p1 = 1.0, x
    if n == 1:
        return p1, 1.0
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    # P'_n(x) = n (x P_n(x) - P_{n-1}(x)) / (x^2 - 1)
    dp = n * (x * p1 - p0) / (x * x - 1.0)
    return p1, dp


def _gauss_legendre(n):
    """纯 Python Newton 法复现 numpy.polynomial.legendre.leggauss。

    返回 (nodes 升序, weights 升序对应)。
    """
    nodes = []
    weights = []
    for i in range(n):
        # Chebyshev 型初值：第 i 个根在 cos(pi*(i+0.75)/(n+0.5)) 附近
        x = _cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(100):
            p, dp = _legendre(n, x)
            dx = p / dp
            x -= dx
            if abs(dx) < 1e-16:
                break
        p, dp = _legendre(n, x)
        w = 2.0 / ((1.0 - x * x) * dp * dp)
        nodes.append(x)
        weights.append(w)
    order = sorted(range(n), key=lambda k: nodes[k])
    return [nodes[k] for k in order], [weights[k] for k in order]


def _cos(x):
    """math.cos 的薄封装（保持模块自洽，便于变异测试定位）。"""
    return math.cos(x)


# ---------------------------------------------------------------- 参数入口
def _parameters(method, n_steps):
    """返回 (step_sizes, alphas, exact_flag)。

    exact_flag=True 表示全为有理数（可逐位判定）；False 表示含无理数（容差判定）。
    """
    if method in SUPPORTED_RIEMANN_METHODS:
        # 官方在 step_sizes / alphas 两个闭包里各 assert 了一次（同一语义），
        # 这里在唯一分派点做一次断言。
        _require_steps(n_steps, 2, "The number of steps has to be larger than one")
        return _riemann_step_sizes(method, n_steps), _riemann_alphas(method, n_steps), True
    if method == "gausslegendre":
        _require_steps(n_steps, 1, "The number of steps has to be larger than zero")
        nodes, weights = _gauss_legendre(n_steps)
        # 官方：step_sizes = 0.5 * leggauss(n)[1]（求积权重），alphas = 0.5 * (1 + leggauss(n)[0])
        step_sizes = [0.5 * w for w in weights]
        alphas = [0.5 * (1.0 + x) for x in nodes]
        return step_sizes, alphas, False
    raise ValueError("Invalid integral approximation method name: {}".format(method))


def _to_float_list(xs):
    return [float(x) for x in xs]


# ---------------------------------------------------------------- mode 实现
def _mode_quadrature_params(method, n_steps):
    step_sizes, alphas, _ = _parameters(method, n_steps)
    return {
        "step_sizes": _to_float_list(step_sizes),
        "alphas": _to_float_list(alphas),
    }


def _mode_scaled_features(input_, baseline, method, n_steps):
    """官方 scaled_features 布局：cat 沿 dim=0 -> (S*B, F)，step-major。

    第 s*B + b 行 = baseline[b] + alphas[s] * (input[b] - baseline[b])
    """
    step_sizes, alphas, exact = _parameters(method, n_steps)
    bsz = len(input_)
    feat = len(input_[0])
    out = []
    for s in range(n_steps):
        alpha = alphas[s]
        for b in range(bsz):
            row = []
            for f in range(feat):
                base = Fraction(baseline[b][f]) if exact else baseline[b][f]
                inp = Fraction(input_[b][f]) if exact else input_[b][f]
                row.append(float(base + alpha * (inp - base)))
            out.append(row)
    return {"scaled_features": out}


def _mode_ig_aggregate(input_, baseline, grads, method, n_steps,
                       multiply_by_inputs=True):
    """官方 _attribute 的聚合段。

    grads : 官方 flatten 后的布局，长度 S*B，第 s*B + b 行是第 s 个 alpha
            下第 b 个样本的梯度（与 scaled_features 同序，step-major）。
    """
    step_sizes, alphas, exact = _parameters(method, n_steps)
    bsz = len(input_)
    feat = len(input_[0])
    steps = n_steps

    # scaled_grads = grad.view(S, -1) * step_sizes.view(S, 1)
    #   -> 第 s 段的每一行都乘上 step_sizes[s]
    scaled = []
    for s in range(steps):
        w = step_sizes[s]
        for b in range(bsz):
            row = grads[s * bsz + b]
            new_row = []
            for f in range(feat):
                g = Fraction(row[f]) if exact else row[f]
                new_row.append(g * w)
            scaled.append(new_row)

    # total_grads = reshape((S, B) + (F,)).sum(dim=0)  -> 沿 step 维求和
    total = []
    for b in range(bsz):
        row = []
        for f in range(feat):
            acc = 0
            for s in range(steps):
                acc = acc + scaled[s * bsz + b][f]
            row.append(acc)
        total.append(row)

    # attributions = total_grads * (input - baseline)
    if multiply_by_inputs:
        attributions = []
        for b in range(bsz):
            row = []
            for f in range(feat):
                base = Fraction(baseline[b][f]) if exact else baseline[b][f]
                inp = Fraction(input_[b][f]) if exact else input_[b][f]
                row.append(total[b][f] * (inp - base))
            attributions.append(row)
    else:
        attributions = [list(row) for row in total]

    return {
        "total_grads": [[float(v) for v in row] for row in total],
        "attributions": [[float(v) for v in row] for row in attributions],
    }


def _mode_ig_completeness(attributions, start_out, end_out):
    """官方 compute_convergence_delta 的账。

    attr_sum[b] = Σ_f attributions[b][f]
    end_out_sum[b] = Σ_f end_out[b][f]      （_sum_rows）
    start_out_sum[b] = Σ_f start_out[b][f]
    delta[b] = attr_sum[b] - (end_out_sum[b] - start_out_sum[b])
    """
    bsz = len(attributions)
    attr_sum = []
    end_out_sum = []
    start_out_sum = []
    for b in range(bsz):
        attr_sum.append(sum(Fraction(v) for v in attributions[b]))
        end_out_sum.append(sum(Fraction(v) for v in end_out[b]))
        start_out_sum.append(sum(Fraction(v) for v in start_out[b]))
    delta = [attr_sum[b] - (end_out_sum[b] - start_out_sum[b]) for b in range(bsz)]
    return {
        "attr_sum": [float(v) for v in attr_sum],
        "end_out_sum": [float(v) for v in end_out_sum],
        "start_out_sum": [float(v) for v in start_out_sum],
        "delta": [float(v) for v in delta],
    }


# ---------------------------------------------------------------- 入口
def solve(mode: str, **kw):
    if mode == "quadrature_params":
        return _mode_quadrature_params(kw["method"], kw["n_steps"])
    if mode == "scaled_features":
        return _mode_scaled_features(kw["input"], kw["baseline"],
                                     kw["method"], kw["n_steps"])
    if mode == "ig_aggregate":
        return _mode_ig_aggregate(kw["input"], kw["baseline"], kw["grads"],
                                  kw["method"], kw["n_steps"],
                                  kw.get("multiply_by_inputs", True))
    if mode == "ig_completeness":
        return _mode_ig_completeness(kw["attributions"], kw["start_out"],
                                     kw["end_out"])
    raise ValueError("unknown mode: {}".format(mode))
