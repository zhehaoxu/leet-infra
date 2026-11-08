"""order 843 参考解：Captum LRP 的 z-rule relevance 传播账。

官方源：pytorch/captum（main）
  - captum/attr/_core/lrp.py            （LRP.attribute / compute_convergence_delta）
  - captum/attr/_utils/lrp_rules.py     （EpsilonRule / GammaRule / Alpha1_Beta0_Rule）

口径（与官方实现的差异已显式化）：
  - 所有输入量都是整数；eps 取**精确有理数 1/10^9**（官方是 float64 的 1e-9）。
  - 全链路用 fractions.Fraction 精确计算，只有最终输出时转一次 float。
  - 官方 LRP 走**两次前向**：第一次用原权重前向并存下每层激活，第二次用
    「改写后的权重 × 第一次的激活」重算。backward hook 捕获到的 outputs 是后者。
  - output hook: relevance = grad / (outputs + sign(outputs) * eps)，且 sign(0) 强制为 +1。
  - input hook : relevance = grad * inputs（inputs 是第一次前向的激活）。
  - attribution = relevance_in(最输入层) * output[target]（output 是原权重下的前向值）。
  - delta = output - sum(attributions)（官方 LRP 符号，output 在前；与 IG 相反）。
"""

from fractions import Fraction

# ---------------------------------------------------------------- 官方常量
EPS = Fraction(1, 10 ** 9)
"""官方 `PropagationRule.STABILITY_FACTOR = 1e-9`。本题按精确有理数 1/10^9 处理。"""

SUPPORTED_RULES = ("epsilon", "gamma", "alpha1_beta0")
"""官方 `lrp_rules.py` 里会改写权重的三条规则。

`IdentityRule` 在官方实现里必然抛 KeyError（`relevance_output` 尚未填充就被读取），
属于官方 bug，本题不收。
"""


# ---------------------------------------------------------------- 有理数工具
def _f(x):
    """把入参转成精确有理数。

    int -> Fraction(x)；str -> Fraction(str)（支持 "1/4" / "0.25" / "-3"）；
    float -> Fraction(str(x))（避免二进制浮点的尾差渗入）。
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        raise TypeError("bool is not a valid numeric value")
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(str(x))
    raise TypeError("unsupported numeric type: %r" % type(x))


def _sign(v):
    """官方 `torch.sign` 后 `sign[sign == 0] = 1` 的口径。"""
    if v == 0:
        return Fraction(1)
    return Fraction(1) if v > 0 else Fraction(-1)


def _one_hot(size, index):
    """one-hot 向量（最顶层的上游梯度）。"""
    if not (0 <= index < size):
        raise ValueError("target %d out of range for %d outputs" % (index, size))
    return [Fraction(1) if i == index else Fraction(0) for i in range(size)]


def _as_matrix(weights):
    """校验并规范化权重矩阵（行列表）。"""
    if not isinstance(weights, (list, tuple)) or len(weights) == 0:
        raise ValueError("weights must be a non-empty list of rows")
    rows = []
    width = None
    for row in weights:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise ValueError("each weight row must be a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("weight rows must all have the same length")
        rows.append([_f(v) for v in row])
    return rows


def _as_vector(vec, name):
    """校验并规范化向量。"""
    if not isinstance(vec, (list, tuple)):
        raise ValueError("%s must be a list" % name)
    return [_f(v) for v in vec]


def _matvec(weights, vec):
    """权重矩阵乘向量（列向量约定：out[i] = sum_j W[i][j] * vec[j]）。"""
    if len(vec) != len(weights[0]):
        raise ValueError("vector length %d does not match weight width %d"
                         % (len(vec), len(weights[0])))
    return [sum(weights[i][j] * vec[j] for j in range(len(vec)))
            for i in range(len(weights))]


# ---------------------------------------------------------------- 规则改写
def _rewrite(rule, weights, bias, gamma, set_bias_to_zero):
    """按官方 `_manipulate_weights` 语义改写权重与偏置。

    - epsilon      : 权重/偏置**原样**（官方 `EpsilonRule._manipulate_weights` 是 pass）
    - gamma        : W'[i][j] = W[i][j] + gamma * max(W[i][j], 0)
                     （官方 quirk：只放大**正**权重，负权重不变）
    - alpha1_beta0 : W'[i][j] = max(W[i][j], 0)（负权重清零，Deep-Taylor）
    - set_bias_to_zero : 偏置全 0
    """
    if rule not in SUPPORTED_RULES:
        raise ValueError("unknown rule: %r" % (rule,))
    W = [list(row) for row in weights]
    B = list(bias)
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


# ---------------------------------------------------------------- 内部：逐层账
def _layer_ledger(weights, bias, activation, upstream):
    """单层的 z-rule 账。

    weights : 已改写的权重（Fraction 矩阵）
    bias    : 已改写的偏置（Fraction 向量）
    activation : 该层的**输入激活**（第一次前向存下的，Fraction 向量）
    upstream : 该层输出侧收到的上游梯度（Fraction 向量）
               —— 最顶层是 one-hot(target)，其余层是上一层的 relevance_in

    返回 (outputs, relevance_out, relevance_in)
    """
    y = _matvec(weights, activation)
    y = [y[i] + bias[i] for i in range(len(y))]
    if len(upstream) != len(y):
        raise ValueError("upstream length %d does not match layer outputs %d"
                         % (len(upstream), len(y)))
    # output hook: relevance = grad / (outputs + sign(outputs) * eps)，sign(0) 强制为 +1
    rel_out = [upstream[i] / (y[i] + _sign(y[i]) * EPS) for i in range(len(y))]
    # input hook: relevance = grad * inputs，其中 grad 是 (rel_out @ W)
    tmp = [sum(rel_out[i] * weights[i][j] for i in range(len(weights)))
           for j in range(len(activation))]
    rel_in = [tmp[j] * activation[j] for j in range(len(activation))]
    return y, rel_out, rel_in


# ---------------------------------------------------------------- mode 1
def _mode_rewrite_rule(rule, weights, bias, gamma, set_bias_to_zero):
    """官方 `_manipulate_weights` 的权重改写账。"""
    W = _as_matrix(weights)
    B = _as_vector(bias, "bias")
    if len(B) != len(W):
        raise ValueError("bias length %d does not match weight rows %d"
                         % (len(B), len(W)))
    Wp, Bp = _rewrite(rule, W, B, gamma, set_bias_to_zero)
    return {
        "weights": [[float(v) for v in row] for row in Wp],
        "bias": [float(v) for v in Bp],
    }


# ---------------------------------------------------------------- mode 2
def _mode_z_rule_layer(weights, bias, inputs, target):
    """单层线性层的 z-rule relevance 分解。"""
    W = _as_matrix(weights)
    B = _as_vector(bias, "bias")
    x = _as_vector(inputs, "inputs")
    if len(B) != len(W):
        raise ValueError("bias length %d does not match weight rows %d"
                         % (len(B), len(W)))
    y, rel_out, rel_in = _layer_ledger(W, B, x, _one_hot(len(W), target))
    attr = [v * y[target] for v in rel_in]
    return {
        "outputs": [float(v) for v in y],
        "relevance_out": [float(v) for v in rel_out],
        "relevance_in": [float(v) for v in rel_in],
        "attribution": [float(v) for v in attr],
    }


# ---------------------------------------------------------------- mode 3
def _mode_stack_relevance(layers, inputs, target):
    """多层线性栈的 z-rule 传播账（严格实现官方的两次前向语义）。

    返回的所有列表都按**前向顺序**（下标 0 = 靠输入层的第一层）。
    """
    if not isinstance(layers, (list, tuple)) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list")

    # ---- 解析每层
    parsed = []
    for lay in layers:
        if not isinstance(lay, dict):
            raise ValueError("each layer must be a dict")
        rule = lay.get("rule", "epsilon")
        if rule not in SUPPORTED_RULES:
            raise ValueError("unknown rule: %r" % (rule,))
        W = _as_matrix(lay["weights"])
        B = _as_vector(lay["bias"], "bias")
        if len(B) != len(W):
            raise ValueError("bias length %d does not match weight rows %d"
                             % (len(B), len(W)))
        gamma = lay.get("gamma", Fraction(1, 4))
        sbz = bool(lay.get("set_bias_to_zero", False))
        parsed.append((W, B, rule, gamma, sbz))

    x = _as_vector(inputs, "inputs")

    # ---- 第一次前向：全用原权重，链式前向（官方 forward_hook_weights 存的就是它）
    acts = [x]
    for (W, B, _r, _g, _s) in parsed:
        y = _matvec(W, acts[-1])
        acts.append([y[i] + B[i] for i in range(len(y))])

    # ---- 权重改写
    rewritten = [_rewrite(rule, W, B, gamma, sbz) for (W, B, rule, gamma, sbz) in parsed]

    # ---- 第二次前向：每层用「改写后的权重 × 对应层的原激活」
    outs = []
    for i in range(len(parsed)):
        Wp, Bp = rewritten[i]
        y = _matvec(Wp, acts[i])
        outs.append([y[k] + Bp[k] for k in range(len(y))])

    # ---- 反向：从输出层往输入层算，最后翻回前向顺序
    rel_outs = []
    rel_ins = []
    upstream = _one_hot(len(parsed[-1][0]), target)
    for i in range(len(parsed) - 1, -1, -1):
        Wp, Bp = rewritten[i]
        _y, ro, ri = _layer_ledger(Wp, Bp, acts[i], upstream)
        rel_outs.append(ro)
        rel_ins.append(ri)
        upstream = ri
    rel_outs.reverse()
    rel_ins.reverse()

    # ---- 归一化输出：attribution = relevance_in(最输入层) * 原权重下的 target 输出
    output_target = acts[-1][target]
    attr = [v * output_target for v in rel_ins[0]]

    return {
        "activations": [[float(v) for v in row] for row in acts[1:]],
        "outputs": [[float(v) for v in row] for row in outs],
        "relevance_out": [[float(v) for v in row] for row in rel_outs],
        "relevance_in": [[float(v) for v in row] for row in rel_ins],
        "attribution": [float(v) for v in attr],
    }


# ---------------------------------------------------------------- mode 4
def _mode_completeness(attributions, output):
    """官方 `LRP.compute_convergence_delta`：delta = output - sum(attributions)。

    官方 `_sum_rows` 是逐行求和；本题单样本，故 attr_sum 就是标量之和。
    """
    attrs = _as_vector(attributions, "attributions")
    out = _f(output)
    attr_sum = sum(attrs, Fraction(0))
    delta = out - attr_sum
    return {
        "attr_sum": float(attr_sum),
        "delta": float(delta),
    }


# ---------------------------------------------------------------- 入口
def solve(mode, **kw):
    """单入口。

    mode:
      rewrite_rule    : 规则权重改写账
      z_rule_layer    : 单层 z-rule 分解
      stack_relevance : 多层栈的 z-rule 传播账
      completeness    : 完备性 delta（output 在前）
    """
    if mode == "rewrite_rule":
        return _mode_rewrite_rule(
            kw["rule"], kw["weights"], kw["bias"],
            kw.get("gamma", Fraction(1, 4)),
            bool(kw.get("set_bias_to_zero", False)))
    if mode == "z_rule_layer":
        return _mode_z_rule_layer(kw["weights"], kw["bias"], kw["inputs"], kw["target"])
    if mode == "stack_relevance":
        return _mode_stack_relevance(kw["layers"], kw["inputs"], kw["target"])
    if mode == "completeness":
        return _mode_completeness(kw["attributions"], kw["output"])
    raise ValueError("unknown mode: %r" % (mode,))
