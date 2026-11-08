"""order 839 参考解（暂存）：xLSTM sLSTM 的 max 稳定器、normalizer state 与点式递推账。

官方回源：NX-AI/xlstm (main)
  xlstm/blocks/slstm/src/vanilla/slstm.py::slstm_forward_pointwise
  xlstm/blocks/slstm/cell.py::sLSTMCellConfig / reset_parameters
  xlstm/components/init.py::bias_linspace_init_

三段纯数值账，不做 GEMM、不做 autograd、不依赖 torch/numpy。
"""

import math

NAN = float("nan")


def _logsigmoid(x):
    """log(sigmoid(x)) = -softplus(-x)，分段避免溢出。"""
    if x >= 0.0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def _sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _tanh(x):
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    return math.tanh(x)


def _split_gates(flat, head_dim):
    """长度 4*head_dim 的扁平列表按 [i, f, z, o] 分成 4 组。"""
    return [list(flat[g * head_dim:(g + 1) * head_dim]) for g in range(4)]


def _gates_from_raw(raw, head_dim):
    """raw = Wx + Ry + b，按官方 view(B, 4, -1) 拆成 iraw/fraw/zraw/oraw。"""
    if len(raw) != 4 * head_dim:
        raise ValueError("raw length must be 4 * head_dim")
    return _split_gates(raw, head_dim)


def _max_stabilizer(iraw, fraw, m_prev, n_prev):
    """sLSTM 的 max 稳定器与两个门。

    官方 vanilla 语义：``torch.all(n == 0.0)`` 是**整帧**判定，
    要么整帧走 ``mnew = iraw``，要么整帧走 ``mnew = max(iraw, logfplusm)``。
    """
    h = len(iraw)
    logfplusm = [m_prev[i] + _logsigmoid(fraw[i]) for i in range(h)]
    zero_n_branch = all(n_prev[i] == 0.0 for i in range(h))
    if zero_n_branch:
        m_new = list(iraw)
    else:
        m_new = [iraw[i] if iraw[i] >= logfplusm[i] else logfplusm[i]
                 for i in range(h)]
    igate = []
    fgate = []
    gate_le_one = []
    for i in range(h):
        gi = math.exp(iraw[i] - m_new[i])
        if gi > 1.0:
            gi = 1.0
        gf = math.exp(logfplusm[i] - m_new[i])
        if gf > 1.0:
            gf = 1.0
        igate.append(gi)
        fgate.append(gf)
        gate_le_one.append(gi <= 1.0 and gf <= 1.0)
    return {
        "logfplusm": logfplusm,
        "m_new": m_new,
        "zero_n_branch": zero_n_branch,
        "igate": igate,
        "fgate": fgate,
        "gate_le_one": gate_le_one,
    }


def _pointwise_step(wx, ry, bias, states, head_dim):
    """官方 slstm_forward_pointwise 的 1-D 化（B=1，H 个通道）。

    states 是 4 组长度 H 的列表：y / c / n / m。
    """
    raw = []
    for i in range(4 * head_dim):
        raw.append(wx[i] + ry[i] + bias[i])
    iraw, fraw, zraw, oraw = _gates_from_raw(raw, head_dim)

    y_prev = states[0]
    c_prev = states[1]
    n_prev = states[2]
    m_prev = states[3]

    stab = _max_stabilizer(iraw, fraw, m_prev, n_prev)
    m_new = stab["m_new"]
    igate = stab["igate"]
    fgate = stab["fgate"]

    ogate = [_sigmoid(oraw[i]) for i in range(head_dim)]
    zval = [_tanh(zraw[i]) for i in range(head_dim)]

    c_new = [fgate[i] * c_prev[i] + igate[i] * zval[i] for i in range(head_dim)]
    n_new = [fgate[i] * n_prev[i] + igate[i] for i in range(head_dim)]

    y_new = []
    for i in range(head_dim):
        if n_new[i] == 0.0:
            y_new.append(NAN)
        else:
            y_new.append(ogate[i] * c_new[i] / n_new[i])

    return {
        "raw_gates": [iraw, fraw, zraw, oraw],
        "logfplusm": stab["logfplusm"],
        "m_new": m_new,
        "zero_n_branch": stab["zero_n_branch"],
        "igate": igate,
        "fgate": fgate,
        "ogate": ogate,
        "zval": zval,
        "c_new": c_new,
        "n_new": n_new,
        "y_new": y_new,
    }


def _bias_powerlaw(head_dim, block_idx, num_blocks, mode="powerlaw"):
    """官方 reset_parameters 的两种 f 门 bias 初始化。"""
    if num_blocks > 1:
        ratio = block_idx / float(num_blocks - 1)
    else:
        ratio = 0.0

    if mode == "powerlaw":
        exponent = 0.3 + 1.3 * ratio
        bias = []
        for d in range(head_dim):
            if head_dim == 1:
                base = NAN
            else:
                base = d / float(head_dim - 1)
            bias.append(-(-5.0 + 12.0 * (base ** exponent)))
        return {"bias": bias, "exponent": exponent, "ratio": ratio}

    if mode == "small_init":
        exponent = 0.0
        bias = []
        for d in range(head_dim):
            if head_dim == 1:
                bias.append(3.0)
            else:
                bias.append(3.0 + (6.0 - 3.0) * d / float(head_dim - 1))
        return {"bias": bias, "exponent": exponent, "ratio": ratio}

    raise ValueError("unknown bias mode: %s" % mode)


def solve(mode: str, **kw):
    if mode == "pointwise_step":
        return _pointwise_step(
            list(kw["wx"]),
            list(kw["ry"]),
            list(kw["bias"]),
            [list(s) for s in kw["states"]],
            int(kw["head_dim"]),
        )
    if mode == "max_stabilizer":
        return _max_stabilizer(
            list(kw["iraw"]),
            list(kw["fraw"]),
            list(kw["m_prev"]),
            list(kw["n_prev"]),
        )
    if mode == "bias_powerlaw":
        return _bias_powerlaw(
            int(kw["head_dim"]),
            int(kw["block_idx"]),
            int(kw["num_blocks"]),
            str(kw.get("bias_mode", "powerlaw")),
        )
    raise ValueError("unknown mode: %s" % mode)
