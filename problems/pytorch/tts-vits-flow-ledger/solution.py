"""VITS normalizing flow 的纯 Python 数值账。

对应官方源码（jaywalnut310/vits, main 分支）：
  modules.py::Log.forward
  modules.py::Flip.forward
  modules.py::ElementwiseAffine.forward
  modules.py::ResidualCouplingLayer.forward
  models.py::ResidualCouplingBlock.forward
  models.py::StochasticDurationPredictor.forward

四个 mode（全部显式接收 shape，不依赖任何隐式状态）：
  flow_primitive  —— 单个 flow 基元的前向/反向与 log-det 账（Log / Flip / ElementwiseAffine）
  coupling_layer  —— ResidualCouplingLayer 的前向/反向与 log-det 账
  coupling_block  —— ResidualCouplingBlock 的 flow 编排账（Coupling 与 Flip 交替）
  duration_nll    —— StochasticDurationPredictor 的 nll 与 logq 账

数值口径：能走精确有理数（Fraction）的部分一律走 Fraction；log / exp 本身是无理数，
按 float 计算。最终所有出参统一转成 float。
"""

import math
from fractions import Fraction


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------


def _to_frac(value):
    """把输入转成精确有理数 Fraction。"""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a valid numeric value")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError("unsupported numeric type: %r" % (type(value),))


def _as_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer, got %r" % (name, value))
    return value


def _flatten(nested):
    """把嵌套列表展平成一维列表。"""
    if not isinstance(nested, (list, tuple)):
        return [nested]
    out = []
    for item in nested:
        out.extend(_flatten(item))
    return out


def _prod(values):
    out = 1
    for v in values:
        out *= v
    return out


def _index(b, c, t, B, C, T):
    return (b * C + c) * T + t


def _f(value):
    """出参统一转 float。"""
    if isinstance(value, Fraction):
        return float(value)
    return float(value)


def _normalize_shape3(shape):
    """校验并归一化 (B, C, T) 形状。"""
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError("shape must be a list of 3 integers (B, C, T)")
    out = []
    for d in shape:
        d = _as_int(d, "shape entry")
        if d <= 0:
            raise ValueError("shape entries must be positive, got %r" % (d,))
        out.append(d)
    return out


def _normalize_tensor3(value, shape, name):
    """把嵌套列表转成展平的 (B, C, T) 有理数张量。"""
    B, C, T = shape
    total = _prod([B, C, T])
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a nested list" % name)
    flat = _flatten(value)
    if len(flat) != total:
        raise ValueError(
            "%s length %d does not match shape size %d" % (name, len(flat), total)
        )
    return [_to_frac(v) for v in flat]


def _normalize_mask(mask, shape):
    """归一化 mask。

    官方 x_mask 的形状是 (B, 1, T)，对 channel 维广播。
    本题接受 (B, 1, T) 或 (B, C, T)。
    """
    B, C, T = shape
    total = _prod([B, C, T])
    if not isinstance(mask, (list, tuple)):
        raise ValueError("mask must be a nested list")
    flat = _flatten(mask)
    if len(flat) == _prod([B, 1, T]):
        out = []
        for b in range(B):
            for c in range(C):
                for t in range(T):
                    out.append(_to_frac(flat[b * T + t]))
        return out
    if len(flat) == total:
        out = []
        for b in range(B):
            for c in range(C):
                base = (b * C + c) * T
                for t in range(T):
                    out.append(_to_frac(flat[base + t]))
        return out
    raise ValueError(
        "mask length %d matches neither (B,1,T)=%d nor (B,C,T)=%d"
        % (len(flat), _prod([B, 1, T]), total)
    )


def _normalize_affine_param(param, channels, name):
    """归一化 ElementwiseAffine 的 m / logs。

    官方是 nn.Parameter(torch.zeros(channels, 1))，即 (channels, 1)。
    本题接受标量 / 长度 channels 的一维列表 / (channels, 1) 嵌套列表。
    """
    if isinstance(param, (list, tuple)):
        flat = _flatten(param)
        if len(flat) != channels:
            raise ValueError(
                "%s must have %d entries (one per channel), got %d"
                % (name, channels, len(flat))
            )
        return [_to_frac(v) for v in flat]
    return [_to_frac(param)] * channels


# --------------------------------------------------------------------------
# mode 1: flow 基元
# --------------------------------------------------------------------------

_LOG_CLAMP = 1e-5


def flow_primitive(kind, shape, x, mask, m=None, logs=None, reverse=False):
    """单个 flow 基元的前向/反向与 log-det。

    kind:
      "log"    —— Log：y = log(clamp_min(x, 1e-5)) * mask；logdet = sum(-y)
      "flip"   —— Flip：y = flip(x, [1])（★ channel 轴）；logdet 恒 0
      "affine" —— ElementwiseAffine：y = (m + exp(logs) * x) * mask；
                   logdet = sum(logs * mask)

    reverse=True 时走官方的反向分支。
    """
    if kind not in ("log", "flip", "affine"):
        raise ValueError("unknown flow kind: %r" % (kind,))

    B, C, T = _normalize_shape3(shape)
    xf = _normalize_tensor3(x, (B, C, T), "x")
    mf = _normalize_mask(mask, (B, C, T))

    if kind == "flip":
        # 官方 Flip.forward 忽略 x_mask；forward 与 reverse 都是 flip（自逆）。
        out = [None] * len(xf)
        for b in range(B):
            for c in range(C):
                for t in range(T):
                    out[_index(b, c, t, B, C, T)] = xf[
                        _index(b, C - 1 - c, t, B, C, T)
                    ]
        return {"y": [_f(v) for v in out], "logdet": [0.0] * B}

    if kind == "affine":
        if m is None or logs is None:
            raise ValueError("affine flow requires m and logs")
        mlist = _normalize_affine_param(m, C, "m")
        logslist = _normalize_affine_param(logs, C, "logs")
        out = [None] * len(xf)
        if reverse:
            # 官方 reverse：x = (x - m) * torch.exp(-logs) * x_mask
            for b in range(B):
                for c in range(C):
                    scale = math.exp(-float(logslist[c]))
                    for t in range(T):
                        i = _index(b, c, t, B, C, T)
                        out[i] = float(
                            (xf[i] - mlist[c]) * Fraction(scale) * mf[i]
                        )
            return {"y": out, "logdet": [0.0] * B}
        logdet = []
        for b in range(B):
            acc = Fraction(0)
            for c in range(C):
                scale = math.exp(float(logslist[c]))
                for t in range(T):
                    i = _index(b, c, t, B, C, T)
                    out[i] = float(
                        (mlist[c] + Fraction(scale) * xf[i]) * mf[i]
                    )
                    # logdet = sum(logs * mask, [1, 2])
                    acc += logslist[c] * mf[i]
            logdet.append(_f(acc))
        return {"y": out, "logdet": logdet}

    # kind == "log"
    if reverse:
        # 官方 reverse：x = torch.exp(x) * x_mask
        out = [None] * len(xf)
        for i in range(len(xf)):
            out[i] = math.exp(float(xf[i])) * float(mf[i])
        return {"y": out, "logdet": [0.0] * B}
    out = [None] * len(xf)
    logdet = []
    for b in range(B):
        acc = 0.0
        for c in range(C):
            for t in range(T):
                i = _index(b, c, t, B, C, T)
                xv = float(xf[i])
                if xv < _LOG_CLAMP:
                    xv = _LOG_CLAMP
                y = math.log(xv) * float(mf[i])
                out[i] = y
                acc += -y
        logdet.append(acc)
    return {"y": out, "logdet": logdet}


# --------------------------------------------------------------------------
# mode 2: ResidualCouplingLayer
# --------------------------------------------------------------------------


def coupling_layer(shape, x, mask, m, logs=None, mean_only=True, reverse=False):
    """ResidualCouplingLayer 的前向/反向与 log-det。

    官方实现（modules.py）：
      x0, x1 = torch.split(x, [half_channels] * 2, 1)
      h = self.pre(x0) * x_mask ; h = self.enc(h, x_mask, g=g)
      stats = self.post(h) * x_mask
      if not mean_only: m, logs = torch.split(stats, [half] * 2, 1)
      else:             m = stats ; logs = torch.zeros_like(m)

      forward:  x1 = m + x1 * torch.exp(logs) * x_mask
                x = torch.cat([x0, x1], 1)
                logdet = torch.sum(logs, [1, 2])       # ★ 没有乘 mask
      reverse:  x1 = (x1 - m) * torch.exp(-logs) * x_mask
                x = torch.cat([x0, x1], 1)

    本题把 `pre/enc/post` 换成直接给定的 m（以及可选的 logs）。
    """
    B, C, T = _normalize_shape3(shape)
    if C % 2 != 0:
        # 官方 assert channels % 2 == 0
        raise ValueError("channels should be divisible by 2")
    half = C // 2
    xf = _normalize_tensor3(x, (B, C, T), "x")
    mf = _normalize_mask(mask, (B, C, T))

    if m is None:
        raise ValueError("coupling layer requires m")
    if not isinstance(m, (list, tuple)):
        raise ValueError("m must be a nested list")
    mflat = _flatten(m)
    if len(mflat) != _prod([B, half, T]):
        raise ValueError(
            "m length %d does not match (B, C/2, T)=%d"
            % (len(mflat), _prod([B, half, T]))
        )
    mlist = [_to_frac(v) for v in mflat]

    if mean_only:
        logslist = [Fraction(0)] * len(mlist)
    else:
        if logs is None:
            raise ValueError("coupling layer requires logs when mean_only=False")
        if not isinstance(logs, (list, tuple)):
            raise ValueError("logs must be a nested list")
        lflat = _flatten(logs)
        if len(lflat) != len(mlist):
            raise ValueError(
                "logs length %d does not match m length %d" % (len(lflat), len(mlist))
            )
        logslist = [_to_frac(v) for v in lflat]

    out = [None] * len(xf)
    logdet = []
    for b in range(B):
        acc = Fraction(0)
        for c in range(C):
            for t in range(T):
                i = _index(b, c, t, B, C, T)
                if c < half:
                    # x0 原样透传（官方不 mask x0）
                    out[i] = _f(xf[i])
                else:
                    j = b * half * T + (c - half) * T + t
                    if not reverse:
                        # ★ 官方是 `x1 = m + x1 * torch.exp(logs) * x_mask`：
                        #   mask 只作用在 x1 项上，m 不被 mask
                        #   → mask=0 的位置输出是 m（不是 0）
                        scale = math.exp(float(logslist[j]))
                        out[i] = float(
                            mlist[j] + Fraction(scale) * xf[i] * mf[i]
                        )
                    else:
                        scale = math.exp(-float(logslist[j]))
                        out[i] = float(
                            (xf[i] - mlist[j]) * Fraction(scale) * mf[i]
                        )
        if not mean_only:
            # logdet = torch.sum(logs, [1, 2])：★ 不乘 mask
            for j in range(b * half * T, b * half * T + half * T):
                acc += logslist[j]
        logdet.append(_f(acc))
    return {"y": out, "logdet": logdet}


# --------------------------------------------------------------------------
# mode 3: ResidualCouplingBlock
# --------------------------------------------------------------------------


def coupling_block(shape, x, mask, m_list, reverse=False):
    """ResidualCouplingBlock 的 flow 编排账。

    官方（models.py）：
      self.flows = []
      for i in range(n_flows):
        self.flows.append(ResidualCouplingLayer(..., mean_only=True))
        self.flows.append(modules.Flip())

      forward: for flow in self.flows:            x, _ = flow(x, x_mask, ...)
      reverse: for flow in reversed(self.flows):  x   = flow(x, x_mask, reverse=True)

    ★ 注意：
      1. 每个 Coupling 都是 mean_only=True → logs 恒 0 → logdet 恒 0；
      2. forward 按 flows 顺序，reverse 按逆序；
      3. Flip 翻的是 channel 轴 [1]；
      4. 官方 reverse 分支只返回 x（没有 logdet），本题仍返回累计 logdet（全 0）。
    """
    B, C, T = _normalize_shape3(shape)
    xf = _normalize_tensor3(x, (B, C, T), "x")
    mf = _normalize_mask(mask, (B, C, T))
    if not isinstance(m_list, (list, tuple)) or len(m_list) == 0:
        raise ValueError("m_list must have at least one entry")
    if C % 2 != 0:
        raise ValueError("channels should be divisible by 2")

    # 展开官方 flows 列表：[Coupling_0, Flip, Coupling_1, Flip, ...]
    steps = []
    for i in range(len(m_list)):
        steps.append(("coupling", i))
        steps.append(("flip", None))
    order = list(reversed(steps)) if reverse else steps

    cur = list(xf)
    logdet_total = [Fraction(0)] * B
    for kind, idx in order:
        if kind == "flip":
            cur = _flip_channels(cur, B, C, T)
            continue
        res = coupling_layer(
            (B, C, T), cur, mf, m_list[idx], mean_only=True, reverse=reverse
        )
        cur = res["y"]
        for b in range(B):
            logdet_total[b] += Fraction(res["logdet"][b])
    return {"y": [_f(v) for v in cur], "logdet": [_f(v) for v in logdet_total]}


def _flip_channels(x, B, C, T):
    """torch.flip(x, [1])：翻 channel 轴。"""
    out = [None] * len(x)
    for b in range(B):
        for c in range(C):
            for t in range(T):
                out[_index(b, c, t, B, C, T)] = x[
                    _index(b, C - 1 - c, t, B, C, T)
                ]
    return out


# --------------------------------------------------------------------------
# mode 4: StochasticDurationPredictor 的 nll / logq 账
# --------------------------------------------------------------------------


def duration_nll(w, e_q, z_u, z1, mask, post_logdets, flow_logdets):
    """StochasticDurationPredictor 的 nll 与 logq 账（官方 forward 的非 reverse 分支）。

    官方（models.py）：
      logdet_tot_q = 0
      e_q = randn(B,2,T) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1)
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += sum((logsigmoid(z_u) + logsigmoid(-z_u)) * x_mask, [1, 2])
      logq = sum(-0.5 * (log(2*pi) + e_q**2) * x_mask, [1, 2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)      # Log flow
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = sum(0.5 * (log(2*pi) + z**2) * x_mask, [1, 2]) - logdet_tot
      return nll + logq

    本题把随机量与 flow 的内部结构换成给定值：
      w            —— (B, 1, T) 原始时长
      e_q          —— (B, 2, T) 正态噪声（官方是 randn * x_mask）
      z_u, z1      —— (B, 1, T) post_flows 之后 z_q 的两半
      post_logdets —— post_flows 各步 logdet 之和（每 batch 一个）
      flow_logdets —— flows 各步 logdet 之和（每 batch 一个）
    """
    B, T = _infer_b1t(w)
    wf = _normalize_tensor3(w, (B, 1, T), "w")
    mf = _normalize_mask(mask, (B, 1, T))
    eq = _normalize_tensor3(e_q, (B, 2, T), "e_q")
    zu = _normalize_tensor3(z_u, (B, 1, T), "z_u")
    z1v = _normalize_tensor3(z1, (B, 1, T), "z1")
    if len(post_logdets) != B or len(flow_logdets) != B:
        raise ValueError("post_logdets and flow_logdets must have one entry per batch")

    half_log_2pi = 0.5 * math.log(2.0 * math.pi)

    # ---- logq 分支 ----
    # 官方：
    #   logdet_tot_q = post_flows 的 logdet 之和
    #   logdet_tot_q += sum((logsigmoid(z_u) + logsigmoid(-z_u)) * x_mask, [1,2])
    #   logq = sum(-0.5 * (log(2*pi) + e_q**2) * x_mask, [1,2]) - logdet_tot_q
    # ★ 注意 logsigmoid 项是【被减去的】（它本身是负数，所以 logq 反而变大）。
    logq = []
    for b in range(B):
        gauss = 0.0
        for c in range(2):
            for t in range(T):
                i = _index(b, c, t, B, 2, T)
                mi = _index(b, 0, t, B, 1, T)
                gauss += -0.5 * (2.0 * half_log_2pi + float(eq[i]) ** 2) * float(mf[mi])
        lsig = 0.0
        for t in range(T):
            mi = _index(b, 0, t, B, 1, T)
            zv = float(zu[mi])
            lsig += (_log_sigmoid(zv) + _log_sigmoid(-zv)) * float(mf[mi])
        logdet_tot_q = float(post_logdets[b]) + lsig
        logq.append(gauss - logdet_tot_q)

    # ---- nll 分支 ----
    # u = sigmoid(z_u) * x_mask ; z0 = (w - u) * x_mask
    z0 = []
    for b in range(B):
        for t in range(T):
            mi = _index(b, 0, t, B, 1, T)
            u = _sigmoid(float(zu[mi])) * float(mf[mi])
            z0.append((float(wf[mi]) - u) * float(mf[mi]))

    # log_flow：z0, logdet = Log(z0, x_mask)
    z0_log = []
    logdet_log = [0.0] * B
    for b in range(B):
        for t in range(T):
            mi = _index(b, 0, t, B, 1, T)
            xv = z0[b * T + t]
            if xv < _LOG_CLAMP:
                xv = _LOG_CLAMP
            y = math.log(xv) * float(mf[mi])
            z0_log.append(y)
            logdet_log[b] += -y

    logdet_tot = [0.0] * B
    for b in range(B):
        logdet_tot[b] += logdet_log[b] + float(flow_logdets[b])

    # z = cat([z0_log, z1])，z1 用给定值
    nll = []
    for b in range(B):
        acc = 0.0
        for t in range(T):
            mi = _index(b, 0, t, B, 1, T)
            zi = z0_log[b * T + t]
            acc += 0.5 * (2.0 * half_log_2pi + zi ** 2) * float(mf[mi])
            zi1 = float(z1v[b * T + t])
            acc += 0.5 * (2.0 * half_log_2pi + zi1 ** 2) * float(mf[mi])
        nll.append(acc - logdet_tot[b])

    total = [nll[b] + logq[b] for b in range(B)]
    return {
        "logq": logq,
        "nll": nll,
        "total": total,
        "logdet_tot_q": [float(v) for v in post_logdets],
        "logdet_tot": logdet_tot,
        "z0": z0,
        "z0_log": z0_log,
    }


def _infer_b1t(w):
    """从嵌套列表推断 (B, T)（w 的形状是 (B, 1, T)）。"""
    if not isinstance(w, (list, tuple)) or len(w) == 0:
        raise ValueError("w must be a non-empty nested list")
    b = len(w)
    first = w[0]
    if not isinstance(first, (list, tuple)) or len(first) != 1:
        raise ValueError("w must have shape (B, 1, T)")
    inner = first[0]
    t = len(inner) if isinstance(inner, (list, tuple)) else 1
    for row in w:
        if len(row) != 1:
            raise ValueError("w must have shape (B, 1, T)")
        inner = row[0]
        n = len(inner) if isinstance(inner, (list, tuple)) else 1
        if n != t:
            raise ValueError("w must have shape (B, 1, T)")
    return b, t


# --------------------------------------------------------------------------
# 数值辅助
# --------------------------------------------------------------------------


def _sigmoid(v):
    if v >= 0.0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def _log_sigmoid(v):
    # 数值稳定写法：log(sigmoid(v))
    if v >= 0.0:
        return -math.log1p(math.exp(-v))
    return v - math.log1p(math.exp(v))


# --------------------------------------------------------------------------
# 单入口
# --------------------------------------------------------------------------


def solve(mode, **kw):
    if mode == "flow_primitive":
        return flow_primitive(
            kw["kind"], kw["shape"], kw["x"], kw["mask"],
            kw.get("m"), kw.get("logs"), bool(kw.get("reverse", False)),
        )
    if mode == "coupling_layer":
        return coupling_layer(
            kw["shape"], kw["x"], kw["mask"], kw.get("m"), kw.get("logs"),
            bool(kw.get("mean_only", True)), bool(kw.get("reverse", False)),
        )
    if mode == "coupling_block":
        return coupling_block(
            kw["shape"], kw["x"], kw["mask"], kw["m_list"],
            bool(kw.get("reverse", False)),
        )
    if mode == "duration_nll":
        return duration_nll(
            kw["w"], kw["e_q"], kw["z_u"], kw["z1"], kw["mask"],
            kw["post_logdets"], kw["flow_logdets"],
        )
    raise ValueError("unknown mode: %r" % (mode,))
