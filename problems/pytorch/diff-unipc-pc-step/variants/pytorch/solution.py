import torch

_ALIAS_TO_BH2 = ("midpoint", "heun", "logrho")


def _tensor(name, value, shape=None, allow_none=False):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} 不能为 None")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() < 2:
        raise ValueError(f"{name} 至少是二维张量（第 0 维为 batch）")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} 含 NaN 或 inf")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} 形状应为 {tuple(shape)}，实际 {tuple(value.shape)}")
    return value.to(torch.float64)


def _int(name, value, low=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 Python int")
    if value < low:
        raise ValueError(f"{name} 不能小于 {low}")
    return value


def _bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是 bool")
    return value


def _solver_type(value):
    if not isinstance(value, str):
        raise ValueError("solver_type 必须是字符串")
    if value in ("bh1", "bh2"):
        return value
    if value in _ALIAS_TO_BH2:
        return "bh2"
    raise ValueError(f"solver_type {value} 未实现")


def _alpha_sigma(sig, flow):
    if flow:
        return 1.0 - sig, sig
    alpha = 1.0 / torch.sqrt(sig * sig + 1.0)
    return alpha, sig * alpha


def _solve(rows, rhs):
    return torch.linalg.solve(rows, rhs)


def _basis(hh, b_h, rks, order):
    """R 与 b：上游 for i in range(1, order+1) 的逐字转写，含 factorial 的更新次序。"""
    phi_1 = torch.expm1(hh)
    phi_k = phi_1 / hh - 1.0
    factorial_i = 1.0
    rows = []
    b = []
    for i in range(1, order + 1):
        rows.append(torch.pow(rks, i - 1))
        b.append(phi_k * factorial_i / b_h)
        factorial_i *= i + 1
        phi_k = phi_k / hh - 1.0 / factorial_i
    return torch.stack(rows), torch.stack(b), phi_1


def solve(m_hist, m_t, x_last, x_this, sigmas, step_index, order_c=1, order_p=1,
          predict_x0=True, solver_type="bh2", use_flow_sigmas=False):
    """UniPC 的 B(h) 预测-校正单步：先用当前输出校正上一步的预测，再外推一步。"""
    mt = _tensor("m_t", m_t)
    shape = tuple(mt.shape)
    if not isinstance(m_hist, (tuple, list)) or len(m_hist) == 0:
        raise ValueError("m_hist 必须是非空的张量序列")
    hist = [_tensor(f"m_hist[{i}]", item, shape) for i, item in enumerate(m_hist)]
    x_this_t = _tensor("x_this", x_this, shape)
    x_last_t = _tensor("x_last", x_last, shape, allow_none=True)

    if not isinstance(sigmas, torch.Tensor):
        raise ValueError("sigmas 必须是 torch.Tensor")
    if sigmas.dim() != 1:
        raise ValueError("sigmas 必须是一维张量")
    if not torch.isfinite(sigmas).all():
        raise ValueError("sigmas 含 NaN 或 inf")
    sig = sigmas.to(torch.float64)
    if bool((sig <= 0).any()):
        raise ValueError("sigmas 必须全部为正：终端 σ=0 的那一步不在本题范围内")

    step = _int("step_index", step_index)
    oc = _int("order_c", order_c, low=1)
    op = _int("order_p", order_p, low=1)
    flow = _bool("use_flow_sigmas", use_flow_sigmas)
    px0 = _bool("predict_x0", predict_x0)
    bh_name = _solver_type(solver_type)

    if step + 1 >= len(sig):
        raise ValueError("sigmas 长度不足，step_index + 1 越界")
    if x_last_t is not None:
        if step < oc:
            raise ValueError("启用校正时需要 step_index >= order_c")
        if len(hist) < oc:
            raise ValueError("m_hist 长度必须 >= order_c")
    if step < op - 1:
        raise ValueError("step_index 太小，预测侧的历史索引会越界")
    if len(hist) < op - 1:
        raise ValueError("m_hist 长度必须 >= order_p - 1")

    alpha, sigma = _alpha_sigma(sig, flow)
    lam = torch.log(alpha) - torch.log(sigma)
    if not bool(torch.isfinite(lam).all()):
        raise ValueError("该 σ 序列下 log-SNR 不有限：use_flow_sigmas 为真时 σ 必须落在 (0, 1) 内")

    x = x_this_t
    corrected = None

    if x_last_t is not None:
        # 校正侧：t = step_index，s0 = step_index - 1，m0 = 上一步的转换输出
        a_t, s_t = alpha[step], sigma[step]
        a_s0, s_s0 = alpha[step - 1], sigma[step - 1]
        h = lam[step] - lam[step - 1]
        hh = -h if px0 else h
        b_h = hh if bh_name == "bh1" else torch.expm1(hh)
        m0 = hist[-1]
        rks, d1s = [], []
        for i in range(1, oc):
            si = step - (i + 1)
            rk = (lam[si] - lam[step - 1]) / h
            rks.append(rk)
            d1s.append((hist[-(i + 1)] - m0) / rk)
        rks.append(torch.ones((), dtype=torch.float64))
        rks = torch.stack(rks)
        R, b, phi_1 = _basis(hh, b_h, rks, oc)
        if oc == 1:
            rhos = torch.full((1,), 0.5, dtype=torch.float64)
        else:
            rhos = _solve(R, b)
        acc = torch.zeros_like(m0)
        if d1s:
            d1s_t = torch.stack(d1s, dim=1)
            acc = torch.einsum("k,bkc...->bc...", rhos[:-1], d1s_t)
        acc = acc + rhos[-1] * (mt - m0)
        if px0:
            x = s_t / s_s0 * x_last_t - a_t * phi_1 * m0 - a_t * b_h * acc
        else:
            x = a_t / a_s0 * x_last_t - s_t * phi_1 * m0 - s_t * b_h * acc
        corrected = x

    # 预测侧：s0 = step_index，t = step_index + 1，m0 = 当前步的转换输出
    a_t, s_t = alpha[step + 1], sigma[step + 1]
    a_s0, s_s0 = alpha[step], sigma[step]
    h = lam[step + 1] - lam[step]
    hh = -h if px0 else h
    b_h = hh if bh_name == "bh1" else torch.expm1(hh)
    m0 = mt
    rks, d1s = [], []
    for i in range(1, op):
        si = step - i
        rk = (lam[si] - lam[step]) / h
        rks.append(rk)
        d1s.append((hist[-i] - m0) / rk)
    rks.append(torch.ones((), dtype=torch.float64))
    rks = torch.stack(rks)
    R, b, phi_1 = _basis(hh, b_h, rks, op)
    acc = torch.zeros_like(m0)
    if d1s:
        d1s_t = torch.stack(d1s, dim=1)
        if op == 2:
            rhos = torch.full((1,), 0.5, dtype=torch.float64)
        else:
            rhos = _solve(R[:-1, :-1], b[:-1])
        acc = torch.einsum("k,bkc...->bc...", rhos, d1s_t)
    if px0:
        prev = s_t / s_s0 * x - a_t * phi_1 * m0 - a_t * b_h * acc
    else:
        prev = a_t / a_s0 * x - s_t * phi_1 * m0 - s_t * b_h * acc

    if corrected is None:
        corrected = x_this_t
    return corrected.to(torch.float32), prev.to(torch.float32)
