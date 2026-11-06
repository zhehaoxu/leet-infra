"""xLSTM mLSTM stabilized gating ledger（order 837）。

回源：NX-AI/xlstm (main)
  xlstm/blocks/mlstm/backends.py
    - recurrent_step_stabilized_simple  -> mode stabilized_step
    - parallel_stabilized_simple        -> mode parallel_decay
    - chunkwise_simple (chunk 内门控段) -> mode chunk_intra_gates

判题约定：单入口 solve(mode, **kw)，3 个 mode，全部纯 Python 数值账：
  stabilized_step    m_state 的 log-sum-exp 稳定化 + 双门激活 + C/n/m 更新 + h_denom 下界
  parallel_decay     cumsum 差分构造 log_fg_matrix / log_D_matrix / rowwise 稳定化 / normalizer
  chunk_intra_gates  chunk 内 loggates / m_loc / kv / ksum（不判 chunk 间递推与三形式恒等）

不进判题域：nn.Linear 投影、MultiHeadLayerNorm、真模型前向、Triton kernel、
chunkwise 的 chunk 间状态与 intra+inter 合成、与 recurrent 的形式间恒等（撞 la-gla-chunked）、
delta rule 矩阵更新（撞 la-gated-deltanet-recurrent）。q/k/v/i/f 全作显式入参。
"""

import math

NEG_INF = float("-inf")


def _logsigmoid(x):
    """log(sigmoid(x)) = -softplus(-x)，分段避免溢出。"""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


# ---------------------------------------------------------------- modes
def _stabilized_step(q, k, v, i, f, C_prev, n_prev, m_prev, eps):
    DH = len(q)
    log_fg_act = _logsigmoid(f)
    m_new = max(log_fg_act + m_prev, i)
    fg_act = math.exp(log_fg_act + m_prev - m_new)
    ig_act = math.exp(i - m_new)
    root = math.sqrt(DH)
    k_scaled = [x / root for x in k]

    C_new = [[fg_act * C_prev[a][b] + ig_act * k_scaled[a] * v[b]
              for b in range(DH)] for a in range(DH)]
    n_new = [fg_act * n_prev[a] + ig_act * k_scaled[a] for a in range(DH)]

    h_num = [sum(q[a] * C_new[a][b] for a in range(DH)) for b in range(DH)]
    qn = sum(q[a] * n_new[a] for a in range(DH))
    max_val = math.exp(-m_new)
    h_denom = max(abs(qn), max_val) + eps
    h = [x / h_denom for x in h_num]

    return {"log_fg_act": log_fg_act, "m_new": m_new,
            "fg_act": fg_act, "ig_act": ig_act,
            "C_new": C_new, "n_new": n_new,
            "h_num": h_num, "qn": qn, "max_val": max_val,
            "h_denom": h_denom, "h": h}


def _parallel_decay(q, k, v, i, f, rowwise, eps):
    S = len(i)
    DH = len(q[0])
    log_fg = [_logsigmoid(x) for x in f]

    # (S+1,) 前缀和，首元素恒 0
    cumsum = [0.0]
    for x in log_fg:
        cumsum.append(cumsum[-1] + x)

    # [1:,1:] 偏移后的差分矩阵 + 因果掩码（不含 j 自身，含 t）
    log_fg_matrix = []
    for t in range(S):
        row = []
        for j in range(S):
            if j <= t:
                row.append(cumsum[t + 1] - cumsum[j + 1])
            else:
                row.append(NEG_INF)
        log_fg_matrix.append(row)

    # igate_preact 转置后按列广播：每个 key 位置 j 加 i[j]
    log_D = [[log_fg_matrix[t][j] + i[j] for j in range(S)] for t in range(S)]
    if rowwise:
        max_log_D = [max(row) for row in log_D]
    else:
        g = max(max(row) for row in log_D)
        max_log_D = [g] * S

    D = []
    for t in range(S):
        row = []
        for j in range(S):
            d = log_D[t][j] - max_log_D[t]
            row.append(math.exp(d) if d != NEG_INF else 0.0)
        D.append(row)

    root = math.sqrt(DH)
    ks = [[x / root for x in row] for row in k]
    C = [[sum(q[t][d] * ks[j][d] for d in range(DH)) * D[t][j]
          for j in range(S)] for t in range(S)]

    normalizer = [max(abs(sum(row)), math.exp(-max_log_D[t]))
                  for t, row in enumerate(C)]
    h_tilde = [[sum(C[t][j] / (normalizer[t] + eps) * v[j][d]
                    for j in range(S)) for d in range(DH)]
               for t in range(S)]

    return {"log_fgates_cumsum": cumsum, "log_fg_matrix": log_fg_matrix,
            "log_D_matrix": log_D, "max_log_D": max_log_D,
            "D_matrix": D, "C_matrix": C,
            "normalizer": normalizer, "h_tilde": h_tilde}


def _chunk_intra(q, k, v, i, f, CS):
    S = len(i)
    DH = len(q[0])
    NS = S // CS
    log_fg = [_logsigmoid(x) for x in f]

    acc_all, lg_all, mloc_all, kv_all, ksum_all = [], [], [], [], []
    for n in range(NS):
        sl = slice(n * CS, (n + 1) * CS)
        lf = log_fg[sl]
        acc = []
        run = 0.0
        for x in lf:
            run += x
            acc.append(run)
        ii = i[sl]
        # loggates 基值 = i_t - acc_t + acc_last，再减 chunk 级 m_loc
        base = [ii[t] - acc[t] + acc[-1] for t in range(CS)]
        m_loc = max(base)
        lg = [x - m_loc for x in base]
        e = [math.exp(x) for x in lg]

        kk, vv = k[sl], v[sl]
        kv = [[sum(kk[t][a] * vv[t][b] * e[t] for t in range(CS))
               for b in range(DH)] for a in range(DH)]
        ksum = [sum(kk[t][a] * e[t] for t in range(CS)) for a in range(DH)]

        acc_all.append(acc)
        lg_all.append(lg)
        mloc_all.append(m_loc)
        kv_all.append(kv)
        ksum_all.append(ksum)

    return {"log_fgates_acc": acc_all, "loggates": lg_all,
            "m_loc": mloc_all, "kv": kv_all, "ksum": ksum_all}


def solve(mode: str, **kw):
    if mode == "stabilized_step":
        return _stabilized_step(
            list(kw["q"]), list(kw["k"]), list(kw["v"]),
            kw["i"], kw["f"],
            [list(r) for r in kw["C_prev"]], list(kw["n_prev"]),
            kw.get("m_prev", 0.0), kw.get("eps", 1e-6))
    if mode == "parallel_decay":
        return _parallel_decay(
            [list(r) for r in kw["q"]], [list(r) for r in kw["k"]],
            [list(r) for r in kw["v"]], list(kw["i"]), list(kw["f"]),
            kw.get("stabilize_rowwise", True), kw.get("eps", 1e-6))
    if mode == "chunk_intra_gates":
        return _chunk_intra(
            [list(r) for r in kw["q"]], [list(r) for r in kw["k"]],
            [list(r) for r in kw["v"]], list(kw["i"]), list(kw["f"]),
            kw.get("chunk_size", 2))
    raise ValueError("unknown mode: %s" % mode)
