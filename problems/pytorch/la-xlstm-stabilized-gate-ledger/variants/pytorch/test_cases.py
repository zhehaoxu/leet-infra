"""order 837 判题用例：独立 oracle 用纯 Python 重写，不 import solution。

3 个 check_*：stabilized_step / parallel_decay / chunk_intra_gates。
每个都做网格对拍 + 手算锚点（锚点独立于 oracle 与 solution，钉死所有严格分支、
默认值、cumsum [1:,1:] 偏移、-inf 变 0、stabilize_rowwise 两路、chunk 级 m_loc）。
"""

import math
import random

NEG = float("-inf")
L2 = math.log(2.0)


# ---------------------------------------------------------------- 独立 oracle
def ref_ls(x):
    """log(sigmoid(x)) = -softplus(-x)，分段避免溢出。"""
    if x < 0:
        return x - math.log1p(math.exp(x))
    return -math.log1p(math.exp(-x))


def ref_step(q, k, v, i, f, C_prev, n_prev, m_prev, eps):
    d = len(q)
    lf = ref_ls(f)
    ml = lf + m_prev
    m = ml if ml >= i else i
    fg = math.exp(ml - m)
    ig = math.exp(i - m)
    rt = math.sqrt(d)

    C = []
    for a in range(d):
        row = []
        for b in range(d):
            row.append(fg * C_prev[a][b] + ig * (k[a] / rt) * v[b])
        C.append(row)
    nn = [fg * n_prev[a] + ig * (k[a] / rt) for a in range(d)]

    hnum = []
    for b in range(d):
        acc = 0.0
        for a in range(d):
            acc += q[a] * C[a][b]
        hnum.append(acc)
    qn = 0.0
    for a in range(d):
        qn += q[a] * nn[a]
    mv = math.exp(-m)
    den = (abs(qn) if abs(qn) >= mv else mv) + eps
    h = [x / den for x in hnum]
    return {"log_fg_act": lf, "m_new": m, "fg_act": fg, "ig_act": ig,
            "C_new": C, "n_new": nn, "h_num": hnum, "qn": qn,
            "max_val": mv, "h_denom": den, "h": h}


def ref_parallel(q, k, v, i, f, rowwise, eps):
    S = len(i)
    d = len(q[0])
    lf = [ref_ls(x) for x in f]
    cs = [0.0]
    for x in lf:
        cs.append(cs[-1] + x)

    lfm = []
    for t in range(S):
        row = []
        for j in range(S):
            row.append(cs[t + 1] - cs[j + 1] if j <= t else NEG)
        lfm.append(row)
    lgd = [[lfm[t][j] + i[j] for j in range(S)] for t in range(S)]
    if rowwise:
        mx = [max(row) for row in lgd]
    else:
        g = max(max(row) for row in lgd)
        mx = [g] * S
    D = []
    for t in range(S):
        row = []
        for j in range(S):
            row.append(math.exp(lgd[t][j] - mx[t])
                       if lgd[t][j] != NEG else 0.0)
        D.append(row)
    rt = math.sqrt(d)
    ks = [[x / rt for x in row] for row in k]
    C = []
    for t in range(S):
        row = []
        for j in range(S):
            acc = 0.0
            for dd in range(d):
                acc += q[t][dd] * ks[j][dd]
            row.append(acc * D[t][j])
        C.append(row)
    norm = []
    for t in range(S):
        sm = 0.0
        for x in C[t]:
            sm += x
        lo = math.exp(-mx[t])
        norm.append(abs(sm) if abs(sm) >= lo else lo)
    ht = []
    for t in range(S):
        row = []
        for dd in range(d):
            acc = 0.0
            for j in range(S):
                acc += C[t][j] / (norm[t] + eps) * v[j][dd]
            row.append(acc)
        ht.append(row)
    return {"log_fgates_cumsum": cs, "log_fg_matrix": lfm,
            "log_D_matrix": lgd, "max_log_D": mx, "D_matrix": D,
            "C_matrix": C, "normalizer": norm, "h_tilde": ht}


def ref_chunk(q, k, v, i, f, CS):
    S = len(i)
    d = len(q[0])
    NS = S // CS
    lf = [ref_ls(x) for x in f]
    acc_all, lg_all, mloc_all, kv_all, ksum_all = [], [], [], [], []
    for n in range(NS):
        lo = n * CS
        acc = []
        run = 0.0
        for t in range(CS):
            run += lf[lo + t]
            acc.append(run)
        base = [i[lo + t] - acc[t] + acc[-1] for t in range(CS)]
        mloc = max(base)
        lg = [base[t] - mloc for t in range(CS)]
        e = [math.exp(lg[t]) for t in range(CS)]
        kv = []
        for a in range(d):
            row = []
            for b in range(d):
                s = 0.0
                for t in range(CS):
                    s += k[lo + t][a] * v[lo + t][b] * e[t]
                row.append(s)
            kv.append(row)
        ksum = []
        for a in range(d):
            s = 0.0
            for t in range(CS):
                s += k[lo + t][a] * e[t]
            ksum.append(s)
        acc_all.append(acc)
        lg_all.append(lg)
        mloc_all.append(mloc)
        kv_all.append(kv)
        ksum_all.append(ksum)
    return {"log_fgates_acc": acc_all, "loggates": lg_all,
            "m_loc": mloc_all, "kv": kv_all, "ksum": ksum_all}


# ---------------------------------------------------------------- 比较工具
def _close(actual, expect, atol=1e-10, rtol=1e-10, tag=""):
    def flat(x):
        if isinstance(x, (list, tuple)):
            out = []
            for it in x:
                out.extend(flat(it))
            return out
        return [x]

    fa, fe = flat(actual), flat(expect)
    assert len(fa) == len(fe), "%s length %d vs %d" % (tag, len(fa), len(fe))
    for a, e in zip(fa, fe):
        if a == e:                      # 同时处理 -inf == -inf
            continue
        assert abs(float(a) - float(e)) <= atol + rtol * abs(float(e)), \
            "%s %r vs %r" % (tag, a, e)


def _cmp(got, exp, tag):
    assert set(got.keys()) == set(exp.keys()), \
        "%s keys %r vs %r" % (tag, sorted(got), sorted(exp))
    for k in exp:
        _close(got[k], exp[k], tag=tag + "." + k)


VALS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
GATES = [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0]
rng = random.Random(8377)


def rvec(d):
    return [rng.choice(VALS) for _ in range(d)]


def rmat(r, c):
    return [[rng.choice(VALS) for _ in range(c)] for _ in range(r)]


# ---------------------------------------------------------------- checks
def check_stabilized_step(solve):
    n = 0
    for _ in range(160):
        d = rng.choice([1, 2, 3])
        q, k, v = rvec(d), rvec(d), rvec(d)
        i, f = rng.choice(GATES), rng.choice(GATES)
        C_prev, n_prev = rmat(d, d), rvec(d)
        m_prev = rng.choice([-2.0, -0.5, 0.0, 1.0, 3.0])
        eps = rng.choice([1e-6, 1e-3, 0.0])
        got = solve(mode="stabilized_step", q=q, k=k, v=v, i=i, f=f,
                    C_prev=C_prev, n_prev=n_prev, m_prev=m_prev, eps=eps)
        exp = ref_step(q, k, v, i, f, C_prev, n_prev, m_prev, eps)
        _cmp(got, exp, "step.grid%d" % n)
        n += 1

    # 锚点 1：不传 m_prev / eps（默认值 0.0 / 1e-6 生效），钉 max 第二分支、
    # C_prev 非对称（h_num 按列取）、n_new / qn / h_denom 的 +eps。
    r1 = solve(mode="stabilized_step", q=[1.0, 0.0], k=[0.0, 0.0],
               v=[1.0, 2.0], i=0.0, f=0.0,
               C_prev=[[1.0, 2.0], [3.0, 4.0]], n_prev=[0.5, 0.25])
    _cmp(r1, {"log_fg_act": -L2, "m_new": 0.0, "fg_act": 0.5,
              "ig_act": 1.0, "C_new": [[0.5, 1.0], [1.5, 2.0]],
              "n_new": [0.25, 0.125], "h_num": [0.5, 1.0], "qn": 0.25,
              "max_val": 1.0, "h_denom": 1.0 + 1e-6,
              "h": [0.5 / (1.0 + 1e-6), 1.0 / (1.0 + 1e-6)]}, "step.a1")

    # 锚点 2：sqrt(DH) 缩放、k_scaled @ v.T 外积方向、h_num 按列、
    # h_denom 的 max(|qn|, max_val) 由 max_val 胜出；eps=0 显式传入。
    r2 = math.sqrt(2.0)
    r2got = solve(mode="stabilized_step", q=[1.0, 0.0], k=[1.0, 2.0],
                  v=[3.0, 4.0], i=0.0, f=0.0,
                  C_prev=[[0.0, 0.0], [0.0, 0.0]], n_prev=[0.0, 0.0],
                  m_prev=0.0, eps=0.0)
    _cmp(r2got, {"log_fg_act": -L2, "m_new": 0.0, "fg_act": 0.5,
                 "ig_act": 1.0,
                 "C_new": [[3.0 / r2, 4.0 / r2], [6.0 / r2, 8.0 / r2]],
                 "n_new": [1.0 / r2, 2.0 / r2],
                 "h_num": [3.0 / r2, 4.0 / r2], "qn": 1.0 / r2,
                 "max_val": 1.0, "h_denom": 1.0,
                 "h": [3.0 / r2, 4.0 / r2]}, "step.a2")

    # 锚点 3：m_new 取第一分支（log_fg_act+m_prev > i），且 |qn| > max_val，
    # 钉 max 不是 min、max_val = exp(-m_new)。
    r3 = solve(mode="stabilized_step", q=[1.0, 0.0], k=[0.0, 0.0],
               v=[0.0, 0.0], i=-2.0, f=0.0,
               C_prev=[[3.0, 0.0], [0.0, 1.0]], n_prev=[3.0, 0.0],
               m_prev=0.0, eps=0.0)
    _cmp(r3, {"log_fg_act": -L2, "m_new": -L2, "fg_act": 1.0,
              "ig_act": math.exp(L2 - 2.0),
              "C_new": [[3.0, 0.0], [0.0, 1.0]], "n_new": [3.0, 0.0],
              "h_num": [3.0, 0.0], "qn": 3.0, "max_val": math.exp(L2),
              "h_denom": 3.0, "h": [1.0, 0.0]}, "step.a3")

    # 锚点 4：m_prev 非零，钉 fg_act / ig_act 的 m_prev 与 i 的相对位置。
    r4 = solve(mode="stabilized_step", q=[1.0], k=[2.0], v=[3.0],
               i=1.0, f=0.0, C_prev=[[1.0]], n_prev=[1.0],
               m_prev=2.0, eps=0.0)
    lf4 = -L2
    m4 = 2.0 + lf4            # = 1.306... > i=1.0
    _cmp(r4, {"log_fg_act": lf4, "m_new": m4,
              "fg_act": math.exp(lf4 + 2.0 - m4),
              "ig_act": math.exp(1.0 - m4),
              "C_new": [[math.exp(lf4 + 2.0 - m4) * 1.0
                         + math.exp(1.0 - m4) * 2.0 * 3.0]],
              "n_new": [math.exp(lf4 + 2.0 - m4) * 1.0
                        + math.exp(1.0 - m4) * 2.0],
              "h_num": [math.exp(lf4 + 2.0 - m4) * 1.0
                        + math.exp(1.0 - m4) * 2.0 * 3.0],
              "qn": math.exp(lf4 + 2.0 - m4) * 1.0
                    + math.exp(1.0 - m4) * 2.0,
              "max_val": math.exp(-m4),
              "h_denom": max(abs(math.exp(lf4 + 2.0 - m4) * 1.0
                                 + math.exp(1.0 - m4) * 2.0),
                             math.exp(-m4)),
              "h": [(math.exp(lf4 + 2.0 - m4) * 1.0
                     + math.exp(1.0 - m4) * 2.0 * 3.0)
                    / max(abs(math.exp(lf4 + 2.0 - m4) * 1.0
                              + math.exp(1.0 - m4) * 2.0),
                          math.exp(-m4))]}, "step.a4")
    print("stabilized_step: %d grid + 4 anchors OK" % n)


def check_parallel_decay(solve):
    n = 0
    for _ in range(160):
        S = rng.choice([1, 2, 3, 4])
        d = rng.choice([1, 2, 3])
        q, k, v = rmat(S, d), rmat(S, d), rmat(S, d)
        iv = [rng.choice(GATES) for _ in range(S)]
        fv = [rng.choice(GATES + [1000.0]) for _ in range(S)]
        rowwise = rng.choice([True, False])
        eps = rng.choice([1e-6, 1e-3, 0.0])
        got = solve(mode="parallel_decay", q=q, k=k, v=v, i=iv, f=fv,
                    stabilize_rowwise=rowwise, eps=eps)
        exp = ref_parallel(q, k, v, iv, fv, rowwise, eps)
        _cmp(got, exp, "par.grid%d" % n)
        n += 1
    # 显式补：C 行和为负且 |sum| 远大于 exp(-max_log_D)，钉 normalizer 的 abs()
    for rowwise in (True, False):
        got = solve(mode="parallel_decay",
                    q=[[2.0], [2.0], [2.0]], k=[[-2.0], [-2.0], [-2.0]],
                    v=[[1.0], [1.0], [1.0]], i=[5.0, 5.0, 5.0],
                    f=[0.0, 0.0, 0.0], stabilize_rowwise=rowwise, eps=1e-6)
        exp = ref_parallel([[2.0], [2.0], [2.0]],
                           [[-2.0], [-2.0], [-2.0]], [[1.0], [1.0], [1.0]],
                           [5.0, 5.0, 5.0], [0.0, 0.0, 0.0], rowwise, 1e-6)
        _cmp(got, exp, "par.neg")
        n += 1

    # 锚点 1：不传 stabilize_rowwise / eps（默认 True / 1e-6），S=2、DH=1。
    # 钉 cumsum 首 0 + [1:,1:] 偏移、-inf 变 0、igate 按列广播、rowwise max、
    # normalizer 下界 exp(-max_log_D) 胜过 |C.sum|、eps 加在分母外。
    r1 = solve(mode="parallel_decay", q=[[0.25], [1.0]],
               k=[[1.0], [1.0]], v=[[2.0], [3.0]],
               i=[0.0, 0.0], f=[0.0, 0.0])
    _cmp(r1, {"log_fgates_cumsum": [0.0, -L2, -2.0 * L2],
              "log_fg_matrix": [[0.0, NEG], [-L2, 0.0]],
              "log_D_matrix": [[0.0, NEG], [-L2, 0.0]],
              "max_log_D": [0.0, 0.0],
              "D_matrix": [[1.0, 0.0], [0.5, 1.0]],
              "C_matrix": [[0.25, 0.0], [0.5, 1.0]],
              "normalizer": [1.0, 1.5],
              "h_tilde": [[0.5 / (1.0 + 1e-6)],
                          [4.0 / (1.5 + 1e-6)]]}, "par.a1")

    # 锚点 2：默认 rowwise=True 且两行 max 不同（f=1000 使 log_fg≈0），
    # 钉逐行 max、igate 列广播（i=[0,3]）、eps 显式 1e-3。
    e3 = math.exp(-3.0)
    r2 = solve(mode="parallel_decay", q=[[1.0], [1.0]],
               k=[[1.0], [1.0]], v=[[2.0], [3.0]],
               i=[0.0, 3.0], f=[1000.0, 1000.0], eps=1e-3)
    _cmp(r2, {"log_fgates_cumsum": [0.0, 0.0, 0.0],
              "log_fg_matrix": [[0.0, NEG], [0.0, 0.0]],
              "log_D_matrix": [[0.0, NEG], [0.0, 3.0]],
              "max_log_D": [0.0, 3.0],
              "D_matrix": [[1.0, 0.0], [e3, 1.0]],
              "C_matrix": [[1.0, 0.0], [e3, 1.0]],
              "normalizer": [1.0, 1.0 + e3],
              "h_tilde": [[2.0 / (1.0 + 1e-3)],
                          [(2.0 * e3 + 3.0) / (1.0 + e3 + 1e-3)]]},
         "par.a2")

    # 锚点 3：显式 rowwise=False（全局 max），与锚点 2 同输入，
    # 钉「全局 max 广播到所有行」这一路（h_tilde 因 eps>0 不再逐行缩放不变）。
    r3 = solve(mode="parallel_decay", q=[[1.0], [1.0]],
               k=[[1.0], [1.0]], v=[[2.0], [3.0]],
               i=[0.0, 3.0], f=[1000.0, 1000.0],
               stabilize_rowwise=False, eps=1e-3)
    _cmp(r3, {"log_fgates_cumsum": [0.0, 0.0, 0.0],
              "log_fg_matrix": [[0.0, NEG], [0.0, 0.0]],
              "log_D_matrix": [[0.0, NEG], [0.0, 3.0]],
              "max_log_D": [3.0, 3.0],
              "D_matrix": [[e3, 0.0], [e3, 1.0]],
              "C_matrix": [[e3, 0.0], [e3, 1.0]],
              "normalizer": [e3, 1.0 + e3],
              "h_tilde": [[2.0 * e3 / (e3 + 1e-3)],
                          [(2.0 * e3 + 3.0) / (1.0 + e3 + 1e-3)]]},
         "par.a3")
    print("parallel_decay: %d grid + 3 anchors OK" % n)


def check_chunk_intra_gates(solve):
    n = 0
    for _ in range(160):
        CS = rng.choice([1, 2, 3])
        NS = rng.choice([1, 2, 3])
        S = CS * NS
        d = rng.choice([1, 2, 3])
        q, k, v = rmat(S, d), rmat(S, d), rmat(S, d)
        iv = [rng.choice(GATES) for _ in range(S)]
        fv = [rng.choice(GATES + [1000.0]) for _ in range(S)]
        got = solve(mode="chunk_intra_gates", q=q, k=k, v=v, i=iv, f=fv,
                    chunk_size=CS)
        exp = ref_chunk(q, k, v, iv, fv, CS)
        _cmp(got, exp, "chunk.grid%d" % n)
        n += 1

    # 锚点 1：不传 chunk_size（默认 2），S=4 → NS=2；钉 chunk 内 cumsum
    # 每个 chunk 重新起算、+acc[-1] 只改 m_loc、m_loc=max(base)、
    # loggates = base - m_loc、kv / ksum 用 exp(loggates) 加权。
    r1 = solve(mode="chunk_intra_gates",
               q=[[1.0], [1.0], [1.0], [1.0]],
               k=[[1.0], [2.0], [3.0], [4.0]],
               v=[[1.0], [1.0], [1.0], [1.0]],
               i=[0.0, 0.0, 0.0, 0.0], f=[0.0, 0.0, 0.0, 0.0])
    _cmp(r1, {"log_fgates_acc": [[-L2, -2.0 * L2], [-L2, -2.0 * L2]],
              "loggates": [[-L2, 0.0], [-L2, 0.0]],
              "m_loc": [0.0, 0.0],
              "kv": [[[2.5]], [[5.5]]],
              "ksum": [[2.5], [5.5]]}, "chunk.a1")

    # 锚点 2：DH=2、f=1000（log_fg≈-0.0）→ acc 全 0、e 全 1，
    # 钉 kv[a][b] = Σ_t k[t][a]·v[t][b]·e[t] 的方向与 ksum 沿 CS 维求和。
    r2 = solve(mode="chunk_intra_gates",
               q=[[1.0, 1.0], [1.0, 1.0]],
               k=[[1.0, 2.0], [3.0, 4.0]],
               v=[[5.0, 6.0], [7.0, 8.0]],
               i=[0.0, 0.0], f=[1000.0, 1000.0], chunk_size=2)
    _cmp(r2, {"log_fgates_acc": [[0.0, 0.0]],
              "loggates": [[0.0, 0.0]],
              "m_loc": [0.0],
              "kv": [[[26.0, 30.0], [38.0, 44.0]]],
              "ksum": [[4.0, 6.0]]}, "chunk.a2")

    # 锚点 3：两个 chunk 的 m_loc 不同（0 与 10），钉 m_loc 是 chunk 级 max
    # 而不是全局 max；若用全局 max，chunk0 的 loggates/kv/ksum 全变。
    e10 = math.exp(-10.0)
    r3 = solve(mode="chunk_intra_gates",
               q=[[1.0], [1.0], [1.0], [1.0]],
               k=[[1.0], [1.0], [1.0], [1.0]],
               v=[[1.0], [1.0], [1.0], [1.0]],
               i=[0.0, 0.0, 10.0, 0.0],
               f=[1000.0, 1000.0, 1000.0, 1000.0], chunk_size=2)
    _cmp(r3, {"log_fgates_acc": [[0.0, 0.0], [0.0, 0.0]],
              "loggates": [[0.0, 0.0], [0.0, -10.0]],
              "m_loc": [0.0, 10.0],
              "kv": [[[2.0]], [[1.0 + e10]]],
              "ksum": [[2.0], [1.0 + e10]]}, "chunk.a3")

    # 锚点 4：chunk_size=1（每个 token 自成一 chunk），NS=S。
    r4 = solve(mode="chunk_intra_gates",
               q=[[1.0], [1.0], [1.0]],
               k=[[1.0], [2.0], [3.0]],
               v=[[1.0], [1.0], [1.0]],
               i=[0.0, 0.0, 0.0], f=[0.0, 0.0, 0.0], chunk_size=1)
    _cmp(r4, {"log_fgates_acc": [[-L2], [-L2], [-L2]],
              "loggates": [[0.0], [0.0], [0.0]],
              "m_loc": [0.0, 0.0, 0.0],
              "kv": [[[1.0]], [[2.0]], [[3.0]]],
              "ksum": [[1.0], [2.0], [3.0]]}, "chunk.a4")
    print("chunk_intra_gates: %d grid + 4 anchors OK" % n)


def check(solve):
    check_stabilized_step(solve)
    check_parallel_decay(solve)
    check_chunk_intra_gates(solve)
    print("all checks passed")
