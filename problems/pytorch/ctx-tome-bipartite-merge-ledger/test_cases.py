"""order 841 判题用例（暂存）：ToMe 的 bipartite soft matching 合并账。

独立 oracle（不 import solution），每个 check_* 含手算锚点 + 网格对拍。

口径：metric / x / size 的元素都是整数；相似度用**精确有理数**比较
（不开根号，交叉相乘），max 并列取下标最小，排序稳定保序。
"""

import math
from fractions import Fraction


# ------------------------------------------------------------------ 独立 oracle
def _cos_pair(u, v):
    """精确余弦：返回 (sign, cos^2 的 Fraction)；None 表示负无穷/零模长。"""
    if u is None or v is None:
        return None
    dot = sum(Fraction(a) * Fraction(b) for a, b in zip(u, v))
    nu = sum(Fraction(a) * Fraction(a) for a in u)
    nv = sum(Fraction(a) * Fraction(a) for a in v)
    if nu == 0 or nv == 0:
        return None
    sign = 1 if dot >= 0 else -1
    return (sign, Fraction(dot * dot, 1) / (nu * nv))


def _cmp_cos(p, q):
    """精确比较两个 (sign, sq) 余弦，返回 -1/0/1。"""
    if p is None:
        return 0 if q is None else -1
    if q is None:
        return 1
    sp, qp = p
    sq_, qq = q
    if sp >= 0 and sq_ < 0:
        return 1
    if sq_ >= 0 and sp < 0:
        return -1
    if sp >= 0:
        return (qp > qq) - (qp < qq)
    return (qp < qq) - (qp > qq)


def _cos_float(p):
    """(sign, sq) -> float 余弦值；None -> -inf。"""
    if p is None:
        return float("-inf")
    sign, sq = p
    return sign * math.sqrt(float(sq))


def oracle_match(metric, r, class_token, distill_token):
    """bipartite_soft_matching 的索引计算段（精确口径）。

    metric : [B][T][C]，元素是整数
    """
    batch = len(metric)
    t = len(metric[0])
    protected = (1 if class_token else 0) + (1 if distill_token else 0)
    r_eff = min(int(r), (t - protected) // 2)

    t1 = (t + 1) // 2
    t2 = t // 2

    src_idx, dst_idx, unm_idx, node_max = [], [], [], []

    if r_eff <= 0:
        # 官方此时返回 do_nothing：不匹配、不合并
        for _ in range(batch):
            src_idx.append([])
            dst_idx.append([])
            unm_idx.append(list(range(t1)))
            node_max.append([])
        return {"r_effective": r_eff, "src_idx": src_idx, "dst_idx": dst_idx,
                "unm_idx": unm_idx, "node_max": node_max,
                "num_src": t1, "num_dst": t2}

    for bi in range(batch):
        rows = metric[bi]
        a = [rows[i] for i in range(0, t, 2)]
        b = [rows[i] for i in range(1, t, 2)]

        scores = []
        for ai in range(t1):
            scores.append([_cos_pair(a[ai], b[bj]) for bj in range(t2)])

        # -inf 保护在两个不同的轴上
        if class_token:
            for bj in range(t2):
                scores[0][bj] = None
        if distill_token:
            for ai in range(t1):
                scores[ai][0] = None

        nmax, nidx = [], []
        for ai in range(t1):
            best = scores[ai][0]
            bj = 0
            for bk in range(1, t2):
                if _cmp_cos(scores[ai][bk], best) > 0:
                    best = scores[ai][bk]
                    bj = bk
            nmax.append(best)
            nidx.append(bj)

        # 稳定降序排序
        order = list(range(t1))
        for i in range(1, t1):
            key = order[i]
            j = i - 1
            while j >= 0 and _cmp_cos(nmax[key], nmax[order[j]]) > 0:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        src = order[:r_eff]
        unm = order[r_eff:]
        if class_token:
            unm = sorted(unm)

        src_idx.append(list(src))
        dst_idx.append([nidx[i] for i in src])
        unm_idx.append(list(unm))
        node_max.append([_cos_float(nmax[i]) for i in range(t1)])

    return {"r_effective": r_eff, "src_idx": src_idx, "dst_idx": dst_idx,
            "unm_idx": unm_idx, "node_max": node_max,
            "num_src": t1, "num_dst": t2}


def oracle_merge(x, src_idx, dst_idx, unm_idx, reduce_mode, distill_token):
    """merge 闭包：gather + scatter_reduce（include_self=True）+ 输出重排。"""
    batch = len(x)
    t = len(x[0])

    merged, dst_values = [], []
    for bi in range(batch):
        rows = x[bi]
        a_rows = [list(rows[i]) for i in range(0, t, 2)]
        b_rows = [list(rows[i]) for i in range(1, t, 2)]

        if not src_idx[bi]:
            # do_nothing：恒等
            merged.append([list(v) for v in rows])
            dst_values.append([list(v) for v in b_rows])
            continue

        unm = [list(a_rows[i]) for i in unm_idx[bi]]
        src = [list(a_rows[i]) for i in src_idx[bi]]
        idxs = dst_idx[bi]

        if reduce_mode == "mean":
            dst = _scatter_mean(b_rows, src, idxs)
        elif reduce_mode == "sum":
            dst = _scatter_sum(b_rows, src, idxs)
        elif reduce_mode == "amax":
            dst = _scatter_amax(b_rows, src, idxs)
        else:
            raise ValueError("unknown reduce: %s" % reduce_mode)

        dst_values.append([list(v) for v in dst])

        if distill_token:
            merged.append([list(unm[0])] + [list(dst[0])]
                          + [list(v) for v in unm[1:]]
                          + [list(v) for v in dst[1:]])
        else:
            merged.append([list(v) for v in unm] + [list(v) for v in dst])

    return {"merged": merged, "merged_len": len(merged[0]) if merged else 0,
            "dst_values": dst_values,
            "num_src": (t + 1) // 2, "num_dst": t // 2}


def _scatter_mean(dst_row, src_rows, idxs):
    """scatter_reduce(reduce='mean')，include_self=True。"""
    counts = [1] * len(dst_row)
    sums = [list(v) for v in dst_row]
    for src, j in zip(src_rows, idxs):
        for c in range(len(sums[j])):
            sums[j][c] += src[c]
        counts[j] += 1
    return [[sums[j][c] / counts[j] for c in range(len(sums[j]))]
            for j in range(len(sums))]


def _scatter_sum(dst_row, src_rows, idxs):
    sums = [list(v) for v in dst_row]
    for src, j in zip(src_rows, idxs):
        for c in range(len(sums[j])):
            sums[j][c] += src[c]
    return sums


def _scatter_amax(dst_row, src_rows, idxs):
    out = [list(v) for v in dst_row]
    for src, j in zip(src_rows, idxs):
        for c in range(len(out[j])):
            if src[c] > out[j][c]:
                out[j][c] = src[c]
    return out


def oracle_wavg(x, size, src_idx, dst_idx, unm_idx, distill_token):
    """merge_wavg：x = merge(x*size,'sum')；size = merge(size,'sum')；x = x/size。"""
    batch = len(x)
    t = len(x[0])
    c = len(x[0][0])

    scaled = []
    for bi in range(batch):
        brows = []
        for ti in range(t):
            brows.append([x[bi][ti][ci] * size[bi][ti][0] for ci in range(c)])
        scaled.append(brows)

    xm = oracle_merge(scaled, src_idx, dst_idx, unm_idx, "sum", distill_token)
    sm = oracle_merge(size, src_idx, dst_idx, unm_idx, "sum", distill_token)

    merged, merged_size = [], []
    for bi in range(batch):
        xs = xm["merged"][bi]
        ss = sm["merged"][bi]
        out, out_s = [], []
        for ti in range(len(xs)):
            denom = ss[ti][0]
            out.append([xs[ti][ci] / denom if denom != 0.0 else float("nan")
                        for ci in range(c)])
            out_s.append([ss[ti][0]])
        merged.append(out)
        merged_size.append(out_s)

    return {"merged": merged, "merged_size": merged_size,
            "merged_len": len(merged[0]) if merged else 0}


def oracle_unmerge(x, src_idx, dst_idx, unm_idx):
    """unmerge 闭包：输入是合并后的张量，输出还原成原始 token 数。"""
    batch = len(x)
    m = len(x[0])
    c = len(x[0][0])
    unm_len = len(unm_idx[0])
    r = len(src_idx[0])
    t = m + r

    out = []
    for bi in range(batch):
        rows = x[bi]
        unm = [list(rows[i]) for i in range(unm_len)]
        dst = [list(rows[i]) for i in range(unm_len, m)]
        src = [list(dst[j]) for j in dst_idx[bi]]

        res = [[0] * c for _ in range(t)]
        for i in range(len(dst)):
            res[2 * i + 1] = list(dst[i])
        for k, ai in enumerate(unm_idx[bi]):
            res[2 * ai] = list(unm[k])
        for k, ai in enumerate(src_idx[bi]):
            res[2 * ai] = list(src[k])
        out.append(res)

    return {"out": out, "out_len": len(out[0]) if out else 0}


# ------------------------------------------------------------------ 工具
def _close(actual, expect, atol=1e-9, rtol=1e-9, tag=""):
    if isinstance(expect, (list, tuple)):
        assert isinstance(actual, (list, tuple)), \
            "%s: expected list got %r" % (tag, type(actual))
        assert len(actual) == len(expect), \
            "%s: length %d != %d" % (tag, len(actual), len(expect))
        for i, item in enumerate(expect):
            _close(actual[i], item, atol=atol, rtol=rtol, tag="%s[%d]" % (tag, i))
        return
    if expect is None:
        assert actual is None, "%s: got %r expected None" % (tag, actual)
        return
    if isinstance(expect, bool) or isinstance(actual, bool):
        assert bool(actual) == bool(expect), \
            "%s: got %r expected %r" % (tag, actual, expect)
        return
    if isinstance(expect, int):
        assert int(actual) == int(expect), \
            "%s: got %r expected %r" % (tag, actual, expect)
        return
    a = float(actual)
    e = float(expect)
    if e != e:
        assert a != a, "%s: got %r expected NaN" % (tag, actual)
        return
    if a != a:
        raise AssertionError("%s: got NaN expected %r" % (tag, e))
    if math.isinf(e):
        assert math.isinf(a) and (a > 0) == (e > 0), \
            "%s: got %r expected %r" % (tag, actual, expect)
        return
    assert abs(a - e) <= atol + rtol * abs(e), \
        "%s: got %r expected %r" % (tag, actual, expect)


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

    def randint(self, lo, hi):
        return lo + int(self.next() * (hi - lo + 1)) % (hi - lo + 1)

    def randrange(self, n):
        return self.randint(0, n - 1)

    def choice(self, seq):
        return seq[int(self.next() * len(seq)) % len(seq)]

    def sample(self, seq, k):
        """从 seq 中不重复地抽 k 个（部分 Fisher-Yates），保持确定性。"""
        pool = list(seq)
        n = len(pool)
        k = min(k, n)
        for i in range(k):
            j = i + self.randint(0, n - i - 1)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]


def _make_metric(rng, batch, t, c, style):
    """整数 metric（每行模长非零）。"""
    out = []
    for _ in range(batch):
        rows = []
        for _ in range(t):
            if style == 0:
                row = [rng.randint(-3, 3) for _ in range(c)]
            elif style == 1:
                row = [rng.randint(0, 3) for _ in range(c)]
            elif style == 2:
                row = [rng.choice([0, 1, -1]) for _ in range(c)]
            elif style == 3:
                row = [rng.randint(-20, 20) for _ in range(c)]
            else:
                # one-hot ±1：制造大量精确并列
                row = [0] * c
                row[rng.randrange(c)] = rng.choice([1, -1])
            if all(v == 0 for v in row):
                row[0] = 1
            rows.append(row)
        out.append(rows)
    return out


# ------------------------------------------------------------------ check_*
def check_match_indices(solve):
    # 手算锚点 1：基本匹配 + max 并列取下标最小
    # metric 4x2 -> a = rows(0,2) = [[1,0],[1,1]], b = rows(1,3) = [[0,1],[2,0]]
    # a0 vs b0: 0/1 = 0        ; a0 vs b1: 2/(1*2) = 1        -> nidx[0]=1, nmax=1
    # a1 vs b0: 1/(sqrt2*1) = 1/sqrt2 ; a1 vs b1: 2/(sqrt2*2)=1/sqrt2 -> 并列 -> nidx[1]=0
    # order(desc, 稳定) = [0, 1] ; r=1 -> src=[0], unm=[1], dst=[1]
    r = solve("match_indices",
              metric=[[[1, 0], [0, 1], [1, 1], [2, 0]]],
              r=1, class_token=False, distill_token=False)
    _cmp(r, {"r_effective": 1,
             "src_idx": [[0]], "dst_idx": [[1]], "unm_idx": [[1]],
             "node_max": [[1.0, 1.0 / math.sqrt(2)]],
             "num_src": 2, "num_dst": 2}, "match.anchor1")

    # 手算锚点 2：class_token 保护第 0 行（a 侧整行 -> -inf）
    # 同metric，class_token=True -> a0 整行 -inf -> nmax[0]=-inf, nidx[0]=0
    # a1 仍 1/sqrt2 -> order = [1, 0] ; r=1 -> src=[1], unm=[0]
    r = solve("match_indices",
              metric=[[[1, 0], [0, 1], [1, 1], [2, 0]]],
              r=1, class_token=True, distill_token=False)
    _cmp(r, {"r_effective": 1,
             "src_idx": [[1]], "dst_idx": [[0]], "unm_idx": [[0]],
             "node_max": [[float("-inf"), 1.0 / math.sqrt(2)]],
             "num_src": 2, "num_dst": 2}, "match.anchor2")

    # 手算锚点 3：distill_token 保护第 0 列（b 侧整列 -> -inf）
    # metric 4x2 = [[1,1],[1,1],[1,1],[0,1]]
    # a = [[1,1],[1,1]], b = [[1,1],[0,1]]
    # 无 distill: a0 vs b0 = 2/(sqrt2*sqrt2) = 1 -> nidx=0
    # 有 distill: b0 被排除 -> a0 vs b1 = 1/(sqrt2*1) = 1/sqrt2 -> nidx=1
    r = solve("match_indices",
              metric=[[[1, 1], [1, 1], [1, 1], [0, 1]]],
              r=1, class_token=False, distill_token=False)
    _cmp(r, {"r_effective": 1,
             "src_idx": [[0]], "dst_idx": [[0]], "unm_idx": [[1]],
             "node_max": [[1.0, 1.0]],
             "num_src": 2, "num_dst": 2}, "match.anchor3a")
    r = solve("match_indices",
              metric=[[[1, 1], [1, 1], [1, 1], [0, 1]]],
              r=1, class_token=False, distill_token=True)
    _cmp(r, {"r_effective": 1,
             "src_idx": [[0]], "dst_idx": [[1]], "unm_idx": [[1]],
             "node_max": [[1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]],
             "num_src": 2, "num_dst": 2}, "match.anchor3b")

    # 手算锚点 4：class_token 时 unm 要排序（保证 class token 在最前）
    # metric 6x2 -> a = rows(0,2,4) = [[1,0],[1,0],[0,1]], b = rows(1,3,5)
    #   = [[1,0],[1,1],[0,1]]
    # a0 被 class_token 保护 -> -inf
    # a1=[1,0]: b0 -> 1.0 ; b1 -> 1/sqrt2 ; b2 -> 0      -> nmax=1.0 @0
    # a2=[0,1]: b0 -> 0 ; b1 -> 1/sqrt2 ; b2 -> 1.0      -> nmax=1.0 @2
    # order(desc, 稳定, a0 是 -inf 在最后) = [1, 2, 0]
    # r=1 -> src=[1] ; unm = [2, 0] -> class_token 排序 -> [0, 2]
    # dst_idx = nidx[1] = 0
    r = solve("match_indices",
              metric=[[[1, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 1]]],
              r=1, class_token=True, distill_token=False)
    _cmp(r, {"r_effective": 1,
             "src_idx": [[1]], "dst_idx": [[0]], "unm_idx": [[0, 2]],
             "node_max": [[float("-inf"), 1.0, 1.0]],
             "num_src": 3, "num_dst": 3}, "match.anchor4")

    # 手算锚点 5：r 被 (t - protected) // 2 截断
    # t=4, 无 protected -> 上限 2 ; r=10 -> r_effective=2
    # order=[0,1] -> src=[0,1], unm=[]
    r = solve("match_indices",
              metric=[[[1, 0], [0, 1], [1, 1], [2, 0]]],
              r=10, class_token=False, distill_token=False)
    _cmp(r, {"r_effective": 2,
             "src_idx": [[0, 1]], "dst_idx": [[1, 0]], "unm_idx": [[]],
             "node_max": [[1.0, 1.0 / math.sqrt(2)]],
             "num_src": 2, "num_dst": 2}, "match.anchor5")

    # 手算锚点 6：r <= 0 -> 官方 do_nothing（不匹配、不合并）
    r = solve("match_indices",
              metric=[[[1, 0], [0, 1], [1, 1], [2, 0]]],
              r=0, class_token=False, distill_token=False)
    _cmp(r, {"r_effective": 0,
             "src_idx": [[]], "dst_idx": [[]], "unm_idx": [[0, 1]],
             "node_max": [[]],
             "num_src": 2, "num_dst": 2}, "match.anchor6a")
    r = solve("match_indices",
              metric=[[[1, 0], [0, 1], [1, 1], [2, 0]]],
              r=-3, class_token=False, distill_token=False)
    _cmp(r, {"r_effective": -3,
             "src_idx": [[]], "dst_idx": [[]], "unm_idx": [[0, 1]],
             "node_max": [[]],
             "num_src": 2, "num_dst": 2}, "match.anchor6b")

    # 手算锚点 7：class_token + distill_token 同时开 -> protected=2
    # t=4 -> 上限 (4-2)//2 = 1 ; r=5 -> r_effective=1
    r = solve("match_indices",
              metric=[[[1, 0], [0, 1], [1, 1], [2, 0]]],
              r=5, class_token=True, distill_token=True)
    _cmp(r, {"r_effective": 1,
             "src_idx": [[1]], "dst_idx": [[1]], "unm_idx": [[0]],
             "node_max": [[float("-inf"), 1.0 / math.sqrt(2)]],
             "num_src": 2, "num_dst": 2}, "match.anchor7")

    # 手算锚点 8：负余弦也要能被选中（符号先判）
    # metric 4x2: a = [[1,0],[1,0]], b = [[0,1],[-1,0]]
    # a0 vs b0 = 0 ; a0 vs b1 = -1 -> nmax[0] = 0 @ b0
    # a1 同a0 -> nmax[1] = 0 @ b0
    # order(desc, 稳定) = [0, 1] ; r=1 -> src=[0], unm=[1], dst=[0]
    r = solve("match_indices",
              metric=[[[1, 0], [0, 1], [1, 0], [-1, 0]]],
              r=1, class_token=False, distill_token=False)
    _cmp(r, {"r_effective": 1,
             "src_idx": [[0]], "dst_idx": [[0]], "unm_idx": [[1]],
             "node_max": [[0.0, 0.0]],
             "num_src": 2, "num_dst": 2}, "match.anchor8")

    # ---- 网格对拍：style / class / distill / r / batch 各维正交
    rng = _Rng(841101)
    for case in range(180):
        style = case % 5
        class_token = (case // 5) % 2 == 1
        distill_token = (case // 10) % 2 == 1
        batch = 1 + (case // 20) % 3
        t = 2 + (case * 7) % 13
        c = 1 + (case * 5) % 4
        r = (case * 3) % 9 - 2
        if class_token and distill_token and t < 4:
            distill_token = False
        metric = _make_metric(rng, batch, t, c, style)
        got = solve("match_indices", metric=metric, r=r,
                    class_token=class_token, distill_token=distill_token)
        exp = oracle_match(metric, r, class_token, distill_token)
        _cmp(got, exp, "match.grid[%d]" % case)


def check_merge_reduce(solve):
    # 手算锚点 1：mean 且 include_self=True
    # x = [[[10],[20],[30],[40]]] ; t=4 -> a = rows(0,2) = [10,30], b = rows(1,3) = [20,40]
    # src=[0] -> a0=10 ; dst=[1] -> b1=40 ; unm=[1] -> a1=30
    # mean: dst1 = (40 + 10)/2 = 25 ; dst0 = 20
    # merged = cat([unm, dst]) = [[30],[20],[25]]
    r = solve("merge_reduce", x=[[[10], [20], [30], [40]]],
              src_idx=[[0]], dst_idx=[[1]], unm_idx=[[1]],
              reduce="mean", distill_token=False)
    _cmp(r, {"merged": [[[30], [20], [25]]],
             "merged_len": 3,
             "dst_values": [[[20], [25]]],
             "num_src": 2, "num_dst": 2}, "merge.anchor1")

    # 手算锚点 2：sum / amax 同输入
    r = solve("merge_reduce", x=[[[10], [20], [30], [40]]],
              src_idx=[[0]], dst_idx=[[1]], unm_idx=[[1]],
              reduce="sum", distill_token=False)
    _cmp(r, {"merged": [[[30], [20], [50]]],
             "merged_len": 3,
             "dst_values": [[[20], [50]]],
             "num_src": 2, "num_dst": 2}, "merge.anchor2a")
    r = solve("merge_reduce", x=[[[10], [20], [30], [40]]],
              src_idx=[[0]], dst_idx=[[1]], unm_idx=[[1]],
              reduce="amax", distill_token=False)
    _cmp(r, {"merged": [[[30], [20], [40]]],
             "merged_len": 3,
             "dst_values": [[[20], [40]]],
             "num_src": 2, "num_dst": 2}, "merge.anchor2b")

    # 手算锚点 3：多个 src 落到同一 dst（scatter_reduce 重复索引）
    # t=6 -> a = rows(0,2,4) = [10,30,50], b = rows(1,3,5) = [20,40,60]
    # src=[0,2] -> a0=10, a2=50 ; dst=[1,1] -> 都并入 b1=40
    # sum: dst1 = 40 + 10 + 50 = 100 ; mean: (40+10+50)/3 = 100/3
    # amax: max(40,10,50) = 50
    # unm=[1] -> a1=30 ; merged = [[30],[20],[dst1],[60]]
    r = solve("merge_reduce", x=[[[10], [20], [30], [40], [50], [60]]],
              src_idx=[[0, 2]], dst_idx=[[1, 1]], unm_idx=[[1]],
              reduce="sum", distill_token=False)
    _cmp(r, {"merged": [[[30], [20], [100], [60]]],
             "merged_len": 4,
             "dst_values": [[[20], [100], [60]]],
             "num_src": 3, "num_dst": 3}, "merge.anchor3a")
    r = solve("merge_reduce", x=[[[10], [20], [30], [40], [50], [60]]],
              src_idx=[[0, 2]], dst_idx=[[1, 1]], unm_idx=[[1]],
              reduce="mean", distill_token=False)
    _cmp(r, {"merged": [[[30], [20], [100.0 / 3.0], [60]]],
             "merged_len": 4,
             "dst_values": [[[20], [100.0 / 3.0], [60]]],
             "num_src": 3, "num_dst": 3}, "merge.anchor3b")
    r = solve("merge_reduce", x=[[[10], [20], [30], [40], [50], [60]]],
              src_idx=[[0, 2]], dst_idx=[[1, 1]], unm_idx=[[1]],
              reduce="amax", distill_token=False)
    _cmp(r, {"merged": [[[30], [20], [50], [60]]],
             "merged_len": 4,
             "dst_values": [[[20], [50], [60]]],
             "num_src": 3, "num_dst": 3}, "merge.anchor3c")

    # 手算锚点 4：distill_token 的交错输出布局
    # t=6 -> a = [10,30,50], b = [20,40,60]
    # src=[0] -> a0=10 ; dst=[2] -> b2=60 ; unm=[1,2] -> a1=30, a2=50
    # sum: dst2 = 60+10 = 70 ; dst = [[20],[40],[70]]
    # merged = cat([unm[:1], dst[:1], unm[1:], dst[1:]])
    #        = [[30],[20],[50],[40],[70]]
    r = solve("merge_reduce", x=[[[10], [20], [30], [40], [50], [60]]],
              src_idx=[[0]], dst_idx=[[2]], unm_idx=[[1, 2]],
              reduce="sum", distill_token=True)
    _cmp(r, {"merged": [[[30], [20], [50], [40], [70]]],
             "merged_len": 5,
             "dst_values": [[[20], [40], [70]]],
             "num_src": 3, "num_dst": 3}, "merge.anchor4")

    # 手算锚点 5：src_idx 为空 -> 官方 do_nothing（恒等）
    r = solve("merge_reduce", x=[[[10], [20], [30], [40]]],
              src_idx=[[]], dst_idx=[[]], unm_idx=[[0, 1]],
              reduce="mean", distill_token=False)
    _cmp(r, {"merged": [[[10], [20], [30], [40]]],
             "merged_len": 4,
             "dst_values": [[[20], [40]]],
             "num_src": 2, "num_dst": 2}, "merge.anchor5")

    # 手算锚点 6：多通道
    # x = [[[1,2],[3,4],[5,6],[7,8]]] ; src=[0] (a0=[1,2]) ; dst=[1] (b1=[7,8])
    # mean -> dst1 = ([7,8]+[1,2])/2 = [4,5] ; dst0 = [3,4]
    # unm=[1] -> a1=[5,6] ; merged = [[5,6],[3,4],[4,5]]
    r = solve("merge_reduce", x=[[[1, 2], [3, 4], [5, 6], [7, 8]]],
              src_idx=[[0]], dst_idx=[[1]], unm_idx=[[1]],
              reduce="mean", distill_token=False)
    _cmp(r, {"merged": [[[5, 6], [3, 4], [4, 5]]],
             "merged_len": 3,
             "dst_values": [[[3, 4], [4, 5]]],
             "num_src": 2, "num_dst": 2}, "merge.anchor6")

    # ---- 网格对拍：索引直接随机（覆盖「多 src 落同一 dst」）
    rng = _Rng(841102)
    for case in range(200):
        batch = 1 + case % 3
        t = 2 + (case * 5) % 12
        c = 1 + (case * 3) % 3
        t1 = (t + 1) // 2
        t2 = t // 2
        r_eff = 1 + (case * 7) % max(1, min(t1, 3))
        reduce_mode = ["mean", "sum", "amax"][case % 3]
        distill_token = (case // 3) % 2 == 1 and t1 >= 2 and t2 >= 2
        x = [[[rng.randint(-20, 20) for _ in range(c)]
              for _ in range(t)] for _ in range(batch)]
        src_idx, dst_idx, unm_idx = [], [], []
        for _ in range(batch):
            # src 是 argsort 的前缀，必互异；dst 可重复（多个 a 指向同一 b）
            src = rng.sample(list(range(t1)), r_eff)
            unm = [i for i in range(t1) if i not in src]
            src_idx.append(src)
            unm_idx.append(unm)
            dst_idx.append([rng.randint(0, t2 - 1) for _ in src])
        got = solve("merge_reduce", x=x, src_idx=src_idx, dst_idx=dst_idx,
                    unm_idx=unm_idx, reduce=reduce_mode,
                    distill_token=distill_token)
        exp = oracle_merge(x, src_idx, dst_idx, unm_idx, reduce_mode,
                           distill_token)
        _cmp(got, exp, "merge.grid[%d]" % case)


def check_size_weighted_merge(solve):
    # 手算锚点 1：分母是 size 之和，不是 token 个数
    # t=4 -> a = rows(0,2) = [2,18], b = rows(1,3) = [8,32]（x*size 后）
    # x = [[2],[4],[6],[8]], size = [[1],[2],[3],[4]] -> scaled = [[2],[8],[18],[32]]
    # src=[0] -> a0=2 ; dst=[0] -> b0=8 ; unm=[1] -> a1=18
    # x merge(sum): dst0 = 8+2 = 10 ; dst1 = 32
    # size merge(sum): a = rows(0,2) of size = [1,3], b = rows(1,3) = [2,4]
    #   unm=[3] ; src=[1] -> dst0 = 2+1 = 3 ; dst1 = 4
    # merged_size = [[3],[3],[4]]
    # merged = [[18],[10],[32]] / [[3],[3],[4]] = [[6],[10/3],[8]]
    r = solve("size_weighted_merge",
              x=[[[2], [4], [6], [8]]], size=[[[1], [2], [3], [4]]],
              src_idx=[[0]], dst_idx=[[0]], unm_idx=[[1]],
              distill_token=False)
    _cmp(r, {"merged": [[[6.0], [10.0 / 3.0], [8.0]]],
             "merged_size": [[[3], [3], [4]]],
             "merged_len": 3}, "wavg.anchor1")

    # 手算锚点 2：unm 的分母就是它自己的 size（不是 1，也不是 token 数）
    # x = [[6],[4],[2],[8]], size = [[1],[1],[1],[5]]
    # scaled = [[6],[4],[2],[40]]
    # a = rows(0,2) = [6,2], b = rows(1,3) = [4,40]
    # src=[1] -> a1=2 ; dst=[1] -> b1=40 ; unm=[0] -> a0=6
    # x merge(sum): dst1 = 40+2 = 42 ; dst0 = 4
    # size merge(sum): a = size rows(0,2) = [1,1], b = size rows(1,3) = [1,5]
    #   unm=[1] ; src=[1] -> dst1 = 5+1 = 6 ; dst0 = 1
    # merged_size = [[1],[1],[6]]
    # merged = [[6],[4],[42]] / [[1],[1],[6]] = [[6],[4],[7]]
    r = solve("size_weighted_merge",
              x=[[[6], [4], [2], [8]]], size=[[[1], [1], [1], [5]]],
              src_idx=[[1]], dst_idx=[[1]], unm_idx=[[0]],
              distill_token=False)
    _cmp(r, {"merged": [[[6.0], [4.0], [7.0]]],
             "merged_size": [[[1], [1], [6]]],
             "merged_len": 3}, "wavg.anchor2")

    # 手算锚点 3：distill_token 的交错布局（size 也走同一套布局）
    # t=6 -> a = rows(0,2,4) of x = [2,6,10], b = rows(1,3,5) = [4,8,12]
    # size = [1,2,3,4,5,6] -> a_size = [1,3,5], b_size = [2,4,6]
    # scaled = x*size = [2,12,30,32,50,72]
    # src=[0] -> a0=2 ; dst=[2] -> b2=72 ; unm=[1,2] -> a1=12, a2=30
    # x merge(sum): dst = [[8],[16],[74]]
    # size merge(sum): unm_size=[3,5], src_size=[1] -> dst_size = [[2],[4],[7]]
    # merged(x)   = cat([unm[:1], dst[:1], unm[1:], dst[1:]]) = [[12],[8],[30],[16],[74]]
    # merged_size = cat([unm_s[:1], dst_s[:1], unm_s[1:], dst_s[1:]])
    #             = [[3],[2],[5],[4],[7]]
    # 归一化：[[18/3],[8/2],[50/5],[32/4],[74/7]] = [[6],[4],[10],[8],[74/7]]
    r = solve("size_weighted_merge",
              x=[[[2], [4], [6], [8], [10], [12]]],
              size=[[[1], [2], [3], [4], [5], [6]]],
              src_idx=[[0]], dst_idx=[[2]], unm_idx=[[1, 2]],
              distill_token=True)
    _cmp(r, {"merged": [[[6.0], [4.0], [10.0], [8.0], [74.0 / 7.0]]],
             "merged_size": [[[3], [2], [5], [4], [7]]],
             "merged_len": 5}, "wavg.anchor3")

    # 手算锚点 4：src_idx 为空 -> do_nothing（x 恒等、size 恒等）
    r = solve("size_weighted_merge",
              x=[[[2], [4]]], size=[[[1], [3]]],
              src_idx=[[]], dst_idx=[[]], unm_idx=[[0]],
              distill_token=False)
    _cmp(r, {"merged": [[[2], [4]]],
             "merged_size": [[[1], [3]]],
             "merged_len": 2}, "wavg.anchor4")

    # 手算锚点 5：多通道
    # x = [[1,2],[3,4],[5,6],[7,8]], size = [[1],[2],[3],[4]]
    # scaled = [[1,2],[6,8],[15,18],[28,32]]
    # a = rows(0,2) = [[1,2],[15,18]], b = rows(1,3) = [[6,8],[28,32]]
    # src=[0] -> a0 ; dst=[1] -> b1 ; unm=[1] -> a1
    # x merge(sum): dst1 = [[29,34]] ; dst0 = [[6,8]]
    # size merge(sum): a_size=[1,3], b_size=[2,4]; unm_s=[3], src_s=[1]
    #   dst_s = [[2],[5]] ; merged_size = [[3],[2],[5]]
    # merged = [[15,18],[6,8],[29,34]] / [[3],[2],[5]]
    #        = [[5,6],[3,4],[29/5,34/5]]
    r = solve("size_weighted_merge",
              x=[[[1, 2], [3, 4], [5, 6], [7, 8]]],
              size=[[[1], [2], [3], [4]]],
              src_idx=[[0]], dst_idx=[[1]], unm_idx=[[1]],
              distill_token=False)
    _cmp(r, {"merged": [[[5.0, 6.0], [3.0, 4.0],
                          [29.0 / 5.0, 34.0 / 5.0]]],
             "merged_size": [[[3], [2], [5]]],
             "merged_len": 3}, "wavg.anchor5")

    # ---- 网格对拍
    rng = _Rng(841103)
    for case in range(160):
        batch = 1 + case % 2
        t = 2 + (case * 5) % 12
        c = 1 + (case * 3) % 3
        t1 = (t + 1) // 2
        t2 = t // 2
        r_eff = 1 + (case * 7) % max(1, min(t1, 3))
        distill_token = (case // 2) % 2 == 1 and t1 >= 2 and t2 >= 2
        x = [[[rng.randint(-20, 20) for _ in range(c)]
              for _ in range(t)] for _ in range(batch)]
        size = [[[rng.randint(1, 6)] for _ in range(t)] for _ in range(batch)]
        src_idx, dst_idx, unm_idx = [], [], []
        for _ in range(batch):
            # src 是 argsort 的前缀，必互异；dst 可重复（多个 a 指向同一 b）
            src = rng.sample(list(range(t1)), r_eff)
            unm = [i for i in range(t1) if i not in src]
            src_idx.append(src)
            unm_idx.append(unm)
            dst_idx.append([rng.randint(0, t2 - 1) for _ in src])
        got = solve("size_weighted_merge", x=x, size=size,
                    src_idx=src_idx, dst_idx=dst_idx, unm_idx=unm_idx,
                    distill_token=distill_token)
        exp = oracle_wavg(x, size, src_idx, dst_idx, unm_idx, distill_token)
        _cmp(got, exp, "wavg.grid[%d]" % case)


def check_unmerge(solve):
    # 手算锚点 1：位置回填
    # merge 后的 x = [[30],[20],[25]]（t=4, r=1 -> m=3, T=4）
    # unm_len = 1 -> unm=[[30]], dst=[[20],[25]]
    # src = dst.gather(dst_idx=[1]) = [[25]]
    # out = zeros(4) ; out[1::2] = dst -> out[1]=[20], out[3]=[25]
    # out[2*unm_idx[0]=2] = unm[0] = [30]
    # out[2*src_idx[0]=0] = src[0] = [25]
    r = solve("unmerge", x=[[[30], [20], [25]]],
              src_idx=[[0]], dst_idx=[[1]], unm_idx=[[1]])
    _cmp(r, {"out": [[[25], [20], [30], [25]]], "out_len": 4},
         "unmerge.anchor1")

    # 手算锚点 2：多个 src 落同一 dst 时回填的是合并后的值
    # t=6, r=2 -> m=4, T=6
    # a = rows(0,2,4), b = rows(1,3,5)
    # src=[0,2] -> a0, a2 ; dst=[1,1] -> 都并入 b1
    # unm=[1] -> a1
    # merged(sum) = cat([unm, dst]) = [[a1],[b0],[b1+a0+a2],[b2]]
    # 令 x = [[10],[20],[30],[40],[50],[60]] -> a=[10,30,50], b=[20,40,60]
    # merged = [[30],[20],[100],[60]]
    # unm_len = 1 -> unm=[[30]], dst=[[20],[100],[60]]
    # src = dst.gather([1,1]) = [[100],[100]]
    # out = zeros(6) ; out[1::2] = dst -> out[1]=[20], out[3]=[100], out[5]=[60]
    # out[2*1=2] = [30]
    # out[2*0=0] = src[0] = [100] ; out[2*2=4] = src[1] = [100]
    r = solve("unmerge", x=[[[30], [20], [100], [60]]],
              src_idx=[[0, 2]], dst_idx=[[1, 1]], unm_idx=[[1]])
    _cmp(r, {"out": [[[100], [20], [30], [100], [100], [60]]], "out_len": 6},
         "unmerge.anchor2")

    # 手算锚点 3：多通道
    # x = [[5,6],[3,4],[4,5]] ; src=[0], dst=[1], unm=[1]
    # unm=[[5,6]], dst=[[3,4],[4,5]], src = dst[1] = [[4,5]]
    # out = zeros(4x2) ; out[1]=[3,4], out[3]=[4,5]
    # out[2] = [5,6] ; out[0] = [4,5]
    r = solve("unmerge", x=[[[5, 6], [3, 4], [4, 5]]],
              src_idx=[[0]], dst_idx=[[1]], unm_idx=[[1]])
    _cmp(r, {"out": [[[4, 5], [3, 4], [5, 6], [4, 5]]], "out_len": 4},
         "unmerge.anchor3")

    # ---- 网格对拍：先 merge 再 unmerge
    rng = _Rng(841104)
    for case in range(160):
        batch = 1 + case % 2
        t = 2 + (case * 5) % 12
        c = 1 + (case * 3) % 3
        t1 = (t + 1) // 2
        t2 = t // 2
        r_eff = 1 + (case * 7) % max(1, min(t1, 3))
        x = [[[rng.randint(-20, 20) for _ in range(c)]
              for _ in range(t)] for _ in range(batch)]
        src_idx, dst_idx, unm_idx = [], [], []
        for _ in range(batch):
            # src 是 argsort 的前缀，必互异；dst 可重复（多个 a 指向同一 b）
            src = rng.sample(list(range(t1)), r_eff)
            unm = [i for i in range(t1) if i not in src]
            src_idx.append(src)
            unm_idx.append(unm)
            dst_idx.append([rng.randint(0, t2 - 1) for _ in src])
        merged = oracle_merge(x, src_idx, dst_idx, unm_idx, "sum", False)
        got = solve("unmerge", x=merged["merged"], src_idx=src_idx,
                    dst_idx=dst_idx, unm_idx=unm_idx)
        exp = oracle_unmerge(merged["merged"], src_idx, dst_idx, unm_idx)
        _cmp(got, exp, "unmerge.grid[%d]" % case)
