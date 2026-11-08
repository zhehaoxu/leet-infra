"""order 841 参考解（暂存）：ToMe 的 bipartite soft matching 合并账。

官方回源：facebookresearch/ToMe (main)
  tome/merge.py::bipartite_soft_matching / merge / unmerge / merge_wavg
  tome/patch/timm.py::ToMeBlock.forward / ToMeAttention.forward

四段纯数值账，不做 ViT 前向、不做 GPU kernel、不依赖 torch/numpy。

**为什么用精确有理数口径**
官方是 `metric / metric.norm(dim=-1)` 之后 `a @ b.T`，全float64。实测在
「精确余弦完全相等」的 case 上，1e-17 的尾数差会让 `scores.max` 与
`argsort` 翻转选中项（例如 `metric=[1,1]` 与 `[2,2]` 归一化后点积是
0.9999999999999998 而不是 1.0）。为了让账本可判定，本题把
  * metric / x / size 的元素全部限定为整数；
  * 相似度比较不开根号，用整数交叉相乘：
        s_ij  = Σ_c a_i[c] * b_j[c]        （精确整数点积）
        m2_i  = Σ_c a_i[c]^2                （精确整数模长平方）
        cos_ij = s_ij / sqrt(m2_i * m2_j)   （真实值）
    比较 u = cos_ij 与 v = cos_pq 时：
        - 符号先判：非负 > 负；
        - 都非负：u >= v  <=>  s_ij^2 * m2_p * m2_q >= s_pq^2 * m2_i * m2_j
        - 都为负：u >= v  <=>  s_ij^2 * m2_p * m2_q <= s_pq^2 * m2_i * m2_j
    于是排序与 max 的选择完全精确，且与官方的**数学语义**完全一致。

**并列规则（官方浮点未定义，本题钉死）**
  * `max` 取到并列最大值时选**下标最小**者（与 torch 实测一致）；
  * `argsort(descending=True)` 视为**稳定**排序，并列时保持原下标升序
    （与 torch CPU 实测一致）。
"""

NEG_INF = None  # 内部用 None 表示「负无穷」（受保护的位置）


def _dot(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def _m2(v):
    s = 0
    for x in v:
        s += x * x
    return s


def _cos_cmp(u, v):
    """精确比较 cos_u 与 cos_v，返回 -1/0/1。

    u = (s_u, m2_u, n2_u)：cos_u = s_u / sqrt(m2_u * n2_u)
    None 表示负无穷（受保护）。
    """
    s_u, m2_u, n2_u = u
    s_v, m2_v, n2_v = v
    if s_u is None:
        return 0 if s_v is None else -1
    if s_v is None:
        return 1
    if s_u >= 0 and s_v < 0:
        return 1
    if s_v >= 0 and s_u < 0:
        return -1
    left = s_u * s_u * m2_v * n2_v
    right = s_v * s_v * m2_u * n2_u
    if s_u >= 0:
        if left > right:
            return 1
        if left < right:
            return -1
        return 0
    if left < right:
        return 1
    if left > right:
        return -1
    return 0


def _cos_value(u):
    """把 (s, m2_a, m2_b) 转成双精度余弦值；None（负无穷）返回 -inf。"""
    if u is None:
        return float("-inf")
    s, m2_a, m2_b = u
    if s is None:
        # 整行被 class_token 保护（或整列被 distill_token 保护），
        # 且它是本行的最优候选 => 官方 node_max 为 -inf。
        return float("-inf")
    if m2_a * m2_b == 0:
        return float("nan")
    return s / ((m2_a * m2_b) ** 0.5)


def _match_indices(metric, r, class_token, distill_token):
    """bipartite_soft_matching 的索引计算段（精确口径）。

    metric : [B][T][C]，元素是整数，且每行模长非零
    """
    batch = len(metric)
    t = len(metric[0])
    protected = (1 if class_token else 0) + (1 if distill_token else 0)
    r_eff = min(int(r), (t - protected) // 2)

    t1 = (t + 1) // 2
    t2 = t // 2

    src_idx = []
    dst_idx = []
    unm_idx = []
    node_max = []

    if r_eff <= 0:
        # 官方此时直接返回 do_nothing，不做任何匹配。
        # 本题口径：src/dst 为空，unm 为全部 a 侧下标（升序），
        # node_max 不计算（空）。
        all_a = list(range(t1))
        for _ in range(batch):
            src_idx.append([])
            dst_idx.append([])
            unm_idx.append(list(all_a))
            node_max.append([])
        return {
            "r_effective": r_eff,
            "src_idx": src_idx,
            "dst_idx": dst_idx,
            "unm_idx": unm_idx,
            "node_max": node_max,
            "num_src": t1,
            "num_dst": t2,
        }

    for b_i in range(batch):
        rows = metric[b_i]
        a = [list(rows[i]) for i in range(0, t, 2)]
        b = [list(rows[i]) for i in range(1, t, 2)]

        m2_a = [_m2(v) for v in a]
        m2_b = [_m2(v) for v in b]

        # scores[a][b]；零模长的行/列按负无穷处理
        scores = []
        for ai in range(t1):
            row = []
            for bi in range(t2):
                if m2_a[ai] == 0 or m2_b[bi] == 0:
                    row.append(None)
                else:
                    row.append(_dot(a[ai], b[bi]))
            scores.append(row)

        # -inf 保护在两个不同的轴上：
        #   class_token  -> scores[..., 0, :]  （第 0 行整行）
        #   distill_token-> scores[..., :, 0]  （第 0 列整列）
        if class_token:
            for bi in range(t2):
                scores[0][bi] = None
        if distill_token:
            for ai in range(t1):
                scores[ai][0] = None

        # 每个 a 元素的最相似 b（并列取下标最小）
        nmax = []
        nidx = []
        for ai in range(t1):
            best = scores[ai][0]
            best_j = 0
            for bi in range(1, t2):
                c = _cos_cmp((scores[ai][bi], m2_a[ai], m2_b[bi]),
                             (best, m2_a[ai], m2_b[best_j]))
                if c > 0:
                    best = scores[ai][bi]
                    best_j = bi
            nmax.append(best)
            nidx.append(best_j)

        # argsort(descending=True)：稳定，并列时按下标升序
        order = list(range(t1))
        for i in range(1, t1):
            key = order[i]
            j = i - 1
            while j >= 0 and _cos_cmp(
                    (nmax[key], m2_a[key], m2_b[nidx[key]]),
                    (nmax[order[j]], m2_a[order[j]], m2_b[nidx[order[j]]])) > 0:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        src = order[:r_eff]
        unm = order[r_eff:]

        # 只有 class_token 时才把 unm 排序（保证 class token 在最前）
        if class_token:
            unm = sorted(unm)

        src_idx.append(list(src))
        dst_idx.append([nidx[i] for i in src])
        unm_idx.append(list(unm))
        node_max.append([
            _cos_value((nmax[i], m2_a[i], m2_b[nidx[i]]))
            for i in range(t1)
        ])

    return {
        "r_effective": r_eff,
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "unm_idx": unm_idx,
        "node_max": node_max,
        "num_src": t1,
        "num_dst": t2,
    }


def _scatter_reduce_mean(dst_row, src_rows, idxs):
    """scatter_reduce(reduce='mean')，include_self=True（官方默认）。

    分母 = 1（dst 自身）+ 落到该位置的 src 个数。
    """
    counts = [1] * len(dst_row)
    sums = [list(v) for v in dst_row]
    for src, j in zip(src_rows, idxs):
        for c in range(len(sums[j])):
            sums[j][c] += src[c]
        counts[j] += 1
    return [[sums[j][c] / counts[j] for c in range(len(sums[j]))]
            for j in range(len(sums))]


def _scatter_reduce_sum(dst_row, src_rows, idxs):
    sums = [list(v) for v in dst_row]
    for src, j in zip(src_rows, idxs):
        for c in range(len(sums[j])):
            sums[j][c] += src[c]
    return sums


def _scatter_reduce_amax(dst_row, src_rows, idxs):
    out = [list(v) for v in dst_row]
    for src, j in zip(src_rows, idxs):
        for c in range(len(out[j])):
            if src[c] > out[j][c]:
                out[j][c] = src[c]
    return out


def _gather(rows, idxs):
    return [list(rows[i]) for i in idxs]


def _merge_reduce(x, src_idx, dst_idx, unm_idx, reduce_mode, distill_token):
    """merge 闭包：gather + scatter_reduce + 输出重排。

    输出布局（官方）：
      distill_token=False -> cat([unm, dst])
      distill_token=True  -> cat([unm[:1], dst[:1], unm[1:], dst[1:]])

    r<=0 时官方返回 do_nothing（恒等），此时 src_idx 为空 => 输出即输入。
    """
    batch = len(x)
    t = len(x[0])

    merged = []
    dst_values = []
    for b_i in range(batch):
        rows = x[b_i]
        a_rows = [list(rows[i]) for i in range(0, t, 2)]
        b_rows = [list(rows[i]) for i in range(1, t, 2)]

        if not src_idx[b_i]:
            # do_nothing：恒等
            merged.append([list(v) for v in rows])
            dst_values.append([list(v) for v in b_rows])
            continue

        unm = _gather(a_rows, unm_idx[b_i])
        src = _gather(a_rows, src_idx[b_i])
        idxs = dst_idx[b_i]

        if reduce_mode == "mean":
            dst = _scatter_reduce_mean(b_rows, src, idxs)
        elif reduce_mode == "sum":
            dst = _scatter_reduce_sum(b_rows, src, idxs)
        elif reduce_mode == "amax":
            dst = _scatter_reduce_amax(b_rows, src, idxs)
        else:
            raise ValueError("unknown reduce: %s" % reduce_mode)

        dst_values.append([list(v) for v in dst])

        if distill_token:
            merged.append([list(unm[0])] + [list(dst[0])]
                          + [list(v) for v in unm[1:]]
                          + [list(v) for v in dst[1:]])
        else:
            merged.append([list(v) for v in unm] + [list(v) for v in dst])

    return {
        "merged": merged,
        "merged_len": len(merged[0]) if merged else 0,
        "dst_values": dst_values,
        "num_src": (t + 1) // 2,
        "num_dst": t // 2,
    }


def _size_weighted_merge(x, size, src_idx, dst_idx, unm_idx, distill_token):
    """merge_wavg：x = merge(x*size, "sum")；size = merge(size, "sum")；x = x / size。

    * 分母是 **size 之和**，不是 token 个数。
    * 官方 merge 闭包内部处理 distill_token 的交错布局，所以 size 的合并
      也走同一套布局。
    * r<=0 时 merge 是 do_nothing => x 恒等、size 恒等。
    """
    batch = len(x)
    t = len(x[0])
    c = len(x[0][0])

    # 官方是 x = merge(x * size, "sum")：先逐元素乘，再整体合并
    scaled = []
    for b_i in range(batch):
        rows = x[b_i]
        srows = size[b_i]
        brows = []
        for ti in range(t):
            row = []
            for ci in range(c):
                row.append(rows[ti][ci] * srows[ti][0])
            brows.append(row)
        scaled.append(brows)

    xm = _merge_reduce(scaled, src_idx, dst_idx, unm_idx, "sum",
                        distill_token)
    sm = _merge_reduce(size, src_idx, dst_idx, unm_idx, "sum",
                       distill_token)

    merged = []
    merged_size = []
    for b_i in range(batch):
        xs = xm["merged"][b_i]
        ss = sm["merged"][b_i]
        out = []
        out_s = []
        for ti in range(len(xs)):
            denom = ss[ti][0]
            row = []
            for ci in range(c):
                row.append(xs[ti][ci] / denom if denom != 0.0 else float("nan"))
            out.append(row)
            out_s.append([ss[ti][0]])
        merged.append(out)
        merged_size.append(out_s)

    return {
        "merged": merged,
        "merged_size": merged_size,
        "merged_len": len(merged[0]) if merged else 0,
    }


def _unmerge(x, src_idx, dst_idx, unm_idx):
    """unmerge 闭包（distill_token=False 口径）。

    输入是**合并后**的张量（m = T - r 行），输出还原成 T 行。
    官方：
      unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
      src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
      out = zeros(n, T, c)
      out[..., 1::2, :] = dst
      out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
      out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
    """
    batch = len(x)
    m = len(x[0])
    c = len(x[0][0])
    unm_len = len(unm_idx[0])
    r = len(src_idx[0])
    t = m + r          # 原始 token 数：合并后 m = T - r

    out = []
    for b_i in range(batch):
        rows = x[b_i]
        unm = [list(rows[i]) for i in range(unm_len)]
        dst = [list(rows[i]) for i in range(unm_len, m)]
        src = _gather(dst, dst_idx[b_i])

        res = [[0] * c for _ in range(t)]
        for i in range(len(dst)):
            res[2 * i + 1] = list(dst[i])
        for k, ai in enumerate(unm_idx[b_i]):
            res[2 * ai] = list(unm[k])
        for k, ai in enumerate(src_idx[b_i]):
            res[2 * ai] = list(src[k])
        out.append(res)

    return {
        "out": out,
        "out_len": len(out[0]) if out else 0,
    }


def solve(mode: str, **kw):
    if mode == "match_indices":
        metric = [[list(v) for v in row] for row in kw["metric"]]
        return _match_indices(
            metric,
            int(kw.get("r", 0)),
            bool(kw.get("class_token", False)),
            bool(kw.get("distill_token", False)),
        )
    if mode == "merge_reduce":
        x = [[list(v) for v in row] for row in kw["x"]]
        return _merge_reduce(
            x,
            [list(v) for v in kw["src_idx"]],
            [list(v) for v in kw["dst_idx"]],
            [list(v) for v in kw["unm_idx"]],
            str(kw.get("reduce", "mean")),
            bool(kw.get("distill_token", False)),
        )
    if mode == "size_weighted_merge":
        x = [[list(v) for v in row] for row in kw["x"]]
        size = [[list(v) for v in row] for row in kw["size"]]
        return _size_weighted_merge(
            x, size,
            [list(v) for v in kw["src_idx"]],
            [list(v) for v in kw["dst_idx"]],
            [list(v) for v in kw["unm_idx"]],
            bool(kw.get("distill_token", False)),
        )
    if mode == "unmerge":
        x = [[list(v) for v in row] for row in kw["x"]]
        return _unmerge(
            x,
            [list(v) for v in kw["src_idx"]],
            [list(v) for v in kw["dst_idx"]],
            [list(v) for v in kw["unm_idx"]],
        )
    raise ValueError("unknown mode: %s" % mode)
