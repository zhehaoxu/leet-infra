"""参考答案：IVF-PQ 非对称 ADC 打分。

查询保持原精度，先按 query 到粗中心的平方欧氏距离选 nprobe 个 cell，
再对每个候选码字用子空间 LUT 求和得到近似距离；最后按
(距离升序, 全局 id 升序) 取 top-k，不足 k 个用 id=-1 / 距离=+inf 补齐。
内部距离一律 float64 累加，输出时再转回 queries 的 dtype。
"""

import torch


def _as_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} 必须是 int，got {type(value).__name__}")
    return value


def solve(queries, centroids, codebooks, invlist_ids, invlist_codes,
          nprobe=1, k=10):
    if queries.dim() != 2:
        raise ValueError(f"queries 必须是 2-D，got {queries.dim()}-D")
    if centroids.dim() != 2:
        raise ValueError(f"centroids 必须是 2-D，got {centroids.dim()}-D")
    if codebooks.dim() != 3:
        raise ValueError(f"codebooks 必须是 3-D，got {codebooks.dim()}-D")

    q, dim = queries.shape
    nc = centroids.shape[0]
    m, ksub, d = codebooks.shape
    if dim <= 0 or d <= 0 or m <= 0 or ksub <= 0:
        raise ValueError(
            f"要求 D>0、m>0、K>0、d>0，got D={dim}, m={m}, K={ksub}, d={d}")
    if dim != m * d:
        raise ValueError(f"D 必须等于 m*d，got D={dim}, m*d={m * d}")
    if centroids.shape[1] != dim:
        raise ValueError(
            f"centroids 第二维必须为 D={dim}，got {centroids.shape[1]}")
    if not (queries.dtype == centroids.dtype == codebooks.dtype):
        raise ValueError(
            "queries/centroids/codebooks 必须同 dtype，got "
            f"{queries.dtype}/{centroids.dtype}/{codebooks.dtype}")
    if queries.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"只支持 float32/float64，got {queries.dtype}")

    if len(invlist_ids) != nc or len(invlist_codes) != nc:
        raise ValueError(
            "invlist_ids/invlist_codes 的长度必须等于粗中心数 C="
            f"{nc}，got {len(invlist_ids)}/{len(invlist_codes)}")
    for c in range(nc):
        ids = invlist_ids[c]
        codes = invlist_codes[c]
        if ids.dim() != 1:
            raise ValueError(f"cell {c} 的 ids 必须是 1-D，got {ids.dim()}-D")
        if ids.dtype != torch.int64:
            raise ValueError(f"cell {c} 的 ids 必须是 int64，got {ids.dtype}")
        if codes.dim() != 2 or codes.shape[1] != m:
            raise ValueError(
                f"cell {c} 的 codes 必须是 (n, {m})，got {tuple(codes.shape)}")
        if codes.dtype != torch.int64:
            raise ValueError(f"cell {c} 的 codes 必须是 int64，got {codes.dtype}")
        if ids.numel() != codes.shape[0]:
            raise ValueError(
                f"cell {c} 的 ids/codes 行数不一致：{ids.numel()} vs {codes.shape[0]}")
        if ids.numel():
            if bool((ids < 0).any()):
                raise ValueError(f"cell {c} 含负的全局 id")
            if bool((codes < 0).any()) or bool((codes >= ksub).any()):
                raise ValueError(f"cell {c} 的码字下标越界，要求 [0, {ksub})")

    nprobe = _as_int("nprobe", nprobe)
    k = _as_int("k", k)
    if nprobe < 0:
        raise ValueError(f"nprobe 必须 >= 0，got {nprobe}")
    if k < 0:
        raise ValueError(f"k 必须 >= 0，got {k}")

    qf = queries.to(torch.float64)
    cf = centroids.to(torch.float64)
    cbf = codebooks.to(torch.float64)

    # (Q, C) query 到粗中心的平方距离；stable sort 保证并列时 cell 号小者在前
    cd2 = ((qf[:, None, :] - cf[None, :, :]) ** 2).sum(dim=-1)
    p = min(nprobe, nc)
    order = torch.argsort(cd2, dim=1, stable=True)[:, :p]

    # (Q, m, K) 每个查询、每个子空间、每个码字的 LUT
    lut = ((qf.reshape(q, m, d)[:, :, None, :] - cbf[None, :, :, :]) ** 2).sum(-1)

    out_ids = torch.full((q, k), -1, dtype=torch.int64)
    out_dist = torch.full((q, k), float("inf"), dtype=queries.dtype)
    ar = torch.arange(m)

    for qi in range(q):
        id_parts, code_parts = [], []
        for pos in range(p):
            c = int(order[qi, pos])
            ids = invlist_ids[c]
            if ids.numel() == 0:
                continue
            id_parts.append(ids)
            code_parts.append(invlist_codes[c])
        if not id_parts:
            continue
        ids_all = torch.cat(id_parts)
        codes_all = torch.cat(code_parts)
        # (n, m) 的 LUT 查表后沿子空间求和
        dist_all = lut[qi][ar[None, :], codes_all].sum(dim=1)

        # 先按 id 升序，再稳定地按距离升序 -> (距离, id) 字典序
        o1 = torch.argsort(ids_all, stable=True)
        ids1, dist1 = ids_all[o1], dist_all[o1]
        o2 = torch.argsort(dist1, stable=True)
        ids2, dist2 = ids1[o2], dist1[o2]

        t = min(k, ids2.numel())
        if t:
            out_ids[qi, :t] = ids2[:t]
            out_dist[qi, :t] = dist2[:t].to(queries.dtype)

    return out_ids, out_dist
