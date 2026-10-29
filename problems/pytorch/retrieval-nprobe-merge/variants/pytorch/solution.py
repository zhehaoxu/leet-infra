"""参考答案：nprobe 多探测与跨 cell 归并。

每个查询先按到粗中心的平方欧氏距离确定 cell 探测顺序（并列取小 cell 号），
按该顺序收集候选 posting；同一全局 id 多次出现时只保留最小距离（用
(id, dist) 双稳定排序后取每个 id 的首行），最后按 (距离, id) 升序取 top-k。
"""

import torch


def _as_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} 必须是 int，got {type(value).__name__}")
    return value


def solve(queries, centroids, invlist_vectors, invlist_ids,
          nprobe=1, k=10):
    if queries.dim() != 2:
        raise ValueError(f"queries 必须是 2-D，got {queries.dim()}-D")
    if centroids.dim() != 2:
        raise ValueError(f"centroids 必须是 2-D，got {centroids.dim()}-D")
    q, dim = queries.shape
    nc = centroids.shape[0]
    if dim <= 0:
        raise ValueError(f"D 必须 > 0，got {dim}")
    if centroids.shape[1] != dim:
        raise ValueError(
            f"centroids 第二维必须为 D={dim}，got {centroids.shape[1]}")
    if queries.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"queries 只支持 float32/float64，got {queries.dtype}")
    if centroids.dtype != queries.dtype:
        raise ValueError(
            f"centroids 必须与 queries 同 dtype，got {centroids.dtype}")
    if len(invlist_vectors) != nc or len(invlist_ids) != nc:
        raise ValueError(
            "invlist_vectors/invlist_ids 的长度必须等于 C="
            f"{nc}，got {len(invlist_vectors)}/{len(invlist_ids)}")
    for c in range(nc):
        vecs = invlist_vectors[c]
        ids = invlist_ids[c]
        if vecs.dim() != 2 or vecs.shape[1] != dim:
            raise ValueError(
                f"cell {c} 的 vectors 必须是 (n, {dim})，got {tuple(vecs.shape)}")
        if vecs.dtype != queries.dtype:
            raise ValueError(
                f"cell {c} 的 vectors dtype 必须为 {queries.dtype}，got {vecs.dtype}")
        if ids.dim() != 1:
            raise ValueError(f"cell {c} 的 ids 必须是 1-D，got {ids.dim()}-D")
        if ids.dtype != torch.int64:
            raise ValueError(f"cell {c} 的 ids 必须是 int64，got {ids.dtype}")
        if ids.numel() != vecs.shape[0]:
            raise ValueError(
                f"cell {c} 的 ids/vectors 行数不一致："
                f"{ids.numel()} vs {vecs.shape[0]}")
        if ids.numel() and bool((ids < 0).any()):
            raise ValueError(f"cell {c} 含负的全局 id")

    nprobe = _as_int("nprobe", nprobe)
    k = _as_int("k", k)
    if nprobe < 0:
        raise ValueError(f"nprobe 必须 >= 0，got {nprobe}")
    if k < 0:
        raise ValueError(f"k 必须 >= 0，got {k}")

    qf = queries.to(torch.float64)
    cf = centroids.to(torch.float64)
    # (Q, C) + stable sort：并列时 cell 号小者优先
    cd2 = ((qf[:, None, :] - cf[None, :, :]) ** 2).sum(dim=-1)
    p = min(nprobe, nc)
    cells = torch.argsort(cd2, dim=1, stable=True)[:, :p].to(torch.int64)

    out_ids = torch.full((q, k), -1, dtype=torch.int64)
    out_dist = torch.full((q, k), float("inf"), dtype=queries.dtype)
    stats = torch.zeros((q, 4), dtype=torch.int64)

    for qi in range(q):
        id_parts, dist_parts = [], []
        for pos in range(p):
            c = int(cells[qi, pos])
            ids = invlist_ids[c]
            if ids.numel() == 0:
                continue
            vecs = invlist_vectors[c].to(torch.float64)
            dd = ((qf[qi] - vecs) ** 2).sum(dim=1)
            id_parts.append(ids)
            dist_parts.append(dd)
        if not id_parts:
            continue
        ids_all = torch.cat(id_parts)
        dist_all = torch.cat(dist_parts)
        n_post = ids_all.numel()

        # 先按距离稳定排序，再按 id 稳定排序：每个 id 的首行即最小距离
        o1 = torch.argsort(dist_all, stable=True)
        ids1, dist1 = ids_all[o1], dist_all[o1]
        o2 = torch.argsort(ids1, stable=True)
        ids2, dist2 = ids1[o2], dist1[o2]
        keep = torch.ones(n_post, dtype=torch.bool)
        if n_post > 1:
            keep[1:] = ids2[1:] != ids2[:-1]
        uniq_ids, uniq_dist = ids2[keep], dist2[keep]

        # 去重后的 id 已升序；对距离做稳定排序 -> (距离, id) 字典序
        o3 = torch.argsort(uniq_dist, stable=True)
        uniq_ids, uniq_dist = uniq_ids[o3], uniq_dist[o3]

        n_uniq = uniq_ids.numel()
        t = min(k, n_uniq)
        if t:
            out_ids[qi, :t] = uniq_ids[:t]
            out_dist[qi, :t] = uniq_dist[:t].to(queries.dtype)
        stats[qi] = torch.tensor(
            [n_post, n_uniq, n_post - n_uniq, t], dtype=torch.int64)

    return out_ids, out_dist, cells, stats
