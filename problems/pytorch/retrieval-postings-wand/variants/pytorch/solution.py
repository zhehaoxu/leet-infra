"""参考答案：skip 指针对齐的 AND 求交 + Block-Max WAND top-k。

AND 与 WAND 共用同一套 postings 块：每条表按 block_size 连续行切块，块尾
docid 即 skip 目标。WAND 的 pivot 只用「当前块」的块内最大贡献作上界，并把
候选 docid 限制在 min(当前块尾) 之内，保证块上界确实覆盖该区间；无法超过
当前 top-k 阈值的整块会被跳过并计入 pruned_blocks。
"""

import heapq
import math

import torch

_INF = float("inf")


def _check(postings, doc_lens, k, k1, b, block_size):
    if not isinstance(doc_lens, torch.Tensor) or doc_lens.dim() != 1:
        raise ValueError("doc_lens 必须是一维整型张量")
    if doc_lens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("doc_lens 必须是整型张量")
    if doc_lens.numel() == 0:
        raise ValueError("N 必须 >= 1")
    if int(doc_lens.min()) < 0:
        raise ValueError("doc_lens 不能为负")
    if not isinstance(postings, (list, tuple)):
        raise ValueError("postings 必须是 list/tuple（每个元素一条 term 的倒排表）")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k 必须是 >= 0 的整数")
    if not isinstance(k1, (int, float)) or isinstance(k1, bool) or not k1 > 0:
        raise ValueError("k1 必须是 > 0 的浮点数")
    if not isinstance(b, (int, float)) or isinstance(b, bool) or not 0.0 <= b <= 1.0:
        raise ValueError("b 必须落在 [0, 1]")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size 必须是 >= 1 的整数")
    n = doc_lens.numel()
    for rows in postings:
        if not isinstance(rows, torch.Tensor) or rows.dim() != 2 or rows.shape[1] != 2:
            raise ValueError("每条 postings 必须是 (n, 2) 的整型张量")
        if rows.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("每条 postings 必须是整型张量")
        prev = -1
        for docid, tf in rows.tolist():
            if docid <= prev:
                raise ValueError("每条 postings 的 docid 必须严格递增")
            if docid < 0 or docid >= n:
                raise ValueError(f"docid 必须落在 [0, {n})")
            if tf < 1:
                raise ValueError("tf 必须 >= 1")
            prev = docid


def _make_terms(postings, dl, avgdl, k1, b, block_size):
    n = len(dl)
    terms = []
    for idx, rows in enumerate(postings):
        pairs = rows.tolist()
        docs = [int(p[0]) for p in pairs]
        tfs = [int(p[1]) for p in pairs]
        df = len(docs)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        contribs = []
        for d, tf in zip(docs, tfs):
            if avgdl > 0.0:
                norm = k1 * (1.0 - b + b * dl[d] / avgdl)
            else:
                norm = k1 * (1.0 - b)
            contribs.append(idf * tf * (k1 + 1.0) / (tf + norm))
        blocks = []
        for start in range(0, df, block_size):
            end_idx = min(start + block_size, df) - 1
            blocks.append((end_idx, docs[end_idx],
                           max(contribs[start:end_idx + 1])))
        terms.append({"idx": idx, "docs": docs, "contribs": contribs,
                      "blocks": blocks, "p": 0})
    return terms


def _advance(term, target, block_size):
    """把指针推进到第一条 docid >= target 的记录，返回整块跳过的块数。"""
    docs = term["docs"]
    n = len(docs)
    p = term["p"]
    pruned = 0
    while p < n and docs[p] < target:
        start = (p // block_size) * block_size
        end = min(start + block_size, n) - 1
        if p == start and docs[end] < target:
            pruned += 1
            p = end + 1
        else:
            p += 1
    term["p"] = p
    return pruned


def _and_candidates(terms, block_size):
    if not terms:
        return []
    if any(len(t["docs"]) == 0 for t in terms):
        return []
    for t in terms:
        t["p"] = 0
    cand = max(t["docs"][0] for t in terms)
    out = []
    while True:
        for t in terms:
            _advance(t, cand, block_size)
        if any(t["p"] >= len(t["docs"]) for t in terms):
            break
        cur = [t["docs"][t["p"]] for t in terms]
        top = max(cur)
        if all(c == top for c in cur):
            out.append(top)
            for t in terms:
                t["p"] += 1
            if any(t["p"] >= len(t["docs"]) for t in terms):
                break
            cand = max(t["docs"][t["p"]] for t in terms)
        else:
            cand = top
    return out


def _block_of(term, block_size):
    return term["blocks"][term["p"] // block_size]


def _wand(terms, k, block_size):
    if k == 0 or not terms:
        return [], 0
    for t in terms:
        t["p"] = 0
    heap = []
    pruned = 0
    while True:
        active = [t for t in terms if t["p"] < len(t["docs"])]
        if not active:
            break
        active.sort(key=lambda t: (t["docs"][t["p"]], t["idx"]))
        lo = active[0]["docs"][active[0]["p"]]
        hi = min(_block_of(t, block_size)[1] for t in active)
        threshold = heap[0][0] if len(heap) == k else 0.0

        acc = 0.0
        pivot = None
        for t in active:
            cur = t["docs"][t["p"]]
            if cur > hi:
                break
            acc += _block_of(t, block_size)[2]
            if acc > threshold:
                pivot = cur
                break

        if pivot is None:
            # [lo, hi] 内所有文档的上界和都不超过阈值：整段跳过
            for t in active:
                if t["docs"][t["p"]] <= hi:
                    pruned += _advance(t, hi + 1, block_size)
            continue

        if pivot == lo:
            score = 0.0
            for t in terms:
                if t["p"] < len(t["docs"]) and t["docs"][t["p"]] == lo:
                    score += t["contribs"][t["p"]]
            item = (score, -lo)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            for t in terms:
                if t["p"] < len(t["docs"]) and t["docs"][t["p"]] == lo:
                    t["p"] += 1
        else:
            for t in active:
                if t["docs"][t["p"]] < pivot:
                    pruned += _advance(t, pivot, block_size)

    ranked = sorted(heap, key=lambda item: (-item[0], -item[1]))
    return [-neg_d for _, neg_d in ranked], pruned


def solve(postings, doc_lens, k=10, k1=1.2, b=0.75, block_size=4):
    _check(postings, doc_lens, k, k1, b, block_size)
    dl = doc_lens.to(torch.int64).tolist()
    avgdl = sum(dl) / len(dl)
    terms = _make_terms(postings, dl, avgdl, float(k1), float(b), block_size)
    candidates = _and_candidates(terms, block_size)
    topk, pruned = _wand(terms, k, block_size)
    return (torch.tensor(candidates, dtype=torch.int64),
            torch.tensor(topk, dtype=torch.int64), pruned)
