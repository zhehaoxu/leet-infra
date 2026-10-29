"""参考答案：BM25 全库打分（向量化）+ 按 (-score, index) 稳定取 top-k。

df 与 avgdl 全部由 docs/doc_lens 现算；idf_mode="robertson" 在 df > N/2 时
按原式为负，不截断。重复 query token 按出现次数累加。
"""

import math

import torch

_IDF_MODES = ("lucene", "robertson")


def _check(docs, doc_lens, query, k, k1, b, idf_mode):
    if not isinstance(docs, torch.Tensor) or docs.dim() != 2:
        raise ValueError("docs 必须是二维整型张量")
    if docs.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("docs 必须是整型张量")
    if not isinstance(doc_lens, torch.Tensor) or doc_lens.dim() != 1:
        raise ValueError("doc_lens 必须是一维整型张量")
    if doc_lens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("doc_lens 必须是整型张量")
    if not isinstance(query, torch.Tensor) or query.dim() != 1:
        raise ValueError("query 必须是一维整型张量")
    if query.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("query 必须是整型张量")
    n, length = docs.shape
    if n == 0:
        raise ValueError("N 必须 >= 1（avgdl 需要至少一篇文档）")
    if doc_lens.numel() != n:
        raise ValueError(f"doc_lens 长度必须为 {n}，got {doc_lens.numel()}")
    if int(doc_lens.min()) < 0 or int(doc_lens.max()) > length:
        raise ValueError(f"doc_lens 必须落在 [0, {length}]")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k 必须是 >= 0 的整数")
    if not isinstance(k1, (int, float)) or isinstance(k1, bool) or not k1 > 0:
        raise ValueError("k1 必须是 > 0 的浮点数")
    if not isinstance(b, (int, float)) or isinstance(b, bool) or not 0.0 <= b <= 1.0:
        raise ValueError("b 必须落在 [0, 1]")
    if idf_mode not in _IDF_MODES:
        raise ValueError(f"idf_mode 必须是 {_IDF_MODES} 之一，got {idf_mode!r}")


def solve(docs, doc_lens, query, k=10, k1=1.2, b=0.75, idf_mode="lucene"):
    _check(docs, doc_lens, query, k, k1, b, idf_mode)
    n, length = docs.shape
    lens = doc_lens.to(torch.int64)
    dl = lens.to(torch.float64)
    avgdl = dl.mean()

    valid = torch.arange(length).unsqueeze(0) < lens.unsqueeze(1)  # (N, L)
    tokens = docs[valid]
    if tokens.numel() and int(tokens.min()) < 0:
        raise ValueError("有效位置上的 token id 必须 >= 0")
    if query.numel() and int(query.min()) < 0:
        raise ValueError("query 中的 token id 必须 >= 0")

    scores = torch.zeros(n, dtype=torch.float64)
    k1f, bf = float(k1), float(b)
    for t in query.tolist():
        present = valid & (docs == int(t))
        tf = present.sum(dim=1).to(torch.float64)
        df = int((tf > 0).sum().item())
        if idf_mode == "lucene":
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        else:
            idf = math.log((n - df + 0.5) / (df + 0.5))
        if float(avgdl) > 0.0:
            norm = k1f * (1.0 - bf + bf * dl / avgdl)
        else:
            norm = torch.full((n,), k1f * (1.0 - bf), dtype=torch.float64)
        denom = tf + norm
        contrib = torch.where(tf > 0,
                              idf * tf * (k1f + 1.0) / denom,
                              torch.zeros_like(tf))
        scores = scores + contrib

    take = min(k, n)
    topk = torch.argsort(-scores, stable=True)[:take].to(torch.int64)
    return scores, topk
