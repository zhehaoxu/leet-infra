import torch


def _norm_rows(x):
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def solve(query_emb, doc_embs, dims, top_k):
    """Matryoshka 截断检索账（规则见 starter 注释）。"""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    D = query_emb.shape[-1]
    if doc_embs.shape[-1] != D:
        raise ValueError("doc_embs dimension mismatch")
    for d in dims:
        if not isinstance(d, int) or isinstance(d, bool) or d < 1 or d > D:
            raise ValueError(f"invalid truncation dim: {d}")
    if top_k > doc_embs.shape[0]:
        raise ValueError("top_k exceeds number of docs")

    q64 = query_emb.double()
    docs64 = doc_embs.double()

    def rank_at(d):
        q = _norm_rows(q64[:d])
        docs = _norm_rows(docs64[:, :d])
        sims = docs @ q
        order = sorted(range(sims.shape[0]), key=lambda i: (-sims[i].item(), i))
        return {
            "sims": [s.item() for s in sims],
            "top": order[:top_k],
        }

    full = rank_at(D)
    truncated = {d: rank_at(d) for d in dims}
    full_set = set(full["top"])
    for d, led in truncated.items():
        led["overlap_full"] = len(set(led["top"]) & full_set)

    return {
        "full": full,
        "truncated": truncated,
        "n_docs": doc_embs.shape[0],
    }
