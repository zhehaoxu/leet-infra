import torch


def solve(scores, rels, k, exp_gain=True):
    """向量化实现：torch.argsort(stable) 拿到名次，张量点乘算 DCG / IDCG。"""
    n = len(scores)
    cut = min(int(k), n)
    if n == 0 or cut <= 0:
        return 0.0, 0.0, 0.0

    sc = torch.tensor(scores, dtype=torch.float64)
    rl = torch.tensor(rels, dtype=torch.float64)

    # 同分按文档下标升序：stable 排序保持相等元素的原始相对次序
    order = torch.argsort(sc, descending=True, stable=True)
    ranked = rl[order][:cut]

    ideal = torch.sort(rl, descending=True, stable=True)[0][:cut]

    def gain(x):
        return torch.pow(2.0, x) - 1.0 if exp_gain else x

    pos = torch.arange(1, cut + 1, dtype=torch.float64)
    discount = torch.log2(pos + 1.0)

    dcg = float((gain(ranked) / discount).sum().item())
    idcg = float((gain(ideal) / discount).sum().item())
    ndcg = dcg / idcg if idcg > 0.0 else 0.0
    return ndcg, dcg, idcg
