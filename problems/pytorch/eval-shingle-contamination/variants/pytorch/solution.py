import torch


def solve(eval_docs, train_docs, n, threshold):
    """元组集合实现：shingle 直接做成 set[tuple[int]]，用集合交并算重叠。"""
    n = int(n)

    def shingles(doc):
        out = set()
        L = len(doc)
        for i in range(L - n + 1):
            out.add(tuple(int(x) for x in doc[i:i + n]))
        return out

    eval_sets = [shingles(d) for d in eval_docs]
    train_sets = [shingles(d) for d in train_docs]

    m_eval = len(eval_sets)
    m_train = len(train_sets)
    contam = torch.zeros(m_eval, m_train, dtype=torch.float32)
    jac = torch.zeros(m_eval, m_train, dtype=torch.float32)

    if m_eval == 0 or m_train == 0:
        return contam, jac, torch.empty(0, dtype=torch.int64)

    b_sizes = torch.tensor([len(s) for s in train_sets], dtype=torch.int64)

    flagged = []
    for i, a in enumerate(eval_sets):
        size_a = len(a)
        if size_a == 0:
            continue                      # 空 shingle 集：整行 0，永不进 flagged
        inter = torch.tensor([len(a & b) for b in train_sets], dtype=torch.float64)
        c = inter / float(size_a)
        union = float(size_a) + b_sizes.double() - inter
        jac_row = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        contam[i] = c.float()
        jac[i] = jac_row.float()
        if float(c.max().item()) >= float(threshold):
            flagged.append(i)

    return contam, jac, torch.tensor(flagged, dtype=torch.int64)
