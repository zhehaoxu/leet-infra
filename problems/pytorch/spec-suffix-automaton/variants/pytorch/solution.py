import torch


def _build_sam(seq):
    """增量构建 SAM，返回 (trans, link, length)；link[0] = -1。"""
    trans = [{}]
    link = [-1]
    length = [0]
    last = 0
    for ch in seq:
        cur = len(trans)
        trans.append({})
        length.append(length[last] + 1)
        link.append(0)
        p = last
        while p != -1 and ch not in trans[p]:
            trans[p][ch] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = trans[p][ch]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(trans)                       # 分裂状态：q 的后缀被两个等价类共用
                trans.append(dict(trans[q]))
                length.append(length[p] + 1)
                link.append(link[q])
                while p != -1 and trans[p].get(ch) == q:
                    trans[p][ch] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        last = cur
    return trans, link, length


def solve(text, pattern):
    """SAM 增量构建 + 最长后缀匹配走转移（复杂度 O(N) 建机、O(P) 查询）。"""
    if text.ndim != 1 or pattern.ndim != 1:
        raise ValueError("text 与 pattern 都必须是一维")
    if int(text.numel()) < 1:
        raise ValueError("text 不能为空")

    trans, link, length = _build_sam([int(x) for x in text.tolist()])
    distinct = sum(length[s] - length[link[s]] for s in range(1, len(trans)))

    matched = []
    state, best = 0, 0
    for ch in [int(x) for x in pattern.tolist()]:
        while state != 0 and ch not in trans[state]:
            state = link[state]                          # 失配就沿后缀链接回退
            best = length[state]
        if ch in trans[state]:
            state = trans[state][ch]
            best += 1
        else:
            best = 0
        matched.append(best)

    return (torch.tensor(len(trans), dtype=torch.int64),
            torch.tensor(distinct, dtype=torch.int64),
            torch.tensor(matched, dtype=torch.int64))
