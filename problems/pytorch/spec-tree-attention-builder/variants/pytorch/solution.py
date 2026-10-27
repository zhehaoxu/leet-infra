import heapq

import torch


def solve(cand_tokens, cand_lps, budget, prefix_len):
    """Medusa/EAGLE 风格候选树：按累积 logprob 做 best-first 扩张，输出树结构。"""
    if cand_tokens.ndim != 2 or cand_lps.ndim != 2:
        raise ValueError("候选必须是二维")
    if cand_tokens.shape != cand_lps.shape:
        raise ValueError("候选 token 与 logprob 形状必须一致")
    heads, k = cand_tokens.shape
    if heads < 1 or k < 1:
        raise ValueError("至少需要一个 head 和一个候选")
    if budget < 1:
        raise ValueError("budget 必须 >= 1")
    if prefix_len < 1:
        raise ValueError("prefix_len 必须 >= 1")
    if bool((cand_lps > 0).any()):
        raise ValueError("候选 logprob 必须 <= 0")

    lps = cand_lps.to(torch.float64)
    # 每个 head 内部按「分数降序 + 候选下标升序」，入队顺序因此确定
    orders = [sorted(range(k), key=lambda j: (-float(lps[h, j]), j)) for h in range(heads)]

    path_parent = {(): -1}
    node_tokens, node_depth, node_parent, paths = [], [], [], []

    heap = []
    for j in orders[0]:
        heapq.heappush(heap, (-float(lps[0, j]), (j,)))
    while heap and len(paths) < budget:
        neg, path = heapq.heappop(heap)
        depth = len(path)
        node_tokens.append(int(cand_tokens[depth - 1, path[-1]]))
        node_depth.append(depth)
        node_parent.append(path_parent[path[:-1]])
        paths.append(path)
        path_parent[path] = len(paths) - 1
        if depth < heads:
            # 只有弹出了父节点才把它挂到队里，保证节点集前缀封闭
            for j in orders[depth]:
                heapq.heappush(heap, (neg - float(lps[depth, j]), path + (j,)))

    n = len(paths)
    mask = torch.zeros(n, n + prefix_len, dtype=torch.bool)
    mask[:, :prefix_len] = True                      # 已确认前缀对所有 draft 位置可见
    for i, path in enumerate(paths):
        for d in range(1, len(path) + 1):
            mask[i, prefix_len + path_parent[path[:d]]] = True   # 自己 + 祖先

    depth = torch.tensor(node_depth, dtype=torch.int64)
    return (torch.tensor(node_parent, dtype=torch.int64),
            torch.tensor(node_tokens, dtype=torch.int64),
            depth,
            depth + prefix_len - 1,
            mask)
