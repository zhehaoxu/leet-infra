import torch


def solve(probs, k, capacity, jitter):
    """中文契约 / English contract
    probs [T,E] 与 jitter [T,E] 都是 CPU float64、非空 2-D、有限值；probs 元素非负，行不必归一化。
    k、capacity 是 python int（bool 不算）：1 <= k <= E，capacity >= 0。
    候选：每行按 probs 降序取 top-k，值相同则专家索引小者优先；jitter 不参与候选筛选 /
        candidates are the top-k of probs[t] (ties -> smaller expert index); jitter never changes the set.
    优先级键 key[t][j] = probs[t][cand[t][j]] + jitter[t][cand[t][j]]；所有槽位 s = t*k + j
    按 (-key, s) 升序**全局**处理，不是按 token 顺序 / dispatch all slots in descending key order, ties by slot id.
    占位：按处理顺序贪心，每个专家最多 capacity 个槽位；自己的专家满了就沿秩 j+1..k-1（不回绕）
    转投第一个还有容量、且尚未被本 token 占用的专家；都不可用则 assign = -1 /
        spill forward along the rank, never backward; a token never occupies the same expert twice.
    权重只在该 token **存活**的槽位上重新归一化：weights[t][j] = probs[t][assign[t][j]] / sum_j'(...)，
    分母为 0 时整行给 0；assign == -1 处给 0（注意 -1 不能直接 gather）/
        renormalize over survivors only; all-zero when nothing survived.
    非法输入（dtype/形状/设备/非有限/负概率/k 或 capacity 越界）抛 ValueError；不修改输入，接受非连续 /
        raise ValueError; mutate nothing; accept noncontiguous tensors.
    返回 / Returns (assign int64 [T,k]（丢弃为 -1）, weights float64 [T,k],
        counts int64 [E], dropped int（Python int）, order int64 [T*k]（处理顺序的槽位编号）)
    """
    raise NotImplementedError
