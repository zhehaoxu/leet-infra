import torch


def solve(hidden, expert_ids, routing_weights, diag):
    """中文契约 / English contract
    hidden [T,D], routing_weights [T,k] float64；expert_ids [T,k] int64；diag [E,D] float64 / all CPU float64 except ids.
    槽位编号 s = t*k + j（token-major），S = T*k；专家编号合法范围 [0, E]，E 本身是哨兵（被丢弃/不在本卡）/
        slot s = t*k + j, S = T*k; valid ids are 0..E where E itself is the sentinel (dropped / not local).
    排序按 (专家编号, 槽位编号) 升序且稳定，哨兵自然落在尾部 / sorted ascending, ties keep slot order, sentinels land last.
    专家 e 对槽位 (t,j) 的输出定义为 hidden[t] * diag[e]（逐元素）；哨兵行的 grouped 输出必须是 0，
    索引 diag 时先把哨兵 clamp 到 E-1 再整行清零 / sentinel rows must be zeroed after clamping the gather index.
    counts[e] 只统计 0..E-1；offsets 是 counts 的闭前缀和（长度 E，int64）/ counts skip the sentinel; offsets is an inclusive cumsum.
    越界 id、形状不匹配、非 float64/int64、非 CPU、空维都抛 ValueError；不修改输入，接受非连续张量 /
        raise ValueError otherwise; mutate nothing and accept noncontiguous tensors.
    返回 / Returns (out [T,D] float64, grouped [S,D] float64, perm [S] int64, inv_perm [S] int64,
        token_rows [S] int64, counts [E] int64, offsets [E] int64)；
        out = (grouped * sorted_weights)[inv_perm].view(T,k,D).sum(1) /
        out is the weighted grouped result unsorted back to token-major and summed over k.
    """
    raise NotImplementedError
