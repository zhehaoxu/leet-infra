"""参考答案：按定义的简化模型闭合式。per_rank=(1-f)A+fA/C；comm=2*fA*(C-1)/C。"""


def solve(total_activation, shardable_fraction, cp_degree):
    A = float(total_activation)
    f = float(shardable_fraction)
    C = float(cp_degree)
    c = (C - 1.0) / C
    A_shard = f * A
    A_rep = (1.0 - f) * A
    per_rank = A_rep + A_shard / C
    comm = 2.0 * A_shard * c
    return (float(per_rank), float(comm))
