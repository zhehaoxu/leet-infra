import torch


def solve(q, k, candidates):
    """MInference 稀疏 pattern 搜索（2407.02490）：每个 head 在候选网格里选保留注意力质量最多的那个。
    q/k:(H,N,D) 同 dtype（float32/float64），N>=64（vertical 统计只看最后 64 个 query）。
    candidates:[(kind,a,b)]，kind ∈ {"stream_llm","vertical_and_slash","block_sparse"}。
    三种掩码都落在因果下三角；score=mean_i sum_j attn[i,j]*mask[i,j]（上游只统计 2500 之后的行）。
    返回 dict：scores(H,C) / best(H,) int64（同分取更小候选下标）/ pattern(H,2) int64 后处理后的
    (vertical_size,slash_size)，block_sparse 胜出时改写为 (1000,6096) / mask(H,N,N) bool（胜出候选的掩码）。
    内部 float64。非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
