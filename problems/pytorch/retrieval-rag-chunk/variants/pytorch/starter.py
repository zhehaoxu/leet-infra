import torch


def solve(num_tokens, chunk_size, overlap):
    """RAG 重叠滑窗分块：step=chunk_size-overlap，起点 0,step,... 且 <num_tokens，
    每块宽 chunk_size 的 token 下标，越界填 -1。返回 int64 (块数, chunk_size)。"""
    # TODO
    raise NotImplementedError
