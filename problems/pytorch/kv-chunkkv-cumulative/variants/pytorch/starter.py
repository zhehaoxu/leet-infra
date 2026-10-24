import torch


def solve(scores, chunk_length, compression_ratio):
    """ChunkKV：跨 head 求和→块内平均得块分→按块分 top-k→整块 token 升序合并。"""
    # TODO
    raise NotImplementedError
