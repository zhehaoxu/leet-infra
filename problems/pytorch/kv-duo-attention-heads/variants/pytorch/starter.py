import torch


def solve(head_scores, seq_len, sink_size, recent_size, head_compression_ratio):
    """DuoAttention：分数最小的若干头降级为流式头（只留 sink+recent），其余留全量。"""
    # TODO
    raise NotImplementedError
