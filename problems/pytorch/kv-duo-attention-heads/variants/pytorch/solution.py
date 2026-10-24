import torch


def solve(head_scores, seq_len, sink_size, recent_size, head_compression_ratio):
    """DuoAttention：最小的 floor(H·ratio+0.5) 个头作流式头，只留 sink+recent；余留全量。"""
    H = int(head_scores.shape[0])
    L = int(seq_len)
    S = int(sink_size)
    R = int(recent_size)
    n_pruned = int(H * head_compression_ratio + 0.5)
    # 分数升序、平局 index 小者优先 → 取前 n_pruned 个作流式头
    order = sorted(range(H), key=lambda h: (float(head_scores[h]), h))
    streaming = set(order[:n_pruned])

    mask = torch.ones(H, L, dtype=torch.bool)
    if L > S + R:  # 首尾窗口未覆盖全序列时才需要挖中间
        for h in streaming:
            mask[h, S:L - R] = False
    return mask
