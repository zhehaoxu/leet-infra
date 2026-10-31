import torch


def solve(attn_caches, q_len, tsp_idx, hidden_states, position_ids,
          window_size=8, retain_rate=0.1, tsp_rate=0.2):
    """FastKV：TSP 层按跨 head 求和分数挑 token 子集，后续层只看该子集且预算按 tsp_rate 放宽。"""
    # TODO
    raise NotImplementedError
