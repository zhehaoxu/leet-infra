import torch


def solve(role_ids, assistant_id, turn_weights=None):
    """逐轮归一的 SFT 权重：每行内把 role==assistant_id 的极大连续段视为一轮(行内从0编号)，
    第 t 轮每 token 权重 = base_t / 该轮token数（base_t=turn_weights[t] 或 1.0），非 assistant 为 0。
    返回 (B,L) float32。"""
    # TODO
    raise NotImplementedError
