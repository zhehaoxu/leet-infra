import torch


def solve(attn, keys, values, max_capacity, window_size, imp_budget=0.5,
          fuse_temperature=1.0, value_norms=None):
    """MorphKV 岛屿融合：importance 加权 → 重要位单例 → 连续岛融成 1 槽 → [fused|important|window]。"""
    # TODO
    raise NotImplementedError
