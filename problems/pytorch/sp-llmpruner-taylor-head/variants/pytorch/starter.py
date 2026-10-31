import torch


def solve(wq, wk, wv, wo, gwq, gwk, gwv, gwo, head_dim,
          mode="param_first", group_reduction="sum", num_prune=1):
    """LLM-Pruner：多头注意力的组粒度一阶泰勒头重要性（待实现）。"""
    raise NotImplementedError
