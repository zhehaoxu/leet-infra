"""Zero-Sum SVD 全局奇异分量选择（starter）。

实现官方单阶段 zero-sum 选择：白化域 svd + 梯度投影 diag(UᵀHV) + δ 估计
（first/second_order、β-mixing）+ break_even 秩下限 + 符号平衡堆选择 +
effective_dense_delta 参数记账。

返回 dict(ranks, keep_masks, truncated_sigma_sq, running_sum,
dense_removed, stage_budget, selections)。
"""

import math

import torch


def solve(modules, global_prune_ratio=0.2, keep_rank_ratio=0.3,
          beta_mixing=None, delta_order="first_order"):
    """TODO: 实现 zero-sum 全局奇异分量选择。"""
    raise NotImplementedError
