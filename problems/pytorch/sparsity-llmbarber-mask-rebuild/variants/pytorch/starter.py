"""LLM-Barber mask 重建（starter）。

对已剪枝稀疏权重做 mask 重建：metric 三选（WD/W/D）+ find_threshold
双阈值交换 + 粒度四选（block/layer/output1/input1）+ retain_reconstruction。
scaler（grad² Fisher 滑动平均）已显式入参。

返回 dict(masks={name: bool 张量}, rebuilt={name: 权重},
sparsity=float, thresholds={name: (growth, prune)})。
"""

import math

import torch


def solve(weights, ref_weights, scalers, metric="WD", threshold=1.0,
          granularity="layer", retain_reconstruction=True):
    """TODO: 实现官方 mask_rebuild 的确定性核心。"""
    raise NotImplementedError
