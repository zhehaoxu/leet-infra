"""LoSparse 低秩+结构化稀疏分解步 -- 待实现骨架.

回源: yxli2123/LoSparse (ICML 2023, arXiv 2401.15024 无关; 2306.11222)
``utils.py``。

1. low_rank_decomposition: svd(full_matrices=False) → rank=count_nonzero(S);
   parameter_ratio 非 None 时 k=ceil(pr·H·W/(H+W)) (覆盖 rank_ratio),
   否则 k=ceil(rank·rank_ratio); L=U@√diag(S)[:,选列], R=√diag(S)[选行,:]@Vh
   (max_eigenvalue 取前 k / min_eigenvalue 取后 k)。
2. 残差 S = W − L@R; neuron_mask 模拟训练置零 (S·mask[:,None]),
   nonzero_idx = nonzero(行和), sparse_pruned = S_masked[nonzero_idx]。
3. 前向 (x 为 (B,L,in)): LRX = x@R.T@L.T; 未剪枝 + x@S.T, 剪枝后把
   x@sparse_pruned.T scatter 到 [..., nonzero_idx]; bias 原样加。

请实现 solve:
* 校验 weight 2 维 / x 3 维且入维一致 / criteria / 比例 / 掩码 0-1 / bias;
* 返回 dict(L, R, sparse_residual, reduced_rank, rank, is_full_rank,
  nonzero_idx, sparse_pruned, output)。
"""

import math

import torch


def solve(weight, x, parameter_ratio=0.15, rank_ratio=0.1,
          remove_criteria="max_eigenvalue", neuron_mask=None, bias=None):
    raise NotImplementedError
