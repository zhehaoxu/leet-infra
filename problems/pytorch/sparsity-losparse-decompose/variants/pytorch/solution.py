"""LoSparse 低秩+结构化稀疏分解步.

回源: yxli2123/LoSparse (ICML 2023, arXiv 2306.11222) ``utils.py`` (399 行)
的 ``low_rank_decomposition`` (L7-66)、``substitute_layer_weights`` 的残差
构造 (L183-194)、``LinearLoSparse.prune_sparse`` (L136-139) 与两路前向
(L91-126)。

LoSparse: W ≈ L @ R + S——SVD 低秩逼近压走神经元的相干部分, 结构化
(神经元/行级)剪枝删走非相干部分, 二者互补; 训练中联合微调后对稀疏
残差按行置零再剪去零行, 前向把剪剩的行 scatter 回全宽输出。

分解 (官方 low_rank_decomposition):
    U, S, Vh = svd(W, full_matrices=False)     # S 降序
    rank = count_nonzero(S)
    parameter_ratio 优先: k = ceil(pr · H·W / (H + W))
    否则:                k = ceil(rank · rank_ratio)
    max_eigenvalue:   L = U @ √diag(S)[:, :k],  R = √diag(S)[:k, :] @ Vh
    min_eigenvalue:   L = U @ √diag(S)[:, -k:], R = √diag(S)[-k:, :] @ Vh
    (L @ R = 所选奇异三元组的重构; √拆分让 L/R 参数量均衡)

残差与剪枝 (官方 substitute_layer_weights + prune_sparse):
    S = W − L @ R                       # 稀疏残差 (out×in)
    训练置零 (判题域以 neuron_mask 模拟): S_masked = S · mask[:, None]
    nonzero_idx = nonzero(S_masked.sum(dim=1))  # 行和非零的幸存神经元
    sparse_pruned = S_masked[nonzero_idx, :]

前向 (官方 LinearLoSparse.forward, x 为 (B, L, in)):
    未剪枝: Y = x @ R.T @ L.T + x @ S.T + bias
    剪枝后: Y = x @ R.T @ L.T + scatter(x @ sparse_pruned.T, nonzero_idx) + bias

与官方实现的判题域约定 (official -> judgement):
* remove_criteria 的 'random' 分支触随机性红线, 剔除 (只收 max/min);
  官方 raise NameError → 判题域 ValueError。
* 官方 masked_fill_ 由 PLATON 训练产生; 判题域以显式 neuron_mask 模拟
  (0/1 行掩码), S 本身不训练更新。
* log/打印与 SX 训练缓存 (Deberta Q/K 双用) 不入。
* 官方不校验; 判题域补权重/x/掩码/criteria/reduced_rank 下界等显式校验。
"""

import math

import torch


def solve(weight, x, parameter_ratio=0.15, rank_ratio=0.1,
          remove_criteria="max_eigenvalue", neuron_mask=None, bias=None):
    """LoSparse 低秩+结构化稀疏分解步。

    参数
    ----
    weight : torch.Tensor
        被分解权重 (out, in) = (H, W), 行主序 nn.Linear 约定。
    x : torch.Tensor
        输入 (B, L, in), in 必须等于 W。
    parameter_ratio : float or None
        低秩参数量占比: k = ceil(pr·H·W/(H+W)); 非 None 时覆盖 rank_ratio。
    rank_ratio : float
        秩占比: k = ceil(rank·rank_ratio) (仅 parameter_ratio 为 None 时)。
    remove_criteria : str
        'max_eigenvalue' 取前 k 个奇异三元组, 'min_eigenvalue' 取后 k 个。
    neuron_mask : torch.Tensor or None
        (H,) 0/1 行掩码, 模拟训练后置零; None = 未剪枝前向 (残差全量)。
    bias : torch.Tensor or None
        (H,) 偏置, 原样加到输出。

    返回
    ----
    dict(L, R, sparse_residual, reduced_rank, rank, is_full_rank,
         nonzero_idx, sparse_pruned, output)
        L (H,k) / R (k,W) 满足 L@R = 所选 SVD 三元组重构; rank 是 S 的
        非零个数; nonzero_idx/sparse_pruned 仅剪枝路径非 None。
    """
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        raise ValueError("weight 必须是 2 维 (out, in) 张量")
    H, W = weight.shape
    if not isinstance(x, torch.Tensor) or x.dim() != 3:
        raise ValueError("x 必须是 3 维 (B, L, in) 张量")
    if x.shape[-1] != W:
        raise ValueError(f"x 的入维必须等于 weight 的 in ({W})")
    if remove_criteria not in ("max_eigenvalue", "min_eigenvalue"):
        raise ValueError("remove_criteria 必须是 max_eigenvalue 或 min_eigenvalue")
    if parameter_ratio is not None:
        if isinstance(parameter_ratio, bool) or not isinstance(parameter_ratio, (int, float)) \
                or parameter_ratio <= 0:
            raise ValueError("parameter_ratio 必须是正数或 None")
    if isinstance(rank_ratio, bool) or not isinstance(rank_ratio, (int, float)) or rank_ratio <= 0:
        raise ValueError("rank_ratio 必须是正数")
    if neuron_mask is not None:
        if not isinstance(neuron_mask, torch.Tensor) or neuron_mask.dim() != 1 \
                or neuron_mask.shape[0] != H:
            raise ValueError(f"neuron_mask 必须是长度 {H} 的一维 0/1 张量或 None")
        if not torch.all((neuron_mask == 0) | (neuron_mask == 1)):
            raise ValueError("neuron_mask 必须是 0/1 掩码")
    if bias is not None and (not isinstance(bias, torch.Tensor)
                             or bias.dim() != 1 or bias.shape[0] != H):
        raise ValueError(f"bias 必须是长度 {H} 的一维张量或 None")

    # 分解
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = int(torch.count_nonzero(S))
    is_full_rank = rank == min(H, W)
    if parameter_ratio is not None:
        reduced_rank = math.ceil(parameter_ratio * (H * W) / (H + W))
    else:
        reduced_rank = math.ceil(rank * rank_ratio)
    # 正比例经 ceil 恒 >= 1, 只需防上界越界
    if reduced_rank > min(H, W):
        raise ValueError(f"reduced_rank ({reduced_rank}) 必须在 [1, {min(H, W)}] 内")

    diag_sqrt = torch.sqrt(torch.diag(S))
    if remove_criteria == "max_eigenvalue":
        L = U @ diag_sqrt[:, :reduced_rank]
        R = diag_sqrt[:reduced_rank, :] @ Vh
    else:
        L = U @ diag_sqrt[:, -reduced_rank:]
        R = diag_sqrt[-reduced_rank:, :] @ Vh

    sparse_residual = weight - L @ R

    # 剪枝 + 前向
    if neuron_mask is not None:
        S_masked = sparse_residual * neuron_mask.unsqueeze(1).to(sparse_residual.dtype)
        nonzero_idx = torch.nonzero(S_masked.sum(dim=1)).flatten()
        if nonzero_idx.numel() == 0:
            raise ValueError("neuron_mask 剪空了全部神经元 (至少保留一行)")
        sparse_pruned = S_masked[nonzero_idx, :]
        LRX = x @ R.T @ L.T
        SX_scattered = torch.zeros(*x.shape[:-1], H, dtype=x.dtype)
        SX_scattered[..., nonzero_idx] = x @ sparse_pruned.T
        output = LRX + SX_scattered
    else:
        nonzero_idx = None
        sparse_pruned = None
        output = x @ R.T @ L.T + x @ sparse_residual.T
    if bias is not None:
        output = output + bias

    return {
        "L": L, "R": R, "sparse_residual": sparse_residual,
        "reduced_rank": reduced_rank, "rank": rank, "is_full_rank": is_full_rank,
        "nonzero_idx": nonzero_idx, "sparse_pruned": sparse_pruned,
        "output": output,
    }
