import torch


def solve(weight, mask, Hinv):
    """SparseGPT 逐列剪枝 + Hessian 误差补偿（Hessian 本身不更新）。
    weight float32 (R,C)；mask float32 (R,C)，0 表示该元素被剪；Hinv float32 (C,C) 是校准激活
    二阶矩 H = X Xᵀ 的逆（对称正定）。对 j = 0..C-1 依次执行：
      cur = W[:, j]；Q = cur * mask[:, j]；e = (cur - Q) / Hinv[j, j]；
      W[:, j] = Q；W[:, j+1:] -= e ⊗ Hinv[j, j+1:]。
    返回 W_out float32 (R,C)。禁止 torch.linalg.inv 与现成 SparseGPT 实现。"""
    # TODO
    raise NotImplementedError
