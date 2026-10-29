import torch


def solve(weight, calib, sparsity):
    """注意力头级结构化剪枝。weight float32 (L,H,d_head,D)；calib float32 (L,H,N,d_head) 为校准激活。
    importance[l,h] = ||weight[l,h]||_F * ||calib[l,h]||_F（Wanda 式权重-激活联合显著度）。
    按 importance 全局升序剪掉 round(sparsity*L*H) 个头（并列按展平索引 l*H+h 升序先剪），
    被剪头的整片 (d_head, D) 权重置零。返回 (pruned_weight, importance)，后者 float32 (L,H)。
    禁止 torch.nn.utils.prune。"""
    # TODO
    raise NotImplementedError
