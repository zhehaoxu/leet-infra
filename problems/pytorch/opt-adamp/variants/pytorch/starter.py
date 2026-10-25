import math

import torch
from torch.nn.functional import cosine_similarity


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, delta=0.1, wd_ratio=0.1):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 AdamP 实例，其 step() 就地更新 params。
    #   固定 nesterov=False, cautious=False, weight_decouple=True。
    #   每个有梯度的参数 p、状态 m, v（初始零），第 t 步（有梯度才自增）：
    #     m ← β1·m+(1−β1)·g ;  v ← β2·v+(1−β2)·g²
    #     bc2sq = sqrt(1−β2**t) ;  step = lr/(1−β1**t)
    #     perturb = bc2sq/(sqrt(v)+eps) * m
    #     r = 1.0
    #     if p.dim() > 1:   # 依 channel view 再 layer view，命中 |cos|.max()<delta/sqrt(列) 则:
    #         n = p/(rownorm+eps); radial = (n*perturb).sum(dim=1) 广播; perturb -= n*radial; r=wd_ratio; break
    #     (wd≠0 时) p ← p·(1−lr·wd·r)
    #     p ← p − step·perturb
    #   全程 float32；grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
