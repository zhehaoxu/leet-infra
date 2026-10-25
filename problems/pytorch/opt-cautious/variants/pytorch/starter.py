import torch


def solve(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 Cautious AdamW 实例，step() 就地更新 params。
    #   第 t 步（从 1 起）：先按 AdamW 求更新方向 u（未乘 lr）：
    #     1) 解耦 wd（≠0 先做）：p ← p·(1 − lr·wd)
    #     2) m ← β1·m + (1−β1)·g ; v ← β2·v + (1−β2)·g²
    #     3) bc1=1−β1^t ; bc2=1−β2^t ; u = (m/bc1) / (√v/√bc2 + eps)
    #   再套谨慎掩码：
    #     4) M = 1[u*g > 0] ; M ← M · numel(M) / (M.sum() + 1)
    #     5) u ← u*M ; p ← p − lr·u
    #   grad 为 None 的参数整体跳过。禁止调用现成优化器。
    raise NotImplementedError
