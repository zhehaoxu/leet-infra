import torch


def solve(params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 Muon 实例，step() 就地更新二维权重 params。
    #   每个有梯度的矩阵 p、缓冲 buf（初始零）：
    #     1) buf ← μ·buf + (1−μ)·g
    #     2) G ← (1−μ)·g + μ·buf            # nesterov，用新 buf
    #     3) Newton-Schulz 正交化 O=NS(G)，系数 (3.4445,-4.7750,2.0315)：
    #          若 rows>cols 先转置；X←X/(||X||_F+1e-7)；重复 ns_steps 次：
    #          A=X Xᵀ; B=b·A+c·A A; X=a·X+B X；若转过再转置回来
    #     4) O ← O · max(1, rows/cols)**0.5
    #     5) 若 wd≠0: p←p·(1−lr·wd) ; p ← p − lr·O
    # float32；grad 为 None 跳过；禁止显式 SVD 或调用现成 Muon。
    raise NotImplementedError
