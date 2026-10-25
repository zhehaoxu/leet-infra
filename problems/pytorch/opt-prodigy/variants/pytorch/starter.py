import torch


def solve(params, lr=1.0, betas=(0.9, 0.999), d0=1e-6, d_coef=1.0, growth_rate=float("inf"),
          weight_decay=0.0, eps=1e-8):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 Prodigy 实例，step() 就地更新 params。
    #   β3 = √β2。组级 d←d0, d_max←d0, num←0（跨步保留）；每参数 s=0, p0=p快照, m=0, v=0。每步：
    #     1) d_lr = d·lr
    #     2) num ← β3·num ; den ← 0
    #     3) 第一趟（每个有梯度参数，梯度 g）：
    #          num += (d/d0)·d_lr·⟨g, p0−p⟩
    #          m ← β1·m + d·(1−β1)·g
    #          v ← β2·v + d²·(1−β2)·g²
    #          s ← β3·s + (d/d0)·d_lr·g ; den += Σ|s|
    #     4) 若 den==0 结束。否则 d̂ = d_coef·num/den
    #          若 d==d0: d←max(d,d̂) ; d_max←max(d_max,d̂) ; d←min(d_max, d·growth_rate)
    #     5) 第二趟（每个有梯度参数）：解耦 wd p←p·(1−d_lr·wd)；denom=√v + d·eps；p ← p − d_lr·m/denom
    #   第二趟步长用旧 d_lr、分母用新 d。grad 为 None 参数两趟都跳过。禁止调用现成优化器。
    raise NotImplementedError
