import torch


def solve(params, proj, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, update_proj_gap=200):
    # TODO: 返回一个继承 torch.optim.Optimizer 的子类实例（GaLore：给定投影基的低秩子空间 AdamW），
    #   其 step() 就地更新 params 里的张量本身。以下为本题唯一口径（约定全部写死在这里）：
    #   入参约定：
    #     - params：待优化张量列表（float32，1D 或 2D，2D 至多 8x8）；
    #     - proj：与 params 同长、逐参数一一对应的列表。每项为 None（该参数不投影），
    #       或形状 [p.numel(), r]（r≥0）的 float32 二维张量 P（约定列已正交归一，
    #       但实现不得依赖、不得重新正交化、不得做 SVD）；
    #     - update_proj_gap：≥1 的整数，基重算间隔（账本参数，不影响数值轨迹）。
    #   对每个 grad 非 None 的参数 p，第 t 步（t 从 1 计，逐参数独立计数；grad 为 None
    #   的参数整步跳过——连权重衰减与账本都不施加）：
    #     1) 状态首见初始化：step=0；矩形状为 [r]（proj 为 None 时 r=p.numel()，即原空间）：
    #        m=0、v=0；投影参数另建空账本 recompute_steps=[]。
    #     2) t ← step+1 写回 step。
    #     3) 基重算账本（仅投影参数，纯整数日程）：若 (t−1) % update_proj_gap == 0
    #        （首步 t=1 建基，此后每隔 gap 步一次），把 t 追加进 state[p]["recompute_steps"]
    #        （升序、无重复；gap==1 则每步都记）。
    #     4) 解耦权重衰减（仅 weight_decay≠0，先于矩估计，作用于原空间参数）：
    #        p ← p·(1 − lr·weight_decay)
    #     5) 投影梯度：g̃ = Pᵀ·flatten(p.grad)，flatten 按 row-major（C 序），g̃ 形状 [r]；
    #        不投影参数即 g̃ = p.grad 展平。
    #     6) 投影空间矩估计：m ← β1·m + (1−β1)·g̃；v ← β2·v + (1−β2)·g̃⊙g̃（逐元素平方）。
    #     7) 偏差修正与方向（均在 [r] 维投影空间）：bc1 = 1−β1^t；bc2 = 1−β2^t；
    #        denom = √v/√bc2 + eps；û = (m/bc1) / denom。
    #     8) 映回并更新（就地）：Δ = (P·û) reshape 回 p 的形状（不投影参数 Δ = û reshape 回 p 形）；
    #        p ← p − lr·Δ。r=0 时 Δ 恒为零向量，参数只剩第 4 步的权重衰减效果。
    #   另需在优化器实例上提供纯函数方法 recompute_at(t)：返回 bool，
    #   (t−1) % update_proj_gap == 0 时为 True（供外部核对日程）。
    #   全程 float32；禁止调用现成优化器/SVD/随机采样。
    raise NotImplementedError
