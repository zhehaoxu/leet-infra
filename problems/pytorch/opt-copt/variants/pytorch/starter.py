import torch


def solve(params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, rho=1.0):
    """返回一个继承 `torch.optim.Optimizer` 的 C-Optimizer 实例，`step()` 就地更新 params。

    C-Optimizer（ByteDance Seed 技术报告思路，论文编号待回源；本 docstring 是本题唯一判题口径）：
    在 AdamW 的骨架上，把 cautious 式的 0/1 方向掩码换成由 cos(g, m) 驱动的**连续插值门**，
    再叠一道 **sign 保持的幅度截幅**。全程 float32；每个参数步计数 t 从 1 起独立维护；
    `grad is None` 的参数整体跳过（连权重衰减也不施加）；禁止调用现成优化器。

    对第 t 步、梯度 g（m、v 为该参数的状态，初始全零）：
      1) 解耦权重衰减（仅当 weight_decay≠0，先于一切矩估计）：p ← p·(1 − lr·weight_decay)
      2) 一阶矩：m ← β1·m + (1−β1)·g
      3) 二阶矩：v ← β2·v + (1−β2)·g⊙g
      4) 对齐余弦（用**本步更新后的** m 与当前 g，按整张量 Frobenius 内积）：
           cos = ⟨g, m⟩_F / (‖g‖_F·‖m‖_F)
         任一范数为 0 时约定 cos = 1（不压制）。
      5) 连续插值门：gate = clamp((cos + 1)/2, 0, 1)
         —— cos=1 完全放行、cos=−1 完全压死、cos=0 放行一半；是 cautious 0/1 掩码的连续化。
      6) 偏差修正与更新方向（与 opt-adamw 同口径）：
           u = [m/(1−β1^t)] / (√(v/(1−β2^t)) + ε)
      7) 先乘门、再截幅（顺序固定）：u ← u·gate；u ← clamp(u, −ρ, ρ)
         —— 截幅保持符号、逐元素夹到 [−ρ, ρ]；无权重衰减时每步 |Δp| ≤ lr·ρ 恒成立。
      8) p ← p − lr·u
    """
    raise NotImplementedError
