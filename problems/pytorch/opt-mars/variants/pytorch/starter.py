import torch


def solve(params, lr=1e-3, betas=(0.9, 0.99), gamma=0.025, eps=1e-8, weight_decay=0.0, ams_bound=False):
    """返回一个继承 `torch.optim.Optimizer` 的 MARS（MARS-AdamW 实例化）实例，`step()` 就地更新 params。

    约定（本题为唯一判题口径，全程 float32；1-D 与 2-D 参数走完全相同的路径，
    信任比按整张量的 Frobenius 范数逐张量计算；`grad is None` 的参数整体跳过，连权重衰减也不施加；
    每个参数的步计数 t 从 1 起、独立维护；禁止调用现成优化器）：

      状态初始化（全部 zeros、与 p 同形状）：m, v, last_g；若 ams_bound=True 再加 v_max。
      last_g 初始为 0，即首步约定 g_0 := 0，因此第一步的梯度差分项不为零。

      对第 t 步、梯度 g：
        1) 解耦权重衰减（仅当 weight_decay≠0，先于一切矩估计）：p ← p·(1 − lr·weight_decay)
        2) 方差缩减梯度估计（修正系数折进 γ′ = gamma·β1/(1−β1)）：
             c = g + γ′·(g − last_g)
           首步 last_g=0 ⇒ c = (1+γ′)·g；gamma=0 ⇒ c = g（精确回到 Adam 的路径）。
        3) 信任比裁剪（trust-ratio，逐张量）：ρ = min(1, 1/‖c‖_F)，c̃ = ρ·c
           （等价于 c̃ = c/max(1, ‖c‖_F)，‖·‖_F 为整张量 Frobenius/2-范数）
        4) VR 一阶矩：m ← β1·m + (1−β1)·c̃
        5) VR 二阶矩（用裁剪后的估计量的逐元素平方）：v ← β2·v + (1−β2)·c̃⊙c̃
        6) 若 ams_bound：v_max ← max(v_max, v)，并以 v_max 作分母基（二阶矩的自适应裁剪：
           运行最大值约束后续步的分母不小于历史峰值）；否则分母基为 v，记 denom_base。
        7) 偏差修正与更新（算子顺序固定，与 opt-adamw 同口径）：
             denom ← √denom_base/√(1−β2^t) + eps
             p ← p − (lr/(1−β1^t))·m/denom
        8) last_g ← g
    """
    raise NotImplementedError
