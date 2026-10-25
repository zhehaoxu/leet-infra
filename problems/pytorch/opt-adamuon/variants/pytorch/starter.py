import torch


def solve(params, lr=0.02, momentum=0.95, ns_steps=5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
    """返回一个继承 `torch.optim.Optimizer` 的 AdaMuon 实例，`step()` 就地更新 params。

    AdaMuon（arXiv:2507.11005）= Muon 的正交化更新之上叠加**逐元素自适应 RMS 预条件**，
    再做**范数还原**保持更新尺度；1-D 参数走 AdamW 路径。本题口径（docstring 为唯一判题标准，
    全程 float32；`grad is None` 的参数整体跳过；禁止调用现成 Muon/AdamW 组合器）：

    对第 t 步、梯度 g（状态 m、v 初始全零，t 从 1 起逐参数独立计数）：
      1) 解耦权重衰减（仅当 weight_decay≠0，先于一切更新）：p ← p·(1 − lr·weight_decay)
    【2-D 参数（dim ≥ 2）：Muon + Ada】
      2) 动量：m ← μ·m + (1−μ)·g
      3) Nesterov：G ← (1−μ)·g + μ·m                     （用更新后的 m）
      4) Newton-Schulz 正交化 O = NS(G)，quintic 系数 (3.4445, −4.7750, 2.0315)、ns_steps 步：
           若 rows>cols 先转置；X ← X/(‖X‖_F + 1e−7)；循环 {A = X Xᵀ; X = a·X + (b·A + c·A²)·X}；
           转过再转置回来。**不做** Muon 的 max(1, rows/cols)^0.5 谱缩放（范数还原会抵消它）。
      5) 逐元素 RMS 二阶矩（驱动信号是**正交化后的 O**，不是 g）：
           v ← β2·v + (1−β2)·O⊙O；V̂ = v/(1−β2^t)
      6) 自适应预条件 + 范数还原（本题的关键两步，顺序固定）：
           U ← O / (√V̂ + ε)
           U ← U · ( ‖O‖_F / (‖U‖_F + 1e−12) )
         —— RMS 相除是逐元素自适应；范数还原把整步位移的 Frobenius 范数拉回 lr·‖O‖_F，
            因此无权重衰减时 ‖Δp‖_F = lr·‖O‖_F 恒成立（Muon 的整体尺度不被 Ada 破坏）。
      7) p ← p − lr·U
    【1-D 参数（dim < 2）：纯 AdamW 路径】
      2') m ← β1·m + (1−β1)·g；v ← β2·v + (1−β2)·g⊙g
      3') p ← p − lr·[m/(1−β1^t)] / (√(v/(1−β2^t)) + ε)
    """
    raise NotImplementedError
