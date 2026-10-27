import torch


def solve(x1, x2, target, mode="cosine", margin=None, reduction="mean"):
    """嵌入 margin 损失分派器：cosine 与 hinge 两种 mode，逐元素计算后按 reduction 归约。

    入参：
    - mode: str，默认 "cosine"，取 "cosine" / "hinge" 之一，其余取值抛 ValueError。
    - mode="cosine"：x1、x2 形状 (N,D)，target 形状 (N,)，取值 ∈ {1,−1}（不做校验）。
      margin 默认 None → 按 0.0 处理。x2 为 None 时抛 ValueError。
      记 cos_i = <x1_i, x2_i> / max(‖x1_i‖₂·‖x2_i‖₂, 1e-8)（分母整体 clamp_min(1e-8)，
      与 F.cosine_embedding_loss 一致；任一向量为零向量时 cos_i = 0）。
      loss_i = target_i == 1 ? 1 − cos_i : max(0, cos_i − margin)。
    - mode="hinge"：x1 形状 (N,) 的输入分数，target (N,) ∈ {1,−1}；x2 约定传 None
      （不做校验，一律忽略）。margin 默认 None → 按 1.0 处理。
      loss_i = target_i == 1 ? x1_i : max(0, margin − x1_i)
      （与 F.hinge_embedding_loss 在 target ∈ {1,−1} 时逐元素一致）。
    - reduction: str，默认 "mean"。"none" 返回形状 (N,) 的逐元素损失；
      "mean" / "sum" 归约为 0 维标量（与原生约定一致）；其余取值抛 ValueError。

    N = 0 的兜底约定：reduction="none" 返回 (0,) 空张量；"sum" 返回 0；
    "mean" 返回标量 0（原生 F 对空集求均值会得到 NaN，本题按兜底约定返回 0）。

    返回：float 张量（dtype 跟随 x1）："none" 时形状 (N,)，否则 0 维标量。
    """
    # TODO
    raise NotImplementedError
