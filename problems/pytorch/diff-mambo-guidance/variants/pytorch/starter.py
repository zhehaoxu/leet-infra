def solve(eps_cond, eps_uncond, guidance_scale, alpha=8.0):
    """MAMBO-G：按「引导更新」与「无条件预测」的幅值比压回引导强度。

    eps_cond / eps_uncond:同形状预测，第 0 维是 batch；
    guidance_scale:w，基础引导强度；
    alpha:抑制强度 λ ≥ 0，越大抑制越激进。

    逐样本（除第 0 维外全部维度）：
      delta = eps_cond - eps_uncond
      r     = ‖delta‖₂ / ‖eps_uncond‖₂
      w'    = 1 + (w - 1) · exp(-λ·r)
      out   = eps_uncond + w' · delta

    ‖eps_uncond‖₂ = 0 时把 r 视为 +∞（即 w' = 1，输出退化为 eps_cond），不得出现 NaN。
    返回 float32 张量，形状与 eps_cond 相同。
    """
    # TODO
    raise NotImplementedError
