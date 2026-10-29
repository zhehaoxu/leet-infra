def solve(eps_cond, eps_uncond, guidance_scale):
    """Tangential CFG (TCFG)：把 eps_uncond 截断投影到第一右奇异方向后再做 CFG 外推。

    eps_cond / eps_uncond:同形状预测，第 0 维是 batch；
    guidance_scale:s，外推基准是投影后的 eps_uncond；
    第一右奇异方向取自把 [eps_cond; eps_uncond] 叠成 2 行矩阵后的 SVD；
    若两个奇异值相等（主方向不唯一），主方向取 eps_cond 的方向。

    返回 float32 张量，形状与 eps_cond 相同。
    """
    # TODO
    raise NotImplementedError
