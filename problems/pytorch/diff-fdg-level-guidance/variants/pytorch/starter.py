def solve(
    cond_levels,
    uncond_levels,
    guidance_scales,
    parallel_weights=None,
    level_rescale=None,
    active_levels=None,
):
    """FDG：逐频层的引导、可选的平行分量加权与逐层 rescale。

    cond_levels / uncond_levels:长度相同的张量列表，按「高频 → 低频」排列，
        每个频率层形状为 (batch, ...)，所有层同形状；
    guidance_scales:每层的引导尺度 s_l（非负有限实数）；
    parallel_weights:可选的每层平行分量权重 w_l（非负有限实数，缺省全为 1.0）；
    level_rescale:可选的每层 rescale 强度 φ_l ∈ [0, 1]（缺省全为 0.0）；
    active_levels:可选的每层开关（bool 列表，缺省全为 True）。

    对每个频率层 l：
      未开启            -> pred_l = cond_l
      否则              -> shift = cond_l - uncond_l
                          若 w_l 与 1.0 不相近：
                              p = (shift·ĉ)ĉ，o = shift - p，shift = w_l·p + o
                          pred_l = uncond_l + s_l · shift
                          若 φ_l > 0：按标准差的比值把 pred_l 拉回 cond_l 的尺度，再与未缩放
                          结果按 (1-φ_l, φ_l) 混合（标准差为 torch 默认的无偏估计）
    最终返回所有 pred_l 之和（对应反变换的求和步骤），float32。
    """
    # TODO
    raise NotImplementedError
