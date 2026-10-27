def solve(eps_cond, eps_uncond, guidance_scale, t, t_low=0.0, t_high=1.0, rescale=1.0, dims=None):
    """带标准差校正与生效区间的 classifier-free guidance。

    eps_cond / eps_uncond:同形状预测，第 0 维是 batch；
    guidance_scale:w，外推 eps_hat = eps_uncond + w·(eps_cond - eps_uncond)；
    t / t_low / t_high:只有 t 落在闭区间 [t_low, t_high] 内才做引导，否则原样返回 eps_cond；
    rescale:φ∈[0, 1] 的缩放强度——把每个样本按 std(eps_cond)/std(eps_hat) 拉回，
        再与未缩放结果按 (1-φ, φ) 混合；φ=0 即不做 rescale；
    dims:求标准差的维度（默认除第 0 维外的全部维度，不可含 0；总体标准差，无偏置）。

    返回 float32 张量，形状与 eps_cond 相同。
    """
    # TODO
    raise NotImplementedError
