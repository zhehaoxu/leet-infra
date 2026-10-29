def solve(
    eps_cond,
    eps_uncond,
    guidance_scale,
    eta=1.0,
    norm_threshold=0.0,
    momentum=None,
    prev_momentum=None,
    dims=None,
):
    """Adaptive Projected Guidance (APG)：先把 CFG 的更新向量拆成平行/正交分量再外推。

    eps_cond / eps_uncond:同形状预测，第 0 维是 batch；
    guidance_scale:s，外推基准是 eps_uncond；
    eta:平行分量（沿 eps_cond 方向）的保留系数，落在 [0, 1]；eta=1 退化为裸 CFG，eta=0 只保留正交分量；
    norm_threshold:τ>0 时先把更新向量按 min(1, τ/‖v‖₂) 逐样本截断范数，0 表示不截断；
    momentum:m∈[0, 1) 的动量系数，None 表示不用动量；
    prev_momentum:上一步的动量缓冲（与输入同形状），None 视作全零；momentum 为 None 时不得给出；
    dims:求范数与内积的维度（默认除第 0 维外的全部维度，不可含 0）。

    返回二元组 (pred, new_momentum)：
    pred 是 float32 张量、形状与 eps_cond 相同；
    new_momentum 是更新后的动量缓冲（float32、同形状，保存截断前的差分），momentum 为 None 时返回 None。
    """
    # TODO
    raise NotImplementedError
