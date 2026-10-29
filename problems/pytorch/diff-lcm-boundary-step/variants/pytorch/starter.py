def solve(
    sample,
    model_output,
    alphas_cumprod,
    timesteps,
    step_index,
    prediction_type="epsilon",
    clip_sample=True,
    clip_sample_range=1.0,
    timestep_scaling=10.0,
    noise=None,
):
    """LCM 单步：返回 (denoised, prev_sample, c_skip, c_out)。

    sample / model_output:同形状非空浮点张量；alphas_cumprod:一维浮点张量，长度 T，元素落在 (0, 1)；
    timesteps:一维整数张量（LCM 推理步的子采样），每个值都在 [0, T-1]；
    step_index:整数，落在 [0, len(timesteps)-1]。

    1) 边界条件系数（σ_d 上游写死 0.5）：
       s = timesteps[step_index] * timestep_scaling
       c_skip = 0.5**2 / (s**2 + 0.5**2)
       c_out  = s / (s**2 + 0.5**2) ** 0.5      （注意没有乘 σ_d）

    2) x0（用 alphas_cumprod 的平方根）：
       epsilon      -> x0 = (sample - sqrt(1 - ᾱ_t) * model_output) / sqrt(ᾱ_t)
       sample       -> x0 = model_output
       v_prediction -> x0 = sqrt(ᾱ_t) * sample - sqrt(1 - ᾱ_t) * model_output
       clip_sample=True 时再把 x0 钳到 [-clip_sample_range, clip_sample_range]。

    3) 边界条件混合：denoised = c_out * x0 + c_skip * sample

    4) 多步重加噪：ṫ = timesteps[step_index + 1]，若 step_index 不是最后一步
       （prev_timestep 不存在时沿用当前 timestep，该值在最后一步不参与计算）
       ᾱ_prev = alphas_cumprod[ṫ]（ṫ < 0 时取 1.0）
       prev_sample = sqrt(ᾱ_prev) * denoised + sqrt(1 - ᾱ_prev) * noise
       最后一步直接 prev_sample = denoised（不消耗 noise）。

    返回 denoised / prev_sample 为与 sample 同形状的 float32，c_skip / c_out 为 python float。
    """
    # TODO
    raise NotImplementedError
