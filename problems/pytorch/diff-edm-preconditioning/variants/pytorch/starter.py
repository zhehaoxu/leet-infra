def solve(
    sample,
    model_output,
    sigmas,
    step_index,
    sigma_data=0.5,
    prediction_type="epsilon",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    noise=None,
):
    """EDM 预条件 + Euler 单步：返回 (model_input, c_noise, prev_sample, denoised, sigma_hat)。

    sample / model_output:同形状浮点张量，model_output 是网络在预条件输入域的裸输出 F(·)；
    sigmas:一维浮点张量，长度 K+1、非增非负，最后一项通常是 0；
    step_index:整数，落在 [0, K-1]；sigma_data:训练数据标准差 σ_d，有限正数。

    记 σ = sigmas[step_index]，σ̂ 为加噪后的噪声级：
      gamma = min(s_churn / K, sqrt(2) - 1)（仅当 s_tmin <= σ <= s_tmax，否则 0；K = len(sigmas) - 1）
      σ̂ = σ · (1 + gamma)
      gamma > 0 时先加噪：sample ← sample + noise · s_noise · sqrt(σ̂² - σ²)（noise 必须显式给出）

    网络输入侧（用**未加噪**的 σ）：
      c_in    = 1 / sqrt(σ² + σ_d²)
      c_noise = 0.25 · ln σ
      model_input = 输入 sample · c_in

    输出合成侧（用**加噪后**的 σ̂）：
      c_skip = σ_d² / (σ̂² + σ_d²)
      c_out  = ±σ̂·σ_d / sqrt(σ̂² + σ_d²)（epsilon 取 +，v_prediction 取 -）
      denoised = c_skip · sample + c_out · model_output

    Euler 单步：
      derivative = (sample - denoised) / σ̂
      dt = sigmas[step_index + 1] - σ̂
      prev_sample = sample + derivative · dt

    返回 (model_input, c_noise, prev_sample, denoised, sigma_hat)：
    前三个张量与 sample 同形状的 float32，c_noise 与 sigma_hat 为 python float。
    """
    # TODO
    raise NotImplementedError
