def solve(sample, model_output, sigmas, step_index, prediction_type="epsilon", noise=None):
    """EulerAncestral 单步：返回 (scaled_input, prev_sample, pred_original_sample, sigma_up, sigma_down)。

    sample / model_output:同形状浮点张量；sigmas:一维浮点张量，长度 K+1、有限非负且单调不增；
    step_index:整数，落在 [0, K-1]；当前步的 sigma 必须为正。

    1) 模型输入端：scaled_input = sample / sqrt(sigma**2 + 1)（上游 scale_model_input）。

    2) 预测 x0（注意 v_prediction 支照抄上游实现）：
       epsilon      -> x0 = sample - sigma * model_output
       v_prediction -> x0 = model_output * (-sigma / sqrt(sigma**2 + 1)) + sample / (sigma**2 + 1)

    3) 祖先采样把 sigma 的下降拆成确定项与随机项：
       sigma_from = sigmas[step_index]，sigma_to = sigmas[step_index + 1]
       sigma_up   = sqrt(sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2)
       sigma_down = sqrt(sigma_to**2 - sigma_up**2)

    4) Euler 步 + 注入噪声：
       derivative = (sample - x0) / sigma_from
       prev_sample = sample + derivative * (sigma_down - sigma_from) + noise * sigma_up

    noise 是显式入参（与 sample 同形状的浮点张量）；sigma_up = 0 时完全不参与计算，
    此时可以不传。返回前三个 float32 张量与 sample 同形状，sigma_up / sigma_down 是 python float。
    """
    # TODO
    raise NotImplementedError
