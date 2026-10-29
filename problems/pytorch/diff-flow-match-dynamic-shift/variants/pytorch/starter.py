def solve(
    num_inference_steps,
    shift=1.0,
    use_dynamic_shifting=False,
    mu=None,
    time_shift_type="exponential",
    shift_terminal=None,
    invert_sigmas=False,
    sigma_min=0.0,
    sigma_max=1.0,
    num_train_timesteps=1000,
):
    """构造 FlowMatchEuler 的 σ 序列与时间刻度，覆盖动态平移与 terminal 拉伸。

    num_inference_steps:N，要求 >= 1；num_train_timesteps:T（默认 1000），要求 >= 1。
    sigma_min / sigma_max:σ 栅格两端，要求 0 <= sigma_min < sigma_max <= 1。

    1) 初始 σ 栅格：sigmas = linspace(sigma_max, sigma_min, N)（从噪声端递减）。
    2) 平移：
       use_dynamic_shifting=False -> 静态平移 sigmas = shift·s / (1 + (shift-1)·s)；
       use_dynamic_shifting=True  -> 必须给 mu，按 time_shift_type 逐元素：
            "exponential" -> e^mu / (e^mu + (1/s - 1))
            "linear"      -> mu / (mu + (1/s - 1))
    3) 若给 shift_terminal（要求 0 < shift_terminal < 1 且 N >= 2）：
       先用平移后的末项算 scale = (1 - sigmas[-1]) / (1 - shift_terminal)，
       再 sigmas = 1 - (1 - sigmas) / scale。注意这一步在平移之后、取 timesteps 之前。
    4) timesteps = sigmas · T。
    5) invert_sigmas=True 时再取 sigmas = 1 - sigmas、timesteps = sigmas · T，
       终端追加 1.0；否则终端追加 0.0。

    返回 (timesteps, sigmas)：timesteps 为 float32、长度 N；sigmas 为 float32、长度 N+1。
    """
    # TODO
    raise NotImplementedError
