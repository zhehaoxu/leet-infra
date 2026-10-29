def solve(
    alphas_cumprod,
    num_inference_steps,
    timestep_spacing="linspace",
    interpolation_type="linear",
    use_karras_sigmas=False,
    use_exponential_sigmas=False,
    use_beta_sigmas=False,
    final_sigmas_type="zero",
    sigma_min=None,
    sigma_max=None,
    steps_offset=0,
):
    """构造 EulerDiscrete 的推理时间栅格：返回 (timesteps, sigmas)。

    alphas_cumprod:一维浮点张量，长度 T = num_train_timesteps，元素落在 (0, 1) 且全部有限；
    num_inference_steps:N，要求 1 <= N <= T。

    1) 时间栅格（timestep_spacing，通常长度 N，一律从噪声侧往干净侧递减）：
       "linspace" -> linspace(0, T-1, N) 再翻转（浮点等距）；
       "leading"  -> step_ratio = T // N（整除），(arange(N) * step_ratio).round() 翻转后 + steps_offset；
       "trailing" -> step_ratio = T / N（真除），取 count = ceil(T / step_ratio) 个点
                     grid[k] = T - k * step_ratio，再 round() - 1；
                     浮点除法不整时 count 会是 N+1（与 numpy 的 arange(T, 0, -step_ratio) 等长）。
   2) σ 基线 sigmas_full = sqrt((1 - abar) / abar)，长度 T。
   3) interpolation_type：
       "linear"     -> 在整数下标 0..T-1 上按 timesteps 线性插值 sigmas_full；
       "log_linear" -> exp(linspace(log(sigmas_full[-1]), log(sigmas_full[0]), N+1))，
                       注意这一支完全忽略时间栅格、且产生 N+1 个点。
   4) σ 转换器（三者至多一个为真，否则 ValueError）：
       use_karras_sigmas       -> (hi^(1/ρ) + ramp·(lo^(1/ρ) - hi^(1/ρ)))^ρ，ramp=linspace(0,1,N)，ρ=7；
                                   hi/lo 默认取 in_sigmas[0] / in_sigmas[-1]，可被 sigma_max/sigma_min 覆盖；
                                   随后 timesteps = 把每个 σ 反查回时间轴（log-σ 线性插值）。
       use_exponential_sigmas  -> exp(linspace(log(hi), log(lo), N))，同样反查 timesteps。
       use_beta_sigmas         -> 上游依赖 scipy.stats.beta.ppf，本题不支持，为真即 ValueError。
   5) 终端 σ（final_sigmas_type）："zero" -> 0；"sigma_min" -> sqrt((1 - abar[0]) / abar[0])。
      追加到 sigmas 末尾（所以 sigmas 比 timesteps 多一个，log_linear 无转换器时多两个）。

    返回 (timesteps, sigmas)，均为 float32 张量。
    """
    # TODO
    raise NotImplementedError
