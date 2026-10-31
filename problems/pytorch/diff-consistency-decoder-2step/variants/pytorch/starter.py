def solve(
    x_init,
    f_first,
    f_second,
    noise,
    num_train_timesteps,
    alpha_transform="cosine",
    sigma_data=0.5,
    timesteps=(1008, 512),
    max_beta=0.999,
):
    """Consistency Decoder 的两步采样：从 betas_for_alpha_bar 一路做到重加噪。

    上游是 diffusers 的 ConsistencyDecoderScheduler：一张训练期调度表导出
    c_skip / c_out / c_in 三组系数，两步「去噪 -> 重加噪 -> 再去噪」就把噪声解成图。
    本题把网络输出直接当输入，只考这套系数与两步更新。

    入参：
      x_init             —— 首步样本 x_{t0}，浮点张量，第 0 维是 batch，必须全为有限值；
      f_first            —— 网络在 t0 处的输出 F1，与 x_init 同形状；
      f_second           —— 网络在 t1 处的输出 F2，与 x_init 同形状；
      noise              —— 重加噪用的标准正态样本 z，与 x_init 同形状（不内部采样）；
      num_train_timesteps—— 调度表长度 T（Python int，>= 2）；
      alpha_transform    —— "cosine" / "laplace" / "exp" 三选一；
      sigma_data         —— 数据分布标准差 σ_d > 0；
      timesteps          —— 恰好两个 Python int (t0, t1)，都落在 [0, T) 且 t0 > t1；
      max_beta           —— β 的上限，落在 (0, 1]。

    调度表（中间量一律 float64）：

      变换 ab(t)：
        cosine  ab(t) = cos((t + 0.008) / 1.008 * pi / 2) ** 2
        laplace lmb = -0.5 * copysign(1, 0.5 - t) * log(1 - 2 * abs(0.5 - t) + 1e-6)
                ab(t) = sqrt(exp(lmb) / (1 + exp(lmb)))
        exp     ab(t) = exp(-12 * t)

      beta_i = min(1 - ab((i + 1) / T) / ab(i / T), max_beta)   (i = 0 .. T-1)
      abar_i = prod_{j <= i} (1 - beta_j)
      sigma_i = sqrt(1 / abar_i - 1)
      c_skip_i = abar_i ** -0.5 * sigma_d**2 / (sigma_i**2 + sigma_d**2)
      c_out_i  = sigma_i * sigma_d / sqrt(sigma_i**2 + sigma_d**2)
      c_in_i   = abar_i ** -0.5 / sqrt(sigma_i**2 + sigma_d**2)

      注意 c_skip 与 c_in 用的是 abar 的**倒数**平方根，不是 sqrt(abar)。

    两步更新：
      scaled_first  = x_init * c_in[t0]
      x0_first      = c_out[t0] * F1 + c_skip[t0] * x_init
      x_second      = sqrt(abar[t1]) * x0_first + sqrt(1 - abar[t1]) * z
      scaled_second = x_second * c_in[t1]
      x_final       = c_out[t1] * F2 + c_skip[t1] * x_second

    重加噪用的是**下一步** t1 的 sqrt(abar) / sqrt(1 - abar)，不是当前步的；末步不加噪，
    x_final 就是 x0 预测。带 * 的 scaled_* 是喂给网络前的输入（x0_first 却用未缩放的 x_init）。

    返回四元组 (scaled_first, x_second, scaled_second, x_final)，都是与 x_init 同形状的
    float32 张量。

    非法输入一律抛 ValueError：非张量、形状不符、含 NaN/inf、0 维张量、T < 2、
    变换名不认识、timesteps 不是长度 2 的整数序列 / 越界 / 不严格递减、σ_d <= 0、
    max_beta 不在 (0, 1]、用 bool 冒充整数；另外若某一步的 abar 落到 0（beta 触顶导致
    调度表退化，sigma 会发散），同样抛 ValueError。
    禁止 import diffusers。
    """
    # TODO
    raise NotImplementedError
