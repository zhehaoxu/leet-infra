def solve(model_output, sample, timestep, at, ct, att, ctt, num_vec_classes, gumbel_uniform):
    """VQ-Diffusion 的掩码离散反扩散单步（diffusers `VQDiffusionScheduler.step`）。

    前向过程把每个隐像素以 γ_t 的概率变成「掩码类」，且掩码是吸收态；本题实现反向
    的一步：给定网络对 x_0 的预测，算出 p(x_{t-1} | x_t) 并从中采样。

    输入
      model_output   (B, C-1, L) 浮点张量：x_0 每个隐像素属于前 C-1 个「已知类」的
                     log 概率。第 1 维故意少一个：x_0 是干净图，不可能是掩码类。
                     不要求归一化，缺的质量会原样带到输出上。
      sample         (B, L) 整数张量，取值 [0, C-1]：当前时刻 x_t 的类别索引。
                     掩码类固定为最后一个索引 C-1。
      timestep       Python int t，0 <= t < T（T = len(at)）。
      at, ct         (T,) 一维浮点张量：单步转移的 α_t（保持原类）与 γ_t（变成掩码）。
      att, ctt       (T+1,) 一维浮点张量：累积转移的 ᾱ_t 与 γ̄_t，按 t=0..T-1 使用。
                     下标 T 是永远读不到的哨兵（上游写 1.0 与 0.0），不参与校验。
      num_vec_classes  整数 C >= 2，类别总数（含掩码类）。
      gumbel_uniform  (B, C, L) 浮点张量、取值 [0, 1)：采样用的均匀随机数。
                     t == 0 时形状是 (B, C-1, L)，因为那一步压根没有掩码类。

    b 表不单独给，由 a、c 推出（k = C-1 是已知类个数）：
        b_t     = (1 - at     - ct     ) / k      行和为 1
        b̄_t     = (1 - att     - ctt     ) / k
    转移矩阵（对某个隐像素，观测值为 x_t = j）：
        q(x_t = j | x_0 = i)            = ᾱ_t + b̄_t  若 i == j，否则 b̄_t；j 为掩码时 = γ̄_t
        q(x_t = j | x_{t-1} = i)        = α_t + b_t   若 i == j，否则 b_t；  j 为掩码时 = γ_t
        q(x_t = j | x_{t-1} = 掩码)     = 1 若 j 为掩码，否则 0

    算法（全部在 log 空间，log(0) 用 log(clamp(0, min=1e-30)) = -69.07755278982137 代替）
      t == 0：log_p_x_t_min_1 = model_output（原样，不做任何变换）。
      t > 0：记 E[i] 为 sample 的 log one-hot（第 0 维是类别，命中为 0，否则为上述钳制值），
        V[i] = logaddexp(E[i] + log ᾱ_t, log b̄_t)，i = 0..C-2；若该像素 x_t 是掩码类，
               整列 V 改成 log γ̄_t
        U[i] = logaddexp(E[i] + log α_t,  log b_t)， i = 0..C-2；掩码像素同样整列改成 log γ_t
        U[C-1] = E[C-1]                        ← 掩码行要在把它丢掉之前保存下来
        q      = model_output - V              (B, C-1, L)
        S      = logsumexp(q, dim=1, keepdim=True)
        q      = q - S                         ← 归一化，末尾再把 S 加回去
        W[i]   = logaddexp(q[i] + log ᾱ_{t-1}, log b̄_{t-1})，i = 0..C-2   ← 注意下标 t-1
        W[C-1] = log γ̄_{t-1}                  （一个纯标量广播成 (B, 1, L)）
        log_p_x_t_min_1 = W + U + S            (B, C, L)
      采样：noise = -log(-log(gumbel_uniform + 1e-30) + 1e-30)
            prev_sample = argmax(log_p_x_t_min_1 + noise, dim=1)   （平票取最小下标）

    输出 (prev_sample, log_p_x_t_min_1)：
      prev_sample        int64 (B, L)，加噪后逐像素 argmax；t == 0 时不可能取到掩码类。
      log_p_x_t_min_1    float64，t > 0 时 (B, C, L)，t == 0 时 (B, C-1, L)。
                         逐像素 exp 之后对类别维求和恒等于该像素 model_output 的 exp 之和。

    入参约束（违反任一抛 ValueError）：model_output 必须三维、浮点、全有限，第 1 维等于
    C-1；sample 必须整数张量、形状 (B, L)、取值 [0, C-1]；at 与 ct 一维同长 T >= 2，
    元素落在 (0, 1) 且 at + ct < 1；att 与 ctt 长度为 T+1，前 T 个元素落在 (0, 1) 且
    和小于 1（末位哨兵不校验）；timestep 是 Python int（bool 非法）且 0 <= t < T；
    num_vec_classes 是 Python int 且 >= 2；gumbel_uniform 浮点、形状如上、取值 [0, 1)。
    内部计算在 float64 下进行。禁止 import diffusers。
    """
    # TODO
    raise NotImplementedError
