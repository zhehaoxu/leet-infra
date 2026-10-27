def solve(x_t, alpha_bar, pred, mode):
    """在三种预测目标之间互转。

    x_t:当前带噪样本；alpha_bar:当前步的 ᾱ（标量，0 < ᾱ < 1，mode="eps" 允许 1）；
    pred:模型给出的那个预测量；mode ∈ {"eps", "x0", "v"}。

    定义（α = sqrt(ᾱ)、σ = sqrt(1 - ᾱ)）：
        x_t = α·x0 + σ·eps
        v   = α·eps - σ·x0

    已知 eps：x0 = (x_t - σ·eps) / α，v = α·eps - σ·x0；
    已知 x0 ：eps = (x_t - α·x0) / σ，v = α·eps - σ·x0；
    已知 v  ：eps = α·v + σ·x_t，x0 = α·x_t - σ·v。

    返回 (x0, eps, v) 三个同形状 float32 张量（内部 float64）。
    """
    # TODO
    raise NotImplementedError
