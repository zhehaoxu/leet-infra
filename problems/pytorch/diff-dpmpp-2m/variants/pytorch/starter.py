def solve(x, x0_pred, alpha_bar_t, alpha_bar_s0, order=1, x0_pred_prev=None, alpha_bar_s1=None):
    """DPM-Solver++ 的单步更新（data-prediction 形式）。

    x:当前样本 x_{s0}；x0_pred:D0（在 s0 处的 x0 预测）；
    alpha_bar_t:下一步的 ᾱ；alpha_bar_s0:当前步的 ᾱ；
    order:1 = Euler 一阶，2 = 2M 二阶（需额外给 x0_pred_prev 与 alpha_bar_s1）。

    记 α = sqrt(ᾱ)、σ = sqrt(1-ᾱ)、λ = log(α/σ)：
        h  = λ_t - λ_s0
        r0 = λ_s0 - λ_s1
        D1 = (D0 - D1_prev) / r0                  （r > 0：s1 比 s0 更噪）
        x_t = (σ_t/σ_s0)·x - α_t·(e^{-h} - 1)·(D0 + [order==2: 0.5·D1])

    ᾱ 必须严格落在 (0, 1) 内；order=2 时缺 x0_pred_prev / alpha_bar_s1 → ValueError。
    返回与 x 同形状的 float32 张量（内部 float64）。
    """
    # TODO
    raise NotImplementedError
