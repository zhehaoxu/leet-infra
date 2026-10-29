import torch


def solve(logp_w, logp_l, ref_logp_w, ref_logp_l, beta):
    """IPO 损失：把隐式奖励差压向目标边际 1/(2β) 的**平方**目标。

    入参（四条序列级对数概率，形状均为 (N,)，N >= 1，dtype 一致且为
    float32/float64）：
    - logp_w / logp_l: 当前策略下偏好 / 被拒回答的序列级对数概率；
    - ref_logp_w / ref_logp_l: 参考策略下的同两条对数概率；
    - beta: float > 0，温度参数。

    口径：
    - 隐式奖励差 Δ = (logp_w - ref_logp_w) - (logp_l - ref_logp_l)，
      逐样本计算，**不做长度归一**（入参已是序列级和）。
    - 损失 = mean( (Δ - 1/(2β))² )，返回 0 维 float64 标量。
    - 与 DPO 的 -logsigmoid(βΔ) 不同：IPO 用平方把 Δ 拉向确定的边际
      1/(2β)，避免 sigmoid 在「已分对」时梯度饱和、继续推大 Δ 导致过拟合；
      这里也**不**乘 β（β 只通过目标边际进入）。

    非法输入抛 ValueError：非张量 / 非一维 / 非浮点 / 形状或 dtype 不一致 /
    空样本 / 含 NaN·Inf / beta <= 0。CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
