import torch


def solve(logp_w, logp_l, ref_logp_w, ref_logp_l, beta, eps):
    """cDPO：按噪声翻转率 eps 做标签平滑的偏好损失。

    入参（四条序列级对数概率，形状均为 (N,)，N >= 1，dtype 一致且为
    float32/float64）：
    - logp_w / logp_l: 当前策略下偏好 / 被拒回答的序列级对数概率；
    - ref_logp_w / ref_logp_l: 参考策略下的同两条对数概率；
    - beta: float > 0，DPO 温度；
    - eps: float，偏好标签被翻转的概率，必须落在 [0, 0.5]。

    口径：
    - 隐式奖励差 Δ = (logp_w - ref_logp_w) - (logp_l - ref_logp_l)；
      logit = β·Δ。
    - 把 DPO 的硬标签 {1, 0} 换成软标签 {1-eps, eps} 后取交叉熵：
        L = -mean( (1-eps)·logsigmoid(logit) + eps·logsigmoid(-logit) )
      等价于 -mean( (1-eps)·log σ + eps·log(1-σ) )，必须用
      `logsigmoid(±logit)` 的稳定式，不许先算 σ 再取 log。
    - eps = 0 时严格退化为 DPO；logit = 0 时无论 eps 取值损失都等于 ln2。
    - 返回 0 维 float64 标量。

    非法输入抛 ValueError：非张量 / 非一维 / 非浮点 / 形状或 dtype 不一致 /
    空样本 / 含 NaN·Inf / beta <= 0 / eps 不在 [0, 0.5]。
    CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
