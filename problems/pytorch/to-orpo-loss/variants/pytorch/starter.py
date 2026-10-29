import torch


def solve(logp_w, logp_l, len_w, len_l, lam=0.1):
    """ORPO 损失（无参考模型）：SFT 项 + λ·赔率项。

    入参（全部长度 N >= 1）：
    - logp_w: (N,) float32/float64，偏好回答 y_w 的**序列级**对数概率
      log π(y_w|x)（未归一）。
    - logp_l: (N,) 同形状同 dtype，被拒回答 y_l 的序列级对数概率。
    - len_w: (N,) int32/int64，|y_w| >= 1；len_l 同理，二者 dtype 一致。
    - lam: float >= 0，赔率项权重 λ（ORPO 默认 0.1）。

    口径（两道归一化必须先做，再做 odds）：
    - 平均对数概率 lp_w = logp_w / len_w，lp_l = logp_l / len_l，
      必须严格 < 0（否则 p = exp(lp) 不是合法概率、log(1-p) 无定义）；
      任一 >= 0 抛 ValueError。
    - 赔率对数 logodds(lp) = lp - log1p(-exp(lp)) = log(p/(1-p))，
      必须用 log1p 稳定式，不许先算 p 再 log(p/(1-p))。
    - SFT 项 L_SFT = -mean(lp_w)（只惩罚**偏好**回答的 NLL）。
    - 赔率项 L_OR = -mean(logsigmoid(logodds(lp_w) - logodds(lp_l)))。
    - 总损失 = L_SFT + lam * L_OR，返回 0 维 float64 标量。

    非法输入抛 ValueError：非张量 / 非一维 / 非浮点 logp / 非整型 len /
    形状不一致 / len < 1 / 含 NaN·Inf / lam < 0 / 平均对数概率 >= 0。
    CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
