import torch


def solve(logp, logp_old, mask, adv, eps=3e-4):
    """GSPO 损失：**序列级**重要性比 s，整条回答共用一个 clip。

    入参：
    - logp / logp_old: (N,T) float32/float64，当前策略与行为策略的逐 token
      对数概率；N >= 1 条回答，T >= 1 是补齐后的宽度；
    - mask: (N,T) bool，有效 token 位置；每条回答至少一个 True；
    - adv: (N,) 与 logp 同 dtype，每条回答一个**序列级**优势（不再逐 token）；
    - eps: float ∈ (0,1)，序列级裁剪半径，GSPO 原论文取 3e-4。

    口径：
    - 序列对数比取**有效 token 上的平均**（长度归一）：
        log s_i = ( Σ_t mask·(logp - logp_old) ) / |o_i|，  s_i = exp(log s_i)
      注意是先平均再取 exp；不是 exp 之后平均，也不是逐 token 各算一个比。
    - 整条回答共用一个被裁剪的比：
        obj_i = min( s_i·A_i, clamp(s_i, 1-eps, 1+eps)·A_i )
    - 损失 = -mean_i obj_i（按回答平均，不再按 token 平均）。
    - 返回 dict：{'loss': 0 维 float64, 'ratio': (N,) float64}，ratio 就是 s，
      便于核对「token 级离群只推动 s 的 1/|o|」这一序列级平滑性质。

    非法输入抛 ValueError：非张量 / 非浮点 / 含 NaN·Inf / 形状不符 /
    N 或 T < 1 / mask 非 bool / 有空回答 / adv 形状或 dtype 不符 /
    eps 不在 (0,1)。CPU、确定性、不调用随机数。
    """
    raise NotImplementedError
